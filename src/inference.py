
import cv2
import torch
import numpy as np
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_inference_transform(img_size=256):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def load_model(checkpoint_path, device):
    """Load trained U-Net from checkpoint."""
    import sys, os
    sys.path.append(os.path.join(os.path.dirname(__file__)))
    from model import UNet

    model = UNet(in_channels=3, out_channels=1).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    print(f"✓ Loaded checkpoint  "
          f"(epoch {checkpoint['epoch']}, "
          f"val Dice {checkpoint['val_dice']:.4f})")
    return model


@torch.no_grad()
def predict_mask(model, image_path, device, img_size=256, threshold=0.5):
    """
    Run inference on a single image.
    Returns:
        original  : np.array [H, W, 3]  original image resized to img_size
        pred_mask : np.array [H, W]     binary mask (0 or 1)
        prob_map  : np.array [H, W]     raw probability map (0.0 to 1.0)
    """
    transform = get_inference_transform(img_size)

    image = cv2.imread(str(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Keep a clean copy for visualization
    original = cv2.resize(image, (img_size, img_size))

    # Transform and add batch dimension
    tensor = transform(image=image)['image']
    tensor = tensor.unsqueeze(0).to(device)     # [1, 3, H, W]

    prob_map  = model(tensor).squeeze().cpu().numpy()   # [H, W]
    pred_mask = (prob_map > threshold).astype(np.uint8) # binary

    return original, pred_mask, prob_map


def overlay_mask(original, pred_mask, gt_mask=None,
                 alpha=0.4, defect_color=(220, 50, 50)):
    """
    Draws defect overlay on the original image using OpenCV.

    Args:
        original     : RGB image [H, W, 3]
        pred_mask    : binary prediction [H, W]
        gt_mask      : binary ground truth [H, W] — optional
        alpha        : overlay transparency (0=invisible, 1=solid)
        defect_color : RGB color for predicted defect region

    Returns:
        overlay : BGR image ready for cv2.imwrite or display
    """
    img_bgr = cv2.cvtColor(original, cv2.COLOR_RGB2BGR)
    overlay = img_bgr.copy()

    # ── Predicted mask — filled color region ─────────────────────────────────
    color_layer = np.zeros_like(img_bgr)
    color_layer[pred_mask == 1] = defect_color[::-1]   # RGB → BGR
    overlay = cv2.addWeighted(overlay, 1-alpha,
                               color_layer, alpha, 0)

    # ── Predicted contour — sharp boundary line ───────────────────────────────
    contours, _ = cv2.findContours(pred_mask,
                                    cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (50, 50, 220), 2)  # BGR

    # ── Ground truth contour — green dashed outline ───────────────────────────
    if gt_mask is not None:
        gt_uint8 = gt_mask.astype(np.uint8)
        gt_contours, _ = cv2.findContours(gt_uint8,
                                           cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, gt_contours, -1, (50, 200, 50), 2)

    return overlay


def run_inference_batch(model, data_root, category,
                        output_dir, device, n_samples=8,
                        img_size=256, threshold=0.5):
    """
    Runs inference on n_samples defect images and saves
    side-by-side visualizations: original | overlay | prob map
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    test_root = Path(data_root) / category / 'test'
    gt_root   = Path(data_root) / category / 'ground_truth'

    # Collect defect images only (skip good/)
    defect_imgs = []
    for defect_type in sorted(test_root.iterdir()):
        if defect_type.name == 'good':
            continue
        for img_path in sorted(defect_type.glob('*.png')):
            mask_path = (gt_root / defect_type.name /
                         (img_path.stem + '_mask.png'))
            defect_imgs.append((img_path, mask_path, defect_type.name))

    defect_imgs = defect_imgs[:n_samples]
    print(f"Running inference on {len(defect_imgs)} images...")

    for idx, (img_path, mask_path, defect_type) in enumerate(defect_imgs):
        original, pred_mask, prob_map = predict_mask(
            model, img_path, device, img_size, threshold)

        # Load ground truth mask
        gt_mask = None
        if mask_path.exists():
            gt = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            gt = cv2.resize(gt, (img_size, img_size))
            gt_mask = (gt > 127).astype(np.uint8)

        # Build overlay
        overlay = overlay_mask(original, pred_mask, gt_mask)

        # Build prob map visualization (heatmap)
        prob_vis = (prob_map * 255).astype(np.uint8)
        prob_vis = cv2.applyColorMap(prob_vis, cv2.COLORMAP_HOT)

        # Stack: original | overlay | probability heatmap
        orig_bgr = cv2.cvtColor(original, cv2.COLOR_RGB2BGR)

        # Add labels
        for panel, label in zip(
            [orig_bgr, overlay, prob_vis],
            ['Original', 'Prediction (red) / GT (green)', 'Probability map']
        ):
            cv2.putText(panel, label, (8, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
            cv2.putText(panel, defect_type, (8, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200,200,200), 1)

        combined = np.hstack([orig_bgr, overlay, prob_vis])

        save_path = output_dir / f"{idx:02d}_{defect_type}_{img_path.stem}.png"
        cv2.imwrite(str(save_path), combined)

    print(f"✓ Saved {len(defect_imgs)} visualizations to {output_dir}")
    return output_dir
