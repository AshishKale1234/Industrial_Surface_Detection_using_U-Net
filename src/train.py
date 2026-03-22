
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import sys, os

sys.path.append(os.path.dirname(__file__))
from losses import BCEDiceLoss, dice_score, iou_score


def train_one_epoch(model, loader, optimizer, criterion, device):
    """Single training epoch — forward pass, loss, backward, update."""
    model.train()
    total_loss, total_dice, total_iou = 0, 0, 0

    loop = tqdm(loader, desc="  Train", leave=False)
    for images, masks in loop:
        images = images.to(device)
        masks  = masks.to(device)

        # ── Forward pass ────────────────────────────────────────────────────
        preds = model(images)

        # ── Loss ────────────────────────────────────────────────────────────
        loss = criterion(preds, masks)

        # ── Backward pass ────────────────────────────────────────────────────
        optimizer.zero_grad()   # clear gradients from previous step
        loss.backward()         # compute gradients via backprop
        optimizer.step()        # update weights

        # ── Metrics ──────────────────────────────────────────────────────────
        total_loss += loss.item()
        total_dice += dice_score(preds.detach(), masks)
        total_iou  += iou_score(preds.detach(), masks)

        loop.set_postfix(loss=f"{loss.item():.4f}")

    n = len(loader)
    return total_loss/n, total_dice/n, total_iou/n


@torch.no_grad()
def val_one_epoch(model, loader, criterion, device):
    """Validation epoch — no gradients, no weight updates."""
    model.eval()
    total_loss, total_dice, total_iou = 0, 0, 0

    for images, masks in loader:
        images = images.to(device)
        masks  = masks.to(device)

        preds = model(images)
        loss  = criterion(preds, masks)

        total_loss += loss.item()
        total_dice += dice_score(preds, masks)
        total_iou  += iou_score(preds, masks)

    n = len(loader)
    return total_loss/n, total_dice/n, total_iou/n


def train(model, train_loader, val_loader, config):
    """
    Full training loop with:
    - ReduceLROnPlateau scheduler (halves LR when val loss plateaus)
    - Best model checkpointing (saves whenever val Dice improves)
    - Per-epoch logging
    """
    device    = config['device']
    epochs    = config['epochs']
    ckpt_path = config['checkpoint_path']

    criterion = BCEDiceLoss(bce_weight=0.5, dice_weight=0.5)
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    scheduler = ReduceLROnPlateau(optimizer, mode='min',
                                  factor=0.5, patience=5)

    best_dice = 0.0
    history   = {'train_loss':[], 'val_loss':[],
                  'train_dice':[], 'val_dice':[],
                  'train_iou':[],  'val_iou':[]}

    print(f"\nTraining on {device} for {epochs} epochs\n")
    print(f"{'Epoch':>6} {'T-Loss':>8} {'V-Loss':>8} "
          f"{'T-Dice':>8} {'V-Dice':>8} {'V-IoU':>7}  LR")
    print("─" * 62)

    for epoch in range(1, epochs + 1):
        # ── Train ────────────────────────────────────────────────────────────
        t_loss, t_dice, t_iou = train_one_epoch(
            model, train_loader, optimizer, criterion, device)

        # ── Validate ─────────────────────────────────────────────────────────
        v_loss, v_dice, v_iou = val_one_epoch(
            model, val_loader, criterion, device)

        # ── Scheduler step ───────────────────────────────────────────────────
        scheduler.step(v_loss)
        current_lr = optimizer.param_groups[0]['lr']

        # ── Log ──────────────────────────────────────────────────────────────
        history['train_loss'].append(t_loss)
        history['val_loss'].append(v_loss)
        history['train_dice'].append(t_dice)
        history['val_dice'].append(v_dice)
        history['train_iou'].append(t_iou)
        history['val_iou'].append(v_iou)

        print(f"{epoch:>6} {t_loss:>8.4f} {v_loss:>8.4f} "
              f"{t_dice:>8.4f} {v_dice:>8.4f} {v_iou:>7.4f}  {current_lr:.0e}")

        # ── Save best checkpoint ─────────────────────────────────────────────
        if v_dice > best_dice:
            best_dice = v_dice
            torch.save({
                'epoch'     : epoch,
                'model_state': model.state_dict(),
                'optim_state': optimizer.state_dict(),
                'val_dice'  : v_dice,
                'val_iou'   : v_iou,
            }, ckpt_path)
            print(f"         ✓ saved checkpoint  (dice={v_dice:.4f})")

    print(f"\nBest val Dice: {best_dice:.4f}")
    return history
