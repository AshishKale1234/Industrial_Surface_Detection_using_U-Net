"""
Industrial Surface Defect Segmentation using U-Net
Trains one U-Net model per MVTec AD category.

Usage:
    python train_all.py --data_root ./data --epochs 50
    python train_all.py --data_root ./data --category leather --epochs 50
"""

import argparse
import torch
import sys
sys.path.append('./src')

from dataset import get_dataloaders
from model   import UNet
from train   import train

CATEGORIES = [
    'bottle', 'cable', 'capsule', 'carpet', 'grid',
    'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
    'tile', 'toothbrush', 'transistor', 'wood', 'zipper'
]

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    categories = [args.category] if args.category else CATEGORIES
    results = {}

    for category in categories:
        print(f'\n{"="*50}\nTraining: {category}\n{"="*50}')

        model = UNet(in_channels=3, out_channels=1).to(device)
        train_loader, val_loader = get_dataloaders(
            data_root  = args.data_root,
            category   = category,
            img_size   = args.img_size,
            batch_size = args.batch_size,
        )

        config = {
            'device'         : device,
            'epochs'         : args.epochs,
            'lr'             : args.lr,
            'checkpoint_path': f'outputs/checkpoints/best_{category}.pth',
        }

        history   = train(model, train_loader, val_loader, config)
        best_dice = max(history['val_dice'])
        best_iou  = history['val_iou'][history['val_dice'].index(best_dice)]
        results[category] = {'dice': best_dice, 'iou': best_iou}

    print(f'\n{"Category":<15} {"Dice":>8} {"IoU":>8}')
    print('-' * 33)
    for cat, r in results.items():
        print(f'{cat:<15} {r["dice"]:>8.4f} {r["iou"]:>8.4f}')
    if len(results) > 1:
        avg_d = sum(r['dice'] for r in results.values()) / len(results)
        avg_i = sum(r['iou']  for r in results.values()) / len(results)
        print('-' * 33)
        print(f'{"Average":<15} {avg_d:>8.4f} {avg_i:>8.4f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root',  default='./data')
    parser.add_argument('--category',   default=None,
                        help='Single category. Omit to train all 15.')
    parser.add_argument('--epochs',     type=int,   default=50)
    parser.add_argument('--batch_size', type=int,   default=8)
    parser.add_argument('--img_size',   type=int,   default=256)
    parser.add_argument('--lr',         type=float, default=1e-3)
    args = parser.parse_args()
    main(args)
