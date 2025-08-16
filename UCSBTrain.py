import argparse
import math
from pathlib import Path

import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
import timm


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_root', type=Path, default=Path(__file__).parent / 'CUB_200_2011')
    p.add_argument('--data_size', type=int, default=384)
    p.add_argument('--num_workers', type=int, default=2)
    p.add_argument('--bs', type=int, default=4)
    p.add_argument('--accum', type=int, default=8)
    p.add_argument('--epochs', type=int, default=80)
    p.add_argument('--lr', type=float, default=5e-4)
    p.add_argument('--wd', type=float, default=3e-4)
    p.add_argument('--warmup_steps', type=int, default=1500)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--save', type=Path, default='best_swin_cub.pth')
    p.add_argument('--log_dir', type=Path, default=Path('runs') / 'swin_cub')
    return p.parse_args()


class CUB(Dataset):
    def __init__(self, df, images_dir: Path, size: int = 384, train: bool = True):
        self.df = df.reset_index(drop=True)
        self.images_dir = images_dir
        norm = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        if train:
            self.tf = T.Compose([
                T.RandomResizedCrop(size, scale=(0.5, 1.0)),
                T.RandAugment(2, 9),
                T.ToTensor(),
                norm,
                T.RandomErasing(p=0.25),
            ])
        else:
            self.tf = T.Compose([
                T.Resize((size, size)),
                T.ToTensor(),
                norm,
            ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        r = self.df.iloc[idx]
        img = Image.open(self.images_dir / r.path).convert('RGB')
        return self.tf(img), int(r.class_id) - 1


class BSHead(nn.Module):
    def __init__(self, in_ch: int, n_cls: int, K: int = 64, lm: float = 1.0, ld: float = 5.0):
        super().__init__()
        self.cls = nn.Conv2d(in_ch, n_cls, 1)
        self.K = K
        self.lm = lm
        self.ld = ld

    def forward(self, feat, target=None):
        logits_map = self.cls(feat)
        topk = logits_map.flatten(2).topk(self.K, dim=2).values.mean(-1)
        logits = topk
        loss = torch.zeros((), device=feat.device)
        if self.training and target is not None:
            loss_m = F.cross_entropy(logits, target)
            loss_d = ((torch.tanh(logits_map) + 1) ** 2).mean()
            loss = self.lm * loss_m + self.ld * loss_d
        return logits, loss


class SwinBS(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.backbone = timm.create_model('swin_base_patch4_window12_384', pretrained=True, num_classes=0)
        self.bs = BSHead(1024, num_classes)

    def forward(self, x, y=None):
        feat = self.backbone.forward_features(x)
        feat = feat.permute(0, 3, 1, 2).contiguous()
        return self.bs(feat, y)


def lr_scale(step: int, total: int, warmup: int):
    if step < warmup:
        return step / warmup
    pct = (step - warmup) / (total - warmup)
    return 0.5 * (1 + math.cos(math.pi * pct))


def run_epoch(model, loader, optim, scaler, device, train: bool, accum: int, schedule_fn=None):
    model.train() if train else model.eval()
    loss_sum = acc_sum = num = 0
    optim.zero_grad(set_to_none=True)
    last_step = len(loader) - 1
    for step, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            logits, bs_loss = model(x, y if train else None)
            ce = F.cross_entropy(logits, y, label_smoothing=0.1)
            loss = ce + bs_loss
        acc = (logits.argmax(1) == y).float().mean().item()
        loss_sum += loss.item() * x.size(0)
        acc_sum += acc * x.size(0)
        num += x.size(0)
        if train:
            scaled_loss = scaler.scale(loss / accum) if scaler else loss / accum
            scaled_loss.backward()
            if (step + 1) % accum == 0 or step == last_step:
                if scaler:
                    scaler.step(optim)
                    scaler.update()
                else:
                    optim.step()
                optim.zero_grad(set_to_none=True)
                if schedule_fn is not None:
                    schedule_fn()
    return loss_sum / num, acc_sum / num


def load_cub_dataframe(root: Path):
    images = pd.read_csv(root / 'images.txt', sep=' ', names=['id', 'path'])
    labels = pd.read_csv(root / 'image_class_labels.txt', sep=' ', names=['id', 'class_id'])
    split = pd.read_csv(root / 'train_test_split.txt', sep=' ', names=['id', 'is_train'])
    boxes = pd.read_csv(root / 'bounding_boxes.txt', sep=' ', names=['id', 'x', 'y', 'bb_width', 'bb_height'])
    return images.merge(labels).merge(split).merge(boxes)


def main():
    args = get_args()
    device = torch.device(args.device)
    torch.backends.cudnn.benchmark = True
    args.log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(args.log_dir))

    df = load_cub_dataframe(args.data_root)
    train_df = df[df.is_train == 1]
    val_df = df[df.is_train == 0]

    img_dir = args.data_root / 'images'
    train_ds = CUB(train_df, img_dir, args.data_size, train=True)
    val_ds = CUB(val_df, img_dir, args.data_size, train=False)

    train_dl = DataLoader(train_ds, batch_size=args.bs, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=args.bs * 2, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    model = SwinBS(df.class_id.nunique()).to(device)

    optim = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd)
    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None

    total_opt_steps = args.epochs * math.ceil(len(train_dl) / args.accum)
    current_step = 0

    def scheduler():
        nonlocal current_step
        current_step += 1
        lr = lr_scale(current_step, total_opt_steps, args.warmup_steps) * args.lr
        for pg in optim.param_groups:
            pg['lr'] = lr

    best_val = 0.0
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = run_epoch(model, train_dl, optim, scaler, device, True, args.accum, scheduler)
        val_loss, val_acc = run_epoch(model, val_dl, optim, scaler, device, False, args.accum)
        writer.add_scalar('Loss/train', tr_loss, epoch)
        writer.add_scalar('Acc/train', tr_acc, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Acc/val', val_acc, epoch)
        writer.add_scalar('LR', optim.param_groups[0]['lr'], epoch)
        if val_acc > best_val:
            best_val = val_acc
            torch.save(model.state_dict(), args.save)
        print(f"{epoch:02d}/{args.epochs} tr_acc {tr_acc:.3f} val_acc {val_acc:.3f} best {best_val:.3f}")

    writer.close()


if __name__ == '__main__':
    main()
