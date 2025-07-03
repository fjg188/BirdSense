import argparse
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets.folder import default_loader
import timm

def get_args():
    p = argparse.ArgumentParser(description="HERBS‑style fine‑tuning with Swin‑T")
    # data
    p.add_argument('--train_root', type=Path, default='./CUB200-2011/train')
    p.add_argument('--val_root',   type=Path, default='./CUB200-2011/test')
    p.add_argument('--data_size',  type=int,  default=384)
    p.add_argument('--num_workers',type=int,  default=2)
    # model / optim
    p.add_argument('--bs',      type=int, default=8, help='physical batch size')
    p.add_argument('--accum',   type=int, default=4, help='grad accumulation steps')
    p.add_argument('--epochs',  type=int, default=80)
    p.add_argument('--lr',      type=float, default=5e-4)
    p.add_argument('--wd',      type=float, default=3e-4)
    p.add_argument('--warmup_steps', type=int, default=1500)
    # misc
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--save',   type=Path, default='best_swin_herbs.pth')
    return p.parse_args()

class CUB(Dataset):
    """CUB‑200‑2011 """

    def __init__(self, root: Path, train: bool = True, size: int = 384):
        self.paths  = sorted([p for p in Path(root).rglob('*.jpg')])
        self.train  = train
        norm = T.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
        if train:
            self.tf = T.Compose([
                T.Resize(int(size*510/384)),
                T.RandomCrop(size),
                T.RandomHorizontalFlip(),
                T.GaussianBlur(3),
                T.RandAugment(2, 9),
                T.ToTensor(),
                norm,
            ])
        else:
            self.tf = T.Compose([
                T.Resize(int(size*510/384)),
                T.CenterCrop(size),
                T.ToTensor(),
                norm,
            ])
        # labels from directory names
        self.cls2idx = {cls:i for i, cls in enumerate(sorted({p.parent.name for p in self.paths}))}

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        img  = default_loader(path)
        y    = self.cls2idx[path.parent.name]
        return self.tf(img), y


class BSHead(nn.Module):
    def __init__(self, in_ch: int, n_cls: int, K: int = 64,
                 lamb_m: float = 1.0, lamb_d: float = 5.0, lamb_l: float = 0.3):
        super().__init__()
        self.cls = nn.Conv2d(in_ch, n_cls, 1)
        self.K = K; self.lm = lamb_m; self.ld = lamb_d; self.ll = lamb_l

    def forward(self, feat, target=None):
        # feature maps → per‑pixel logits
        logits_map = self.cls(feat)                # B×C×H×W
        B,C,H,W = logits_map.shape
        # Foreground merge (top‑K)
        topk = logits_map.flatten(2).topk(self.K, dim=2).values.mean(-1)  # B×C
        logits = topk
        bs_loss = torch.tensor(0., device=logits.device)
        if self.training and target is not None:
            loss_m = F.cross_entropy(logits, target)
            dropped = torch.tanh(logits_map)
            loss_d = ((dropped + 1) ** 2).mean()
            bs_loss = self.lm*loss_m + self.ld*loss_d  # single‑layer ⇒ no layer‑logits loss
        return logits, bs_loss


class SwinBS(nn.Module):
    def __init__(self, num_classes: int = 200):
        super().__init__()
        self.backbone = timm.create_model('swin_tiny_patch4_window7_224', pretrained=True, num_classes=0)
        self.bs       = BSHead(in_ch=768, n_cls=num_classes)  # Swin‑T last stage = 768 ch

    def forward(self, x, y=None):
        feat = self.backbone.forward_features(x)   # B×768×(H/32)×(W/32)  (12×12 for 384²)
        logits, bs_loss = self.bs(feat, y)
        return logits, bs_loss


def cosine_with_warmup(step, total, warmup):
    if step < warmup:
        return step / warmup
    pct = (step - warmup) / (total - warmup)
    return 0.5 * (1 + torch.cos(torch.tensor(pct * 3.1415926535)))

def run_epoch(model, loader, optim, scaler, device, train=True, accum=1, sch_fn=None):
    if train:
        model.train()
    else:
        model.eval()
    loss_tot = acc_tot = n = 0
    optim.zero_grad()
    for step, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            out, bs_loss = model(x, y if train else None)
            ce = F.cross_entropy(out, y, label_smoothing=0.1)
            loss = ce + bs_loss
            acc = (out.argmax(1)==y).float().mean().item()
        if train:
            if scaler is not None:
                scaler.scale(loss/accum).backward()
            else:
                (loss/accum).backward()
            if (step+1) % accum == 0:
                if scaler is not None:
                    scaler.step(optim)
                    scaler.update()
                else:
                    optim.step()
                optim.zero_grad()
                if sch_fn is not None:
                    sch_fn()
        loss_tot += loss.item()*x.size(0)
        acc_tot  += acc*x.size(0)
        n += x.size(0)
    return loss_tot/n, acc_tot/n


def main():
    args = get_args()
    device = torch.device(args.device)
    torch.backends.cudnn.benchmark = True

    # data
    tr_ds = CUB(args.train_root, train=True,  size=args.data_size)
    va_ds = CUB(args.val_root,   train=False, size=args.data_size)
    tr_dl = DataLoader(tr_ds, batch_size=args.bs, shuffle=True,  num_workers=args.num_workers, pin_memory=True)
    va_dl = DataLoader(va_ds, batch_size=args.bs*2, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # model
    model = SwinBS().to(device)

    # optim + scaler
    optim = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd)
    scaler = torch.cuda.amp.GradScaler(init_scale=1024.) if torch.cuda.is_available() else None

    # LR schedule with warm‑up
    total_steps = args.epochs * len(tr_dl) // args.accum
    global_step = 0
    def lr_sched():
        nonlocal global_step
        global_step += 1
        lr_scale = cosine_with_warmup(global_step, total_steps, args.warmup_steps)
        for pg in optim.param_groups:
            pg['lr'] = lr_scale * args.lr

    best = 0.
    for epoch in range(1, args.epochs+1):
        tr_loss, tr_acc = run_epoch(model, tr_dl, optim, scaler, device, True, args.accum, lr_sched)
        va_loss, va_acc = run_epoch(model, va_dl, optim, scaler, device, False)
        best = max(best, va_acc)
        print(f"{epoch:02d}/{args.epochs}  train {tr_acc:.3f}  val {va_acc:.3f}  best {best:.3f}")
        if va_acc >= best:
            torch.save(model.state_dict(), args.save)

if __name__ == '__main__':
    main()
