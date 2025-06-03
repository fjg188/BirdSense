"""Train TinyViT-21m on CUB‑200‑2011."""

import os
from pathlib import Path
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

import timm

# paths
DATA_ROOT    = Path("./CUB_200_2011")
IMAGES_FILE  = DATA_ROOT / "images.txt"
SPLIT_FILE   = DATA_ROOT / "train_test_split.txt"
BBOX_FILE    = DATA_ROOT / "bounding_boxes.txt"
IMAGES_DIR   = DATA_ROOT / "images"

# hyper‑params
IMG_SIZE     = 384
BATCH        = 32
N_WORKERS    = 4
SMOOTH       = 0.1

FREEZE_EPOCH = 5
PART_EPOCH   = 20
FULL_EPOCH   = 10

class CUB(Dataset):
    def __init__(self, df, img_dir, size, train=True):
        self.df, self.dir, self.size, self.train = df.reset_index(drop=True), img_dir, size, train
        if train:
            self.tf = transforms.Compose([
                transforms.RandomResizedCrop(size, scale=(0.5,1.0)),
                transforms.RandAugment(num_ops=2, magnitude=9),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                transforms.RandomErasing(p=0.25)
            ])

        else:
            self.tf = transforms.Compose([
                transforms.Resize((size,size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
            ])
    def __len__(self):
        return len(self.df)
    def __getitem__(self, i):
        r   = self.df.iloc[i]
        img = Image.open(self.dir / r.path).convert("RGB")
        if not pd.isna(r.x):
            x,y,w,h = map(int,[r.x,r.y,r.bb_width,r.bb_height])
            img = img.crop((x,y,x+w,y+h))
        return self.tf(img), int(r.class_id)-1

def freeze_bn(m):
    for mod in m.modules():
        if isinstance(mod, nn.BatchNorm2d):
            mod.eval()
            for p in mod.parameters():
                p.requires_grad_(False)

def acc(out, tgt):
    return (out.argmax(1)==tgt).float().mean()

def run_epoch(model, loader, crit, opt, train, dev):
    if train:
        model.train()
    else:
        model.eval()
    loss_tot = correct = n = 0
    with torch.set_grad_enabled(train):
        for x,y in loader:
            x,y = x.to(dev), y.to(dev)
            if train: opt.zero_grad()
            out = model(x)
            loss = crit(out,y)
            if train:
                loss.backward(); opt.step()
            bs = x.size(0)
            loss_tot += loss.item()*bs
            correct  += (out.argmax(1)==y).sum().item()
            n += bs
    return loss_tot/n, correct/n

def main():
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # data
    df   = (pd.read_csv(IMAGES_FILE,sep=' ',header=None,names=['id','path'])
            .merge(pd.read_csv(SPLIT_FILE,sep=' ',header=None,names=['id','is_train']))
            .merge(pd.read_csv(BBOX_FILE,sep=' ',header=None,
                               names=['id','x','y','bb_width','bb_height'])))
    train_df, val_df = df[df.is_train==1], df[df.is_train==0]
    train_dl = DataLoader(CUB(train_df, IMAGES_DIR, IMG_SIZE, True),
                          batch_size=BATCH, shuffle=True, num_workers=N_WORKERS, pin_memory=True)
    val_dl   = DataLoader(CUB(val_df,   IMAGES_DIR, IMG_SIZE, False),
                          batch_size=BATCH, shuffle=False, num_workers=N_WORKERS, pin_memory=True)
    # model
    model = timm.create_model('tiny_vit_21m_384', pretrained=True, num_classes=200).to(dev)
    crit  = nn.CrossEntropyLoss(label_smoothing=SMOOTH)
    writer= SummaryWriter('runs/tinyvit')
    best  = 0

    def stage(epochs, unfreeze):
        if unfreeze=='head':
            for n,p in model.named_parameters():
                p.requires_grad_(n.startswith('head'))
            freeze_bn(model)
            opt = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=3e-4, weight_decay=1e-2,)
            total_steps = len(train_dl) * epochs
            sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-4, total_steps=total_steps, pct_start=0.3,
                                                      div_factor=25)
        elif unfreeze=='partial':
            for n,p in model.named_parameters():
                p.requires_grad_(('blocks.9' in n) or n.startswith('head'))
            opt = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4, weight_decay=5e-3)
            sch = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=max(epochs // 2, 1), T_mult=1)
        else:
            for p in model.parameters():
                p.requires_grad_(True)
            opt = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=3e-5, weight_decay=1e-4)
            sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=len(train_dl) * epochs, eta_min=1e-6)
        nonlocal best

        for e in range(epochs):
            tr_loss,tr_acc = run_epoch(model, train_dl, crit, opt, True, dev)
            val_loss,val_acc= run_epoch(model, val_dl, crit, opt, False, dev)
            sch.step(e)
            writer.add_scalars('loss',{'train':tr_loss,'val':val_loss},writer._get_next_global_step())
            writer.add_scalars('acc', {'train':tr_acc,'val':val_acc},writer._get_next_global_step())
            if val_acc>best:
                best=val_acc
                torch.save(model.state_dict(),'best_tinyvit_cub.pth')
            print(f"e{e+1:02d} {tr_acc:.3f}/{val_acc:.3f}")

    stage(FREEZE_EPOCH,'head')
    stage(PART_EPOCH,'partial')
    stage(FULL_EPOCH,'full')

if __name__=='__main__':
    main()
