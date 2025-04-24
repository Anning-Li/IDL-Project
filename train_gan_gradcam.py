#!/usr/bin/env python3
# train_gan_gradcam.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.models import resnet18, vgg16
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from skimage.metrics import structural_similarity as ssim
from scipy.stats import pearsonr
from collections import OrderedDict

from models import ResNet18, VGG
from models import PreActResNet18


# -------------------------------
# 1) DCGAN generator & discriminator
# -------------------------------
class DCGAN_Generator(nn.Module):
    def __init__(self, nz=100, ngf=64, nc=3):
        super().__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf*8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf*8), nn.ReLU(True),
            nn.ConvTranspose2d(ngf*8, ngf*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*4), nn.ReLU(True),
            nn.ConvTranspose2d(ngf*4, ngf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*2), nn.ReLU(True),
            nn.ConvTranspose2d(ngf*2, ngf,   4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),   nn.ReLU(True),
            nn.ConvTranspose2d(ngf,   nc,    3, 1, 1, bias=False),
            nn.Tanh()
        )
    def forward(self, z):
        return self.main(z)

class DCGAN_Discriminator(nn.Module):
    def __init__(self, nc=3, ndf=64):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*2), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf*2, ndf*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*4), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf*4, 1,     4, 1, 0, bias=False),
            nn.Sigmoid()
        )
    def forward(self, img):
        return self.main(img).view(-1)

# -------------------------------
# 2) CGAN (label‐conditioned) models
# -------------------------------
class CGAN_Generator(nn.Module):
    def __init__(self, nz=100, ngf=64, nc=3, n_classes=10):
        super().__init__()
        self.label_emb = nn.Embedding(n_classes, n_classes)
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz + n_classes, ngf*8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf*8), nn.ReLU(True),
            nn.ConvTranspose2d(ngf*8, ngf*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*4), nn.ReLU(True),
            nn.ConvTranspose2d(ngf*4, ngf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*2), nn.ReLU(True),
            nn.ConvTranspose2d(ngf*2, ngf,   4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),   nn.ReLU(True),
            nn.ConvTranspose2d(ngf,   nc,    3, 1, 1, bias=False),
            nn.Tanh()
        )
    def forward(self, z, labels):
        c = self.label_emb(labels).view(labels.size(0), -1, 1, 1)
        z = z.view(z.size(0), -1, 1, 1)
        x = torch.cat([z, c], dim=1)
        return self.main(x)

class CGAN_Discriminator(nn.Module):
    def __init__(self, nc=3, ndf=64, n_classes=10):
        super().__init__()
        self.label_emb = nn.Embedding(n_classes, n_classes)
        self.main = nn.Sequential(
            nn.Conv2d(nc + n_classes, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*2), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf*2, ndf*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*4), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf*4, 1,     4, 1, 0, bias=False),
            nn.Sigmoid()
        )
    def forward(self, img, labels):
        c = self.label_emb(labels).view(labels.size(0), -1, 1, 1)
        cmap = c.expand(-1, -1, img.size(2), img.size(3))
        x = torch.cat([img, cmap], dim=1)
        return self.main(x).view(-1)

# -------------------------------
# 3) Grad‐CAM heatmap helper
# -------------------------------
def generate_gradcam_heatmap(cam, img, cls, device):
    """
    cam: a pytorch_grad_cam.GradCAM instance
    img: 1×3×32×32 tensor, already normalized to [−1,1]
    cls: integer class index
    """
    # map back to [0,1] then re-normalize for GradCAM preprocess
    inp = (img.cpu().squeeze(0) + 1.0)/2.0
    inp = transforms.functional.normalize(inp, mean=[0.5]*3, std=[0.5]*3)
    inp = inp.unsqueeze(0).to(device)
    targets = [ClassifierOutputTarget(int(cls))]
    heatmap = cam(input_tensor=inp, targets=targets)[0]
    return heatmap

# -------------------------------
# 4) Training function
# -------------------------------
def train_gan_gradcam(
    gen, disc, loader, classifier, cam, device,
    num_epochs=150, lr=2e-4, beta1=0.5,
    lambda_adv=1.0, lambda_perc=0.0, lambda_gc=1.0,
    conditional=False
):
    bce = nn.BCELoss()
    # perceptual loss model
    perc_model = vgg16(pretrained=True).features[:16].to(device).eval()
    for p in perc_model.parameters(): p.requires_grad = False

    optD = optim.Adam(disc.parameters(), lr=lr, betas=(beta1,0.999))
    optG = optim.Adam(gen.parameters(), lr=lr, betas=(beta1,0.999))
    nz = 100
    history = []

    for epoch in range(1, num_epochs+1):
        running_d, running_g = 0.0, 0.0
        for reals, labels in loader:
            bs = reals.size(0)
            reals = reals.to(device); labels = labels.to(device)

            # Discriminator step
            disc.zero_grad()
            out_real = disc(reals, labels) if conditional else disc(reals)
            lossD_real = bce(out_real, torch.ones(bs, device=device))
            noise = torch.randn(bs, nz, device=device)
            fake = gen(noise, labels) if conditional else gen(noise)
            out_fake = disc(fake.detach(), labels) if conditional else disc(fake.detach())
            lossD_fake = bce(out_fake, torch.zeros(bs, device=device))
            lossD = 0.5*(lossD_real + lossD_fake)
            lossD.backward(); optD.step()

            # Generator step
            gen.zero_grad()
            out_g = disc(fake, labels) if conditional else disc(fake)
            lossG_adv = bce(out_g, torch.ones(bs, device=device))

            # optional perceptual
            loss_perc = torch.tensor(0., device=device)
            if lambda_perc>0:
                f_real = perc_model(reals)
                f_fake = perc_model(fake)
                loss_perc = F.mse_loss(f_fake, f_real)

            # GradCAM loss
            batch_gc = []
            for i in range(bs):
                h_r = generate_gradcam_heatmap(cam, reals[i:i+1], labels[i], device)
                h_f = generate_gradcam_heatmap(cam, fake[i:i+1], labels[i], device)
                Lss = 1 - ssim(h_r, h_f)
                c, _ = pearsonr(h_r.flatten(), h_f.flatten())
                Lpe = 1 - c
                batch_gc.append((Lss+Lpe)/2)
            loss_gc = torch.tensor(batch_gc, device=device).mean()

            lossG = lambda_adv*lossG_adv + lambda_perc*loss_perc + lambda_gc*loss_gc
            lossG.backward(); optG.step()

            running_d += lossD.item()*bs
            running_g += lossG.item()*bs

        avgD = running_d / len(loader.dataset)
        avgG = running_g / len(loader.dataset)
        history.append((epoch, avgG, avgD))

        if epoch in (50, 100, 150):
            tag = "CGAN" if conditional else "DCGAN"
            print(f"[{tag}] Epoch {epoch:3d} → G_loss {avgG:.4f}, D_loss {avgD:.4f}")

    return history

# -------------------------------
# 5) Plotting function
# -------------------------------
def plot_losses(record, title):
    epochs, g, d = zip(*record)
    xs    = [e for e in epochs if e%5==0]
    yg    = [g[i] for i,e in enumerate(epochs) if e%5==0]
    yd    = [d[i] for i,e in enumerate(epochs) if e%5==0]

    plt.figure(); plt.plot(xs, yg, marker='o')
    plt.title(f"{title} Generator Loss"); plt.xlabel("Epoch"); plt.ylabel("G Loss")
    plt.show()

    plt.figure(); plt.plot(xs, yd, marker='o')
    plt.title(f"{title} Discriminator Loss"); plt.xlabel("Epoch"); plt.ylabel("D Loss")
    plt.show()


# -------------------------------
# 6) Main
# -------------------------------
if __name__ == "__main__":
    device = torch.device("cuda")

    # ──────────────── DataLoader setup ────────────────
    transform = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize((0.5,)*3, (0.5,)*3)
    ])
    ds     = CIFAR10(root='./data', train=True, download=True, transform=transform)
    loader = DataLoader(ds, batch_size=128, shuffle=True, num_workers=4)

    # 1) Instantiate *your* CIFAR‐ResNet exactly, with num_classes=10
    clf = PreActResNet18().to(device)

    # 2) Load the checkpoint
    ckpt = torch.load('./checkpoint/preresnet18_ckpt.pth', map_location=device)
    raw_sd = ckpt['net']

    # 3) Strip off any "module." prefix left by DataParallel
    # new_sd = OrderedDict()
    # for k, v in raw_sd.items():
    #     name = k.replace('module.', '')     # remove the "module." 
    #     new_sd[name] = v
    from collections import OrderedDict

    ckpt = torch.load('./checkpoint/preresnet18_ckpt.pth', map_location=device)
    raw = ckpt['net']
    new_sd = OrderedDict()
    for k,v in raw.items():
        new_sd[k.replace('module.', '')] = v

    # 4) Load into your model
    clf.load_state_dict(new_sd)
    clf.eval()

    # 5) Then wrap with GradCAM as before
    cam = GradCAM(model=clf, target_layers=[clf.layer4])


    # DCGAN + GradCAM
    print("\n## DCGAN + Grad-CAM")
    g1, d1 = DCGAN_Generator().to(device), DCGAN_Discriminator().to(device)
    rec1 = train_gan_gradcam(g1, d1, loader, clf, cam, device,
                              num_epochs=150, lambda_adv=1.0, lambda_perc=0.0, lambda_gc=1.0,
                              conditional=False)

    # CGAN + GradCAM
    print("\n## CGAN + Grad-CAM")
    g2, d2 = CGAN_Generator().to(device), CGAN_Discriminator().to(device)
    rec2 = train_gan_gradcam(g2, d2, loader, clf, cam, device,
                              num_epochs=150, lambda_adv=1.0, lambda_perc=0.0, lambda_gc=1.0,
                              conditional=True)

    # Print Table 4 results
    print("\n=== Table 4 Values ===")
    print("Model               | Epoch |    G Loss |    D Loss")
    print("--------------------|-------|-----------|----------")
    for e,gl,dl in rec1 + rec2:
        tag = "CGAN" if (e,gl,dl) in rec2 else "DCGAN"
        print(f"{tag:20s} | {e:5d} | {gl:9.4f} | {dl:8.4f}")

    # Plot losses
    plot_losses(rec1, "DCGAN + Grad-CAM")
    plot_losses(rec2, "CGAN + Grad-CAM")