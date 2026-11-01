"""PyTorch U-Net for semantic segmentation (dissertation architecture family).
SYNTHETIC demo only in demo_metrics.py; dissertation used DeepParticle dataset."""
try:
    import torch, torch.nn as nn
    class DoubleConv(nn.Module):
        def __init__(self, cin, cout):
            super().__init__(); self.net=nn.Sequential(nn.Conv2d(cin,cout,3,padding=1),nn.ReLU(inplace=True),nn.Conv2d(cout,cout,3,padding=1),nn.ReLU(inplace=True))
        def forward(self,x): return self.net(x)
    class UNet(nn.Module):
        def __init__(self, in_channels=3, num_classes=4, base=32):
            super().__init__()
            self.d1=DoubleConv(in_channels,base); self.d2=DoubleConv(base,base*2); self.d3=DoubleConv(base*2,base*4)
            self.pool=nn.MaxPool2d(2); self.up2=nn.ConvTranspose2d(base*4,base*2,2,stride=2); self.c2=DoubleConv(base*4,base*2)
            self.up1=nn.ConvTranspose2d(base*2,base,2,stride=2); self.c1=DoubleConv(base*2,base); self.out=nn.Conv2d(base,num_classes,1)
        def forward(self,x):
            x1=self.d1(x); x2=self.d2(self.pool(x1)); b=self.d3(self.pool(x2))
            u=self.c2(torch.cat([self.up2(b),x2],dim=1)); u=self.c1(torch.cat([self.up1(u),x1],dim=1)); return self.out(u)
except ImportError:
    UNet=None
if __name__=="__main__":
    print("U-Net defined. Install torch to instantiate:", UNet is not None)
