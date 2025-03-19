from UNet import UNet
import torch

if __name__ == '__main__':
    x = torch.randn(1, 3, 256, 256)
    model = UNet(in_channels=3, out_channels=1)
    preds = model(x)
    print(f"输入尺寸: {x.shape}")
    print(f"输出尺寸: {preds.shape}")