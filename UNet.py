import torch
import torch.nn as nn
from tensorboard.data.proto.data_provider_pb2 import Downsample


#双卷积结构
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


#编码器（下采样）
class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.down_sample = nn.Sequential(
    #池化，减少空间维度，增加通道数
            nn.MaxPool2d(2,2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.down_sample(x)

#解码器（上采样）
class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up_sample = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    # x1为上采样特征图，x2为跳跃链接特征图
    def forward(self,x1, x2):
        print(f"[调试] 上采样前 x1 尺寸: {x1.shape}")
        x1 = self.up_sample(x1) # 调用上采样操作
        print(f"[调试] 上采样后 x1 尺寸: {x1.shape}")
        print(f"[调试] 裁剪前 x2 尺寸: {x2.shape}")
        # 调整尺寸以匹配跳跃连接的特征图
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x2 = nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2,
                                    diffY // 2, diffY - diffY // 2])
        print(f"[调试] 裁剪后 x2 尺寸: {x2.shape}")
        # 拼接跳跃连接的特征图
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

#全部流程
class UNet(nn.Module):
    #输入RGB三通道，输出分辨结果一通道
    def __init__(self, in_channels=3, out_channels=1, features = [64, 128, 256, 512]):
        super().__init__()
        #编码器部分（3-64-128-256-512）
        self.encoder = nn.ModuleList()
        self.pool = nn.MaxPool2d(2,2)
        for feature in features:
            self.encoder.append(DoubleConv(in_channels, feature))
            in_channels = feature
        #最底层部分（512-1024）
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
        #解码器部分（1024-512-256-128-64）
        self.decoder = nn.ModuleList()
        for feature in reversed(features):
            self.decoder.append(Decoder(feature * 2 , feature))
        #输出层(64-1)
        self.final = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        #编码器，记录跳跃连接
        skip_connections = []
        for down in self.encoder:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
        #中间层
        x = self.bottleneck(x)
        #反转跳跃连接顺序以匹配解码器
        for cons in skip_connections:
            print(f"反转前卷积记录: {cons.shape}")
        skip_connections.reverse()
        for cons in skip_connections:
            print(f"反转后卷积记录: {cons.shape}")
        #解码器，拼接跳跃连接
        for idx, up in enumerate(self.decoder):
            x = up(x, skip_connections[idx])
        #输出
        return torch.sigmoid(self.final(x))









