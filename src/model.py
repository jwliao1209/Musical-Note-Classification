import torch
from torch import nn


class Conv2dBlock(nn.Module):
    def __init__(
        self,
        input_channels,
        output_channels,
        shape=3,
        stride=1,
        pooling=2,
    ):
        super(Conv2dBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(
                input_channels,
                output_channels,
                kernel_size=shape,
                stride=stride,
                padding=shape // 2,
            ),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(),
            nn.MaxPool2d(pooling)
        )

    def forward(self, x):
        return self.conv_block(x)


class ShortChunkCNN(nn.Module):
    def __init__(
        self,
        n_channels=128,
        sample_rate=16000,
        n_fft=512,
        f_min=0.0,
        f_max=8000.0,
        n_mels=128,
        n_class=11,
    ):
        super(ShortChunkCNN, self).__init__()
        self.spec_bn = nn.BatchNorm2d(1)
        
        channel_list = [
            (1, n_channels),
            (n_channels, n_channels),
            (n_channels, n_channels * 2),
            (n_channels * 2, n_channels * 2),
            (n_channels * 2, n_channels * 2),
            (n_channels * 2, n_channels * 2),
            (n_channels * 2, n_channels * 4),
        ]
        
        self.conv_blocks = nn.ModuleList([
                Conv2dBlock(*channel) for channel in channel_list
        ])
        
        self.dense_blocks = nn.Sequential(
            nn.Linear(n_channels * 4, n_channels * 4),
            nn.BatchNorm1d(n_channels * 4),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(n_channels * 4, n_class),
        )

    def forward(self, x):
        x = self.spec_bn(x)
        for conv in self.conv_blocks:
            x = conv(x)
        return self.dense_blocks(x.view(x.size(0), -1))


if __name__ == '__main__':
    x = torch.randn(4, 1, 128, 128)
    model = ShortChunkCNN(n_channels=128)
    y = model(x)
    print(model)
