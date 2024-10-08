import torch
import torch.nn.functional as F
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


class Res2dBlock(nn.Module):
    def __init__(
        self,
        input_channels,
        output_channels,
        shape=3,
        stride=2,
    ):
        super(Res2dBlock, self).__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(
                input_channels,
                output_channels,
                kernel_size=shape,
                stride=stride,
                padding=shape // 2,
            ),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(),
            nn.Conv2d(
                output_channels,
                output_channels,
                kernel_size=shape,
                padding=shape // 2,
            ),
            nn.BatchNorm2d(output_channels),
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(
                input_channels,
                output_channels,
                kernel_size=shape,
                stride=stride,
                padding=shape // 2,
            ),
            nn.BatchNorm2d(output_channels)
            
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        out1 = self.conv_block_1(x)
        out2 = self.conv_block_2(x)
        return self.relu(out1 + out2)


class BaseModel(nn.Module):
    def predict_proba(self, x):
        return F.softmax(self(x), dim=-1)

    def predict(self, x):
        return torch.argmax(self(x), dim=-1)
    
    


class ShortChunkCNN(BaseModel):
    def __init__(
        self,
        n_channels=128,
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


class ShortChunkResCNN(BaseModel):
    def __init__(
        self,
        n_channels=128,
        n_class=11,
    ):
        super(ShortChunkResCNN, self).__init__()
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
        
        self.res_blocks = nn.ModuleList([
            Res2dBlock(*channel) for channel in channel_list
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
        for res in self.res_blocks:
            x = res(x)
        x = x.squeeze(2)
        x = F.max_pool1d(x, x.size(-1))
        return self.dense_blocks(x.view(x.size(0), -1))


def get_model(name: str) -> BaseModel:
    if name == 'short_chunk_cnn':
        return ShortChunkCNN()
    elif name == 'short_chunk_res_cnn':
        return ShortChunkResCNN()


if __name__ == '__main__':
    x = torch.randn(4, 1, 128, 128)
    model = ShortChunkCNN(n_channels=128)
    y = model(x)
    print(model)
