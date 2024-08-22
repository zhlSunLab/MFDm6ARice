import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=(1,), dilation=(1,), if_bias=False,
                 same_padding=True, relu=True, bn=True):
        super(ConvBlock, self).__init__()
        p0 = 'same' if same_padding else 0

        self.conv1d = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                stride=stride, padding=p0, dilation=dilation, bias=True if if_bias else False)
        self.batchnorm1d = nn.BatchNorm1d(out_channels) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv1d(x)
        if self.batchnorm1d is not None:
            x = self.batchnorm1d(x)
        if self.relu is not None:
            x = self.relu(x)
        x = F.dropout(x, 0.3, training=self.training)
        return x
