import torch
from math import log
from conv_layer import *


class GLDF(nn.Module):
    def __init__(self, in_channels, hidden_channels, b=1, gamma=2, r=4):
        super(GLDF, self).__init__()

        self.conv1 = ConvBlock(in_channels, hidden_channels, kernel_size=1, stride=1, same_padding=False, relu=False,
                               bn=False)

        adp_kernel_size = int(abs((log(hidden_channels, 2) + b) / gamma))
        adp_kernel_size = adp_kernel_size if adp_kernel_size % 2 else adp_kernel_size + 1

        out_channel = hidden_channels // r
        self.conv_l1 = ConvBlock(hidden_channels, out_channel, kernel_size=adp_kernel_size, stride=1)
        self.conv_l2 = ConvBlock(out_channel, hidden_channels, kernel_size=adp_kernel_size, stride=1, relu=False)
        self.softmax_l = nn.Softmax(dim=1)

        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.conv_g = ConvBlock(1, 1, kernel_size=adp_kernel_size, relu=False)
        self.sigmoid_g = nn.Sigmoid()

        self.conv2 = ConvBlock(hidden_channels * 2, in_channels, kernel_size=1, stride=1, same_padding=False,
                               relu=False, bn=False)

    def forward(self, x1, x2, x3, x4):
        x12 = torch.cat([x1, x2, x3, x4], dim=1)
        x12_fc = self.conv1(x12)

        # Local
        localF = self.conv_l1(x12_fc)
        localF = self.conv_l2(localF)
        localF = self.softmax_l(localF)  # torch.Size([N, 32, 800])

        # Global
        avg_pool = self.avg_pool(x12_fc)
        globalF = self.conv_g(avg_pool.transpose(-1, -2)).transpose(-1, -2)
        globalF = self.sigmoid_g(globalF)  # torch.Size([N, 32, 1])

        localFNew = x12_fc * localF.expand_as(x12_fc)  # torch.Size([N, 32, 800])
        globalFNew = x12_fc * globalF.expand_as(x12_fc)  # torch.Size([N, 32, 800])

        output = torch.cat([localFNew, globalFNew], dim=1)
        output = self.conv2(output)

        return output


class MKFF(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(MKFF, self).__init__()

        self.kernel1 = ConvBlock(in_channel, out_channel, kernel_size=1, same_padding=False)

        self.kernel2 = nn.Sequential(ConvBlock(in_channel, out_channel, kernel_size=1, same_padding=False, bn=False),
                                     ConvBlock(out_channel, out_channel, kernel_size=3))

        self.kernel3 = nn.Sequential(ConvBlock(in_channel, out_channel, kernel_size=1, same_padding=False),
                                     ConvBlock(out_channel, out_channel, kernel_size=5))

        self.kernel4 = nn.Sequential(nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
                                     ConvBlock(in_channel, out_channel, kernel_size=1, same_padding=False))

        self.GLDF = GLDF(out_channel * 4, out_channel)

    def forward(self, x):
        kernel1 = self.kernel1(x)
        kernel2 = self.kernel2(x)
        kernel3 = self.kernel3(x)
        kernel4 = self.kernel4(x)

        kernel_all = self.GLDF(kernel1, kernel2, kernel3, kernel4)

        return kernel_all + x


class DR_block(nn.Module):
    def __init__(self, filter_num, kernel_size, dilation):
        super(DR_block, self).__init__()

        self.padding_pool = nn.ConstantPad1d((0, 1), 0)
        self.max_pooling = nn.MaxPool1d(kernel_size=(3,), stride=2)

        self.padding_conv = nn.ConstantPad1d(((kernel_size - 1) // 2) * dilation, 0)
        self.conv0 = ConvBlock(filter_num, filter_num, kernel_size=kernel_size, stride=1, dilation=dilation,
                               same_padding=False)
        self.conv1 = ConvBlock(filter_num, filter_num, kernel_size=kernel_size, stride=1, dilation=dilation,
                               same_padding=False)

    def forward(self, x):
        x = self.padding_pool(x)
        px = self.max_pooling(x)

        x = self.padding_conv(px)
        x = self.conv0(x)
        x = self.padding_conv(x)
        x = self.conv1(x)

        return x + px


class DRFE(nn.Module):

    def __init__(self, filter_num):
        super(DRFE, self).__init__()

        self.kernel_size_list = [5, 5, 5]
        self.dilation_list = [1, 1, 1]

        self.padding_conv = nn.ConstantPad1d(((self.kernel_size_list[0] - 1) // 2), 0)
        self.conv0 = ConvBlock(filter_num, filter_num, self.kernel_size_list[0], stride=1, dilation=1,
                               same_padding=False)
        self.conv1 = ConvBlock(filter_num, filter_num, self.kernel_size_list[0], stride=1, dilation=1,
                               same_padding=False)

        self.DR_blocklist = nn.ModuleList(
            [DR_block(filter_num, kernel_size=self.kernel_size_list[i],
                      dilation=self.dilation_list[i]) for i in range(len(self.kernel_size_list))])

    def forward(self, x):
        x = self.padding_conv(x)
        x = self.conv0(x)
        x = self.padding_conv(x)
        x = self.conv1(x)

        for i in range(len(self.DR_blocklist)):
            x = self.DR_blocklist[i](x)
        x = x.squeeze(-1).squeeze(-1)

        return x


class MFDm6ARice(nn.Module):
    def __init__(self):
        super(MFDm6ARice, self).__init__()

        self.Embed = nn.Embedding(800, 5)
        self.Stem = ConvBlock(5, 128, kernel_size=1, stride=1)  # filter * cat number in multi-kernel
        self.Stem_test2 = ConvBlock(1, 128, kernel_size=3, stride=1, same_padding=False)
        self.MKFF = MKFF(128, 32)
        self.MKFF_test2 = MKFF(32, 32)

        self.DRFE = DRFE(128)

        self.Classifier = nn.Sequential(nn.Flatten(), nn.Linear(128 * 100, 1))

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, class_fea):
        x = class_fea.long()  # (N, 800)

        # MKFF
        x = self.Embed(x)
        x = x.permute(0, 2, 1)
        x = self.Stem(x)
        mkff = self.MKFF(x)

        # DRFE
        drfe = self.DRFE(mkff)
        return drfe

    # Ref: https://awi.cuhk.edu.cn/~dbAMP/AVP, https://github.com/LZYHZAU/PTM-CMGMS
    def trainModel(self, x):
        output = self.forward(x)
        return self.Classifier(output), output
