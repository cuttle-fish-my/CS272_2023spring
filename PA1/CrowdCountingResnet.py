from torchvision.models import resnet
import torch.nn as nn


class UpSamplingBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = resnet.conv3x3(in_channels, out_channels)
        self.conv2 = resnet.conv3x3(out_channels, out_channels)
        self.conv1x1 = resnet.conv1x1(in_channels, out_channels)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x):
        identity = self.upsample(x)
        out = self.conv1(identity)
        out = self.bn(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn(out)
        # identity = self.conv1x1(identity)
        # out += identity
        out = self.relu(out)
        return out


class CrowdCountingResnet(resnet.ResNet):
    def __init__(self):
        super().__init__(resnet.BasicBlock, [2, 2, 2, 2])
        # weight = resnet.ResNet18_Weights.IMAGENET1K_V1
        # self.load_state_dict(weight.get_state_dict(progress=True))
        self.layer3 = UpSamplingBlock(in_channels=128, out_channels=64)
        self.layer4 = UpSamplingBlock(in_channels=64, out_channels=32)
        self.layer5 = UpSamplingBlock(in_channels=32, out_channels=16)
        self.conv1x1 = resnet.conv1x1(in_planes=16, out_planes=1)
        self.sigmoid = nn.Sigmoid()

    def _forward_impl(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.conv1x1(x)
        x = self.sigmoid(x)
        return x
