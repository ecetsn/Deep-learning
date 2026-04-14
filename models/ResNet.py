import torch.nn as nn
import torch.nn.functional as F


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)

class BasicBlock(nn.Module):
    """Basic residual block for ResNet-18/34."""
  
    expansion = 1 # For BasicBlock, output channels = channels * expansion = channels
    def __init__(self, in_channels, channels, stride=1,norm=nn.BatchNorm2d, option='B'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = norm(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = norm(channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != channels:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, channels//4, channels//4), "constant", 0))
                # The slicing x[:, :, ::2, ::2] performs downsampling by taking every second pixel in height and width dimensions.
                # (1, 16, 32, 32) → (1, 16, 16, 16) after slicing.
                # Format: (padding_left, padding_right, padding_top, padding_bottom, padding_channel_left, padding_channel_right).
                # (1, 16, 16, 16) → (1, 32, 16, 16) after padding channels from 16 to 32, channels//4 = 32//4 = 8 zeros added to the left and right of the channel dimension
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_channels, self.expansion * channels, kernel_size=1, stride=stride, bias=False),
                     norm(self.expansion * channels)
                )
                # (1, 16, 32, 32) → (1, 32, 16, 16) after 1×1 convolution with stride=2 and output channels=32.

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    """ResNet implementation for CIFAR-10 classification."""
    def __init__(self, block, num_blocks, norm=nn.BatchNorm2d, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = norm(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], norm=norm,stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], norm=norm,stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], norm=norm,stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], norm=norm,stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, channels, num_blocks, norm, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, channels, stride,norm))
            self.in_channels = channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out