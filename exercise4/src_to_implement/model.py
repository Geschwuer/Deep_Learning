import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()

        # main path operations
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # skip (identity path)
        if stride != 1 or in_channels != out_channels:
            # apply 1x1 conv + batch norm to get same shape as in main path
            self.skip = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.skip = nn.Identity()


    def forward(self, in_tensor):
        identity = self.skip(in_tensor)

        # main path
        out1 = F.relu(self.bn1(self.conv1(in_tensor)))
        out2 = self.bn2(self.conv2(out1))

        out_tensor = F.relu(identity + out2)

        return out_tensor
    

class ResNet(nn.Module):
    def __init__(self, num_classes: int = 2):
        super().__init__()
        # standard blocks
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        # residual blocks
        self.layer1 = ResBlock(64, 64, 1)
        self.layer2 = ResBlock(64, 128, 2)
        self.layer3 = ResBlock(128, 256, 2)
        self.layer4 = ResBlock(256, 512, 2)

        # classification
        self.global_avg_pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.fc = nn.Linear(512, num_classes)
        self.activation = nn.Sigmoid()

        self._initialize_weights()


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)   # gamma = 1
                nn.init.zeros_(m.bias)    # beta  = 0


    def forward(self, in_tensor: torch.Tensor) -> torch.Tensor:
        x = self.stem(in_tensor)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = self.activation(x)

        return x