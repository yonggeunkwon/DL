import torch
from torch import nn
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
from my_functions import *

class BasicBlock(nn.Module):
    expansion = 1 
    def __init__(self, in_channels, inner_channels, stride = 1, projection = None):
        super().__init__()

        self.residual = nn.Sequential(nn.Conv2d(in_channels, inner_channels, 3, stride=stride, padding=1, bias=False),
                                      nn.BatchNorm2d(inner_channels),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(inner_channels, inner_channels * self.expansion, 3, padding=1, bias=False),
                                      nn.BatchNorm2d(inner_channels)
                                      )
        self.projection = projection
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):

        residual = self.residual(x) 

        if self.projection is not None:
            shortcut = self.projection(x) 
        else:
            shortcut = x

        out = self.relu(residual + shortcut) 
        return out
    

class Bottleneck(nn.Module):

    expansion = 4 
    def __init__(self, in_channels, inner_channels, stride = 1, projection = None):
        super().__init__()

        self.residual = nn.Sequential(nn.Conv2d(in_channels, inner_channels, 1, bias=False),
                                      nn.BatchNorm2d(inner_channels),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(inner_channels, inner_channels, 3, stride=stride, padding=1, bias=False),
                                      nn.BatchNorm2d(inner_channels),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(inner_channels, inner_channels * self.expansion, 1, bias=False),
                                      nn.BatchNorm2d(inner_channels * self.expansion))

        self.projection = projection
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = self.residual(x)

        if self.projection is not None:
            shortcut = self.projection(x)
        else:
            shortcut = x

        out = self.relu(residual + shortcut)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_block_list, num_classes = 120, zero_init_residual = True):
        super().__init__()

        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True) # 좀더 메모리 효율적
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self.make_layers(block, 64, num_block_list[0], stride=1)
        self.layer2 = self.make_layers(block, 128, num_block_list[1], stride=2)
        self.layer3 = self.make_layers(block, 256, num_block_list[2], stride=2)
        self.layer4 = self.make_layers(block, 512, num_block_list[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3%p according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, block):
                    nn.init.constant_(m.residual[-1].weight, 0)

    def make_layers(self, block, inner_channels, num_blocks, stride = 1):

        if stride != 1 or self.in_channels != inner_channels * block.expansion:
            # stride = 1 이여도 채널 수가 다르면 (layer1의 첫번째 BottleNeck) projection 해야함 (이 때는 resoltion은 그대로, 채널 수만 늘어남)
            projection = nn.Sequential(
                nn.Conv2d(self.in_channels, inner_channels * block.expansion, 1, stride=stride, bias=False),
                nn.BatchNorm2d(inner_channels * block.expansion)) # 점선 connection 임
        else:
            projection = None

        layers = []
        layers += [block(self.in_channels, inner_channels, stride, projection)] # projection은 첫 block에서만
        self.in_channels = inner_channels * block.expansion
        for _ in range(1, num_blocks):
            layers += [block(self.in_channels, inner_channels)]

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
    

def ResNet50():
    return ResNet(Bottleneck, [3,4,6,3])

def ResNet101():
    return ResNet(Bottleneck, [3, 4, 23, 3])

def ResNet152(**kwargs):
    return ResNet(Bottleneck, [3, 8, 36, 3])


# class PretrainedResNet(nn.Module):

#     def __init__(self, pretrained_resnet, n_classes, freeze=True):
#         super().__init__()
#         self.pretrained_resnet = pretrained_resnet
#         if freeze: # freeze
#             for p in pretrained_resnet.parameters():
#                 p.requires_grad = False
#         out_features = self.pretrained_resnet.fc.out_features
#         # 학습시킬 파라미터
#         self.fc1 = nn.Linear(out_features, 1024)
#         self.fc2 = nn.Linear(1024, n_classes)

#     def forward(self, x):
#         # 전처리된 이미지를 pretrained 모델에 통과시키기
#         x = self.pretrained_resnet(x)
#         # 추가적인 레이어에 통과시켜서 최종 클래스 예측을 만들기
#         x = nn.functional.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x

class PretrainedResNet(nn.Module):

    def __init__(self, pretrained_resnet, n_classes, freeze=True):
        super().__init__()
        self.pretrained_resnet = pretrained_resnet
        if freeze: # freeze
            for name, child in self.pretrained_resnet.named_children():
                for param in child.parameters():
                    if name not in ['fc']:
                        param.requires_grad = False
        out_features = self.pretrained_resnet.fc.out_features
        # 학습시킬 파라미터
        self.fc1 = nn.Linear(out_features, 1024)
        self.fc2 = nn.Linear(1024, n_classes)

    def forward(self, x):
        # 전처리된 이미지를 pretrained 모델에 통과시키기
        x = self.pretrained_resnet(x)
        # 추가적인 레이어에 통과시켜서 최종 클래스 예측을 만들기
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class PretrainedResNetFreeze(nn.Module):

    def __init__(self, pretrained_resnet, n_classes, freeze=True):
        super().__init__()
        self.pretrained_resnet = pretrained_resnet
        if freeze: # freeze
            for name, child in pretrained_resnet.named_children():
                if name in ['conv1', 'bn1', 'relu']:
                    for param in child.parameters():
                        param.requires_grad = False
        out_features = self.pretrained_resnet.fc.in_features
        # 학습시킬 파라미터
        self.fc1 = nn.Linear(out_features, 1024)
        self.fc2 = nn.Linear(1024, n_classes)

    def forward(self, x):
        # 전처리된 이미지를 pretrained 모델에 통과시키기
        x = self.pretrained_resnet.conv1(x)
        x = self.pretrained_resnet.bn1(x)
        x = self.pretrained_resnet.relu(x)

        # 추가적인 레이어에 통과시켜서 최종 클래스 예측을 만들기
        x = self.pretrained_resnet.maxpool(x)
        x = self.pretrained_resnet.layer1(x)
        x = self.pretrained_resnet.layer2(x)
        x = self.pretrained_resnet.layer3(x)
        x = self.pretrained_resnet.layer4(x)
        x = self.pretrained_resnet.avgpool(x)

        x = torch.flatten(x ,1)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    

class CustomDenseNet169(nn.Module):
    def __init__(self, models, num_classes=120):
        super(CustomDenseNet169, self).__init__()
        self.densenet = models.densenet169(pretrained=True)
        num_features = self.densenet.classifier.in_features
        self.densenet.classifier = nn.Sequential(
            nn.Linear(num_features, num_classes)
        )

    def forward(self, x):
        return self.densenet(x)

# model = CustomDenseNet169().to(DEVICE)