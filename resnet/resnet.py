import torch
import torch.nn as nn


class Block(nn.Module):
    expansion = 1

    def __init__(self, input_dim, output_dim, stride=1, is_conv2_0=False):
        super().__init__()
        self.block1 = nn.Conv2d(
            in_channels=input_dim,
            out_channels=output_dim,
            kernel_size=3,
            stride=stride,
            padding=1,
        )  # conv net의 첫번째 block에서는 downsampling을 위해 stride를 2로 설정할 때가 있다.
        self.block2 = nn.Conv2d(
            in_channels=output_dim,
            out_channels=output_dim,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(output_dim)
        self.bn2 = nn.BatchNorm2d(output_dim)
        self.stride = stride
        self.downsampling = nn.Conv2d(
            in_channels=input_dim, out_channels=output_dim, kernel_size=1, stride=2
        )

    def forward(self, x):
        identity = x.clone()
        x = self.relu(self.bn1(self.block1(x)))
        x = self.bn2(self.block2(x))
        if self.stride != 1:
            identity = self.downsampling(identity)
        x = self.relu(x + identity)
        return x


class Bottlececk(nn.Module):
    expansion = 4

    def __init__(self, input_dim, output_dim, stride=1, is_conv2_0=False):
        super().__init__()
        self.block1 = nn.Conv2d(
            in_channels=input_dim, out_channels=output_dim, kernel_size=1, stride=stride
        )  # conv net의 첫번째 block에서는 downsampling을 위해 stride를 2로 설정할 때가 있다.
        self.block2 = nn.Conv2d(
            in_channels=output_dim,
            out_channels=output_dim,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.block3 = nn.Conv2d(
            in_channels=output_dim,
            out_channels=output_dim * self.expansion,
            kernel_size=1,
            stride=1,
        )
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(output_dim)
        self.bn2 = nn.BatchNorm2d(output_dim)
        self.bn3 = nn.BatchNorm2d(output_dim * self.expansion)
        self.stride = stride
        self.downsampling = nn.Conv2d(
            in_channels=input_dim,
            out_channels=output_dim * self.expansion,
            kernel_size=1,
            stride=2,
        )
        self.is_conv2_0 = is_conv2_0
        self.identity_dim_x_4 = nn.Conv2d(
            in_channels=input_dim,
            out_channels=output_dim * self.expansion,
            kernel_size=1,
        )

    def forward(self, x):
        identity = x.clone()
        if self.is_conv2_0:
            # conv2의 첫번째 block이면, shortcut의 차원이 4배가 되어야 block의 output과 더할 수 있다.
            identity = self.identity_dim_x_4(identity)
        x = self.relu(self.bn1(self.block1(x)))
        x = self.relu(self.bn2(self.block2(x)))
        x = self.bn3(self.block3(x))
        if self.stride != 1:
            identity = self.downsampling(identity)
        x = self.relu(x + identity)
        return x


class Resnet(nn.Module):
    def __init__(self, block, output_dim, input_dim, layer_list):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=input_dim, out_channels=64, kernel_size=7, stride=2, padding=3
        )
        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2 = self.conv_layer(
            block, 64, layer_list[0], True
        )  # conv2 는 블록 내부에서 downsampling을 안해주기 때문에 구분해준다.
        self.conv3 = self.conv_layer(block, 128, layer_list[1])
        self.conv4 = self.conv_layer(block, 256, layer_list[2])
        self.conv5 = self.conv_layer(block, 512, layer_list[3])
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_features=512 * block.expansion, out_features=output_dim)
        # self.softmax = nn.Softmax(output_dim)

    def forward(self, x):
        x = self.relu(self.bn(self.conv1(x)))
        x = self.conv2(self.max_pool(x))
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.avg_pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x

    def conv_layer(self, block, dim, layer_num, is_conv2=False):
        layer = []
        for i in range(layer_num):
            if is_conv2:
                if i == 0:  # Bottlneck 에서 첫 block은 input 차원이 그대로 이다.
                    layer.append(block(dim, dim, stride=1, is_conv2_0=True))
                else:  # 하지만 다음 block 부터는 input은 4배 증가한다.
                    layer.append(block(dim * block.expansion, dim, stride=1))
            else:
                if i == 0:  # conv3 부터 첫 block은 downsampling을 해줘야 한다.
                    input_dim = int(dim/2) # 각 block의 첫 input은 전 channel을 받는다.
                    layer.append(block(input_dim * block.expansion, dim, stride=2)) 
                else:
                    layer.append(block(dim * block.expansion, dim, stride=1))
        result = nn.Sequential(*layer)
        return result


def Resnet18(output_dim, input_dim=3):
    return Resnet(Block, output_dim, input_dim, [2, 2, 2, 2])


def Resnet34(output_dim, input_dim=3):
    return Resnet(Block, output_dim, input_dim, [3, 4, 6, 3])


def Resnet50(output_dim, input_dim=3):
    return Resnet(Bottlececk, output_dim, input_dim, [3, 4, 6, 3])


def Resnet101(output_dim, input_dim=3):
    return Resnet(Bottlececk, output_dim, input_dim, [3, 4, 23, 3])


def Resnet152(output_dim, input_dim=3):
    return Resnet(Bottlececk, output_dim, input_dim, [3, 8, 36, 3])
