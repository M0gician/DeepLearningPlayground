import torch
import typing
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, Adam
from typing import Union, TypeVar, Tuple, Optional
from collections import OrderedDict

T = TypeVar('T')
Layer_Type = Union[typing.OrderedDict[str, nn.Module], nn.Module]


def truncated_normal_(tensor: torch.Tensor, mean=0, std=1):
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)


class BatchNormaledConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, **kwargs):
        super(BatchNormaledConv2d, self).__init__()
        self.convolution = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.normalization = nn.BatchNorm2d(out_channels, eps=0.001)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = self.convolution(X)
        X = self.normalization(X)
        return self.relu(X)


class Inception(nn.Module):
    def __init__(self, in_channels: int,
                 ch_1x1: int, ch_3x3: int, ch_3x3_reduce: int, ch_5x5: int, ch_5x5_reduce: int,
                 pool_proj: int, conv_type=None):
        super(Inception, self).__init__()
        if conv_type is None:
            conv_type = BatchNormaledConv2d

        self.branch1 = conv_type(
            in_channels=in_channels,
            out_channels=ch_1x1,
            kernel_size=(1, 1)
        )

        self.branch2 = nn.Sequential(
            conv_type(
                in_channels=in_channels,
                out_channels=ch_3x3_reduce,
                kernel_size=(1, 1),
                padding=1
            ),
            conv_type(
                in_channels=ch_3x3_reduce,
                out_channels=ch_3x3,
                kernel_size=(3, 3),
                padding=1
            )
        )

        self.branch3 = nn.Sequential(
            conv_type(
                in_channels=in_channels,
                out_channels=ch_5x5_reduce,
                kernel_size=(1, 1),
                padding=1
            ),
            conv_type(
                in_channels=ch_5x5_reduce,
                out_channels=ch_5x5,
                kernel_size=(5, 5),
                padding=1
            )
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(
                kernel_size=(3, 3),
                stride=2,
                padding=1,
                # ceil_mode=True
            ),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=pool_proj,
                kernel_size=(1, 1)
            )
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        b1 = self.branch1(X)
        b2 = self.branch2(X)
        b3 = self.branch3(X)
        b4 = self.branch4(X)

        return torch.cat([b1, b2, b3, b4], 1)


class AuxInception(nn.Module):
    def __init__(self, in_channels: int, num_classes: int):
        super(AuxInception, self).__init__()
        self.avgPooling = nn.AdaptiveAvgPool2d((4, 4))
        self.conv = nn.Conv2d(
            in_channels=in_channels, out_channels=128, kernel_size=(1, 1)
        )
        self.fc1 = nn.Linear(4*4*128, 1024)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, X:torch.Tensor) -> torch.Tensor:
        # AUX@4a: 4x4x512, AUX@4d: 4x4x528
        X = self.avgPooling(X)
        X = self.conv(X)
        X = torch.flatten(X, 1)
        X = F.relu(input=self.fc1(X), inplace=True)
        X = F.dropout(input=X, p=0.7, training=self.training)
        X = self.fc2(X)

        return X


class GoogLeNet(nn.Module):
    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            truncated_normal_(m.weight)
            # nn.init.kaiming_normal_(
            #     tensor=m.weight,
            #     mode='fan_out',
            #     nonlinearity='relu'
            # )
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def __init__(self, num_classes: int, enable_aux=False, conv_type=None):
        super(GoogLeNet, self).__init__()
        self.criterion = None
        self.optimizer = None
        self.scheduler = None
        self.enable_aux = enable_aux

        # Input size: 224x224x3 (RGB color space w/ zero mean)
        # Kernel size: 7x7
        # Padding size: 3x3
        # Stride = 2
        # floor((n_h - k_h + 2*p_h)/s_h) + 1 = floor((224 - 7 + 2*3)/2) + 1 = 112
        #
        # Output size: 112x112x64
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=64,
            kernel_size=(7, 7), stride=2, padding=(3, 3)
        )

        # Input size: 112x112x64
        # Kernel size: 3x3
        # Padding size: 1x1
        # Stride: 2
        # floor((n_h - k_h + 2*p_h)/s_h) + 1 = floor((112 - 3 + 1*2)/2) + 1 = 56
        #
        # Output size: 56x56x64
        self.maxPooling1 = nn.MaxPool2d(
            kernel_size=(3, 3), stride=2, padding=(1, 1)
        )

        # Input size: 56x56x64
        # Kernel size: 1x1
        # Padding size: 0
        # Stride: 1
        # floor((n_h - k_h + 2*p_h)/s_h) + 1 = floor((56 - 1 + 0)/1) + 1 = 56
        #
        # Output size: 56x56x64
        self.conv2_1 = nn.Conv2d(
            in_channels=64, out_channels=64,
            kernel_size=(1, 1), stride=1, padding=0
        )
        # Input size: 56x56x64
        # Kernel size: 3x3
        # Padding size: 1x1
        # Stride: 1
        # floor((n_h - k_h + 2*p_h)/s_h) + 1 = floor((56 - 3 + 1*2)/1) + 1 = 56
        #
        # Output size: 56x56x192
        self.conv2_2 = nn.Conv2d(
            in_channels=64, out_channels=192,
            kernel_size=(3, 3), stride=1, padding=(1, 1)
        )

        # Input size: 56x56x192
        # Kernel size: 3x3
        # Padding size: 1x1
        # Stride: 2
        # floor((n_h - k_h + 2*p_h)/s_h) + 1 = floor((56 - 3 + 1*2)/2) + 1 = 28
        #
        # Output size: 28x28x192
        self.maxPooling2 = nn.MaxPool2d(
            kernel_size=(3, 3), stride=2, padding=(1, 1)
        )

        self.inception_3a = Inception(
            in_channels=192,
            ch_1x1=64, ch_3x3_reduce=96, ch_3x3=128, ch_5x5_reduce=16, ch_5x5=32, pool_proj=32,
            conv_type=conv_type
        )
        self.inception_3b = Inception(
            in_channels=256,
            ch_1x1=128, ch_3x3_reduce=128, ch_3x3=192, ch_5x5_reduce=32, ch_5x5=96, pool_proj=64,
            conv_type=conv_type
        )

        # Input size: 28x28x480
        # Kernel size: 3x3
        # Padding size: 1x1
        # Stride: 2
        # floor((n_h - k_h + 2*p_h)/s_h) + 1 = floor((28 - 3 + 1*2)/2) + 1 = 14
        #
        # Output size: 14x14x480
        self.maxPooling3 = nn.MaxPool2d(
            kernel_size=(3, 3), stride=2, padding=(1, 1)
        )

        self.inception_4a = Inception(
            in_channels=480,
            ch_1x1=192, ch_3x3_reduce=96, ch_3x3=208, ch_5x5_reduce=16, ch_5x5=48, pool_proj=64,
            conv_type=conv_type
        )
        self.inception_4b = Inception(
            in_channels=512,
            ch_1x1=160, ch_3x3_reduce=112, ch_3x3=224, ch_5x5_reduce=24, ch_5x5=64, pool_proj=64,
            conv_type=conv_type
        )
        self.inception_4c = Inception(
            in_channels=512,
            ch_1x1=128, ch_3x3_reduce=128, ch_3x3=256, ch_5x5_reduce=24, ch_5x5=64, pool_proj=64,
            conv_type=conv_type
        )
        self.inception_4d = Inception(
            in_channels=512,
            ch_1x1=112, ch_3x3_reduce=144, ch_3x3=288, ch_5x5_reduce=32, ch_5x5=64, pool_proj=64,
            conv_type=conv_type
        )
        self.inception_4e = Inception(
            in_channels=528,
            ch_1x1=256, ch_3x3_reduce=160, ch_3x3=320, ch_5x5_reduce=32, ch_5x5=128, pool_proj=128,
            conv_type=conv_type
        )

        # Input size: 14x14x832
        # Kernel size: 3x3
        # Padding size: 1x1
        # Stride: 2
        # floor((n_h - k_h + 2*p_h)/s_h) + 1 = floor((14 - 3 + 1*2)/2) + 1 = 7
        #
        # Output size: 7x7x832
        self.maxPooling4 = nn.MaxPool2d(
            kernel_size=(3, 3), stride=2, padding=(1, 1)
        )

        self.inception_5a = Inception(
            in_channels=832,
            ch_1x1=256, ch_3x3_reduce=160, ch_3x3=320, ch_5x5_reduce=32, ch_5x5=128, pool_proj=128,
            conv_type=conv_type
        )
        self.inception_5b = Inception(
            in_channels=832,
            ch_1x1=384, ch_3x3_reduce=192, ch_3x3=384, ch_5x5_reduce=48, ch_5x5=128, pool_proj=128,
            conv_type=conv_type
        )

        # Input size: 7x7x1024
        # Kernel size: 7x7
        # Padding size: 0
        # Stride: 1
        # floor((n_h - k_h + 2*p_h)/s_h) + 1 = floor((7 - 7 + 0)/2) + 1 = 1
        #
        # Output size: 1x1x1024
        self.avgPooling1 = nn.AvgPool2d(
            kernel_size=(3, 3), stride=2, padding=0
        )

        self.dropout = nn.Dropout(p=0.4, inplace=True)
        self.fc = nn.Linear(in_features=1024, out_features=num_classes, bias=True)

        if enable_aux:
            self.aux1 = AuxInception(in_channels=512, num_classes=num_classes)
            self.aux2 = AuxInception(in_channels=528, num_classes=num_classes)
        else:
            self.aux1 = None
            self.aux2 = None

    def initialize(self, criterion=None, optimizer=None, scheduler=None, weight_init=None, learning_rate=1e-2) -> None:
        if criterion is None:
            self.criterion = nn.CrossEntropyLoss()
        else:
            self.criterion = criterion

        if optimizer is None:
            self.optimizer = SGD(self.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
        else:
            self.optimizer = optimizer

        if scheduler is None:
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer=self.optimizer,
                step_size=4, gamma=0.04
            )
        else:
            self.scheduler = scheduler

        if weight_init is None:
            for m in self.modules():
                m.apply(self.init_weights)
        else:
            for m in self.modules():
                m.apply(weight_init)

    def _forward(self, X: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        # input: 224x224x3
        X = self.conv1(X)
        # input: 112x112x64
        X = self.maxPooling1(X)
        # input 56x56x64
        X = self.conv2_1(X)
        X = self.conv2_2(X)
        # input: 56x56x192
        X = self.maxPooling2(X)

        # input: 28x28x192
        X = self.inception_3a(X)
        # input: 28x28x256
        X = self.inception_3b(X)
        # input: 28x28x480
        X = self.maxPooling3(X)

        # input: 14x14x480
        X = self.inception_4a(X)

        aux1 = self.aux1(X) if (self.aux1 is not None) else None

        # input: 14x14x512
        X = self.inception_4b(X)
        # input: 14x14x512
        X = self.inception_4c(X)
        # input: 14x14x512
        X = self.inception_4d(X)

        aux2 = self.aux2(X) if (self.aux1 is not None) else None

        # input: 14x14x528
        X = self.inception_4e(X)
        # input: 14x14x528
        X = self.maxPooling4(X)

        # input: 7x7x832
        X = self.inception_5a(X)
        # input: 7x7x832
        X = self.inception_5b(X)
        # input: 7x7x1024
        X = self.avgPooling1(X)
        X = torch.flatten(X, 1)

        # input: 1x1x1024
        X = self.dropout(X)
        # input: 1x1x1000
        X = self.fc(X)

        # output: 1 x 1 x num_classes
        return X, aux1, aux2

    def forward(self, X:torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]]:
        X, aux1, aux2 = self._forward(X)
        if self.training and self.enable_aux:
            return X, aux2, aux1
        else:
            return X

    def train(self, mode=True, data=None, epochs=10) -> 'GoogLeNet':
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(device)

        if (data is None) and (mode):
            raise FileNotFoundError("\"data\" has to be a valid Dataloader object!")

        self.training = mode
        for m in self.modules():
            m.train(mode)

        if mode:
            running_loss = 0.0
            for epoch in range(0, epochs):
                for i, datum in enumerate(data, 0):
                    features, labels = datum[0].to(device), datum[1].to(device)
                    loss = self.criterion(self(features), labels)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    running_loss += loss.item()
                    batch_split = int(len(data.dataset) / data.batch_size / 5)
                    batch_split = 1 if batch_split < 1 else batch_split
                    if i % batch_split == batch_split - 1:
                        if self.verbose:
                            print(f"[epoch {epoch + 1}, batch {i + 1}] loss: {running_loss / batch_split}")
                        self.scheduler.step(epoch)
                        running_loss = 0.0

            if self.verbose:
                print('Finished Training')

        return self


