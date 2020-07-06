import torch
import typing
import torch.nn as nn
from torch.optim import SGD, Adam
from typing import Union, TypeVar
from collections import OrderedDict

T = TypeVar('T')
Layer_Type = Union[typing.OrderedDict[str, nn.Module], nn.Module]


class VGG(nn.Module):
    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(
                tensor=m.weight,
                mode='fan_out',
                nonlinearity='relu'
            )
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(
                tensor=m.weight,
                mean=0,
                std=1e-2
            )
            nn.init.zeros_(tensor=m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    @staticmethod
    def get_vgg_block(in_channels: int, out_channels: int, block_size: int) -> nn.Sequential:
        module_list = [
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(3, 3),
                padding=(1, 1)
            ),
            nn.ReLU(inplace=True)
        ]

        for i in range(block_size - 1):
            module_list.append(
                nn.Conv2d(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    kernel_size=(3, 3),
                    padding=(1, 1)
                )
            )
            module_list.append(nn.ReLU(inplace=True))

        module_list.append(
            nn.MaxPool2d(
                kernel_size=(2, 2),
                stride=2
            )
        )
        return nn.Sequential(
            OrderedDict([(str(k), v) for k, v in enumerate(module_list)])
        )

    def __init__(self, block_size: int, block_inc: int, num_class: int, verbose=False):
        super(VGG, self).__init__()
        self.verbose = verbose
        self.num_class = num_class
        self.training = False

        self.block_size = block_size
        self.block_inc = block_inc

        self.convolution1 = self.get_vgg_block(3, 64, self.block_size)
        self.convolution2 = self.get_vgg_block(64, 128, self.block_size)
        self.convolution3 = self.get_vgg_block(128, 256, self.block_size + self.block_inc)
        self.convolution4 = self.get_vgg_block(256, 512, self.block_size + self.block_inc)
        self.convolution5 = self.get_vgg_block(512, 512, self.block_size + self.block_inc)

        self.convolution = [
            self.convolution1,
            self.convolution2,
            self.convolution3,
            self.convolution4,
            self.convolution5
        ]
        self.classifier = nn.Sequential(
            nn.Linear(in_features=(512 * 7 * 7), out_features=4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=4096, out_features=self.num_class)
        )

        self.criterion = None
        self.optimizer = None
        self.scheduler = None

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
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer=self.optimizer,
                mode='min', factor=0.1, patience=10, threshold=0.02
            )
        else:
            self.scheduler = scheduler

        if weight_init is None:
            for vgg_block in self.convolution:
                vgg_block.apply(self.init_weights)
        else:
            for vgg_block in self.convolution:
                vgg_block.apply(weight_init)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = self.convolution1(X)
        X = self.convolution2(X)
        X = self.convolution3(X)
        X = self.convolution4(X)
        X = self.convolution5(X)
        X = torch.flatten(X, 1)
        return self.classifier(X)

    def train(self, mode=True, data=None, epochs=10) -> 'VGG':
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(device)

        if (data is None) and (mode):
            raise FileNotFoundError("\"data\" has to be a valid Dataloader object!")

        self.training = mode
        for vgg_block in self.convolution:
            for module in vgg_block:
                module.train(mode)
        for module in self.classifier:
            module.train(mode)

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
                        self.scheduler.step(running_loss / batch_split)
                        running_loss = 0.0

            if self.verbose:
                print('Finished Training')

        return self
