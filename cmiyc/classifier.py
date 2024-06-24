import numpy as np
import cv2
import torch


def conv_layer(in_channels: int, out_channels: int, kernel_size) -> torch.nn.Sequential:
    return torch.nn.Sequential(
        torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=(1, 1), padding='same'),
        torch.nn.BatchNorm2d(out_channels, eps=1e-6, momentum=0.99),
        torch.nn.ReLU()
    )


def conv_block(in_channels: int, spec: list) -> torch.nn.Sequential:
    layers = []
    for out_channels, kernel_size in spec:
        layers.append(conv_layer(in_channels, out_channels, kernel_size))
        in_channels = out_channels
    return torch.nn.Sequential(*layers)


class Encoder(torch.nn.Module):

    def __init__(self):
        super(Encoder, self).__init__()
        self.stem = conv_block(3, [(32, (3, 3)), (48, (7, 1)), (48, (1, 7))])
        self.pool = torch.nn.MaxPool2d(2, stride=2, padding=0)
        self.dropout_1 = torch.nn.Dropout(0.2)
        self.branch_1 = conv_block(48, [(64, (3, 1)), (64, (1, 3))])
        self.branch_2 = conv_block(48, [(64, (1, 7)), (64, (7, 1))])
        self.dropout_2 = torch.nn.Dropout(0.2)
        self.block = conv_block(128, [(128, (3, 3)), (256, (3, 3))])
        self.dropout_3 = torch.nn.Dropout(0.3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # `x` has shape `(N, 3, 48, 48)`
        y = self.stem(x)                    # `(N, 48, 48, 48)`
        y = self.pool(y)                    # `(N, 48, 24, 24)`
        y = self.dropout_1(y)
        b1 = self.branch_1(y)               # `(N, 64, 24, 24)`
        b2 = self.branch_2(y)               # `(N, 64, 24, 24)`
        y = torch.cat([b1, b2], dim=1)      # `(N, 128, 24, 24)`
        y = self.pool(y)                    # `(N, 128, 12, 12)`
        y = self.dropout_2(y)
        y = self.block(y)
        y = self.pool(y)                    # `(N, 256, 6, 6)`
        y = self.dropout_3(y)
        return torch.flatten(y, start_dim=1)   # `(N, 9216)`


class Head(torch.nn.Module):

    def __init__(self, n_classes: int):
        super(Head, self).__init__()
        self.linear_1 = torch.nn.Linear(9216, 256)
        self.batch_norm = torch.nn.BatchNorm1d(256, eps=1e-6, momentum=0.99)
        self.dropout = torch.nn.Dropout(0.4)
        self.linear_2 = torch.nn.Linear(256, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # `x` has shape `(N, 9216)`
        y = self.linear_1(x)
        y = self.batch_norm(y)
        y = torch.relu(y)
        y = self.dropout(y)
        return self.linear_2(y)


class Classifier(torch.nn.Module):

    def __init__(self, n_classes: int):
        super(Classifier, self).__init__()
        self.mean = torch.nn.Parameter(torch.zeros(1, 3, 1, 1), requires_grad=False)
        self.std = torch.nn.Parameter(torch.ones(1, 3, 1, 1), requires_grad=False)
        self.encoder = Encoder()
        self.head = Head(n_classes)
        self.input_size = 48

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.mean) / self.std
        return self.head(self.encoder(x))
    
    def classify(self, img: np.ndarray) -> int:
        """
        This is a mere convenience method, for handling Numpy input images of arbitrary size.
        """

        img = cv2.resize(img, (self.input_size, self.input_size))

        with torch.inference_mode():
            pred = self.forward(torch.tensor(img.astype(np.float32).transpose(2, 0, 1)).unsqueeze(0))[0]

        return torch.argmax(pred).item()
