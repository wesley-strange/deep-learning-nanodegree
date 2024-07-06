import torch
import torch.nn as nn


# define the CNN architecture
class MyModel(nn.Module):
    def __init__(self, num_classes: int = 1000, dropout: float = 0.7) -> None:

        super().__init__()

        # YOUR CODE HERE
        # Define a CNN architecture. Remember to use the variable num_classes
        # to size appropriately the output of your classifier, and if you use
        # the Dropout layer, use the variable "dropout" to indicate how much
        # to use (like nn.Dropout(p=dropout))
        
        
        # convolutional layer 1. It sees 3x224x224 image tensor
        # and produces 16 feature maps 224x224 (i.e., a tensor 16x224x224)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),  # 16 x 224 x 224
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # 16 x 112 x 112
            nn.Dropout2d(dropout)
        )
        
        # convolutional layer 2
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1), # -> 32x112x112
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # -> 32x56x56
            nn.Dropout2d(dropout)
        )
        
        # convolutional layer 3
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1), # -> 64x56x56
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # -> 64x28x28
            nn.Dropout2d(dropout)
        )
        
        # convolutional layer 4
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1), # -> 128x28x28
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # -> 128x14x14
            nn.Dropout2d(dropout)
        )
        
        # convolutional layer 5
        self.conv5 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1), # -> 256x14x14
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # -> 256x7x7
            nn.Dropout2d(dropout)
        )
        
        self.fc = nn.Sequential(
            nn.Linear(256 * 7 * 7, 5120),
            nn.BatchNorm1d(5120),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(5120, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(512, num_classes)
        )
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # YOUR CODE HERE: process the input tensor through the
        # feature extractor, the pooling and the final linear
        # layers (if appropriate for the architecture chosen)
        
        x=self.conv1(x)
        x=self.conv2(x)
        x=self.conv3(x)
        x=self.conv4(x)
        x=self.conv5(x)
        x=torch.flatten(x, 1)
        x=self.fc(x)
        
        return x


######################################################################################
#                                     TESTS
######################################################################################
import pytest


@pytest.fixture(scope="session")
def data_loaders():
    from .data import get_data_loaders

    return get_data_loaders(batch_size=2)


def test_model_construction(data_loaders):

    model = MyModel(num_classes=23, dropout=0.3)

    dataiter = iter(data_loaders["train"])
    images, labels = dataiter.next()

    out = model(images)

    assert isinstance(
        out, torch.Tensor
    ), "The output of the .forward method should be a Tensor of size ([batch_size], [n_classes])"

    assert out.shape == torch.Size(
        [2, 23]
    ), f"Expected an output tensor of size (2, 23), got {out.shape}"
