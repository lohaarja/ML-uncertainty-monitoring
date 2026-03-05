import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, img_size=28, n_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(img_size * img_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, n_classes)
        )
    def forward(self, x):
        return self.net(x)