import torch
from torch.utils.data import Dataset
class SyntheticImageDataset(Dataset):
    def __init__(self, n_samples=5000, img_size=28, n_classes=10):
        self.n_samples = n_samples
        self.img_size = img_size
        self.n_classes = n_classes
        self.images = torch.zeros(n_samples, 1, img_size, img_size)
        self.labels = torch.randint(0, n_classes, (n_samples,))
        for i in range(n_samples):
            label = self.labels[i]
            x = (label % 5) * 5
            y = (label // 5) * 5
            self.images[i, 0, x:x+5, y:y+5] = 1.0
        self.images += 0.05 * torch.randn_like(self.images)
        self.images = torch.clamp(self.images, 0, 1)
    def __len__(self):
        return self.n_samples
    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]