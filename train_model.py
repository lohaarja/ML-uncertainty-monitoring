import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from synthetic_dataset import SyntheticImageDataset
from cnn import SimpleCNN
device = "cuda" if torch.cuda.is_available() else "cpu"
dataset = SyntheticImageDataset(n_samples=5000)
loader = DataLoader(dataset, batch_size=64, shuffle=True)

model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

model.train()
for epoch in range(3):  
    total_loss = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss:.2f}")
torch.save(model.state_dict(), "cnn_trained.pth")