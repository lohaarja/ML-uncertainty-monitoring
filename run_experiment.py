import torch
import matplotlib.pyplot as plt
from synthetic_dataset import SyntheticImageDataset
from perturbations import apply_perturbation
from cnn import SimpleCNN
from behavior_tracker import BehaviorTracker
from entropy import entropy

device = "cuda" if torch.cuda.is_available() else "cpu"
dataset = SyntheticImageDataset()

model = SimpleCNN().to(device)
model.load_state_dict(torch.load("cnn_trained.pth", map_location=device))
model.eval()

x, y = dataset[0]
x = x.unsqueeze(0).to(device)
tracker = BehaviorTracker()
true_label = y.item()
failed = False

for step in range(1, 20):
    x_perturbed = apply_perturbation(x, step)
    logits = model(x_perturbed)
    pred = logits.argmax(dim=-1).item()
    tracker.log(logits, pred)
    if entropy(logits).item() > 1.5:
        print(f"Warning: model uncertainty spike at step {step}")
    if pred != true_label and not failed:
        print(f"Failure at step {step}")
        failed = True

timeline = tracker.get_timeline()
print(timeline)
entropy_values = [entry["entropy"] for entry in timeline]
confidence_values = [entry["confidence"] for entry in timeline]
plt.figure(figsize=(8,5))
plt.plot(entropy_values, label="Entropy")
plt.plot(confidence_values, label="Confidence")
plt.xlabel("Perturbation Step")
plt.ylabel("Value")
plt.title("Model Behavior Under Perturbations")
plt.legend()
plt.show()