import torch
from entropy import entropy
from confidence import confidence

class BehaviorTracker:
    def __init__(self):
        self.timeline = []
    def log(self, logits, prediction):
        self.timeline.append({
            "entropy": entropy(logits).item(),
            "confidence": confidence(logits).item(),
            "prediction": prediction
        })
    def get_timeline(self):
        return self.timeline