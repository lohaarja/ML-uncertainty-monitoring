# ML-uncertainty-monitoring
Tracking entropy and confidence of model predictions under progressive input perturbations to detect prediction failure.
Instead of only checking accuracy, the goal is to observe **how model confidence and entropy change before the model makes a wrong prediction**.

The system tracks prediction behavior step-by-step and detects:
- prediction failure (when the predicted class changes)
- uncertainty spikes using entropy
- confidence drop during perturbations

## How it works?
1. Train a CNN on a synthetic dataset.
2. Apply increasing perturbations to an input image.
3. Track entropy and confidence of the model predictions.
4. Detect when the prediction changes (failure).
5. Visualize model behavior with a graph.

## Example behavior
As perturbation increases:
- entropy increases (model uncertainty)
- confidence decreases
- prediction eventually flips

This helps understand model reliability under noisy inputs.
