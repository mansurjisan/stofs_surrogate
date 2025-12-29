# Future Enhancements for STOFS-GNN

## 1. Depth-Based Loss Weighting (Priority: Medium)

### Background
The 25k mesh has the following depth distribution:
- 65% of nodes have negative depth (land/intertidal)
- Negative values = elevation above datum (can become wet during surge)
- Positive values = bathymetric depth (ocean)

### Why Weighting > Hard Masking
1. **Wetting/drying physics**: Land nodes aren't irrelevant - they're where inundation happens
2. **GNN message passing**: Hard masks create edge effects at boundaries
3. **Mass conservation**: Physics loss needs whole domain for `pred.sum() vs target.sum()`

### Implementation

Add depth-based weighting to `PhysicsLoss`:

```python
class PhysicsLoss(nn.Module):
    def __init__(self, mass_weight=0.01, smooth_weight=0.001):
        super().__init__()
        self.mass_weight = mass_weight
        self.smooth_weight = smooth_weight

    def forward(self, pred, target, edge_index, depth=None):
        # pred, target: [N, 1]
        # depth: [N, 1] - negative = land, positive = ocean

        if depth is not None:
            d = depth.squeeze(-1)

            # Weight by importance for surge prediction
            # |depth| < 20m  -> coastal/intertidal (highest priority)
            # 20-200m        -> shelf
            # > 200m         -> deep ocean (lower priority)
            shallow = (d.abs() < 20.0).float()
            mid = ((d.abs() >= 20.0) & (d.abs() < 200.0)).float()
            deep = (d.abs() >= 200.0).float()

            weights = 3.0 * shallow + 1.5 * mid + 0.5 * deep
            weights = weights / (weights.mean() + 1e-8)  # normalize
        else:
            weights = torch.ones_like(pred.squeeze(-1))

        # Weighted MSE
        mse_loss = (weights * (pred.squeeze(-1) - target.squeeze(-1))**2).mean()

        # Mass term (unchanged - uses whole domain)
        pred_sum = pred.sum()
        target_sum = target.sum()
        mass_diff = (pred_sum - target_sum).abs() / (pred.shape[0] + 1e-8)
        mass_loss = torch.clamp(mass_diff, max=10.0)

        # Smoothness term (unchanged)
        row, col = edge_index
        smooth_loss = ((pred[row] - pred[col]) ** 2).mean()

        total_loss = mse_loss + self.mass_weight * mass_loss + self.smooth_weight * smooth_loss
        return total_loss, {
            'mse': mse_loss.item(),
            'mass': mass_loss.item(),
            'smooth': smooth_loss.item()
        }
```

### Training Loop Update

```python
loss, components = criterion(pred, y, edge_index, depth=batch['raw_depth'][i].to(device))

# For 2-step curriculum:
if num_steps >= 2:
    loss2, _ = criterion(pred2, y_next, edge_index, depth=batch['raw_depth'][i].to(device))
```

---

## 2. Wetness-Aware Weighting (Priority: Low)

### Concept
Weight nodes based on whether they're actually wet in the target state.

```python
if depth is not None:
    d = depth.squeeze(-1)
    eta_true = target.squeeze(-1) * ETA_SCALE  # surge in meters
    H = d + eta_true  # Total water level (height above datum)

    # More weight where there is actual water
    wet = (H > 0.05).float()   # node has water
    dry = (H <= 0.05).float()  # node is dry

    weights = 2.0 * wet + 0.5 * dry
    weights = weights / (weights.mean() + 1e-8)
```

### Benefits
- Focuses learning on actual inundation signal
- Nodes that never get wet contribute less
- Can combine with depth-based weights (multiply and renormalize)

---

## 3. Implementation Order

1. **DONE**: Fix depth vs depth_norm bug in curriculum learning
2. **DONE**: Clarify pressure normalization (already done in preprocessing)
3. **NEXT**: Run fixed scripts on ParallelWorks to establish baseline
4. **FUTURE**: Add depth-based weighting if coastal skill needs improvement
5. **FUTURE**: Add wetness-aware weighting for inundation-focused training

---

## 4. Tunable Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `shallow_thresh` | 20.0 m | Boundary between shallow and mid depths |
| `deep_thresh` | 200.0 m | Boundary between mid and deep |
| `shallow_weight` | 3.0 | Weight for coastal/intertidal nodes |
| `mid_weight` | 1.5 | Weight for shelf nodes |
| `deep_weight` | 0.5 | Weight for deep ocean nodes |
| `wet_thresh` | 0.05 m | Water depth threshold for "wet" classification |

These can be tuned based on where surge/inundation signal actually lives in the data.
