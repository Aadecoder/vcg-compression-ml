# z-score normalization

## What is z-score normalization
- z-score normalization rescales a signal so that:
  - mean = 0
  - Standard Deviation = 1
- Mathematically for a signal x:
    `z = (x - μ)/σ`
  Where:
  - μ = mean of signal
  - σ = standard deviation of the signal

## Why normalization is necessary
Problems Without normalization:
- One channel (e.g., Z) may dominate due to larger amplitude.
- Gradients explode or vanish .
- Training becomes unstable or very slow.
- Bottleneck learns scale instead of structure.
- PRD and NMSE becomes meaningless.

Normalization fixes: 
- x,y,z having different amplitude ranges.
- Baseline offsets vary per patient.
- Energy differs across recordings.

## Implementation
### Mean and standard deviation Computation

```python
if fit:
  self.mean = np.mean(data, axis=0)
  self.std = np.std(data, axis=0)
```

- `axis = 0` -> computes statistics per channel
- `self.mean` shape -> (3,)
- `self.std` shape -> (3,)

### Normalization Formula

```python
normalized = (data - self.mean) / (self.std + 1e-8)
```

This is the exact mathematical z-score formula

Why 1e-8?
- Prevents division by zero.
- Numerical Stability.
- Especially important for flat segments. 

### Inverse Normalization

```python
def inverse_normalize(self, normalized_data):
  return normalized_data * self.std + self.mean
```

Why this matters?
- Autoencoder works in normalized space.
- Metrics like PRD, SNR must be computed in original amplitude space.
- Reconstruction plots must reflect real signal magnitudes.

## Conclusion
Z-score normalization was applied to the filtered VCG signals to ensure zero mean and unit variance for each orthogonal component. 
This normalization stabilizes network training, prevents dominance of high-amplitude channels, and improves convergence of the CNN–LSTM autoencoder. 
An inverse normalization step was applied after reconstruction to compute performance metrics in the original signal domain.
