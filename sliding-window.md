# Sliding Window

## What is Sliding Window Segmentation
A sliding window splits a long, continuous signal into fixed-length overlapping segments.

So instead of feeding the network the entire VCG signal at once:
- We feed hundreds of thousands of samples.
- Each window becomes one training sample.

## Why Sliding windows are used?
### Constraints
- CNN expects fixed-size inputs.
- LSTM expects fixed-length sequences.
- GPU/CPU memory is limited.

### Without Sliding window
- Model cannot accept variable-length signals.
- Memory usage explodes.

## Implementation
### Window Configuration

```python
self.window_size = 1250
self.stride = int(window_size * (1 - overlap)) 
# where overlap = 0.5
```

- this gives stride = 625 samples.
- Each new window starts 625 samples after the previous.
- 50% overlap between windows.
- `window_size = 1250` because 1.25s at 1000Hz
