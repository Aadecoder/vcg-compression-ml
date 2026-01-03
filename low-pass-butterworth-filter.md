# Low Pass Butterworth Filter

- A Butterworth filter is a type of signal processing filter that allows low-frequency components of a signal to pass through while attenuating high-frequency components
- aka Maximally flat magnitude filter.
- Nyquist Frequency: Highest representable frequency.

## Terminology
- Decades: On frequency scale, a Decade is a tenfold increase or tenfold decrease.
- Roll-off: How quickly a filter reduces signal strength for frequencies above its cutoff frequency.

## Key Features
- Flat response in the passband (bandpass filter).
- Roll-offs towards zero in the stopband.
- The ***Capacitor*** is only the reactive element used in this filter.
- The number of Capacitors will decide the order of the filter.
- A simple first-order filter has a standard roll-off rate of 20dB/decade*.
- No ripples.
- No Oscillations.
- Smooth frequency response.
- No Artificial emphasis of any frequency inside the passband.

## Why is it used here?
- ECG/VCG has useful information below 40-50 Hz.
- Above that only noise is present and if not removed then Autoencoder wastes capacity learning noise while also increasing reconstruction error.

## Implementation
- We use Scipy a python library to achieve make this filter.

```python
scipy.signal.butter()
```

- We choose cutoff_frequency to be 40 and Nyquist Frequency to be 500.
- This results in the normal_cutoff to be 0.08 (cutoff_frequency / Nyquist)
- scipy.signal.butterworth() exprects a value between 0 and 1.
- Each channel is filtered independently.

### Inputs

```python
b, a = signal.butter(
    self.order,
    normal_cutoff,
    btype='low',
    analog=False
)
```

- The above code creates a ***Digital IIR Butterworth Filter***.
- order = 5 // for moderate roll-off.
- btype = 'low' // fow low pass filter.
- analog=False // Digital (z-domain) filter, not analog.

### Outputs
- b: Numerator coefficient
- a: denominator coefficient
- Coefficients of the Difference Equation.

## Zero Phase Filtering

```python
filtered_data[:, i] = signal.filtfilt(b, a, data[:, i])
```

- filtfilt(): Filters forward, then backward.
- This cancels phase distortion completely.

## Conclusion
A low-pass Butterworth filter was employed to remove high-frequency noise components while preserving the physiological morphology of the VCG signal. 
The Butterworth filter was chosen due to its maximally flat passband response, ensuring no amplitude ripple distortion, which is critical for minimizing reconstruction error in compression tasks. 
Zero-phase filtering using forward-backward filtering (`filterfilter`) was applied to eliminate phase distortion and preserve temporal alignment of cardiac features.
