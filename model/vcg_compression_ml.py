"""
VCG Signal Compression using CNN-LSTM Autoencoder
Optimized for Raspberry Pi deployment
"""

import numpy as np
import wfdb
import pywt
from scipy import signal
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, regularizers
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
from pathlib import Path

# ============================================================================
# PART 1: ECG to VCG Conversion using Frank's Transform
# ============================================================================

class ECGtoVCGConverter:
    """
    Converts ECG signals to VCG using Frank's transformation
    Frank's lead system provides orthogonal X, Y, Z components
    """
    
    def __init__(self):
        # Frank's transformation coefficients for standard 12-lead ECG
        # These convert standard ECG leads to orthogonal VCG (X, Y, Z)
        self.frank_coefficients = {
            'X': {'I': 0.610, 'II': -0.171, 'V1': -0.781, 'V2': -0.516, 
                  'V3': -0.044, 'V4': 0.456, 'V5': 0.815, 'V6': 0.891},
            'Y': {'I': -0.233, 'II': 0.887, 'V1': 0.022, 'V2': -0.106, 
                  'V3': -0.229, 'V4': -0.310, 'V5': -0.246, 'V6': -0.063},
            'Z': {'I': 0.127, 'II': 0.022, 'V1': -0.229, 'V2': -0.310, 
                  'V3': -0.246, 'V4': -0.063, 'V5': 0.055, 'V6': 0.108}
        }
    
    def load_ptb_record(self, record_path):
        """
        Load PTB Diagnostic ECG record (.dat format)
        
        Args:
            record_path: Path to PTB record (without extension)
        
        Returns:
            signals: ECG signals array
            fields: Metadata dictionary
        """
        try:
            record = wfdb.rdrecord(record_path)
            return record.p_signal, record.__dict__
        except Exception as e:
            print(f"Error loading record {record_path}: {e}")
            return None, None
    
    def convert_to_vcg(self, ecg_signals, lead_names):
        """
        Convert multi-lead ECG to 3-lead VCG (X, Y, Z)
        
        For PTB database which has 15 leads, we'll use a simplified approach:
        - Use leads I, II, and V1-V6 if available
        - If not all leads available, use available leads with adjusted coefficients
        
        Args:
            ecg_signals: Array of shape (samples, leads)
            lead_names: List of lead names corresponding to columns
        
        Returns:
            vcg_signals: Array of shape (samples, 3) representing X, Y, Z
        """
        n_samples = ecg_signals.shape[0]
        vcg_signals = np.zeros((n_samples, 3))
        
        # Create a mapping of lead names to signal columns
        lead_map = {name.strip().upper(): idx for idx, name in enumerate(lead_names)}
        
        # Simplified VCG derivation for PTB database
        # PTB has leads: 'i', 'ii', 'iii', 'avr', 'avl', 'avf', 'v1'-'v6'
        
        # X component (Left-Right): Primarily from lead I and chest leads
        if 'I' in lead_map:
            vcg_signals[:, 0] += 0.5 * ecg_signals[:, lead_map['I']]
        if 'V1' in lead_map:
            vcg_signals[:, 0] += -0.3 * ecg_signals[:, lead_map['V1']]
        if 'V6' in lead_map:
            vcg_signals[:, 0] += 0.4 * ecg_signals[:, lead_map['V6']]
        
        # Y component (Superior-Inferior): Primarily from lead II and aVF
        if 'II' in lead_map:
            vcg_signals[:, 1] += 0.6 * ecg_signals[:, lead_map['II']]
        if 'AVF' in lead_map:
            vcg_signals[:, 1] += 0.4 * ecg_signals[:, lead_map['AVF']]
        
        # Z component (Anterior-Posterior): Primarily from chest leads
        if 'V1' in lead_map:
            vcg_signals[:, 2] += 0.3 * ecg_signals[:, lead_map['V1']]
        if 'V2' in lead_map:
            vcg_signals[:, 2] += 0.2 * ecg_signals[:, lead_map['V2']]
        if 'V5' in lead_map:
            vcg_signals[:, 2] += -0.3 * ecg_signals[:, lead_map['V5']]
        if 'V6' in lead_map:
            vcg_signals[:, 2] += -0.2 * ecg_signals[:, lead_map['V6']]
        
        return vcg_signals
    
    def process_ptb_database(self, data_dir, output_dir, max_records=None):
        """
        Process entire PTB database and convert to VCG
        
        PTB database structure: patient folders containing record files
        Each record has .dat, .hea, and .xyz files
        
        Args:
            data_dir: Directory containing PTB patient folders
            output_dir: Directory to save VCG signals
            max_records: Maximum number of records to process (None for all)
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Find all patient directories
        data_path = Path(data_dir)
        patient_dirs = [d for d in data_path.iterdir() if d.is_dir() and d.name.startswith('patient')]
        
        vcg_data = []
        processed_count = 0
        
        print(f"Found {len(patient_dirs)} patient directories")
        
        for patient_dir in patient_dirs:
            # Find all .hea files in this patient directory
            hea_files = list(patient_dir.glob("*.hea"))
            
            for hea_file in hea_files:
                if max_records and processed_count >= max_records:
                    break
                
                record_path = str(hea_file).replace('.hea', '')
                record_name = hea_file.stem
                
                print(f"Processing {processed_count + 1}: {patient_dir.name}/{record_name}")
                
                signals, fields = self.load_ptb_record(record_path)
                
                if signals is not None and signals.shape[0] > 0:
                    lead_names = fields.get('sig_name', [])
                    
                    # Convert to VCG
                    vcg = self.convert_to_vcg(signals, lead_names)
                    
                    # Only keep if we got valid VCG data
                    if np.abs(vcg).max() > 0:
                        vcg_data.append(vcg)
                        
                        # Save individual VCG file
                        np.save(f"{output_dir}/vcg_{patient_dir.name}_{record_name}.npy", vcg)
                        processed_count += 1
                    else:
                        print(f"  Warning: Generated VCG has all zeros, skipping")
                else:
                    print(f"  Warning: Could not load signals")
            
            if max_records and processed_count >= max_records:
                break
        
        # Combine all VCG signals
        if vcg_data:
            all_vcg = np.concatenate(vcg_data, axis=0)
            np.save(f"{output_dir}/all_vcg_signals.npy", all_vcg)
            print(f"\nSuccessfully processed {processed_count} records")
            print(f"Total VCG data shape: {all_vcg.shape}")
            print(f"Data saved to: {output_dir}")
        else:
            print("\nWarning: No VCG data was generated!")
            
        return vcg_data


# ============================================================================
# PART 2: Preprocessing - Butterworth Filter & Z-Score Normalization
# ============================================================================

class VCGPreprocessor:
    """
    Preprocessing for VCG signals:
    1. Low-pass Butterworth filter
    2. Z-score normalization
    """
    
    def __init__(self, cutoff_freq=40, fs=1000, order=5):
        """
        Args:
            cutoff_freq: Cutoff frequency for low-pass filter (Hz)
            fs: Sampling frequency (Hz)
            order: Filter order
        """
        self.cutoff_freq = cutoff_freq
        self.fs = fs
        self.order = order
        self.mean = None
        self.std = None
    
    def butterworth_filter(self, data):
        """
        Apply low-pass Butterworth filter to remove high-frequency noise
        """
        nyquist = 0.5 * self.fs
        normal_cutoff = self.cutoff_freq / nyquist
        b, a = signal.butter(self.order, normal_cutoff, btype='low', analog=False)
        
        # Apply filter to each channel
        filtered_data = np.zeros_like(data)
        for i in range(data.shape[1]):
            filtered_data[:, i] = signal.filtfilt(b, a, data[:, i])
        
        return filtered_data
    
    def z_score_normalize(self, data, fit=True):
        """
        Z-score normalization: (x - mean) / std
        
        Args:
            data: Input data
            fit: If True, calculate mean and std; if False, use stored values
        """
        if fit:
            self.mean = np.mean(data, axis=0)
            self.std = np.std(data, axis=0)
        
        normalized = (data - self.mean) / (self.std + 1e-8)
        return normalized
    
    def inverse_normalize(self, normalized_data):
        """
        Inverse Z-score normalization
        """
        return normalized_data * self.std + self.mean
    
    def preprocess(self, data, fit=True):
        """
        Complete preprocessing pipeline
        """
        # Step 1: Butterworth filter
        filtered = self.butterworth_filter(data)
        
        # Step 2: Z-score normalization
        normalized = self.z_score_normalize(filtered, fit=fit)
        
        return normalized


# ============================================================================
# PART 3: Data Preparation for Training
# ============================================================================

class VCGDataGenerator:
    """
    Prepare VCG data for autoencoder training
    Creates sliding windows from continuous signals
    """
    
    def __init__(self, window_size=1250, overlap=0.5):
        """
        Args:
            window_size: Number of samples per window (1250 ≈ 1.25s at 1000Hz)
            overlap: Overlap ratio between consecutive windows
        """
        self.window_size = window_size
        self.stride = int(window_size * (1 - overlap))
    
    def create_windows(self, vcg_data):
        """
        Create sliding windows from VCG signal
        
        Args:
            vcg_data: Array of shape (samples, 3) for X, Y, Z
        
        Returns:
            windows: Array of shape (n_windows, window_size, 3)
        """
        n_samples = vcg_data.shape[0]
        n_windows = (n_samples - self.window_size) // self.stride + 1
        
        windows = np.zeros((n_windows, self.window_size, 3))
        
        for i in range(n_windows):
            start_idx = i * self.stride
            end_idx = start_idx + self.window_size
            windows[i] = vcg_data[start_idx:end_idx]
        
        return windows
    
    def prepare_dataset(self, vcg_files, preprocessor, test_size=0.15):
        """
        Prepare complete dataset for training
        
        Args:
            vcg_files: List of VCG .npy files
            preprocessor: VCGPreprocessor instance
            test_size: Fraction of data for testing
        
        Returns:
            X_train, X_test: Training and test sets
        """
        all_windows = []
        
        for vcg_file in vcg_files:
            # Load VCG data
            vcg_data = np.load(vcg_file)
            
            # Preprocess
            preprocessed = preprocessor.preprocess(vcg_data, fit=True)
            
            # Create windows
            windows = self.create_windows(preprocessed)
            all_windows.append(windows)
        
        # Concatenate all windows
        all_windows = np.concatenate(all_windows, axis=0)
        
        # Split into train/test
        X_train, X_test = train_test_split(
            all_windows, 
            test_size=test_size, 
            random_state=42
        )
        
        print(f"Training samples: {X_train.shape[0]}")
        print(f"Test samples: {X_test.shape[0]}")
        print(f"Window shape: {X_train.shape[1:]}")
        
        return X_train, X_test


# ============================================================================
# PART 4: CNN-LSTM Autoencoder Model (Based on Methodology)
# ============================================================================

class VCGAutoencoder:
    """
    CNN-LSTM Autoencoder for VCG signal compression
    Architecture based on the research methodology
    
    Encoder: Conv1D blocks → LSTM layers → Dense (bottleneck)
    Decoder: Dense → LSTM layers → Conv1DTranspose blocks
    """
    
    def __init__(self, input_shape=(1250, 3), compression_ratio=30):
        """
        Args:
            input_shape: Shape of input VCG window (samples, channels)
            compression_ratio: Target compression ratio
        """
        self.input_shape = input_shape
        self.compression_ratio = compression_ratio
        
        # Calculate bottleneck size based on compression ratio
        self.bottleneck_size = int(
            (input_shape[0] * input_shape[1]) / compression_ratio
        )
        
        self.model = None
        self.encoder = None
        self.decoder = None
    
    def build_encoder(self):
        """
        Build encoder network: Conv1D blocks → LSTM → Dense
        
        Architecture from methodology:
        - 3 Conv1D blocks (32, 64, 128 filters)
        - 3 LSTM layers (128, 128, 64 units) with dropout
        - Dense bottleneck layer
        """
        inputs = layers.Input(shape=self.input_shape, name='encoder_input')
        
        # Convolutional Block 1: 32 filters, kernel=5
        x = layers.Conv1D(32, kernel_size=5, padding='same', 
                         name='encoder_conv1')(inputs)
        x = layers.BatchNormalization(name='encoder_bn1')(x)
        x = layers.ReLU(name='encoder_relu1')(x)
        x = layers.MaxPooling1D(pool_size=2, name='encoder_pool1')(x)  # 1250 → 625
        
        # Convolutional Block 2: 64 filters, kernel=5
        x = layers.Conv1D(64, kernel_size=5, padding='same', 
                         name='encoder_conv2')(x)
        x = layers.BatchNormalization(name='encoder_bn2')(x)
        x = layers.ReLU(name='encoder_relu2')(x)
        x = layers.MaxPooling1D(pool_size=2, name='encoder_pool2')(x)  # 625 → 312
        
        # Convolutional Block 3: 128 filters, kernel=5
        x = layers.Conv1D(128, kernel_size=5, padding='same', 
                         name='encoder_conv3')(x)
        x = layers.BatchNormalization(name='encoder_bn3')(x)
        x = layers.ReLU(name='encoder_relu3')(x)
        x = layers.MaxPooling1D(pool_size=2, name='encoder_pool3')(x)  # 312 → 156
        
        # LSTM layers for temporal dependencies
        # LSTM 1: 128 units
        x = layers.LSTM(128, return_sequences=True, dropout=0.2,
                       name='encoder_lstm1')(x)
        x = layers.BatchNormalization(name='encoder_lstm_bn1')(x)
        
        # LSTM 2: 128 units
        x = layers.LSTM(128, return_sequences=True, dropout=0.2,
                       name='encoder_lstm2')(x)
        x = layers.BatchNormalization(name='encoder_lstm_bn2')(x)
        
        # LSTM 3: 64 units (final sequence layer)
        x = layers.LSTM(64, return_sequences=False, dropout=0.2,
                       name='encoder_lstm3')(x)
        x = layers.BatchNormalization(name='encoder_lstm_bn3')(x)
        
        # Flatten and create bottleneck (compressed representation)
        x = layers.Dense(self.bottleneck_size, 
                        activation='relu',
                        name='bottleneck')(x)
        
        encoder = models.Model(inputs, x, name='encoder')
        return encoder
    
    def build_decoder(self):
        """
        Build decoder network: Dense → LSTM → Conv1DTranspose
        
        Mirrors the encoder architecture to reconstruct the signal
        """
        # Calculate intermediate shape after encoding
        # After 3 MaxPooling layers: 1250 -> 625 -> 312 -> 156
        intermediate_length = 156  # Explicit calculation
        intermediate_features = 64  # Last LSTM output
        
        inputs = layers.Input(shape=(self.bottleneck_size,), name='decoder_input')
        
        # Dense layer to expand bottleneck
        x = layers.Dense(intermediate_length * intermediate_features, 
                        activation='relu',
                        name='decoder_dense')(inputs)
        x = layers.Reshape((intermediate_length, intermediate_features),
                          name='decoder_reshape')(x)
        
        # LSTM layers (reverse of encoder)
        # LSTM 1: 64 units
        x = layers.LSTM(64, return_sequences=True, dropout=0.2,
                       name='decoder_lstm1')(x)
        x = layers.BatchNormalization(name='decoder_lstm_bn1')(x)
        
        # LSTM 2: 128 units
        x = layers.LSTM(128, return_sequences=True, dropout=0.2,
                       name='decoder_lstm2')(x)
        x = layers.BatchNormalization(name='decoder_lstm_bn2')(x)
        
        # LSTM 3: 128 units
        x = layers.LSTM(128, return_sequences=True, dropout=0.2,
                       name='decoder_lstm3')(x)
        x = layers.BatchNormalization(name='decoder_lstm_bn3')(x)
        
        # Conv1DTranspose blocks (reverse of encoder)
        # Upsampling 1: 156 → 312
        x = layers.UpSampling1D(size=2, name='decoder_upsample1')(x)
        x = layers.Conv1DTranspose(128, kernel_size=5, padding='same',
                                   name='decoder_conv1')(x)
        x = layers.BatchNormalization(name='decoder_bn1')(x)
        x = layers.ReLU(name='decoder_relu1')(x)
        
        # Upsampling 2: 312 → 624
        x = layers.UpSampling1D(size=2, name='decoder_upsample2')(x)
        x = layers.Conv1DTranspose(64, kernel_size=5, padding='same',
                                   name='decoder_conv2')(x)
        x = layers.BatchNormalization(name='decoder_bn2')(x)
        x = layers.ReLU(name='decoder_relu2')(x)
        
        # Upsampling 3: 624 → 1248
        x = layers.UpSampling1D(size=2, name='decoder_upsample3')(x)
        x = layers.Conv1DTranspose(32, kernel_size=5, padding='same',
                                   name='decoder_conv3')(x)
        x = layers.BatchNormalization(name='decoder_bn3')(x)
        x = layers.ReLU(name='decoder_relu3')(x)
        
        # Now at 1248, need to get to 1250
        # Use ZeroPadding1D to add 2 samples (1 on each side)
        x = layers.ZeroPadding1D(padding=(1, 1), name='decoder_padding')(x)  # 1248 → 1250
        
        # Final layer to get 3 channels (X, Y, Z)
        x = layers.Conv1D(self.input_shape[1], kernel_size=3, 
                         padding='same',
                         activation='linear',
                         name='decoder_output')(x)
        
        decoder = models.Model(inputs, x, name='decoder')
        return decoder
    
    def build_model(self):
        """
        Build complete autoencoder by connecting encoder and decoder
        """
        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()
        
        # Connect encoder and decoder
        inputs = layers.Input(shape=self.input_shape, name='autoencoder_input')
        encoded = self.encoder(inputs)
        decoded = self.decoder(encoded)
        
        self.model = models.Model(inputs, decoded, name='vcg_autoencoder')
        
        print("\n" + "="*70)
        print("ENCODER ARCHITECTURE")
        print("="*70)
        self.encoder.summary()
        
        print("\n" + "="*70)
        print("DECODER ARCHITECTURE")
        print("="*70)
        self.decoder.summary()
        
        print("\n" + "="*70)
        print("COMPLETE AUTOENCODER")
        print("="*70)
        self.model.summary()
        
        return self.model
    
    def compile_model(self, learning_rate=0.001):
        """
        Compile model with optimizer and loss function
        Using Mean Squared Error (MSE) as per the paper
        """
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        
        self.model.compile(
            optimizer=optimizer,
            loss='mse',  # Mean Squared Error
            metrics=['mae']  # Mean Absolute Error for monitoring
        )
        
        print(f"\nModel compiled with:")
        print(f"  - Optimizer: Adam (lr={learning_rate})")
        print(f"  - Loss: MSE")
        print(f"  - Bottleneck size: {self.bottleneck_size}")
        print(f"  - Compression ratio: ~{self.compression_ratio}:1")
    
    def train(self, X_train, X_test, epochs=120, batch_size=32):
        """
        Train the autoencoder model
        
        Args:
            X_train: Training data
            X_test: Validation data
            epochs: Number of training epochs (paper uses 120)
            batch_size: Batch size for training
        """
        # Check if we have enough data
        if X_train.shape[0] < 100:
            print(f"\n⚠️  WARNING: Very small training set ({X_train.shape[0]} samples)")
            print("    Reducing epochs and adjusting learning rate for stability")
            epochs = min(epochs, 50)
            learning_rate = 0.0005
            self.model.optimizer.learning_rate.assign(learning_rate)
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=20,  # Increased patience
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=10,
                min_lr=1e-7,
                verbose=1
            ),
            keras.callbacks.ModelCheckpoint(
                'best_vcg_autoencoder.keras',  # Use .keras format
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            )
        ]
        
        print(f"\nStarting training...")
        print(f"  - Training samples: {X_train.shape[0]}")
        print(f"  - Validation samples: {X_test.shape[0]}")
        print(f"  - Epochs: {epochs}")
        print(f"  - Batch size: {batch_size}")
        print(f"  - Input range: [{X_train.min():.3f}, {X_train.max():.3f}]")
        
        history = self.model.fit(
            X_train, X_train,  # Autoencoder: input = target
            validation_data=(X_test, X_test),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def compress(self, vcg_signal):
        """
        Compress VCG signal using encoder
        
        Args:
            vcg_signal: Input VCG signal (batch, samples, channels)
        
        Returns:
            compressed: Compressed representation
        """
        return self.encoder.predict(vcg_signal, verbose=0)
    
    def decompress(self, compressed_signal):
        """
        Decompress VCG signal using decoder
        
        Args:
            compressed_signal: Compressed representation
        
        Returns:
            reconstructed: Reconstructed VCG signal
        """
        return self.decoder.predict(compressed_signal, verbose=0)
    
    def save_models(self, encoder_path='encoder.h5', decoder_path='decoder.h5'):
        """Save encoder and decoder separately for deployment"""
        self.encoder.save(encoder_path)
        self.decoder.save(decoder_path)
        print(f"Models saved: {encoder_path}, {decoder_path}")


# ============================================================================
# PART 5: Performance Evaluation Metrics (As per Paper)
# ============================================================================

class PerformanceEvaluator:
    """
    Calculate all performance metrics as described in the paper:
    - Compression Ratio (CR)
    - Mean Squared Error (MSE)
    - Root Mean Square Error (RMSE)
    - Normalized MSE (NMSE)
    - Percentage RMS Difference (PRD)
    - PRD Normalized (PRDN)
    - Signal-to-Noise Ratio (SNR)
    - Peak Signal-to-Noise Ratio (PSNR)
    - Quality Score (QS)
    """
    
    @staticmethod
    def compression_ratio(original_size, compressed_size):
        """
        CR = Original Size / Compressed Size
        """
        return original_size / compressed_size
    
    @staticmethod
    def mse(original, reconstructed):
        """
        MSE = (1/n) * Σ(xi - x̂i)²
        """
        return np.mean((original - reconstructed) ** 2)
    
    @staticmethod
    def rmse(original, reconstructed):
        """
        RMSE = √(MSE)
        """
        return np.sqrt(PerformanceEvaluator.mse(original, reconstructed))
    
    @staticmethod
    def nmse(original, reconstructed):
        """
        NMSE = Σ(xi - x̂i)² / Σ(xi - x̄)²
        """
        numerator = np.sum((original - reconstructed) ** 2)
        denominator = np.sum((original - np.mean(original)) ** 2)
        return numerator / (denominator + 1e-10)
    
    @staticmethod
    def prd(original, reconstructed):
        """
        PRD = √(Σ(xi - x̂i)² / Σ(xi)²) * 100
        """
        numerator = np.sum((original - reconstructed) ** 2)
        denominator = np.sum(original ** 2)
        return np.sqrt(numerator / (denominator + 1e-10)) * 100
    
    @staticmethod
    def prdn(original, reconstructed):
        """
        PRDN = PRD * Range / 100
        where Range = max(xi) - min(xi)
        """
        prd_value = PerformanceEvaluator.prd(original, reconstructed)
        signal_range = np.max(original) - np.min(original)
        return (prd_value * signal_range) / 100
    
    @staticmethod
    def snr(original, reconstructed):
        """
        SNR = 10 * log10(Signal Power / Noise Power)
        """
        signal_power = np.sum(original ** 2)
        noise_power = np.sum((original - reconstructed) ** 2)
        return 10 * np.log10(signal_power / (noise_power + 1e-10))
    
    @staticmethod
    def psnr(original, reconstructed):
        """
        PSNR = 10 * log10(MAX² / MSE)
        """
        max_value = np.max(np.abs(original))
        mse_value = PerformanceEvaluator.mse(original, reconstructed)
        return 10 * np.log10((max_value ** 2) / (mse_value + 1e-10))
    
    @staticmethod
    def quality_score(cr, prd):
        """
        QS = CR / PRD
        Higher QS indicates better compression quality
        """
        return cr / (prd + 1e-10)
    
    @staticmethod
    def evaluate_all(original, reconstructed, compressed_size):
        """
        Calculate all metrics and return as dictionary
        
        Args:
            original: Original VCG signal
            reconstructed: Reconstructed VCG signal
            compressed_size: Size of compressed representation
        
        Returns:
            metrics: Dictionary of all performance metrics
        """
        original_size = original.size
        
        cr = PerformanceEvaluator.compression_ratio(original_size, compressed_size)
        mse = PerformanceEvaluator.mse(original, reconstructed)
        rmse = PerformanceEvaluator.rmse(original, reconstructed)
        nmse = PerformanceEvaluator.nmse(original, reconstructed)
        prd = PerformanceEvaluator.prd(original, reconstructed)
        prdn = PerformanceEvaluator.prdn(original, reconstructed)
        snr = PerformanceEvaluator.snr(original, reconstructed)
        psnr = PerformanceEvaluator.psnr(original, reconstructed)
        qs = PerformanceEvaluator.quality_score(cr, prd)
        
        metrics = {
            'CR': cr,
            'MSE': mse,
            'RMSE': rmse,
            'NMSE': nmse,
            'PRD': prd,
            'PRDN': prdn,
            'SNR': snr,
            'PSNR': psnr,
            'QS': qs
        }
        
        return metrics
    
    @staticmethod
    def print_metrics(metrics):
        """Pretty print all metrics"""
        print("\n" + "="*70)
        print("PERFORMANCE METRICS")
        print("="*70)
        print(f"Compression Ratio (CR):           {metrics['CR']:.2f}")
        print(f"Mean Squared Error (MSE):         {metrics['MSE']:.6f}")
        print(f"Root Mean Square Error (RMSE):    {metrics['RMSE']:.4f}")
        print(f"Normalized MSE (NMSE):            {metrics['NMSE']:.6f}")
        print(f"Percentage RMS Difference (PRD):  {metrics['PRD']:.2f}%")
        print(f"PRD Normalized (PRDN):            {metrics['PRDN']:.2f}")
        print(f"Signal-to-Noise Ratio (SNR):      {metrics['SNR']:.2f} dB")
        print(f"Peak SNR (PSNR):                  {metrics['PSNR']:.2f} dB")
        print(f"Quality Score (QS):               {metrics['QS']:.2f}")
        print("="*70)


# ============================================================================
# PART 6: Visualization Functions
# ============================================================================

def plot_training_history(history):
    """Plot training and validation loss"""
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Training History')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Training MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.title('Mean Absolute Error')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    print("  ✓ Saved: training_history.png")
    plt.close()


def plot_vcg_comparison(original, reconstructed, sample_idx=0):
    """
    Plot original vs reconstructed VCG signals for all 3 channels
    """
    fig, axes = plt.subplots(3, 1, figsize=(15, 10))
    channels = ['X (Frontal)', 'Y (Sagittal)', 'Z (Horizontal)']
    
    for i, (ax, channel) in enumerate(zip(axes, channels)):
        time_axis = np.arange(len(original[sample_idx, :, i])) / 1000  # Convert to seconds
        
        ax.plot(time_axis, original[sample_idx, :, i], 
                label='Original', linewidth=1.5, alpha=0.7)
        ax.plot(time_axis, reconstructed[sample_idx, :, i], 
                label='Reconstructed', linewidth=1.5, alpha=0.7, linestyle='--')
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude')
        ax.set_title(f'VCG Channel {channel}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('vcg_comparison.png', dpi=300, bbox_inches='tight')
    print("  ✓ Saved: vcg_comparison.png")
    plt.close()


def plot_3d_vcg(vcg_signal, title='VCG Signal', filename='vcg_3d_trajectory.png'):
    """
    Plot 3D vectorcardiogram (X, Y, Z trajectory)
    """
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot VCG trajectory
    ax.plot(vcg_signal[:, 0], vcg_signal[:, 1], vcg_signal[:, 2], 
            linewidth=1.5, alpha=0.7)
    
    # Mark start and end points
    ax.scatter(vcg_signal[0, 0], vcg_signal[0, 1], vcg_signal[0, 2], 
               c='green', s=100, label='Start', marker='o')
    ax.scatter(vcg_signal[-1, 0], vcg_signal[-1, 1], vcg_signal[-1, 2], 
               c='red', s=100, label='End', marker='x')
    
    ax.set_xlabel('X (Frontal)')
    ax.set_ylabel('Y (Sagittal)')
    ax.set_zlabel('Z (Horizontal)')
    ax.set_title(title)
    ax.legend()
    
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {filename}")
    plt.close()


# ============================================================================
# PART 7: Raspberry Pi Optimization
# ============================================================================

def optimize_for_raspberry_pi(model_path='best_vcg_autoencoder.keras', 
                               output_path='vcg_model_optimized.tflite'):
    """
    Convert Keras model to TensorFlow Lite for Raspberry Pi deployment
    
    TFLite optimizations:
    - Quantization: Reduces model size and increases inference speed
    - Pruning: Removes unnecessary weights
    """
    # Load the trained model
    model = keras.models.load_model(model_path)
    
    # Convert to TensorFlow Lite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Apply optimizations
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # Optional: Quantize to int8 for even faster inference
    # This requires representative dataset
    # converter.target_spec.supported_types = [tf.int8]
    
    # Convert
    tflite_model = converter.convert()
    
    # Save the optimized model
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    
    # Print size comparison
    original_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
    optimized_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
    
    print(f"\nModel Optimization for Raspberry Pi:")
    print(f"  Original model size: {original_size:.2f} MB")
    print(f"  Optimized TFLite size: {optimized_size:.2f} MB")
    print(f"  Size reduction: {((original_size - optimized_size) / original_size * 100):.1f}%")
    print(f"  Saved to: {output_path}")


# ============================================================================
# PART 8: Main Execution Pipeline
# ============================================================================

def main():
    """
    Complete pipeline for VCG signal compression
    """
    print("="*70)
    print("VCG SIGNAL COMPRESSION USING CNN-LSTM AUTOENCODER")
    print("="*70)
    
    # ========== STEP 1: ECG to VCG Conversion ==========
    print("\n[STEP 1] Converting ECG to VCG...")
    converter = ECGtoVCGConverter()
    
    # IMPORTANT: Set your PTB database path here
    data_dir = "../ptb-diagnostic-ecg-database-1.0.0"  # Update this path!
    output_dir = "vcg_data"
    
    # Check if data directory exists
    if not Path(data_dir).exists():
        print(f"\n⚠️  ERROR: Data directory not found: {data_dir}")
        print("\nPlease download the PTB Diagnostic ECG Database:")
        print("1. Visit: https://www.physionet.org/content/ptbdb/1.0.0/")
        print("2. Download and extract the database")
        print("3. Update the 'data_dir' variable in main() function")
        print("\nFor now, creating a LARGER synthetic dataset for demonstration...")
        print("Note: Results will be poor with synthetic data!")
        
        # Create more realistic synthetic VCG data
        os.makedirs(output_dir, exist_ok=True)
        np.random.seed(42)
        
        # Generate multiple synthetic records with more data
        for i in range(10):  # Create 10 synthetic records
            # Create synthetic VCG with some cardiac-like patterns
            n_samples = 50000  # 50 seconds at 1000 Hz
            t = np.linspace(0, 50, n_samples)
            
            # Simulate heartbeat-like patterns (rough approximation)
            heart_rate = 70  # bpm
            num_beats = int(50 * heart_rate / 60)
            
            vcg = np.zeros((n_samples, 3))
            for beat in range(num_beats):
                beat_center = int((beat / num_beats) * n_samples)
                beat_width = int(0.2 * 1000)  # 200ms beat width
                
                # Create a simple QRS-like complex for each channel
                start_idx = max(0, beat_center - beat_width // 2)
                end_idx = min(n_samples, beat_center + beat_width // 2)
                beat_len = end_idx - start_idx
                
                if beat_len > 0:
                    # X channel
                    vcg[start_idx:end_idx, 0] += np.sin(np.linspace(0, 2*np.pi, beat_len)) * 1.5
                    # Y channel
                    vcg[start_idx:end_idx, 1] += np.sin(np.linspace(0, 2*np.pi, beat_len)) * 1.2
                    # Z channel
                    vcg[start_idx:end_idx, 2] += np.sin(np.linspace(0, 2*np.pi, beat_len)) * 0.8
            
            # Add some noise
            vcg += np.random.randn(n_samples, 3) * 0.05
            
            np.save(f"{output_dir}/synthetic_vcg_{i}.npy", vcg)
        
        print(f"  Created 10 synthetic VCG files in {output_dir}/")
    else:
        # Process real PTB database
        print(f"  Found PTB database at: {data_dir}")
        print("  Processing records...")
        vcg_data = converter.process_ptb_database(
            data_dir, 
            output_dir, 
            max_records=50  # Process up to 50 records
        )
        
        if not vcg_data:
            print("\n⚠️  ERROR: No VCG data was generated from PTB database!")
            print("Please check if the database path is correct.")
            return None, None
    
    # ========== STEP 2: Preprocessing ==========
    print("\n[STEP 2] Preprocessing VCG signals...")
    preprocessor = VCGPreprocessor(cutoff_freq=40, fs=1000, order=5)
    
    # ========== STEP 3: Prepare Dataset ==========
    print("\n[STEP 3] Preparing training dataset...")
    data_generator = VCGDataGenerator(window_size=1250, overlap=0.5)
    
    # Get list of VCG files
    vcg_files = list(Path(output_dir).glob("*.npy"))
    vcg_files = [f for f in vcg_files if 'all_vcg' not in f.name]  # Exclude combined file
    print(f"  Found {len(vcg_files)} VCG files")
    
    if len(vcg_files) == 0:
        print("\n⚠️  ERROR: No VCG files found!")
        return None, None
    
    # Prepare train/test split
    X_train, X_test = data_generator.prepare_dataset(
        vcg_files, 
        preprocessor, 
        test_size=0.15
    )
    
    print(f"\n  Dataset Statistics:")
    print(f"    Training samples: {X_train.shape[0]}")
    print(f"    Test samples: {X_test.shape[0]}")
    print(f"    Signal range: [{X_train.min():.3f}, {X_train.max():.3f}]")
    print(f"    Signal std: {X_train.std():.3f}")
    
    # ========== STEP 4: Build and Train Model ==========
    print("\n[STEP 4] Building CNN-LSTM Autoencoder...")
    autoencoder = VCGAutoencoder(
        input_shape=(1250, 3),
        compression_ratio=30  # Target: 30:1 compression
    )
    
    autoencoder.build_model()
    autoencoder.compile_model(learning_rate=0.001)
    
    print("\n[STEP 5] Training model...")
    history = autoencoder.train(
        X_train, 
        X_test, 
        epochs=120,  # Use fewer epochs if testing
        batch_size=32
    )
    
    # ========== STEP 6: Evaluate Performance ==========
    print("\n[STEP 6] Evaluating performance...")
    
    # Get predictions on test set
    X_test_reconstructed = autoencoder.model.predict(X_test, verbose=0)
    
    # Get compressed representation
    compressed = autoencoder.compress(X_test)
    
    # Calculate metrics for each test sample
    all_metrics = []
    for i in range(min(10, len(X_test))):  # Evaluate first 10 samples
        metrics = PerformanceEvaluator.evaluate_all(
            original=X_test[i],
            reconstructed=X_test_reconstructed[i],
            compressed_size=compressed[i].size
        )
        all_metrics.append(metrics)
    
    # Average metrics
    avg_metrics = {
        key: np.mean([m[key] for m in all_metrics])
        for key in all_metrics[0].keys()
    }
    
    print("\n[AVERAGE PERFORMANCE ACROSS TEST SAMPLES]")
    PerformanceEvaluator.print_metrics(avg_metrics)
    
    # ========== STEP 7: Visualizations ==========
    print("\n[STEP 7] Creating visualizations...")
    
    # Plot training history
    plot_training_history(history)
    
    # Plot signal comparison
    plot_vcg_comparison(X_test, X_test_reconstructed, sample_idx=0)
    
    # Plot 3D VCG trajectory
    plot_3d_vcg(X_test[0], title='Original VCG Signal', 
                filename='vcg_3d_original.png')
    plot_3d_vcg(X_test_reconstructed[0], title='Reconstructed VCG Signal',
                filename='vcg_3d_reconstructed.png')
    
    # ========== STEP 8: Save Models ==========
    print("\n[STEP 8] Saving models...")
    autoencoder.save_models('vcg_encoder.keras', 'vcg_decoder.keras')
    
    # ========== STEP 9: Raspberry Pi Optimization ==========
    # print("\n[STEP 9] Optimizing for Raspberry Pi deployment...")
    # optimize_for_raspberry_pi('best_vcg_autoencoder.keras', 
    #                           'vcg_autoencoder_pi.tflite')
    
    print("\n" + "="*70)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*70)
    print("\nGenerated files:")
    print("  - best_vcg_autoencoder.keras (Complete model)")
    print("  - vcg_encoder.keras (Encoder only)")
    print("  - vcg_decoder.keras (Decoder only)")
    print("  - vcg_autoencoder_pi.tflite (Raspberry Pi optimized)")
    print("\nVisualization files:")
    print("  - training_history.png")
    print("  - vcg_comparison.png")
    print("  - vcg_3d_original.png")
    print("  - vcg_3d_reconstructed.png")
    print("\nTo view visualizations, open the PNG files in your file explorer.")
    print("\n📊 Performance Summary:")
    print(f"    Compression Ratio: {avg_metrics['CR']:.2f}:1")
    print(f"    PRD: {avg_metrics['PRD']:.2f}%")
    print(f"    SNR: {avg_metrics['SNR']:.2f} dB")
    print(f"    Quality Score: {avg_metrics['QS']:.2f}")
    print("="*70)
    
    return autoencoder, avg_metrics


# ============================================================================
# Usage Example for Inference
# ============================================================================

def inference_example():
    """
    Example of how to use the trained model for compression/decompression
    """
    # Load the models
    encoder = keras.models.load_model('vcg_encoder.h5')
    decoder = keras.models.load_model('vcg_decoder.h5')
    
    # Load and preprocess new VCG signal
    preprocessor = VCGPreprocessor()
    new_vcg = np.load('path/to/new_vcg.npy')
    preprocessed = preprocessor.preprocess(new_vcg, fit=False)
    
    # Create windows
    data_generator = VCGDataGenerator()
    windows = data_generator.create_windows(preprocessed)
    
    # Compress
    compressed = encoder.predict(windows)
    print(f"Compressed shape: {compressed.shape}")
    
    # Decompress
    reconstructed = decoder.predict(compressed)
    print(f"Reconstructed shape: {reconstructed.shape}")
    
    # Inverse normalization
    reconstructed_denorm = preprocessor.inverse_normalize(
        reconstructed.reshape(-1, 3)
    )
    
    # Evaluate
    evaluator = PerformanceEvaluator()
    metrics = evaluator.evaluate_all(
        windows.reshape(-1, 3),
        reconstructed.reshape(-1, 3),
        compressed.size
    )
    evaluator.print_metrics(metrics)


if __name__ == "__main__":
    # Run the complete pipeline
    autoencoder, metrics = main()
    
    # Uncomment below to run inference example later
    # inference_example()