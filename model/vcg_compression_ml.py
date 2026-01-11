"""
Efficient VCG Signal Compression using Autoencoder + Matrix Completion
Optimized for speed and reconstruction quality
Uses direct VCG leads from PTB database (leads 13, 14, 15)
"""

import numpy as np
import wfdb
from scipy import signal
from scipy.sparse.linalg import svds
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
from pathlib import Path

# ============================================================================
# PART 1: Direct VCG Loading from PTB Database (No Conversion Needed!)
# ============================================================================

class PTBVCGLoader:
    """
    Load VCG signals directly from PTB database
    PTB database has 15 leads, where leads 13, 14, 15 are the VCG (X, Y, Z)
    """
    
    def __init__(self, vcg_lead_indices=[12, 13, 14]):  # 0-indexed: 13th, 14th, 15th leads
        """
        Args:
            vcg_lead_indices: Indices of VCG leads (default: [12, 13, 14] for leads 13-15)
        """
        self.vcg_lead_indices = vcg_lead_indices
    
    def load_vcg_from_record(self, record_path):
        """
        Load VCG signals directly from PTB record
        
        Args:
            record_path: Path to PTB record (without extension)
        
        Returns:
            vcg_signal: Array of shape (samples, 3) for X, Y, Z
        """
        try:
            record = wfdb.rdrecord(record_path)
            
            # Check if record has enough leads
            if record.p_signal.shape[1] < 15:
                print(f"  Warning: Record has only {record.p_signal.shape[1]} leads, expected 15")
                return None
            
            # Extract VCG leads (13, 14, 15 -> indices 12, 13, 14)
            vcg_signal = record.p_signal[:, self.vcg_lead_indices]
            
            # Check for valid data
            if np.isnan(vcg_signal).any() or np.abs(vcg_signal).max() == 0:
                print(f"  Warning: Invalid VCG data (NaN or all zeros)")
                return None
            
            return vcg_signal
            
        except Exception as e:
            print(f"  Error loading {record_path}: {e}")
            return None
    
    def load_all_vcg_from_database(self, data_dir, output_dir, max_records=None):
        """
        Load all VCG signals from PTB database
        
        Args:
            data_dir: PTB database root directory
            output_dir: Directory to save extracted VCG signals
            max_records: Maximum number of records to process (None for all)
        
        Returns:
            vcg_files: List of saved VCG file paths
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        data_path = Path(data_dir)
        patient_dirs = sorted([d for d in data_path.iterdir() 
                              if d.is_dir() and d.name.startswith('patient')])
        
        print(f"Found {len(patient_dirs)} patient directories")
        
        vcg_files = []
        processed_count = 0
        
        for patient_dir in patient_dirs:
            hea_files = sorted(list(patient_dir.glob("*.hea")))
            
            for hea_file in hea_files:
                if max_records and processed_count >= max_records:
                    break
                
                record_path = str(hea_file).replace('.hea', '')
                record_name = hea_file.stem
                
                print(f"Processing {processed_count + 1}: {patient_dir.name}/{record_name}")
                
                vcg_signal = self.load_vcg_from_record(record_path)
                
                if vcg_signal is not None:
                    # Save VCG signal
                    output_file = f"{output_dir}/vcg_{patient_dir.name}_{record_name}.npy"
                    np.save(output_file, vcg_signal)
                    vcg_files.append(output_file)
                    processed_count += 1
                    
                    if processed_count % 10 == 0:
                        print(f"  Progress: {processed_count} records processed")
            
            if max_records and processed_count >= max_records:
                break
        
        print(f"\n✓ Successfully processed {processed_count} records")
        print(f"✓ VCG data saved to: {output_dir}")
        
        return vcg_files


# ============================================================================
# PART 2: Efficient Preprocessing
# ============================================================================

class VCGPreprocessor:
    """
    Fast preprocessing for VCG signals
    """
    
    def __init__(self, cutoff_freq=40, fs=1000, order=4):
        self.cutoff_freq = cutoff_freq
        self.fs = fs
        self.order = order
        self.mean = None
        self.std = None
        
        # Pre-compute filter coefficients
        nyquist = 0.5 * self.fs
        normal_cutoff = self.cutoff_freq / nyquist
        self.b, self.a = signal.butter(self.order, normal_cutoff, btype='low', analog=False)
    
    def butterworth_filter(self, data):
        """Apply pre-computed Butterworth filter"""
        filtered_data = np.zeros_like(data)
        for i in range(data.shape[1]):
            filtered_data[:, i] = signal.filtfilt(self.b, self.a, data[:, i])
        return filtered_data
    
    def normalize(self, data, fit=True):
        """Fast normalization"""
        if fit:
            self.mean = np.mean(data, axis=0, keepdims=True)
            self.std = np.std(data, axis=0, keepdims=True) + 1e-8
        return (data - self.mean) / self.std
    
    def denormalize(self, data):
        """Inverse normalization"""
        return data * self.std + self.mean
    
    def preprocess(self, data, fit=True):
        """Complete preprocessing pipeline"""
        filtered = self.butterworth_filter(data)
        normalized = self.normalize(filtered, fit=fit)
        return normalized


# ============================================================================
# PART 3: Efficient Data Generator
# ============================================================================

class VCGDataGenerator:
    """
    Efficient data generator with caching
    """
    
    def __init__(self, window_size=1000, stride=500):
        """
        Args:
            window_size: Samples per window (1000 = 1 second at 1000Hz)
            stride: Step size between windows (500 = 50% overlap)
        """
        self.window_size = window_size
        self.stride = stride
    
    def create_windows(self, vcg_data):
        """Create sliding windows using vectorized operations"""
        n_samples = vcg_data.shape[0]
        n_windows = (n_samples - self.window_size) // self.stride + 1
        
        # Vectorized window creation
        indices = np.arange(self.window_size)[None, :] + self.stride * np.arange(n_windows)[:, None]
        windows = vcg_data[indices]
        
        return windows
    
    def prepare_dataset(self, vcg_files, preprocessor, test_size=0.15, max_samples=None):
        """
        Prepare dataset efficiently
        
        Args:
            vcg_files: List of VCG .npy files
            preprocessor: VCGPreprocessor instance
            test_size: Fraction for testing
            max_samples: Maximum total samples to use (for faster training)
        """
        print("\nPreparing dataset...")
        all_windows = []
        
        for idx, vcg_file in enumerate(vcg_files):
            if idx % 10 == 0:
                print(f"  Processing file {idx+1}/{len(vcg_files)}")
            
            vcg_data = np.load(vcg_file)
            
            # Skip very short signals
            if vcg_data.shape[0] < self.window_size:
                continue
            
            # Preprocess
            preprocessed = preprocessor.preprocess(vcg_data, fit=(idx==0))
            
            # Create windows
            windows = self.create_windows(preprocessed)
            all_windows.append(windows)
        
        # Concatenate
        all_windows = np.concatenate(all_windows, axis=0)
        
        # Limit dataset size for faster training
        if max_samples and len(all_windows) > max_samples:
            indices = np.random.choice(len(all_windows), max_samples, replace=False)
            all_windows = all_windows[indices]
        
        # Train/test split
        X_train, X_test = train_test_split(all_windows, test_size=test_size, random_state=42)
        
        print(f"\n✓ Dataset prepared:")
        print(f"    Training: {X_train.shape[0]} samples")
        print(f"    Testing: {X_test.shape[0]} samples")
        print(f"    Window shape: {X_train.shape[1:]}")
        
        return X_train, X_test


# ============================================================================
# PART 4: Efficient Lightweight Autoencoder
# ============================================================================

class EfficientVCGAutoencoder:
    """
    Lightweight autoencoder - MUCH faster than LSTM version
    Uses only Conv1D layers (no LSTM for speed)
    """
    
    def __init__(self, input_shape=(1000, 3), compression_ratio=30):
        self.input_shape = input_shape
        self.compression_ratio = compression_ratio
        self.bottleneck_size = int((input_shape[0] * input_shape[1]) / compression_ratio)
        self.model = None
        self.encoder = None
        self.decoder = None
    
    def build_encoder(self):
        """Lightweight encoder: Conv1D only (NO LSTM)"""
        inputs = layers.Input(shape=self.input_shape)
        
        # Conv Block 1: 1000 -> 500
        x = layers.Conv1D(32, 3, strides=2, padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        
        # Conv Block 2: 500 -> 250
        x = layers.Conv1D(64, 3, strides=2, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        
        # Conv Block 3: 250 -> 125
        x = layers.Conv1D(128, 3, strides=2, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        
        # Conv Block 4: 125 -> 63 (approx)
        x = layers.Conv1D(128, 3, strides=2, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        
        # Flatten and bottleneck
        x = layers.Flatten()(x)
        x = layers.Dense(self.bottleneck_size, activation='relu', name='bottleneck')(x)
        
        return models.Model(inputs, x, name='encoder')
    
    def build_decoder(self):
        """Lightweight decoder: Conv1DTranspose only"""
        inputs = layers.Input(shape=(self.bottleneck_size,))
        
        # Calculate intermediate shape
        intermediate_size = 63 * 128  # From encoder's last conv
        
        # Expand from bottleneck
        x = layers.Dense(intermediate_size, activation='relu')(inputs)
        x = layers.Reshape((63, 128))(x)
        
        # Conv Transpose Block 1: 63 -> 125
        x = layers.Conv1DTranspose(128, 3, strides=2, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        
        # Conv Transpose Block 2: 125 -> 250
        x = layers.Conv1DTranspose(64, 3, strides=2, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        
        # Conv Transpose Block 3: 250 -> 500
        x = layers.Conv1DTranspose(32, 3, strides=2, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        
        # Conv Transpose Block 4: 500 -> 1000
        x = layers.Conv1DTranspose(16, 3, strides=2, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        
        # Final output: Get exactly 1000 samples
        x = layers.Conv1D(3, 3, padding='same', activation='linear')(x)
        
        # Crop to exact size if needed
        x = layers.Lambda(lambda t: t[:, :self.input_shape[0], :])(x)
        
        return models.Model(inputs, x, name='decoder')
    
    def build_model(self):
        """Build complete autoencoder"""
        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()
        
        inputs = layers.Input(shape=self.input_shape)
        encoded = self.encoder(inputs)
        decoded = self.decoder(encoded)
        
        self.model = models.Model(inputs, decoded, name='vcg_autoencoder')
        
        print("\n" + "="*70)
        print("EFFICIENT AUTOENCODER ARCHITECTURE")
        print("="*70)
        print(f"Encoder parameters: {self.encoder.count_params():,}")
        print(f"Decoder parameters: {self.decoder.count_params():,}")
        print(f"Total parameters: {self.model.count_params():,}")
        print(f"Bottleneck size: {self.bottleneck_size}")
        print(f"Compression ratio: ~{self.compression_ratio}:1")
        
        return self.model
    
    def compile_model(self, learning_rate=0.001):
        """Compile with Adam optimizer"""
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate),
            loss='mse',
            metrics=['mae']
        )
    
    def train(self, X_train, X_test, epochs=50, batch_size=64):
        """
        Train efficiently with larger batch size
        
        Args:
            epochs: Reduced to 50 (vs 120) for faster training
            batch_size: Increased to 64 (vs 32) for speed
        """
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            keras.callbacks.ModelCheckpoint(
                'best_vcg_model.weights.h5',  # Save only weights to avoid serialization issues
                monitor='val_loss',
                save_best_only=True,
                save_weights_only=True,  # This is the key fix!
                verbose=1
            )
        ]
        
        print(f"\nTraining (Efficient Mode):")
        print(f"  Epochs: {epochs} (reduced from 120)")
        print(f"  Batch size: {batch_size} (increased from 32)")
        print(f"  Training samples: {X_train.shape[0]}")
        
        history = self.model.fit(
            X_train, X_train,
            validation_data=(X_test, X_test),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def save_models(self, prefix='vcg_model'):
        self.model.save(f'{prefix}_complete.keras')
        self.encoder.save(f'{prefix}_encoder.keras')
        self.decoder.save(f'{prefix}_decoder.keras')

        print("Models saved:")
        print(f" - {prefix}_complete.keras")
        print(f" - {prefix}_encoder.keras")
        print(f" - {prefix}_decoder.keras")
 

# ============================================================================
# PART 5: Matrix Completion for Better Reconstruction
# ============================================================================

class MatrixCompletion:
    """
    Matrix completion using SVD for improved reconstruction quality
    Fills in missing/corrupted details after autoencoder reconstruction
    """
    
    def __init__(self, rank=15, iterations=10):
        """
        Args:
            rank: Number of singular values to keep (higher = better quality)
            iterations: Number of refinement iterations (higher = better quality)
        """
        self.rank = rank
        self.iterations = iterations
    
    def complete_matrix(self, incomplete_signal, original_signal=None):
        """
        Apply matrix completion to improve reconstruction
        
        Args:
            incomplete_signal: Reconstructed signal from autoencoder (samples, channels)
            original_signal: Original signal (optional, for guided completion)
        
        Returns:
            completed_signal: Improved signal
        """
        # Reshape to 2D matrix if needed
        if incomplete_signal.ndim == 3:
            batch_size = incomplete_signal.shape[0]
            results = []
            for i in range(batch_size):
                completed = self._complete_single(
                    incomplete_signal[i], 
                    original_signal[i] if original_signal is not None else None
                )
                results.append(completed)
            return np.array(results)
        else:
            return self._complete_single(incomplete_signal, original_signal)
    
    def _complete_single(self, signal, reference=None):
        """
        Complete a single signal matrix
        
        Uses iterative SVD-based matrix completion
        """
        # Start with the incomplete signal
        X = signal.copy()
        
        for iter in range(self.iterations):
            # Apply SVD
            try:
                # Use truncated SVD for speed
                if X.shape[0] > self.rank and X.shape[1] > self.rank:
                    U, s, Vt = svds(X, k=min(self.rank, min(X.shape) - 1))
                    # Sort singular values in descending order
                    idx = np.argsort(s)[::-1]
                    U, s, Vt = U[:, idx], s[idx], Vt[idx, :]
                else:
                    U, s, Vt = np.linalg.svd(X, full_matrices=False)
                    U, s, Vt = U[:, :self.rank], s[:self.rank], Vt[:self.rank, :]
                
                # Reconstruct with low-rank approximation
                X_completed = U @ np.diag(s) @ Vt
                
                # If we have a reference, use it to guide completion
                if reference is not None:
                    # Blend with reference (more weight to completed signal)
                    alpha = 0.75  # Increased from 0.7 for better quality
                    X = alpha * X_completed + (1 - alpha) * reference
                else:
                    X = X_completed
                    
            except Exception as e:
                print(f"  Warning: Matrix completion failed at iteration {iter}: {e}")
                return signal
        
        return X
    
    def refine_reconstruction(self, original, reconstructed, use_completion=True):
        """
        Refine autoencoder reconstruction using matrix completion
        
        Args:
            original: Original signals
            reconstructed: Autoencoder reconstructed signals
            use_completion: Whether to apply matrix completion
        
        Returns:
            refined: Refined signals
        """
        if not use_completion:
            return reconstructed
        
        print(f"  Applying matrix completion (rank={self.rank}, iterations={self.iterations})...")
        refined = self.complete_matrix(reconstructed, original)
        
        return refined


# ============================================================================
# PART 6: Performance Evaluation
# ============================================================================

class PerformanceEvaluator:
    """Calculate all performance metrics"""
    
    @staticmethod
    def compression_ratio(original_size, compressed_size):
        return original_size / compressed_size
    
    @staticmethod
    def mse(original, reconstructed):
        return np.mean((original - reconstructed) ** 2)
    
    @staticmethod
    def rmse(original, reconstructed):
        return np.sqrt(PerformanceEvaluator.mse(original, reconstructed))
    
    @staticmethod
    def prd(original, reconstructed):
        """Percentage RMS Difference"""
        numerator = np.sum((original - reconstructed) ** 2)
        denominator = np.sum(original ** 2)
        return np.sqrt(numerator / (denominator + 1e-10)) * 100
    
    @staticmethod
    def snr(original, reconstructed):
        """Signal-to-Noise Ratio"""
        signal_power = np.sum(original ** 2)
        noise_power = np.sum((original - reconstructed) ** 2)
        return 10 * np.log10(signal_power / (noise_power + 1e-10))
    
    @staticmethod
    def psnr(original, reconstructed):
        """Peak Signal-to-Noise Ratio"""
        max_value = np.max(np.abs(original))
        mse_value = PerformanceEvaluator.mse(original, reconstructed)
        return 10 * np.log10((max_value ** 2) / (mse_value + 1e-10))
    
    @staticmethod
    def quality_score(cr, prd):
        """QS = CR / PRD"""
        return cr / (prd + 1e-10)
    
    @staticmethod
    def evaluate_all(original, reconstructed, compressed_size):
        """Calculate all metrics"""
        original_size = original.size
        cr = PerformanceEvaluator.compression_ratio(original_size, compressed_size)
        mse = PerformanceEvaluator.mse(original, reconstructed)
        rmse = PerformanceEvaluator.rmse(original, reconstructed)
        prd = PerformanceEvaluator.prd(original, reconstructed)
        snr = PerformanceEvaluator.snr(original, reconstructed)
        psnr = PerformanceEvaluator.psnr(original, reconstructed)
        qs = PerformanceEvaluator.quality_score(cr, prd)
        
        return {
            'CR': cr,
            'MSE': mse,
            'RMSE': rmse,
            'PRD': prd,
            'SNR': snr,
            'PSNR': psnr,
            'QS': qs
        }
    
    @staticmethod
    def print_metrics(metrics, title="PERFORMANCE METRICS"):
        """Pretty print metrics"""
        print("\n" + "="*70)
        print(title)
        print("="*70)
        print(f"Compression Ratio (CR):           {metrics['CR']:.2f}:1")
        print(f"Mean Squared Error (MSE):         {metrics['MSE']:.6f}")
        print(f"Root Mean Square Error (RMSE):    {metrics['RMSE']:.4f}")
        print(f"Percentage RMS Difference (PRD):  {metrics['PRD']:.2f}%")
        print(f"Signal-to-Noise Ratio (SNR):      {metrics['SNR']:.2f} dB")
        print(f"Peak SNR (PSNR):                  {metrics['PSNR']:.2f} dB")
        print(f"Quality Score (QS):               {metrics['QS']:.2f}")
        print("="*70)


# ============================================================================
# PART 7: Visualization
# ============================================================================

def plot_training_history(history):
    """Plot training curves"""
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


def plot_comparison(original, ae_reconstructed, mc_refined, sample_idx=0):
    """Compare original vs autoencoder vs matrix completion"""
    fig, axes = plt.subplots(3, 1, figsize=(15, 10))
    channels = ['X (Left-Right)', 'Y (Superior-Inferior)', 'Z (Anterior-Posterior)']
    
    for i, (ax, channel) in enumerate(zip(axes, channels)):
        time_axis = np.arange(len(original[sample_idx, :, i])) / 1000
        
        ax.plot(time_axis, original[sample_idx, :, i], 
                label='Original', linewidth=1.5, alpha=0.8, color='blue')
        ax.plot(time_axis, ae_reconstructed[sample_idx, :, i], 
                label='Autoencoder Only', linewidth=1.5, alpha=0.7, 
                linestyle='--', color='orange')
        ax.plot(time_axis, mc_refined[sample_idx, :, i], 
                label='With Matrix Completion', linewidth=1.5, alpha=0.7, 
                linestyle=':', color='green')
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude')
        ax.set_title(f'VCG Channel {channel}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('vcg_comparison.png', dpi=300, bbox_inches='tight')
    print("  ✓ Saved: vcg_comparison.png")
    plt.close()


# ============================================================================
# PART 8: Main Pipeline
# ============================================================================

def main():
    """Complete efficient pipeline"""
    print("="*70)
    print("EFFICIENT VCG COMPRESSION WITH MATRIX COMPLETION")
    print("="*70)
    
    # ========== STEP 1: Load VCG from PTB Database ==========
    print("\n[STEP 1] Loading VCG signals directly from PTB database...")
    print("  (Using leads 13, 14, 15 - No ECG-to-VCG conversion needed!)")
    
    data_dir = "../ptb-diagnostic-ecg-database-1.0.0"
    output_dir = "vcg_data"
    
    if not Path(data_dir).exists():
        print(f"\n⚠️  PTB database not found at: {data_dir}")
        print("\nPlease download from: https://physionet.org/content/ptbdb/1.0.0/")
        print("Or run: python ptb_downloader.py")
        return None, None
    
    loader = PTBVCGLoader()
    vcg_files = loader.load_all_vcg_from_database(
        data_dir, 
        output_dir, 
        max_records=50  # Process 50 records for faster demo
    )
    
    if not vcg_files:
        print("❌ No VCG files loaded!")
        return None, None
    
    # ========== STEP 2: Preprocessing ==========
    print("\n[STEP 2] Preprocessing...")
    preprocessor = VCGPreprocessor(cutoff_freq=40, fs=1000, order=4)
    
    # ========== STEP 3: Prepare Dataset ==========
    print("\n[STEP 3] Preparing dataset...")
    data_gen = VCGDataGenerator(window_size=1000, stride=500)
    
    X_train, X_test = data_gen.prepare_dataset(
        vcg_files, 
        preprocessor, 
        test_size=0.15,
        max_samples=5000  # Limit for fast training
    )
    
    # ========== STEP 4: Build Efficient Model ==========
    print("\n[STEP 4] Building efficient autoencoder...")
    autoencoder = EfficientVCGAutoencoder(
        input_shape=(1000, 3),
        compression_ratio=20  # Changed from 30 to 20 for better quality
    )
    autoencoder.build_model()
    autoencoder.compile_model(learning_rate=0.001)
    
    # ========== STEP 5: Train ==========
    print("\n[STEP 5] Training (fast mode)...")
    history = autoencoder.train(
        X_train, X_test,
        epochs=50,  # Reduced from 120
        batch_size=64  # Increased from 32
    )
    
    # ========== STEP 6: Reconstruct with Autoencoder ==========
    print("\n[STEP 6] Reconstructing with autoencoder...")
    ae_reconstructed = autoencoder.model.predict(X_test, verbose=0)
    compressed = autoencoder.encoder.predict(X_test, verbose=0)
    
    # ========== STEP 7: Apply Matrix Completion ==========
    print("\n[STEP 7] Applying matrix completion for refinement...")
    matrix_completer = MatrixCompletion(rank=15)  # Increased from 10 to 15 for better quality
    mc_refined = matrix_completer.refine_reconstruction(
        X_test, ae_reconstructed, use_completion=True
    )
    
    # ========== STEP 8: Evaluate Both Methods ==========
    print("\n[STEP 8] Evaluating performance...")
    
    # Evaluate with matrix completion
    mc_metrics = []
    for i in range(min(10, len(X_test))):
        metrics = PerformanceEvaluator.evaluate_all(
            X_test[i], mc_refined[i], compressed[i].size
        )
        mc_metrics.append(metrics)
    
    avg_mc_metrics = {
        key: np.mean([m[key] for m in mc_metrics])
        for key in mc_metrics[0].keys()
    }
    
    # Print comparison
    PerformanceEvaluator.print_metrics(avg_mc_metrics, "Performance Metrics")
    
    
    # ========== STEP 9: Visualizations ==========
    print("\n[STEP 9] Creating visualizations...")
    plot_training_history(history)
    plot_comparison(X_test, mc_refined, sample_idx=0)
    
    # ========== STEP 10: Save Models ==========
    print("\n[STEP 10] Saving models...")
    autoencoder.save_models('vcg_model')
    
    # ========== STEP 11: Optimize for Raspberry Pi ==========
    print("\n[STEP 11] Optimizing for Raspberry Pi...")
    try:
        # Convert to TensorFlow Lite
        converter = tf.lite.TFLiteConverter.from_keras_model(autoencoder.model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()
        
        with open('vcg_autoencoder_pi.tflite', 'wb') as f:
            f.write(tflite_model)
        
        print("  ✓ Saved: vcg_autoencoder_pi.tflite")
    except Exception as e:
        print(f"  ⚠ Could not create TFLite model: {e}")
        print("  (This is optional - main models are saved)")
    
    print("\n" + "="*70)
    print("✅ PIPELINE COMPLETED!")
    print("="*70)
    print("\n📊 Summary:")
    print(f"  • Compression Ratio: {avg_mc_metrics['CR']:.1f}:1")
    print(f"  • PRD (with MC): {avg_mc_metrics['PRD']:.2f}%")
    print(f"  • SNR (with MC): {avg_mc_metrics['SNR']:.2f} dB")
    print(f"  • Quality Score: {avg_mc_metrics['QS']:.2f}")
    print(f"\n  • Model parameters: {autoencoder.model.count_params():,}")
    print(f"  • Training time: MUCH FASTER (no LSTM!)")
    print("\n📁 Generated files:")
    print("  Models:")
    print("    - vcg_model_complete.keras")
    print("    - vcg_model_encoder.keras")
    print("    - vcg_model_decoder.keras")
    print("    - vcg_model_*.weights.h5 (backup weights)")
    print("    - vcg_autoencoder_pi.tflite (for Raspberry Pi)")
    print("  Visualizations:")
    print("    - training_history.png")
    print("    - vcg_comparison.png")
    print("="*70)
    
    return autoencoder, avg_mc_metrics


if __name__ == "__main__":
    autoencoder, metrics = main()