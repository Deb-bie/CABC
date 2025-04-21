import os
import numpy as np
import tensorflow as tf
import datetime
import uuid
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, Callback, TensorBoard
import tensorflow_hub as hub
import io
import gc
import math


tf.config.run_functions_eagerly(True)


# Constants
data_path = "./data/BreaKHis_Total_dataset"
labels = ['benign', 'malignant']
img_size = 224  
batch_size = 8  # Reduced from 10 for better memory handling
initial_batch_size = 16  # For phase 1 training
epochs = 10
mixed_precision = True


# Configure memory management
def configure_memory():
    # Configure mixed precision
    if mixed_precision:
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)
        print('Mixed precision enabled')

    # Set memory growth for GPU
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        try:
            for device in physical_devices:
                tf.config.experimental.set_memory_growth(device, True)
            
            # Optionally set memory limit (adjust based on your GPU)
            # tf.config.experimental.set_virtual_device_configuration(
            #     physical_devices[0], 
            #     [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)]
            # )
            print(f"Found {len(physical_devices)} GPU(s). Memory growth enabled.")
        except Exception as e:
            print(f"Error setting memory growth: {e}")

    # Disable eager execution for better graph optimization
    tf.config.run_functions_eagerly(False)


# Gradient Accumulation Model for simulating larger batch sizes
class GradientAccumulationModel(tf.keras.Model):
    def __init__(self, model, accumulation_steps=4):
        super().__init__()
        self.model = model
        self.accumulation_steps = accumulation_steps
        self.accumulated_gradients = None
        
    def call(self, inputs, training=None):
        return self.model(inputs, training=training)
        
    def train_step(self, data):
        # Initialize accumulated gradients on first call
        if self.accumulated_gradients is None:
            self.accumulated_gradients = [tf.zeros_like(v) for v in self.model.trainable_variables]
        
        # Unpack the data
        x, y = data
        
        # Split batch if needed to save memory
        batch_size = tf.shape(x)[0]
        split_size = batch_size // self.accumulation_steps
        
        # Track loss across mini-batches
        total_loss = 0
        
        # Process batch in smaller chunks
        for i in range(self.accumulation_steps):
            start_idx = i * split_size
            # Handle last chunk which might be different size
            end_idx = start_idx + split_size if i < self.accumulation_steps - 1 else batch_size
            
            # Extract mini-batch
            if start_idx < end_idx:  # Check to avoid empty batches
                x_mini = x[start_idx:end_idx]
                y_mini = y[start_idx:end_idx]
                
                # Forward pass and calculate loss
                with tf.GradientTape() as tape:
                    y_pred = self.model(x_mini, training=True)
                    # Scale loss by accumulation steps
                    loss = self.compiled_loss(y_mini, y_pred)
                    scaled_loss = loss / self.accumulation_steps
                
                # Calculate gradients
                gradients = tape.gradient(scaled_loss, self.model.trainable_variables)
                
                # Accumulate gradients
                for i, gradient in enumerate(gradients):
                    if gradient is not None:  # Handle None gradients
                        self.accumulated_gradients[i] += gradient
                
                # Track total loss
                total_loss += loss
        
        # Apply accumulated gradients
        self.optimizer.apply_gradients(zip(self.accumulated_gradients, self.model.trainable_variables))
        
        # Reset accumulated gradients
        self.accumulated_gradients = [tf.zeros_like(v) for v in self.model.trainable_variables]
        
        # Update metrics - use original batch for consistent metrics
        y_pred = self.model(x, training=False)
        self.compiled_metrics.update_state(y, y_pred)
        
        # Return metrics
        results = {m.name: m.result() for m in self.metrics}
        results.update({"loss": total_loss})
        return results
    
    def test_step(self, data):
        # Direct pass-through to base model for evaluation
        x, y = data
        y_pred = self.model(x, training=False)
        
        # Update metrics
        self.compiled_loss(y, y_pred)
        self.compiled_metrics.update_state(y, y_pred)
        
        return {m.name: m.result() for m in self.metrics}


# Custom Callbacks
class GPUMemoryCallback(Callback):
    """Monitor GPU memory usage during training"""
    def on_epoch_end(self, epoch, logs=None):
        try:
            physical_devices = tf.config.list_physical_devices('GPU')
            if not physical_devices:
                return
            
            for i, device in enumerate(physical_devices):
                memory = tf.config.experimental.get_memory_info(device.name)
                current_memory_mb = memory['current'] / (1024 * 1024)
                peak_memory_mb = memory['peak'] / (1024 * 1024)
                
                print(f"Epoch {epoch}: GPU-{i} Memory - "
                      f"Current: {current_memory_mb:.2f} MB, "
                      f"Peak: {peak_memory_mb:.2f} MB")
        except Exception as e:
            print(f"Error in GPU memory monitoring: {e}")


class MemoryCleanupCallback(Callback):
    """Clean up memory after each epoch"""
    def on_epoch_end(self, epoch, logs=None):
        gc.collect()
        
        if tf.config.list_physical_devices('GPU'):
            # Try to clear GPU memory cache - may not work on all systems
            try:
                tf.keras.backend.clear_session()
                gc.collect()
            except:
                pass
        print("Memory cleanup performed")


class ConfusionMatrixCallback(Callback):
    """Log confusion matrix to TensorBoard with custom threshold tuning"""
    def __init__(self, validation_data, class_names, log_dir, batch_size, threshold=0.5, freq=1):
        super().__init__()
        self.validation_data = validation_data
        self.class_names = class_names
        self.log_dir = log_dir
        self.freq = freq
        self.batch_size = batch_size
        self.threshold = threshold
        self.file_writer = tf.summary.create_file_writer(os.path.join(self.log_dir, 'cm'))
        
    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.freq != 0:
            return
        
        # Get predictions and true labels
        all_preds = []
        all_labels = []
        
        # Use a reduced set of validation data to save memory
        max_samples = min(1000, self.batch_size * 50)
        batch_count = 0
        
        for x, y in self.validation_data:
            preds = self.model.predict(x, verbose=0)
            all_preds.append(preds)
            all_labels.append(y)
            
            batch_count += 1
            if batch_count * self.batch_size >= max_samples:
                break
                
        # Convert to numpy arrays
        y_pred = np.vstack(all_preds)
        y_true = np.vstack(all_labels)
        
        # Find optimal threshold
        thresholds = np.linspace(0.1, 0.9, 9)
        best_f1 = -1
        best_threshold = 0.5
        
        for threshold in thresholds:
            y_pred_binary = (y_pred > threshold).astype(int)
            
            # Calculate per-class metrics for this threshold
            cm = confusion_matrix(y_true, y_pred_binary)
            
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
                
                # Calculate F1 scores with safeguards against division by zero
                precision_0 = tn / (tn + fn) if (tn + fn) > 0 else 0
                recall_0 = tn / (tn + fp) if (tn + fp) > 0 else 0
                f1_0 = 2 * precision_0 * recall_0 / (precision_0 + recall_0) if (precision_0 + recall_0) > 0 else 0
                
                precision_1 = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall_1 = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1_1 = 2 * precision_1 * recall_1 / (precision_1 + recall_1) if (precision_1 + recall_1) > 0 else 0
                
                # Calculate macro F1 (average of both classes)
                macro_f1 = (f1_0 + f1_1) / 2
                
                # Update if better
                if macro_f1 > best_f1:
                    best_f1 = macro_f1
                    best_threshold = threshold
        
        # Use best threshold for confusion matrix
        y_pred_best = (y_pred > best_threshold).astype(int)
        cm = confusion_matrix(y_true, y_pred_best)
        
        # Log the confusion matrix and best threshold
        with self.file_writer.as_default():
            tf.summary.scalar("Best Threshold", best_threshold, step=epoch)
            
            # Log confusion matrix
            figure = plot_confusion_matrix(cm, class_names=self.class_names)
            cm_image = plot_to_image(figure)
            tf.summary.image("Confusion Matrix", cm_image, step=epoch)


class ThresholdTuningCallback(Callback):
    """Analyze different thresholds and find optimal classification point"""
    def __init__(self, validation_data, log_dir, batch_size, freq=2):
        super().__init__()
        self.validation_data = validation_data
        self.log_dir = log_dir
        self.batch_size = batch_size
        self.freq = freq
        self.file_writer = tf.summary.create_file_writer(os.path.join(self.log_dir, 'thresholds'))
        
    def on_epoch_end(self, epoch, logs=None):
        # Only run every few epochs to save computation
        if (epoch + 1) % self.freq != 0:
            return
            
        # Get predictions on validation data
        all_preds = []
        all_labels = []
        max_samples = min(1000, self.batch_size * 50)
        batch_count = 0
        
        for x, y in self.validation_data:
            preds = self.model.predict(x, verbose=0)
            all_preds.append(preds)
            all_labels.append(y)
            
            batch_count += 1
            if batch_count * self.batch_size >= max_samples:
                break
                
        y_pred = np.vstack(all_preds)
        y_true = np.vstack(all_labels)
        
        # Try different thresholds
        thresholds = np.linspace(0.1, 0.9, 9)
        
        # Create plots for precision, recall, f1 for each class at different thresholds
        figure = plt.figure(figsize=(12, 8))
        plt.subplot(3, 1, 1)
        plt.title("Precision at different thresholds")
        plt.subplot(3, 1, 2)
        plt.title("Recall at different thresholds")
        plt.subplot(3, 1, 3)
        plt.title("F1 score at different thresholds")
        
        # Track metrics for each threshold
        precision_benign = []
        recall_benign = []
        f1_benign = []
        precision_malignant = []
        recall_malignant = []
        f1_malignant = []
        
        for threshold in thresholds:
            y_pred_binary = (y_pred > threshold).astype(int)
            cm = confusion_matrix(y_true, y_pred_binary)
            
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
                
                # Calculate metrics for benign class (class 0)
                p_benign = tn / (tn + fn) if (tn + fn) > 0 else 0
                r_benign = tn / (tn + fp) if (tn + fp) > 0 else 0
                f1_b = 2 * p_benign * r_benign / (p_benign + r_benign) if (p_benign + r_benign) > 0 else 0
                
                # Calculate metrics for malignant class (class 1)
                p_malignant = tp / (tp + fp) if (tp + fp) > 0 else 0
                r_malignant = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1_m = 2 * p_malignant * r_malignant / (p_malignant + r_malignant) if (p_malignant + r_malignant) > 0 else 0
                
                precision_benign.append(p_benign)
                recall_benign.append(r_benign)
                f1_benign.append(f1_b)
                precision_malignant.append(p_malignant)
                recall_malignant.append(r_malignant)
                f1_malignant.append(f1_m)
        
        # Plot precision
        plt.subplot(3, 1, 1)
        plt.plot(thresholds, precision_benign, label='Benign', marker='o')
        plt.plot(thresholds, precision_malignant, label='Malignant', marker='x')
        plt.xlabel('Threshold')
        plt.ylabel('Precision')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot recall
        plt.subplot(3, 1, 2)
        plt.plot(thresholds, recall_benign, label='Benign', marker='o')
        plt.plot(thresholds, recall_malignant, label='Malignant', marker='x')
        plt.xlabel('Threshold')
        plt.ylabel('Recall')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot F1 score
        plt.subplot(3, 1, 3)
        plt.plot(thresholds, f1_benign, label='Benign', marker='o')
        plt.plot(thresholds, f1_malignant, label='Malignant', marker='x')
        plt.plot(thresholds, [(f_b + f_m)/2 for f_b, f_m in zip(f1_benign, f1_malignant)], 
                 label='Macro Avg', marker='^', linestyle='--')
        plt.xlabel('Threshold')
        plt.ylabel('F1 Score')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Convert to image and log to TensorBoard
        threshold_image = plot_to_image(figure)
        
        # Find optimal threshold (maximize macro avg F1)
        avg_f1 = [(f_b + f_m)/2 for f_b, f_m in zip(f1_benign, f1_malignant)]
        optimal_idx = np.argmax(avg_f1)
        optimal_threshold = thresholds[optimal_idx]
        
        with self.file_writer.as_default():
            tf.summary.image("Threshold Analysis", threshold_image, step=epoch)
            tf.summary.scalar("Optimal Threshold", optimal_threshold, step=epoch)


# Helper Functions
def plot_confusion_matrix(cm, class_names):
    """Create a matplotlib figure containing the confusion matrix"""
    figure = plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    
    # Normalize the confusion matrix
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.nan_to_num(cm_norm)  # Replace NaN with 0
    
    # Use white text for darker cells, black for lighter cells
    threshold = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, f"{cm[i, j]}\n({cm_norm[i, j]:.2f})", 
                     horizontalalignment="center", 
                     color="white" if cm[i, j] > threshold else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return figure


def plot_to_image(figure):
    """Convert a matplotlib figure to a PNG image as a tensor"""
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(figure)
    buf.seek(0)
    
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    
    # Add batch dimension
    image = tf.expand_dims(image, 0)
    
    return image


def weighted_binary_crossentropy(benign_weight=2.0):
    """Custom loss function that applies higher weight to benign class errors"""
    def loss(y_true, y_pred):
        # Convert to float32 to avoid numeric issues
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        
        # Standard binary crossentropy
        bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        
        # Apply class weights (benign=0, malignant=1)
        # Higher weight for benign samples (class 0)
        weights = (1.0 - y_true) * benign_weight + y_true * 1.0
        
        # Apply weights to loss
        weighted_bce = bce * weights
        
        return tf.reduce_mean(weighted_bce)
    return loss


def focal_loss(gamma=2.0, alpha=0.75):
    """
    Focal Loss implementation that focuses more on hard examples
    and underrepresented classes
    """
    def loss(y_true, y_pred):
        # Cast to float32
        y_true = tf.cast(y_true, dtype=tf.float32)
        y_pred = tf.cast(y_pred, dtype=tf.float32)
        
        # Clip for stability
        epsilon = 1e-7
        y_pred = tf.clip_by_value(y_pred, epsilon, 1 - epsilon)
        
        # Binary cross entropy calculation
        bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        
        # Apply focal weighting
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        alpha_factor = y_true * alpha + (1 - y_true) * (1 - alpha)
        modulating_factor = tf.pow(1.0 - p_t, gamma)
        
        # Apply focal loss formula
        loss = alpha_factor * modulating_factor * bce
        return tf.reduce_mean(loss)
    
    return loss


def memory_efficient_loading_data(data_dir):
    """Load only image paths instead of loading all images into memory"""
    image_paths = []
    labels_list = []
    
    for label in labels:
        path = os.path.join(data_dir, label)
        class_num = labels.index(label)
        try:
            files = os.listdir(path)
            total_files = len(files)

            print(f"Processing {label} image paths ({total_files} files)")

            for img in files:
                img_path = os.path.join(path, img)
                if os.path.isfile(img_path):  # Only add if it's a file
                    image_paths.append(img_path)
                    labels_list.append(class_num)
                    
        except Exception as e:
            print(f"Error processing {path}: {e}")
    
    return np.array(image_paths), np.array(labels_list)


def optimized_dataset(image_paths, labels, is_training=False, batch_size=32):
    """
    Create an optimized dataset with efficient preprocessing and caching
    """
    # Function to load and preprocess an image
    def preprocess(path, label):
        # Read the image file
        img = tf.io.read_file(path)
        # Decode the image
        img = tf.io.decode_image(img, channels=1, expand_animations=False)
        # Resize the image
        img = tf.image.resize(img, [img_size, img_size])
        # Normalize pixel values to [0,1]
        img = tf.cast(img, tf.float32) / 255.0
        # Convert grayscale to RGB (for DeiT compatibility)
        img = tf.image.grayscale_to_rgb(img)
        return img, label
    
    # Function for data augmentation - separate from basic preprocessing for efficiency
    def augment(img, label):
        # Basic augmentations
        if tf.random.uniform(()) > 0.5:
            img = tf.image.flip_left_right(img)
        if tf.random.uniform(()) > 0.5:
            img = tf.image.flip_up_down(img)
            
        # Apply stronger augmentation to benign class (minority class)
        if label == 0:
            # Random rotation using tf.image.rot90
            k = tf.cast(tf.random.uniform([], 0, 4), tf.int32)
            img = tf.image.rot90(img, k)
            
            # Random brightness and contrast adjustments
            img = tf.image.random_brightness(img, 0.2)
            img = tf.image.random_contrast(img, 0.8, 1.2)
        else:
            # Lighter augmentation for majority class
            img = tf.image.random_brightness(img, 0.1)
        
        # Ensure values stay in valid range
        img = tf.clip_by_value(img, 0.0, 1.0)
        
        return img, label
    
    # Create dataset from paths and labels
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    
    # Apply preprocessing
    dataset = dataset.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Apply augmentation if training
    if is_training:
        dataset = dataset.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Apply memory-efficient operations
    if is_training:
        # For training: shuffle, repeat, batch
        dataset = dataset.shuffle(min(1000, len(image_paths)))
        dataset = dataset.batch(batch_size)
        dataset = dataset.repeat()  # Repeat to prevent running out of data
    else:
        # For validation/testing: just batch
        dataset = dataset.batch(batch_size)
    
    # Enable prefetching for better performance
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset


def balanced_dataset(image_paths, labels, batch_size=32, is_training=False):
    """
    Create a dataset with class balancing to handle imbalance between benign and malignant
    """
    # Separate paths by class
    benign_mask = labels == 0
    malignant_mask = labels == 1
    
    benign_paths = image_paths[benign_mask]
    benign_labels = labels[benign_mask]
    malignant_paths = image_paths[malignant_mask]
    malignant_labels = labels[malignant_mask]
    
    print(f"Benign samples: {len(benign_paths)}")
    print(f"Malignant samples: {len(malignant_paths)}")
    
    # Create preprocessing function
    def preprocess(path, label):
        # Read the image file
        img = tf.io.read_file(path)
        # Decode the image
        img = tf.io.decode_image(img, channels=1, expand_animations=False)
        # Resize the image
        img = tf.image.resize(img, [img_size, img_size])
        # Normalize pixel values
        img = tf.cast(img, tf.float32) / 255.0
        # Convert to RGB (DeiT expects 3 channels)
        img = tf.image.grayscale_to_rgb(img)
        return img, label
    
    # Function for augmentations - separate from basic preprocessing for efficiency
    def augment(img, label):
        # Basic augmentations for all images
        if tf.random.uniform(()) > 0.5:
            img = tf.image.flip_left_right(img)
        if tf.random.uniform(()) > 0.5:
            img = tf.image.flip_up_down(img)
            
        # Apply specific augmentation depending on class
        if label == 0:  # Benign class - more augmentation
            # Random rotation
            k = tf.cast(tf.random.uniform([], 0, 4), tf.int32)
            img = tf.image.rot90(img, k)
            
            # Random brightness and contrast
            img = tf.image.random_brightness(img, 0.2)
            img = tf.image.random_contrast(img, 0.8, 1.2)
        else:  # Malignant - less augmentation
            img = tf.image.random_brightness(img, 0.1)
        
        # Ensure values stay in valid range
        img = tf.clip_by_value(img, 0.0, 1.0)
        return img, label
        
    # Create separate datasets for each class
    benign_ds = tf.data.Dataset.from_tensor_slices((benign_paths, benign_labels))
    malignant_ds = tf.data.Dataset.from_tensor_slices((malignant_paths, malignant_labels))
    
    # Apply preprocessing
    benign_ds = benign_ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    malignant_ds = malignant_ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Apply augmentation if training
    if is_training:
        benign_ds = benign_ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
        malignant_ds = malignant_ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
        
        # Calculate class weights for balancing
        # We oversample the minority class (benign)
        malignant_to_benign_ratio = len(malignant_paths) / len(benign_paths)
        repeat_factor = int(malignant_to_benign_ratio) + 1
        
        # Repeat benign samples to balance classes
        benign_ds = benign_ds.repeat(repeat_factor)
        malignant_ds = malignant_ds.repeat()
        
        # Apply shuffle to each dataset
        benign_ds = benign_ds.shuffle(min(1000, len(benign_paths)))
        malignant_ds = malignant_ds.shuffle(min(1000, len(malignant_paths)))
        
        # Sample from both datasets with equal probability
        dataset = tf.data.experimental.sample_from_datasets([benign_ds, malignant_ds])
        
    else:
        # For validation/test, we maintain the original distribution
        dataset = tf.data.Dataset.concatenate(benign_ds, malignant_ds)
    
    # Batch dataset
    dataset = dataset.batch(batch_size)
    
    # Add repeat if training
    if is_training:
        dataset = dataset.repeat()
    
    # Prefetch for performance
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset


def check_class_balance(y):
    """Print and return class distribution"""
    unique, counts = np.unique(y, return_counts=True)
    print(f"Class distribution: {dict(zip([labels[i] for i in unique], counts))}")
    return counts


def create_deit_model(trainable_base=False):
    """
    Create DeiT model for image classification
    
    Args:
        trainable_base: Whether to make the DeiT base trainable
    """
    print("Creating DeiT model...")
    
    # Start with float32 inputs to ensure compatibility with the hub model
    inputs = layers.Input(shape=(img_size, img_size, 3), dtype=tf.float32)
    
    # Load DeiT-Tiny model for better memory efficiency
    deit_url = "https://tfhub.dev/sayakpaul/deit_tiny_patch16_224/1"
    
    # Apply simple normalization to match DeiT requirements
    x = inputs / 255.0  # Simple rescaling instead of imagenet preprocessing
    
    # Load the DeiT layer
    deit_layer = hub.KerasLayer(deit_url, trainable=trainable_base)
    # x = deit_layer(x)

    deit_outputs = deit_layer(x)

    # Extract the class token (first output) for classification
    x = deit_outputs[0] 
    
    # Add classification head
    x = layers.Dropout(0.5)(x)  # Strong dropout for better generalization
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    model = models.Model(inputs, outputs)
    
    # Print model summary
    model.summary()
    
    return model


def create_custom_deit_model():
    """
    Alternative model: Create a custom DeiT-inspired model
    This avoids TF Hub compatibility issues with mixed precision
    """
    print("Creating custom ViT/DeiT-like model (not using Hub)...")
    
    # MLP block commonly used in Vision Transformer
    def mlp(x, hidden_units, dropout_rate):
        for units in hidden_units:
            x = layers.Dense(units, activation=tf.nn.gelu)(x)
            x = layers.Dropout(dropout_rate)(x)
        return x
    
    # Parameters - smaller than original DeiT for memory efficiency
    input_shape = (img_size, img_size, 3)
    patch_size = 16  # Size of the patches
    num_patches = (img_size // patch_size) ** 2
    projection_dim = 192  # Smaller dimension
    num_heads = 3  # Reduced number of attention heads
    transformer_units = [projection_dim * 2, projection_dim]
    transformer_layers = 4  # Reduced number of layers
    
    # Input
    inputs = layers.Input(shape=input_shape)
    
    # Create patches
    patches = layers.Conv2D(
        filters=projection_dim,
        kernel_size=(patch_size, patch_size),
        strides=(patch_size, patch_size),
        padding="VALID",
    )(inputs)
    
    # Reshape patches
    batch_size = tf.shape(patches)[0]
    patches = tf.reshape(patches, [batch_size, -1, projection_dim])
    
    # Create positional embeddings
    positions = tf.range(start=0, limit=num_patches, delta=1)
    pos_embedding = layers.Embedding(
        input_dim=num_patches, output_dim=projection_dim
    )(positions)
    
    # Add positional embeddings
    x = patches + pos_embedding
    
    # Create multiple transformer blocks
    for _ in range(transformer_layers):
        # Layer normalization 1
        x1 = layers.LayerNormalization(epsilon=1e-6)(x)
        
        # Multi-head attention
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim // num_heads, dropout=0.1
        )(x1, x1)
        
        # Skip connection 1
        x2 = layers.Add()([attention_output, x])
        
        # Layer normalization 2
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        
        # MLP
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)
        
        # Skip connection 2
        x = layers.Add()([x3, x2])
    
    # Create a [batch_size, projection_dim] tensor
    representation = layers.LayerNormalization(epsilon=1e-6)(x)
    representation = layers.GlobalAveragePooling1D()(representation)
    
    # Add MLP head with strong regularization
    features = mlp(representation, hidden_units=[projection_dim, projection_dim // 2], dropout_rate=0.5)
    
    # Classification layer
    outputs = layers.Dense(1, activation="sigmoid")(features)
    
    # Create the model
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.summary()
    return model


def plot_training_history(history):
    # Plot training & validation accuracy values
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    # Plot training & validation loss values
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.tight_layout()
    plt.savefig('deit_training_history.png')
    plt.close()


def evaluate_model(model, test_ds, y_test, log_dir=None, epoch=0):
    # Get predictions
    y_pred_prob = model.predict(test_ds)
    
    # Extract probabilities and convert to flat array
    y_pred_prob_flat = []
    for batch in y_pred_prob:
        for prob in batch:
            y_pred_prob_flat.append(prob)
    y_pred_prob_flat = np.array(y_pred_prob_flat)[:len(y_test)]
    
    # Convert probabilities to class predictions
    y_pred = (y_pred_prob_flat > 0.5).astype(int)
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=labels))
    
    # If a log directory is provided, log confusion matrix and ROC curve to TensorBoard
    if log_dir:
        # Create confusion matrix figure and log it
        cm = confusion_matrix(y_test, y_pred)
        cm_figure = plot_confusion_matrix(cm, class_names=labels)
        cm_image = plot_to_image(cm_figure)
        
        # Create ROC curve figure and log it
        roc_figure = plot_roc_curve(y_test, y_pred_prob_flat)
        roc_image = plot_to_image(roc_figure)
        
        # Write to TensorBoard
        file_writer = tf.summary.create_file_writer(os.path.join(log_dir, 'final_evaluation'))
        with file_writer.as_default():
            tf.summary.image("Final Confusion Matrix", cm_image, step=0)
            tf.summary.image("Final ROC Curve", roc_image, step=0)
    
    return y_pred, y_pred_prob_flat


def train_model_with_memory_optimizations():
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    print("Loading image paths...")
    image_paths, image_labels = memory_efficient_loading_data(data_path)
    
    # Check class balance and calculate class weights
    class_counts = check_class_balance(image_labels)
    if len(np.unique(image_labels)) == 2:  # Binary classification
        class_weight = {
            0: len(image_labels) / (2.0 * np.sum(image_labels == 0)),
            1: len(image_labels) / (2.0 * np.sum(image_labels == 1))
        }
        print(f"Using class weights: {class_weight}")
    else:
        class_weight = None

    # Split data - test set 20%
    train_paths, test_paths, train_labels, test_labels = train_test_split(
        image_paths,
        image_labels,
        test_size=0.2,
        random_state=42,
        stratify=image_labels
    )
    
    # Further split training set into train/val - val set 20% of remaining data (16% of total)
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_paths,
        train_labels,
        test_size=0.2,
        random_state=42,
        stratify=train_labels
    )

    print(f"Train set: {train_paths.shape[0]} images")
    print(f"Validation set: {val_paths.shape[0]} images")
    print(f"Test set: {test_paths.shape[0]} images")

    # Create datasets with smaller batch size for DeiT
    train_ds = balanced_dataset(train_paths, train_labels, batch_size=batch_size, is_training=True)
    val_ds = balanced_dataset(val_paths, val_labels, batch_size=batch_size, is_training=False)
    test_ds = balanced_dataset(test_paths, test_labels, batch_size=batch_size, is_training=False)

    # Create model with memory-saving techniques
    print("Creating DeiT-inspired model...")
    
    try:
        # Clear any previous model from memory
        tf.keras.backend.clear_session()
        import gc
        gc.collect()
        
        # Try to create DeiT model first
        try:
            print("Attempting to create hub-based DeiT model with dtype handling...")
            model = create_deit_model()
            model.summary()
        except Exception as e:
            print(f"Error creating hub-based model: {e}")
            print("Falling back to custom DeiT-inspired model...")
            model = create_custom_deit_model()
            model.summary()
        
        # Use a relatively low learning rate for stability
        initial_learning_rate = 1e-5
        
        # Compile model with binary classification metrics
        model.compile(
            optimizer=tf.keras.optimizers.Adam(initial_learning_rate),
            # loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
            loss=focal_loss(gamma=3.0, alpha=0.85),
            metrics=[
                'accuracy',
                tf.keras.metrics.AUC(name='auc'),
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall')
            ]
        )

        # Create unique model filename and log directory
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        model_filename = f"DeiT_{timestamp}_{unique_id}.keras"
        log_dir = f"logs/fit/{timestamp}_{unique_id}"
        
        # Define callbacks with memory optimization and TensorBoard
        callbacks = [
            TensorBoard(
                log_dir=log_dir,
                histogram_freq=1,
                write_graph=True,
                update_freq='epoch',
                profile_batch=0  # Disable profiling to save memory
            ),
            EarlyStopping(
                monitor='val_auc', 
                patience=4,
                mode='max',
                restore_best_weights=True,
                verbose=1
            ),
            ModelCheckpoint(
                model_filename, 
                monitor='val_auc',
                mode='max',
                save_best_only=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=2,
                min_lr=1e-6,
                verbose=1
            ),
            ConfusionMatrixCallback(
                validation_data=val_ds,  
                class_names=labels,
                log_dir=log_dir,
                freq=1,  # Log every epoch
                batch_size=batch_size
            ),
            ROCCurveCallback(
                validation_data=val_ds,
                log_dir=log_dir,
                freq=1,
                batch_size=batch_size
            ),
            ThresholdTuningCallback(
                validation_data=val_ds,  # Your validation dataset
                log_dir=log_dir,
                batch_size=batch_size        
            ),
            GPUMemoryCallback(),
            MemoryCleanupCallback()
        ]

        # Calculate steps per epoch to limit iterations and save memory
        # This allows you to see progress faster and potentially stop if there are memory issues
        steps_per_epoch = min(len(train_paths) // batch_size, 50)  # Cap at 50 steps
        validation_steps = min(len(val_paths) // batch_size, 20)   # Cap at 20 steps
        
        print(f"Training with {steps_per_epoch} steps per epoch and {validation_steps} validation steps")
        print(f"TensorBoard logs will be saved to {log_dir}")
        print("Starting training with memory optimizations...")
        
        # Train the model
        history = model.fit(
            train_ds, 
            validation_data=val_ds, 
            epochs=epochs, 
            callbacks=callbacks,
            class_weight=class_weight,
            steps_per_epoch=steps_per_epoch,      # Limit steps to save memory
            validation_steps=validation_steps      # Limit validation steps
        )
        
        # Plot training history
        plot_training_history(history)
        
        # Evaluate on test set
        print("\n======= Final Evaluation on Test Set =======")
        test_metrics = model.evaluate(test_ds, steps=min(len(test_paths) // batch_size, 30))
        metric_names = ['Loss', 'Accuracy', 'AUC', 'Precision', 'Recall']
        
        for name, value in zip(metric_names, test_metrics):
            print(f"Test {name}: {value:.4f}")

        y_pred, y_pred_prob = evaluate_model(model, test_ds, test_labels, log_dir=log_dir)
        
        # Save final model
        final_model_filename = f'final_DeiT_model_{timestamp}_{unique_id}.keras'
        model.save(final_model_filename)
        print(f"Model successfully saved as '{final_model_filename}'")
        print(f"To view the TensorBoard logs, run: tensorboard --logdir {log_dir}")
        
    except tf.errors.ResourceExhaustedError as e:
        print(f"Memory error during training: {e}")
        print("\nDeiT models are very memory-intensive. Here are some suggestions:")
        print("1. Further reduce batch_size (try 4 or 2)")
        print("2. Use a smaller version of DeiT (DeiT-Tiny is already being used)")
        print("3. Consider gradient accumulation to simulate larger batch sizes")
        print("4. Reduce image dimensions (e.g., img_size=160)")
        print("5. Use a subset of your data for the initial comparison")
        
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()  # Print full traceback for debugging



# Progressive model fitting
def progressive_training():
    """Train model in phases to better manage memory and improve performance"""
    # Load and preprocess data
    image_paths, image_labels = memory_efficient_loading_data(data_path)
    
    train_paths, test_paths, train_labels, test_labels = train_test_split(
        image_paths, 
        image_labels, 
        test_size=0.2, 
        stratify=image_labels
    )
    
    # Phase 1: Train with frozen DeiT base
    print("Phase 1: Training classification head only...")
    model = create_deit_model()  # DeiT is frozen by default
    
    # Use smaller batch size initially
    batch_size_phase1 = 16
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss=weighted_binary_crossentropy(),
        metrics=['accuracy', tf.keras.metrics.AUC()]
    )
    

    num_train_samples = len(train_paths) # Assuming you have the number of training samples
    steps_per_epoch = math.ceil(num_train_samples / batch_size_phase1)
    num_test_samples = len(test_paths) # Assuming you have the number of test samples
    validation_steps = math.ceil(num_test_samples / batch_size_phase1)

    
    model.fit(
        optimized_dataset(train_paths, train_labels, True, batch_size_phase1),
        epochs=5,
        validation_data=optimized_dataset(test_paths, test_labels, False, batch_size_phase1),
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps
    )
    
    # Phase 2: Fine-tune the entire model
    print("Phase 2: Fine-tuning entire model...")
    # Unfreeze the DeiT layer
    for layer in model.layers:
        if isinstance(layer, hub.KerasLayer):
            layer.trainable = True
    
    # Use gradient accumulation for effectively larger batch size
    model_with_accum = GradientAccumulationModel(model, accumulation_steps=4)
    
    model_with_accum.compile(
        optimizer=tf.keras.optimizers.Adam(5e-5),  # Lower learning rate
        loss=weighted_binary_crossentropy(),
        metrics=['accuracy', tf.keras.metrics.AUC()]
    )
    
    model_with_accum.fit(
        optimized_dataset(train_paths, train_labels, True, batch_size=batch_size),
        epochs=epochs,
        validation_data=optimized_dataset(test_paths, test_labels, False, batch_size),
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps
    )

    # After training completes (after Phase 2), add this evaluation code:
    print("Evaluating model on test set...")
    
    # Use the final trained model for evaluation
    final_model = model_with_accum.get_model() if 'model_with_accum' in locals() else model
    
    # Create test dataset with an appropriate batch size for evaluation
    test_dataset = optimized_dataset(test_paths, test_labels, False, batch_size)
    
    # Evaluate and get results
    test_results = final_model.evaluate(
        test_dataset,
        steps=validation_steps,
        verbose=1
    )
    
    # Print results in a readable format
    metrics_names = final_model.metrics_names
    print("\nTest Results:")
    for name, value in zip(metrics_names, test_results):
        print(f"{name}: {value:.4f}")
    
    return final_model, test_results


if __name__ == "__main__":
    # Clear any existing TF sessions
    tf.keras.backend.clear_session()
    
    # Train with memory optimizations
    # train_model_with_memory_optimizations()
    progressive_training()




