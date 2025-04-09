import os
import io
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import seaborn as sns
import datetime
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers, mixed_precision
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.applications import EfficientNetB0  # Changed from ResNet50 to EfficientNetB0
import tensorflow_addons as tfa

# Enable mixed precision for memory efficiency
mixed_precision.set_global_policy('mixed_float16')

# Constants
data_path = "../../../data/BreaKHis_Total_dataset"
labels = ['benign', 'malignant']
img_size = 224
batch_size = 16  # Reduced batch size for memory efficiency
epochs = 40      # Increased for better convergence
grad_accumulation_steps = 4  # Simulate larger batch size with gradient accumulation

# Create output directory for results
os.makedirs('results', exist_ok=True)

def loading_data(data_dir):
    data = []
    labels_list = []

    for label in labels:
        path = os.path.join(data_dir, label)
        class_num = labels.index(label)
        files = os.listdir(path)
        total_files = len(files)

        print(f"Loading {label} images ({total_files} files)")

        for i, img in enumerate(files):
            if i % 100 == 0:
                print(f" Progress: {i}/{total_files}")
            
            img_path = os.path.join(path, img)
            img_arr = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            if img_arr is not None:
                resized_arr = cv2.resize(img_arr, (img_size, img_size))
                data.append(resized_arr)
                labels_list.append(class_num)
            else:
                print(f"Warning: Unable to read image {img_path}")

    return np.array(data), np.array(labels_list)

def preprocess_data(data, labels):
    # Enhanced preprocessing with standardization
    X_data = np.array(data).astype('float32')
    
    # Apply CLAHE for contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_data = []
    
    for img in X_data:
        enhanced_img = clahe.apply(img.astype(np.uint8))
        enhanced_data.append(enhanced_img)
    
    X_data = np.array(enhanced_data).astype('float32')
    
    # Standardize data
    X_data = (X_data - X_data.mean()) / (X_data.std() + 1e-7)
    X_data = X_data.reshape(-1, img_size, img_size, 1)
    print(f"Data shape after preprocessing: {X_data.shape}")
    
    y_data = np.array(labels)
    
    return X_data, y_data

def check_class_balance(y):
    unique, counts = np.unique(y, return_counts=True)
    print(f"Class distribution: {dict(zip([labels[i] for i in unique], counts))}")
    return counts

def create_dataset(X, y, augment=False):
    def _generator():
        for i in range(len(X)):
            rgb = tf.image.grayscale_to_rgb(tf.convert_to_tensor(X[i]))
            label = y[i]
            yield rgb, label

    def _augment(image, label):
        # Enhanced augmentation pipeline with histopathology-specific transformations
        # Random flips
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_flip_up_down(image)
        
        # Color augmentations - mild for histopathology
        image = tf.image.random_brightness(image, max_delta=0.2)
        image = tf.image.random_contrast(image, 0.8, 1.2)
        
        # Histopathology-specific augmentations
        # Random rotation with various angles
        angle = tf.random.uniform([], -0.25, 0.25)  # Random rotation between -15 and 15 degrees
        image = tfa.image.rotate(image, angles=angle, fill_mode='reflect')
        
        # Random zoom to simulate magnification changes
        zoom_factor = tf.random.uniform([], 0.85, 1.15)
        image_shape = tf.shape(image)
        crop_size = tf.cast(tf.cast(image_shape[:-1], tf.float32) * zoom_factor, tf.int32)
        if crop_size[0] > 0 and crop_size[1] > 0 and crop_size[0] <= image_shape[0] and crop_size[1] <= image_shape[1]:
            image = tf.image.random_crop(image, [crop_size[0], crop_size[1], 3])
            image = tf.image.resize(image, [img_size, img_size])
        
        # Gaussian noise to simulate staining variations
        noise = tf.random.normal(shape=tf.shape(image), mean=0.0, stddev=0.02)
        image = tf.clip_by_value(image + noise, 0.0, 1.0)
        
        # Color shifting to simulate stain variations
        color_shift = tf.random.uniform([], -0.05, 0.05, dtype=tf.float32)
        image = tf.clip_by_value(image + color_shift, 0.0, 1.0)
        
        return image, label

    ds = tf.data.Dataset.from_generator(
        _generator,
        output_signature=(
            tf.TensorSpec(shape=(img_size, img_size, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.int32)
        )
    )

    if augment:
        ds = ds.map(_augment, num_parallel_calls=tf.data.AUTOTUNE)

    return ds.shuffle(1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)

def create_model():
    # Use EfficientNetB0 for better accuracy with lower memory footprint
    base_model = EfficientNetB0(
        include_top=False, 
        weights='imagenet', 
        input_shape=(img_size, img_size, 3), 
        pooling='avg'
    )
    
    # Freeze early layers
    for layer in base_model.layers[:-20]:
        layer.trainable = False
        
    inputs = base_model.input
    x = base_model.output
    
    # Memory-efficient classification head with fewer parameters
    x = layers.Dense(512, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.4)(x)
    
    x = layers.Dense(256, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.3)(x)
    
    outputs = layers.Dense(1, activation='sigmoid')(x)
    return tf.keras.Model(inputs, outputs)

def log_images_to_tensorboard(log_dir):
    """Create a TensorBoard image logger"""
    file_writer = tf.summary.create_file_writer(log_dir + '/images')
    return file_writer

def log_image(file_writer, name, figure, step=0):
    """Log a matplotlib figure to TensorBoard"""
    with file_writer.as_default():
        # Convert figure to PNG image
        buffer = io.BytesIO()
        figure.savefig(buffer, format='png')
        buffer.seek(0)
        
        # Convert PNG buffer to TF image
        image = tf.image.decode_png(buffer.getvalue(), channels=4)
        
        # Add batch dimension and log
        image = tf.expand_dims(image, 0)
        tf.summary.image(name, image, step=step)
        
    plt.close(figure)

def plot_training_history(history, fold):
    # Plot training & validation accuracy values
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title(f'Model Accuracy - Fold {fold + 1}')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    # Plot training & validation loss values
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title(f'Model Loss - Fold {fold + 1}')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.tight_layout()
    plt.savefig(f'results/history_fold{fold+1}.png')
    plt.close()

def evaluate_model(model, test_ds, y_test, log_dir, epoch=0):
    # Create image writer for this evaluation
    tb_image_writer = log_images_to_tensorboard(log_dir)
    
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
    report = classification_report(y_test, y_pred, target_names=labels)
    print(report)
    
    # Save report to file
    with open(f"results/classification_report.txt", "w") as f:
        f.write(report)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    fig = plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels, cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig('results/final_confusion_matrix.png')
    
    # Log the confusion matrix to TensorBoard
    log_image(tb_image_writer, 'confusion_matrix', fig, step=epoch)
    
    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob_flat)
    roc_auc = auc(fpr, tpr)
    
    fig = plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig('results/roc_curve.png')
    
    # Log the ROC curve to TensorBoard
    log_image(tb_image_writer, 'roc_curve', fig, step=epoch)
    
    return y_pred, y_pred_prob_flat

# Custom training loop with gradient accumulation for memory efficiency
class GradientAccumulationModel:
    def __init__(self, model, optimizer, loss_fn, metrics, accumulation_steps=1):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.metrics = metrics
        self.accumulation_steps = accumulation_steps
        
    def train_step(self, x, y, class_weight=None):
        with tf.GradientTape() as tape:
            pred = self.model(x, training=True)
            
            # Apply class weights if provided
            if class_weight is not None:
                sample_weights = tf.gather(
                    tf.constant([class_weight[0], class_weight[1]], dtype=tf.float32),
                    tf.cast(y, tf.int32)
                )
                loss = self.loss_fn(y, pred, sample_weight=sample_weights)
            else:
                loss = self.loss_fn(y, pred)
            
            # Scale the loss for gradient accumulation
            scaled_loss = loss / self.accumulation_steps
            
        # Get gradients and scale them
        gradients = tape.gradient(scaled_loss, self.model.trainable_variables)
        
        # Accumulate gradients
        if not hasattr(self, 'accumulated_gradients'):
            self.accumulated_gradients = [tf.zeros_like(var) for var in self.model.trainable_variables]
            self.steps = 0
            
        for i, grad in enumerate(gradients):
            if grad is not None:
                self.accumulated_gradients[i] += grad
                
        self.steps += 1
        
        # Apply accumulated gradients
        if self.steps >= self.accumulation_steps:
            self.optimizer.apply_gradients(zip(self.accumulated_gradients, self.model.trainable_variables))
            self.accumulated_gradients = [tf.zeros_like(var) for var in self.model.trainable_variables]
            self.steps = 0
            
        # Update metrics
        for metric in self.metrics:
            metric.update_state(y, pred)
            
        return loss
    
    def reset_metrics(self):
        for metric in self.metrics:
            metric.reset_states()
            
    def get_metrics_result(self):
        return {metric.name: metric.result().numpy() for metric in self.metrics}
        
    def test_step(self, x, y):
        pred = self.model(x, training=False)
        loss = self.loss_fn(y, pred)
        
        for metric in self.metrics:
            metric.update_state(y, pred)
            
        return loss

def train_model():
    # Load and preprocess data
    print("Loading data...")
    data, labels_data = loading_data(data_path)
    X, y = preprocess_data(data, labels_data)
    
    # Check class balance and calculate class weights
    class_counts = check_class_balance(y)
    if len(np.unique(y)) == 2:  # Binary classification
        # More aggressive class weighting to address imbalance
        minority_class = 0 if class_counts[0] < class_counts[1] else 1
        majority_class = 1 - minority_class
        imbalance_ratio = class_counts[majority_class] / class_counts[minority_class]
        
        class_weight = {
            minority_class: imbalance_ratio,
            majority_class: 1.0
        }
        print(f"Using class weights: {class_weight}")
    else:
        class_weight = None

    # Split data into train/validation/test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, 
        y, 
        test_size=0.15,
        random_state=42,
        stratify=y  # Ensure balanced split
    )

    kf = KFold(
        n_splits=5, 
        shuffle=True, 
        random_state=42
    )
    
    # Tracking metrics
    fold_results = []
    best_model = None
    best_acc = 0
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_temp, y_temp)):
        print(f"\n=========== Fold {fold + 1}/5 ===========")

        X_train, X_val = X_temp[train_idx], X_temp[val_idx]
        y_train, y_val = y_temp[train_idx], y_temp[val_idx]
        
        print(f"Train set: {X_train.shape}, {y_train.shape}")
        print(f"Validation set: {X_val.shape}, {y_val.shape}")

        # Create datasets with augmentation for training
        train_ds = create_dataset(X_train, y_train, augment=True)
        val_ds = create_dataset(X_val, y_val, augment=False)
        
        # Create model
        print("Creating model...")
        model = create_model()
        
        # Learning rate schedule with warmup
        total_steps = epochs * (len(X_train) // batch_size)
        warmup_steps = int(0.1 * total_steps)  # 10% of total steps for warmup
        
        def lr_schedule(step):
            if step < warmup_steps:
                return 1e-6 + (step / warmup_steps) * (1e-4 - 1e-6)  # Linear warmup
            else:
                return 1e-4 * tf.math.exp(0.1 * (1 - (step - warmup_steps) / (total_steps - warmup_steps)))
        
        optimizer = Adam(learning_rate=1e-4)
        
        # Compile model with additional metrics
        model.compile(
            optimizer=optimizer, 
            loss='binary_crossentropy', 
            metrics=[
                'accuracy', 
                tf.keras.metrics.AUC(name='auc'),
                tf.keras.metrics.Precision(name='precision'), 
                tf.keras.metrics.Recall(name='recall'),
                # F1 Score for better balance between precision and recall
                tf.keras.metrics.F1Score(name='f1_score')
            ]
        )
        
        # Setup callbacks
        log_dir = f"logs/EfficientNet_fold{fold+1}_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tb_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
        
        callbacks = [
            EarlyStopping(
                monitor='val_f1_score', 
                patience=15,  # More patience for better convergence
                mode='max',
                restore_best_weights=True,
                verbose=1
            ),
            ModelCheckpoint(
                f"results/EfficientNet_fold{fold + 1}.h5", 
                monitor='val_f1_score',  # F1 score better for imbalanced data
                mode='max',
                save_best_only=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=7,
                min_lr=1e-7,
                verbose=1
            ),
            tb_callback
        ]
        
        # Create image writer for this fold
        tb_image_writer = log_images_to_tensorboard(log_dir)
        
        # Create gradient accumulation model wrapper
        ga_model = GradientAccumulationModel(
            model=model,
            optimizer=optimizer,
            loss_fn=tf.keras.losses.BinaryCrossentropy(),
            metrics=model.metrics,
            accumulation_steps=grad_accumulation_steps
        )
        
        # Custom training loop with gradient accumulation
        print("Training model with gradient accumulation...")
        
        # Initialize history dictionary
        history = {
            'loss': [], 'accuracy': [], 'auc': [], 'precision': [], 'recall': [], 'f1_score': [],
            'val_loss': [], 'val_accuracy': [], 'val_auc': [], 'val_precision': [], 'val_recall': [], 'val_f1_score': []
        }
        
        for epoch in range(epochs):
            print(f"Epoch {epoch+1}/{epochs}")
            
            # Train
            ga_model.reset_metrics()
            train_loss = 0
            train_steps = 0
            
            for x_batch, y_batch in train_ds:
                batch_loss = ga_model.train_step(x_batch, y_batch, class_weight=class_weight)
                train_loss += batch_loss
                train_steps += 1
                
            train_metrics = ga_model.get_metrics_result()
            train_loss = train_loss / train_steps
            
            # Validate
            ga_model.reset_metrics()
            val_loss = 0
            val_steps = 0
            
            for x_batch, y_batch in val_ds:
                batch_loss = ga_model.test_step(x_batch, y_batch)
                val_loss += batch_loss
                val_steps += 1
                
            val_metrics = ga_model.get_metrics_result()
            val_loss = val_loss / val_steps
            
            # Update history
            history['loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            
            for metric_name, value in train_metrics.items():
                history[metric_name].append(value)
                
            for metric_name, value in val_metrics.items():
                history[f'val_{metric_name}'].append(value)
                
            # Print metrics
            print(f"Train loss: {train_loss:.4f}, Val loss: {val_loss:.4f}")
            print(f"Train accuracy: {train_metrics['accuracy']:.4f}, Val accuracy: {val_metrics['accuracy']:.4f}")
            print(f"Train AUC: {train_metrics['auc']:.4f}, Val AUC: {val_metrics['auc']:.4f}")
            print(f"Train F1: {train_metrics['f1_score']:.4f}, Val F1: {val_metrics['f1_score']:.4f}")
            
            # Log to TensorBoard
            with tb_callback.writer.as_default():
                tf.summary.scalar('train_loss', train_loss, step=epoch)
                tf.summary.scalar('val_loss', val_loss, step=epoch)
                
                for metric_name, value in train_metrics.items():
                    tf.summary.scalar(f'train_{metric_name}', value, step=epoch)
                    
                for metric_name, value in val_metrics.items():
                    tf.summary.scalar(f'val_{metric_name}', value, step=epoch)
            
            # Early stopping and model checkpoint logic
            current_val_f1 = val_metrics['f1_score']
            
            # Keep track of best model
            if current_val_f1 > best_acc:
                best_acc = current_val_f1
                best_model = model
                print(f"New best model with val_f1_score: {best_acc:.4f}")
                model.save(f"results/EfficientNet_fold{fold + 1}_best.h5")
            
        # Plot training history
        # Convert history dict to a History-like object for compatibility
        class History:
            def __init__(self, history_dict):
                self.history = history_dict
                
        history_obj = History(history)
        plot_training_history(history_obj, fold)
        
        # Evaluate on validation set
        val_loss = history['val_loss'][-1]
        val_acc = history['val_accuracy'][-1]
        val_auc = history['val_auc'][-1]
        val_precision = history['val_precision'][-1]
        val_recall = history['val_recall'][-1]
        val_f1 = history['val_f1_score'][-1]
        
        print(f"Fold {fold + 1} - Validation Metrics:")
        print(f"  Loss: {val_loss:.4f}")
        print(f"  Accuracy: {val_acc:.4f}")
        print(f"  AUC: {val_auc:.4f}")
        print(f"  Precision: {val_precision:.4f}")
        print(f"  Recall: {val_recall:.4f}")
        print(f"  F1 Score: {val_f1:.4f}")
        
        # Store results
        fold_results.append({
            'fold': fold + 1,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'val_auc': val_auc,
            'val_precision': val_precision,
            'val_recall': val_recall,
            'val_f1': val_f1
        })

    # Print summary of cross-validation
    print("\n======= Cross-Validation Results =======")
    metrics = ['val_loss', 'val_acc', 'val_auc', 'val_precision', 'val_recall', 'val_f1']
    for metric in metrics:
        values = [result[metric] for result in fold_results]
        print(f"Mean {metric}: {np.mean(values):.4f} (±{np.std(values):.4f})")

    # Create test dataset
    test_ds = create_dataset(X_test, y_test, augment=False)
    
    # Evaluate best model on test set
    print("\n======= Final Evaluation on Test Set =======")
    test_loss, test_acc, test_auc, test_precision, test_recall, test_f1 = best_model.evaluate(test_ds)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test AUC: {test_auc:.4f}")
    print(f"Test Precision: {test_precision:.4f}")
    print(f"Test Recall: {test_recall:.4f}")
    print(f"Test F1 Score: {test_f1:.4f}")
    
    # Detailed evaluation with confusion matrix and ROC curve
    final_log_dir = "logs/final_evaluation_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    y_pred, y_pred_prob = evaluate_model(best_model, test_ds, y_test, final_log_dir)
    
    # Save best model
    best_model.save('results/best_histopathology_model_final.h5')
    print("Best model saved as 'results/best_histopathology_model_final.h5'")


if __name__ == "__main__":
    # Set memory growth for GPU
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        try:
            for device in physical_devices:
                tf.config.experimental.set_memory_growth(device, True)
            print(f"Found {len(physical_devices)} GPU(s). Memory growth enabled.")
        except Exception as e:
            print(f"Error setting memory growth: {e}")
    
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Create results directory
    os.makedirs('results', exist_ok=True)
    
    # Train model
    train_model()