import os
import cv2 
import io
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import seaborn as sns
import datetime
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import tensorflow_addons as tfa

# Constants
data_path = "../../../data/BreaKHis_Total_dataset"
labels = ['benign', 'malignant']
img_size = 224
batch_size = 10 
epochs = 10

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
    X_data = np.array(zaq   data).astype('float32')
    # Global standardization
    X_data = (X_data - X_data.mean()) / (X_data.std() + 1e-7)
    X_data = X_data.reshape(-1, img_size, img_size, 1)
    print(f"Data shape after preprocessing: {X_data.shape}")
    
    y_data = np.array(labels)
    
    return X_data, y_data

def check_class_balance(y):
    unique, counts = np.unique(y, return_counts=True)
    print(f"Class distribution: {dict(zip([labels[i] for i in unique], counts))}")
    return counts

def focal_loss(gamma=2.0, alpha=0.75):
    """
    Focal Loss for addressing class imbalance.
    alpha: weighs the importance of positive class (set higher for the minority class)
    gamma: focuses more on hard examples
    """
    def focal_loss_with_logits(logits, targets, alpha, gamma, y_pred):
        targets = tf.cast(targets, dtype=tf.float32)
        
        # Standard binary cross entropy calculation
        BCE = tf.keras.losses.binary_crossentropy(targets, y_pred)
        
        # Focal loss weights
        alpha_t = targets * alpha + (1 - targets) * (1 - alpha)
        p_t = targets * y_pred + (1 - targets) * (1 - y_pred)
        FL = alpha_t * tf.pow(1 - p_t, gamma) * BCE
        
        return tf.reduce_mean(FL)
    
    def loss(y_true, y_pred):
        return focal_loss_with_logits(y_pred, y_true, alpha, gamma, y_pred)
    
    return loss

def _strong_augment(image):
    # Strong augmentation for underrepresented class
    
    # Random flips
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    
    # Multiple random rotations (more varied angles)
    angle = tf.random.uniform([], -0.4, 0.4)  # About ±23 degrees
    image = tfa.image.rotate(image, angles=angle, fill_mode='reflect')
    
    # Random zoom and crop
    zoom_factor = tf.random.uniform([], 0.7, 1.3)
    image_shape = tf.shape(image)
    crop_size = tf.cast(tf.cast(image_shape[:-1], tf.float32) * zoom_factor, tf.int32)
    if crop_size[0] > 0 and crop_size[1] > 0 and crop_size[0] <= image_shape[0] and crop_size[1] <= image_shape[1]:
        image = tf.image.random_crop(image, [crop_size[0], crop_size[1], 3])
        image = tf.image.resize(image, [img_size, img_size])
    
    # Color transformations
    image = tf.image.random_brightness(image, max_delta=0.5)
    image = tf.image.random_contrast(image, 0.5, 1.5)
    image = tf.image.random_saturation(image, 0.6, 1.4)
    
    # Add random noise
    noise = tf.random.normal(shape=tf.shape(image), mean=0.0, stddev=0.08)
    image = tf.clip_by_value(image + noise, 0.0, 1.0)
    
    # Random translation
    translation = tf.random.uniform([2], -0.2, 0.2, dtype=tf.float32)
    image = tfa.image.translate(image, translation * img_size, fill_mode='reflect')
    
    return image

def create_balanced_dataset(X, y, augment=False):
    # Separate samples by class
    benign_indices = np.where(y == 0)[0]
    malignant_indices = np.where(y == 1)[0]
    
    # Calculate how many extra benign samples we need through augmentation
    augmentation_factor = len(malignant_indices) // len(benign_indices)
    
    def _generator():
        # Include all original samples
        for i in range(len(X)):
            rgb = tf.image.grayscale_to_rgb(tf.convert_to_tensor(X[i]))
            label = y[i]
            yield rgb, label
            
        # Extra augmentation for benign class if needed
        if augment:
            for _ in range(augmentation_factor - 1):  # -1 because we already included original samples
                for i in benign_indices:
                    rgb = tf.image.grayscale_to_rgb(tf.convert_to_tensor(X[i]))
                    # Apply strong augmentation to create diverse samples
                    rgb = _strong_augment(rgb)
                    yield rgb, 0  # benign class
    
    def _augment(image, label):
        # Standard augmentation for all samples
        if tf.random.uniform([], 0, 1) > 0.5:
            image = tf.image.random_flip_left_right(image)
        if tf.random.uniform([], 0, 1) > 0.5:
            image = tf.image.random_flip_up_down(image)
            
        # Random rotation
        angle = tf.random.uniform([], -0.2, 0.2)
        image = tfa.image.rotate(image, angles=angle, fill_mode='reflect')
        
        # Color jitter
        image = tf.image.random_brightness(image, max_delta=0.2)
        image = tf.image.random_contrast(image, 0.8, 1.2)
        
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

def create_efficient_model():
    # Use EfficientNetB2 which has better performance than ResNet50 for medical imaging
    base_model = tf.keras.applications.EfficientNetB2(
        include_top=False, 
        weights='imagenet', 
        input_shape=(img_size, img_size, 3), 
        pooling=None
    )
    
    # Keep base model frozen initially
    base_model.trainable = False
        
    inputs = base_model.input
    x = base_model.output
    
    # Add spatial attention mechanism
    x = layers.GlobalAveragePooling2D()(x)
    
    # Feature extraction layers with regularization
    x = layers.Dense(512)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('swish')(x)  # EfficientNet uses swish activation
    x = layers.Dropout(0.5)(x)
    
    x = layers.Dense(256)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('swish')(x)
    x = layers.Dropout(0.3)(x)
    
    # Output layer
    outputs = layers.Dense(1, activation='sigmoid')(x)
    model = tf.keras.Model(inputs, outputs)
    
    return model

def train_with_progressive_strategy(model, train_ds, val_ds, epochs=30, callbacks=None):
    # Stage 1: Train only the top classification layers
    print("Stage 1: Training classification head only...")
    for layer in model.layers:
        if isinstance(layer, tf.keras.models.Model):  # This is the base model
            layer.trainable = False
    
    # Use higher learning rate for initial training
    model.compile(
        optimizer=tf.keras.optimizers.Adam(3e-4),
        loss=focal_loss(gamma=2.0, alpha=0.75),  # Focus more on benign class
        metrics=[
            'accuracy',
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall')
        ]
    )
    
    history1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=10,
        callbacks=callbacks
    )
    
    # Stage 2: Fine-tune upper layers of the base model
    print("Stage 2: Fine-tuning upper layers...")
    
    # Unfreeze the top layers of the base model
    base_model = model.layers[0]  # Assuming base model is the first layer
    for layer in model.layers[-30:]:  # Unfreeze last 30 layers
        layer.trainable = True
    
    # Lower learning rate for fine-tuning
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),
        loss=focal_loss(gamma=2.0, alpha=0.75),
        metrics=[
            'accuracy',
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall')
        ]
    )
    
    history2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs - 10,
        callbacks=callbacks
    )
    
    # Combine histories
    full_history = {}
    for k in history1.history:
        if k in history2.history:
            full_history[k] = history1.history[k] + history2.history[k]
    
    return model, full_history

def plot_training_history(history, fold):
    # Plot training & validation accuracy values
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['accuracy'])
    plt.plot(history['val_accuracy'])
    plt.title(f'Model Accuracy - Fold {fold + 1}')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    # Plot training & validation loss values
    plt.subplot(1, 2, 2)
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title(f'Model Loss - Fold {fold + 1}')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.tight_layout()
    plt.savefig(f'history_fold{fold+1}.png')
    plt.close()

def test_time_augmentation(model, X_test_ds, n_augmentations=5):
    """Apply augmentation at test time for more robust predictions"""
    # Get the original predictions
    orig_preds = model.predict(X_test_ds)
    all_preds = [orig_preds]
    
    # Create augmented versions of the test set
    for i in range(n_augmentations):
        # Create an augmented dataset
        def _light_augment(image, label):
            # Simple augmentations for test-time
            if tf.random.uniform([], 0, 1) > 0.5:
                image = tf.image.flip_left_right(image)
            if tf.random.uniform([], 0, 1) > 0.5:
                image = tf.image.flip_up_down(image)
            
            # Small rotation
            angle = tf.random.uniform([], -0.15, 0.15)
            image = tfa.image.rotate(image, angles=angle, fill_mode='reflect')
            
            return image, label
        
        # Apply the augmentation to the dataset
        aug_ds = X_test_ds.map(_light_augment)
        
        # Get predictions on the augmented dataset
        aug_preds = model.predict(aug_ds)
        all_preds.append(aug_preds)
    
    # Stack and average the predictions
    stacked_preds = np.stack(all_preds)
    avg_preds = np.mean(stacked_preds, axis=0)
    
    return avg_preds

def optimize_threshold(y_true, y_pred_prob):
    """Find the optimal threshold for classification"""
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob)
    
    # Calculate the geometric mean of sensitivity and specificity
    gmeans = np.sqrt(tpr * (1-fpr))
    
    # Locate the index of the largest g-mean
    ix = np.argmax(gmeans)
    best_threshold = thresholds[ix]
    print(f"Best Threshold: {best_threshold:.4f}, G-Mean: {gmeans[ix]:.4f}")
    print(f"At this threshold - Sensitivity: {tpr[ix]:.4f}, Specificity: {1-fpr[ix]:.4f}")
    
    return best_threshold

def evaluate_model(model, test_ds, y_test, log_dir, epoch=0, threshold=0.5):
    # Create image writer for this evaluation
    tb_image_writer = log_images_to_tensorboard(log_dir)

    # Get predictions with test-time augmentation
    print("Applying test-time augmentation...")
    y_pred_prob = test_time_augmentation(model, test_ds, n_augmentations=3)
    
    # Extract probabilities and convert to flat array
    y_pred_prob_flat = []
    for batch in y_pred_prob:
        for prob in batch:
            y_pred_prob_flat.append(prob)
    y_pred_prob_flat = np.array(y_pred_prob_flat)[:len(y_test)]
    
    # Find optimal threshold
    print("Finding optimal classification threshold...")
    optimal_threshold = optimize_threshold(y_test, y_pred_prob_flat)
    
    # Convert probabilities to class predictions using the optimal threshold
    y_pred = (y_pred_prob_flat > optimal_threshold).astype(int)
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=labels))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    fig = plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels, cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig('final_confusion_matrix.png')
    
    log_image(tb_image_writer, 'confusion_matrix', fig, step=epoch)
    
    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob_flat)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    fig = plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig('roc_curve.png')

    log_image(tb_image_writer, 'roc_curve', fig, step=epoch)
    
    return y_pred, y_pred_prob_flat, optimal_threshold

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

def train_model():
    # Load and preprocess data
    print("Loading data...")
    data, labels_data = loading_data(data_path)
    X, y = preprocess_data(data, labels_data)
    
    # Check class balance
    class_counts = check_class_balance(y)
    
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
    best_auc = 0
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_temp, y_temp)):
        print(f"\n=========== Fold {fold + 1}/5 ===========")

        X_train, X_val = X_temp[train_idx], X_temp[val_idx]
        y_train, y_val = y_temp[train_idx], y_temp[val_idx]
        
        print(f"Train set: {X_train.shape}, {y_train.shape}")
        print(f"Validation set: {X_val.shape}, {y_val.shape}")

        # Create balanced datasets with augmentation
        train_ds = create_balanced_dataset(X_train, y_train, augment=True)
        val_ds = create_balanced_dataset(X_val, y_val, augment=False)
        
        # Create EfficientNet model
        print("Creating EfficientNetB2 model...")
        model = create_efficient_model()
        
        # Setup callbacks
        log_dir = f"logs/EfficientNet_fold{fold+1}_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        
        callbacks = [
            EarlyStopping(
                monitor='val_auc', 
                patience=10,
                mode='max',
                restore_best_weights=True,
                verbose=1
            ),
            ModelCheckpoint(
                f"EfficientNet_fold{fold + 1}.h5", 
                monitor='val_auc',
                mode='max',
                save_best_only=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1
            ),
            tf.keras.callbacks.TensorBoard(log_dir=log_dir)
        ]
        
        # Train model with progressive strategy
        print("Training model with progressive strategy...")
        model, history = train_with_progressive_strategy(
            model, 
            train_ds, 
            val_ds, 
            epochs=epochs, 
            callbacks=callbacks
        )
        
        # Plot training history
        plot_training_history(history, fold)
        
        # Evaluate on validation set
        val_loss, val_acc, val_auc, val_precision, val_recall = model.evaluate(val_ds)
        print(f"Fold {fold + 1} - Validation Metrics:")
        print(f"  Loss: {val_loss:.4f}")
        print(f"  Accuracy: {val_acc:.4f}")
        print(f"  AUC: {val_auc:.4f}")
        print(f"  Precision: {val_precision:.4f}")
        print(f"  Recall: {val_recall:.4f}")
        
        # Store results
        fold_results.append({
            'fold': fold + 1,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'val_auc': val_auc,
            'val_precision': val_precision,
            'val_recall': val_recall
        })
        
        # Keep track of best model (using AUC as primary metric)
        if val_auc > best_auc:
            best_auc = val_auc
            best_acc = val_acc
            best_model = model
            print(f"New best model found with AUC: {best_auc:.4f}")

    # Print summary of cross-validation
    print("\n======= Cross-Validation Results =======")
    metrics = ['val_loss', 'val_acc', 'val_auc', 'val_precision', 'val_recall']
    for metric in metrics:
        values = [result[metric] for result in fold_results]
        print(f"Mean {metric}: {np.mean(values):.4f} (±{np.std(values):.4f})")

    # Create test dataset
    test_ds = create_balanced_dataset(X_test, y_test, augment=False)
    
    # Evaluate best model on test set
    print("\n======= Final Evaluation on Test Set =======")
    test_loss, test_acc, test_auc, test_precision, test_recall = best_model.evaluate(test_ds)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test AUC: {test_auc:.4f}")
    print(f"Test Precision: {test_precision:.4f}")
    print(f"Test Recall: {test_recall:.4f}")
    
    # Detailed evaluation with confusion matrix and ROC curve
    final_log_dir = "logs/final_evaluation_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    y_pred, y_pred_prob, optimal_threshold = evaluate_model(best_model, test_ds, y_test, final_log_dir)
    
    # Save best model
    best_model.save('best_histopathology_model.h5')
    print("Best model saved as 'best_histopathology_model.h5'")
    
    # Save the optimal threshold for future use
    np.save('optimal_threshold.npy', optimal_threshold)
    print(f"Optimal threshold ({optimal_threshold:.4f}) saved as 'optimal_threshold.npy'")

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
    
    # Train model
    train_model()

