import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import seaborn as sns
import datetime
import pandas as pd
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.applications import EfficientNetB3
import tensorflow_addons as tfa

# Constants
data_path = "../../../data/BreaKHis_Total_dataset"
labels = ['benign', 'malignant']
img_size = 256  # Increased from 224 to 256
batch_size = 16  # Reduced to ensure stable training
epochs = 50      # Increased from 30 to 50

def loading_data(data_dir):
    data = []
    labels_list = []
    file_paths = []  # Store file paths for debugging

    for label in labels:
        path = os.path.join(data_dir, label)
        class_num = labels.index(label)
        
        if not os.path.exists(path):
            print(f"Warning: Path {path} does not exist")
            continue
            
        files = os.listdir(path)
        total_files = len(files)

        print(f"Loading {label} images ({total_files} files)")

        for i, img in enumerate(files):
            if i % 100 == 0:
                print(f" Progress: {i}/{total_files}")
            
            img_path = os.path.join(path, img)
            img_arr = cv2.imread(img_path)  # Read as color image

            if img_arr is not None:
                # Convert BGR to RGB
                img_arr = cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB)
                resized_arr = cv2.resize(img_arr, (img_size, img_size))
                data.append(resized_arr)
                labels_list.append(class_num)
                file_paths.append(img_path)
            else:
                print(f"Warning: Unable to read image {img_path}")

    # Save file paths for debugging
    with open('processed_files.txt', 'w') as f:
        for path in file_paths:
            f.write(f"{path}\n")
            
    return np.array(data), np.array(labels_list)

def preprocess_data(data, labels):
    # Enhanced preprocessing with per-channel standardization
    X_data = np.array(data).astype('float32')
    
    # Per-channel normalization
    for i in range(X_data.shape[0]):
        for c in range(3):  # Normalize each channel separately
            channel = X_data[i, :, :, c]
            channel_mean = np.mean(channel)
            channel_std = np.std(channel)
            X_data[i, :, :, c] = (channel - channel_mean) / (channel_std + 1e-7)
    
    print(f"Data shape after preprocessing: {X_data.shape}")
    
    y_data = np.array(labels)
    
    return X_data, y_data

def check_class_balance(y):
    unique, counts = np.unique(y, return_counts=True)
    class_distribution = dict(zip([labels[i] for i in unique], counts))
    print(f"Class distribution: {class_distribution}")
    
    # Plot class distribution
    plt.figure(figsize=(8, 5))
    sns.barplot(x=list(class_distribution.keys()), y=list(class_distribution.values()))
    plt.title('Class Distribution')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig('class_distribution.png')
    
    return counts

def create_dataset(X, y, augment=False):
    def _parse_data(image, label):
        # Convert to float32
        image = tf.cast(image, dtype=tf.float32)
        return image, label

    def _strong_augment(image, label):
        # Enhanced augmentation pipeline with stronger transformations
        
        # Random flips
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_flip_up_down(image)
        
        # Color augmentations (more aggressive)
        image = tf.image.random_brightness(image, max_delta=0.4)
        image = tf.image.random_contrast(image, 0.6, 1.4)
        image = tf.image.random_saturation(image, 0.7, 1.3)
        image = tf.image.random_hue(image, 0.1)
        
        # Random rotation with various angles
        angle = tf.random.uniform([], -0.4, 0.4)  # Random rotation between -23 and 23 degrees
        image = tfa.image.rotate(image, angles=angle, fill_mode='reflect')
        
        # Random zoom and crop
        zoom_factor = tf.random.uniform([], 0.7, 1.3)
        image_shape = tf.shape(image)
        crop_size = tf.cast(tf.cast(image_shape[:-1], tf.float32) * zoom_factor, tf.int32)
        if crop_size[0] > 0 and crop_size[1] > 0 and crop_size[0] <= image_shape[0] and crop_size[1] <= image_shape[1]:
            image = tf.image.random_crop(image, [crop_size[0], crop_size[1], 3])
            image = tf.image.resize(image, [img_size, img_size])
        
        # Add Gaussian noise
        noise = tf.random.normal(shape=tf.shape(image), mean=0.0, stddev=0.05)
        image = tf.clip_by_value(image + noise, 0.0, 1.0)
        
        # Random cutout (remove random rectangular areas)
        if tf.random.uniform(()) > 0.5:
            cut_size = tf.cast(tf.cast(image_shape[0], tf.float32) * 0.2, tf.int32)
            x = tf.random.uniform([], 0, image_shape[0] - cut_size, dtype=tf.int32)
            y = tf.random.uniform([], 0, image_shape[1] - cut_size, dtype=tf.int32)
            mask = tf.ones((cut_size, cut_size, 3))
            padding = [[x, image_shape[0] - x - cut_size], 
                       [y, image_shape[1] - y - cut_size],
                       [0, 0]]
            mask = tf.pad(mask, padding)
            image = image * (1 - mask) + mask * tf.random.uniform(())
        
        return image, label

    # Create dataset from numpy arrays
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    dataset = dataset.map(_parse_data, num_parallel_calls=tf.data.AUTOTUNE)
    
    if augment:
        # Apply augmentation with probability 0.8
        aug_dataset = dataset.map(_strong_augment, num_parallel_calls=tf.data.AUTOTUNE)
        # Combine original and augmented datasets (80% augmented, 20% original)
        dataset = tf.data.Dataset.sample_from_datasets([aug_dataset, dataset], weights=[0.8, 0.2])
    
    return dataset.shuffle(2000).batch(batch_size).prefetch(tf.data.AUTOTUNE)

def create_efficientnet_model():
    # Use EfficientNetB3 instead of ResNet50
    base_model = EfficientNetB3(
        include_top=False, 
        weights='imagenet', 
        input_shape=(img_size, img_size, 3), 
        pooling='avg'
    )
    
    # Freeze the base model initially
    base_model.trainable = False
        
    inputs = base_model.input
    x = base_model.output
    
    # Dropout before the dense layers
    x = layers.Dropout(0.2)(x)
    
    # Classification head with batch normalization and more regularization
    x = layers.Dense(512, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.5)(x)
    
    x = layers.Dense(128, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.3)(x)
    
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs, outputs)
    
    # Two-phase training strategy
    return model, base_model

def plot_training_history(history, fold):
    # Plot training & validation metrics
    metrics = ['accuracy', 'loss', 'auc', 'precision', 'recall']
    plt.figure(figsize=(15, 10))
    
    for i, metric in enumerate(metrics):
        plt.subplot(2, 3, i+1)
        plt.plot(history.history[metric])
        plt.plot(history.history[f'val_{metric}'])
        plt.title(f'Model {metric} - Fold {fold + 1}')
        plt.ylabel(metric.capitalize())
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='best')
    
    plt.tight_layout()
    plt.savefig(f'history_fold{fold+1}.png')
    plt.close()

def plot_predictions_histogram(y_pred_prob, y_test):
    # Plot histogram of prediction probabilities
    plt.figure(figsize=(10, 6))
    
    benign_probs = y_pred_prob[y_test == 0]
    malignant_probs = y_pred_prob[y_test == 1]
    
    plt.hist(benign_probs, bins=50, alpha=0.5, label='Benign')
    plt.hist(malignant_probs, bins=50, alpha=0.5, label='Malignant')
    
    plt.xlabel('Prediction Probability')
    plt.ylabel('Count')
    plt.title('Histogram of Prediction Probabilities')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('prediction_histogram.png')
    plt.close()

def evaluate_model(model, X_test, y_test):
    # Get predictions
    y_pred_prob = model.predict(X_test)
    y_pred_prob = y_pred_prob.flatten()
    
    # Find optimal threshold using ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    
    # Find the optimal threshold (maximizes tpr - fpr)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    print(f"Optimal threshold: {optimal_threshold:.4f}")
    
    # Convert probabilities to class predictions using optimal threshold
    y_pred = (y_pred_prob > optimal_threshold).astype(int)
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=labels))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels, cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig('final_confusion_matrix.png')
    plt.close()
    
    # ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.scatter(fpr[optimal_idx], tpr[optimal_idx], marker='o', color='red', 
                label=f'Optimal threshold: {optimal_threshold:.2f}')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.savefig('roc_curve.png')
    plt.close()
    
    # Plot prediction histogram
    plot_predictions_histogram(y_pred_prob, y_test)
    
    # Calculate additional metrics
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    
    print(f"Sensitivity: {sensitivity:.4f}")
    print(f"Specificity: {specificity:.4f}")
    
    return y_pred, y_pred_prob, optimal_threshold

def visualize_incorrect_predictions(X_test, y_test, y_pred, file_paths=None):
    # Find incorrect predictions
    incorrect_indices = np.where(y_test != y_pred)[0]
    
    if len(incorrect_indices) == 0:
        print("No incorrect predictions found!")
        return
    
    # Create a directory for saving misclassified images
    os.makedirs('misclassified', exist_ok=True)
    
    # Visualize a sample of the incorrect predictions
    sample_size = min(10, len(incorrect_indices))
    sample_indices = np.random.choice(incorrect_indices, sample_size, replace=False)
    
    plt.figure(figsize=(15, 10))
    for i, idx in enumerate(sample_indices):
        plt.subplot(2, 5, i+1)
        
        # Denormalize image for visualization
        img = X_test[idx]
        img = (img - img.min()) / (img.max() - img.min())
        
        plt.imshow(img)
        plt.title(f'True: {labels[y_test[idx]]}\nPred: {labels[y_pred[idx]]}')
        plt.axis('off')
        
        # Save individual images
        plt.figure(figsize=(6, 6))
        plt.imshow(img)
        plt.title(f'True: {labels[y_test[idx]]}, Pred: {labels[y_pred[idx]]}')
        plt.axis('off')
        path_info = f" ({file_paths[idx]})" if file_paths is not None else ""
        plt.savefig(f'misclassified/misclassified_{idx}{path_info}.png')
        plt.close()
    
    plt.tight_layout()
    plt.savefig('misclassified_samples.png')
    plt.close()

def train_model():
    # Load and preprocess data
    print("Loading data...")
    data, labels_data = loading_data(data_path)
    print(f"Data loaded with shape: {data.shape}, Labels shape: {labels_data.shape}")
    
    # Check if data was loaded successfully
    if len(data) == 0:
        print("Error: No data was loaded. Please check data path and file formats.")
        return
    
    # Basic analysis of loaded data
    print(f"Data statistics - Min: {data.min()}, Max: {data.max()}, Mean: {data.mean():.2f}, Std: {data.std():.2f}")
    
    X, y = preprocess_data(data, labels_data)
    
    # Store file paths for debugging if loading_data is updated to return them
    file_paths = None
    
    # Check class balance and calculate class weights
    class_counts = check_class_balance(y)
    total_samples = sum(class_counts)
    n_classes = len(class_counts)
    
    # Calculate balanced class weights
    class_weight = {}
    for i in range(n_classes):
        class_weight[i] = total_samples / (n_classes * class_counts[i])
    print(f"Using class weights: {class_weight}")

    # Split data with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, 
        y, 
        test_size=0.15,
        random_state=42,
        stratify=y  # Ensure balanced split
    )
    
    # Save some test images for visual inspection
    os.makedirs('sample_images', exist_ok=True)
    for i in range(min(5, len(X_test))):
        img = X_test[i].copy()
        img = (img - img.min()) / (img.max() - img.min())  # Normalize to [0,1] for visualization
        plt.figure(figsize=(6, 6))
        plt.imshow(img)
        plt.title(f"Class: {labels[y_test[i]]}")
        plt.axis('off')
        plt.savefig(f'sample_images/test_sample_{i}.png')
        plt.close()

    # Create datasets
    train_ds = create_dataset(X_train, y_train, augment=True)
    test_ds = create_dataset(X_test, y_test, augment=False)
    
    # Create model
    print("Creating model...")
    model, base_model = create_efficientnet_model()
    
    # Compile model with additional metrics
    model.compile(
        optimizer=Adam(learning_rate=1e-3), 
        loss='binary_crossentropy', 
        metrics=[
            'accuracy', 
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.Precision(name='precision'), 
            tf.keras.metrics.Recall(name='recall')
        ]
    )
    
    # Setup callbacks
    log_dir = "logs/EfficientNet_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    
    callbacks = [
        EarlyStopping(
            monitor='val_auc', 
            patience=15,
            mode='max',
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            "best_model.h5", 
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
    
    # Phase 1: Train only the top layers
    print("\n===== Phase 1: Training only top layers =====")
    history1 = model.fit(
        train_ds, 
        validation_data=test_ds, 
        epochs=10,  # Shorter initial training 
        callbacks=callbacks,
        class_weight=class_weight
    )
    
    # Phase 2: Fine-tune the entire model
    print("\n===== Phase 2: Fine-tuning entire model =====")
    # Unfreeze the base model
    base_model.trainable = True
    
    # Recompile with lower learning rate
    model.compile(
        optimizer=Adam(learning_rate=1e-4),  # Lower learning rate for fine-tuning
        loss='binary_crossentropy', 
        metrics=[
            'accuracy', 
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.Precision(name='precision'), 
            tf.keras.metrics.Recall(name='recall')
        ]
    )
    
    history2 = model.fit(
        train_ds, 
        validation_data=test_ds, 
        epochs=epochs, 
        callbacks=callbacks,
        class_weight=class_weight,
        initial_epoch=history1.epoch[-1] + 1  # Continue from where we left off
    )
    
    # Combine histories for plotting
    combined_history = {}
    for key in history1.history.keys():
        combined_history[key] = history1.history[key] + history2.history[key]
    
    class HistoryWrapper:
        def __init__(self, history):
            self.history = history
    
    plot_training_history(HistoryWrapper(combined_history), 0)
    
    # Detailed evaluation with confusion matrix and ROC curve
    print("\n===== Final Evaluation on Test Set =====")
    test_loss, test_acc, test_auc, test_precision, test_recall = model.evaluate(test_ds)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test AUC: {test_auc:.4f}")
    print(f"Test Precision: {test_precision:.4f}")
    print(f"Test Recall: {test_recall:.4f}")
    
    # Get predictions on raw test data for detailed analysis
    y_pred, y_pred_prob, optimal_threshold = evaluate_model(model, X_test, y_test)
    
    # Visualize incorrect predictions
    visualize_incorrect_predictions(X_test, y_test, y_pred, file_paths)
    
    # Save best model
    model.save('best_histopathology_model_final.h5')
    
    # Save the optimal threshold for future use
    with open('model_threshold.txt', 'w') as f:
        f.write(f"{optimal_threshold}")
    
    print("Training complete. Model saved as 'best_histopathology_model_final.h5'")
    print(f"Optimal threshold saved as: {optimal_threshold}")

def predict_single_image(image_path, model_path, threshold_path=None):
    """
    Function to predict a single image using the trained model
    """
    # Load the model
    model = tf.keras.models.load_model(model_path)
    
    # Load the optimal threshold if available
    threshold = 0.5
    if threshold_path and os.path.exists(threshold_path):
        with open(threshold_path, 'r') as f:
            threshold = float(f.read().strip())
    
    # Load and preprocess the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Cannot read image {image_path}")
        return None
    
    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (img_size, img_size))
    
    # Normalize the image
    img = img.astype('float32')
    for c in range(3):
        channel = img[:, :, c]
        channel_mean = np.mean(channel)
        channel_std = np.std(channel)
        img[:, :, c] = (channel - channel_mean) / (channel_std + 1e-7)
    
    # Add batch dimension
    img = np.expand_dims(img, axis=0)
    
    # Get prediction
    pred = model.predict(img)[0][0]
    
    # Apply threshold
    pred_class = 1 if pred > threshold else 0
    
    print(f"Prediction for {image_path}:")
    print(f"Class: {labels[pred_class]} (Probability: {pred:.4f}, Threshold: {threshold:.4f})")
    
    return pred, pred_class

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
    
    # Example of using the prediction function (uncomment when needed)
    # predict_single_image("path_to_image.jpg", "best_histopathology_model_final.h5", "model_threshold.txt")