import os
import cv2
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
from tensorflow.keras.applications import ResNet50
import tensorflow_addons as tfa

# Constants
data_path = "../../../data/BreaKHis_Total_dataset"
labels = ['benign', 'malignant']
img_size = 224
batch_size = 24  # Increased from 16 to 32
epochs = 30      # Increased from 10 to 30

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
            
            # Removed the 2000 image limit to use all available data
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
    X_data = (X_data - X_data.mean()) / (X_data.std() + 1e-7)  # Adding epsilon to avoid division by zero
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
        # Enhanced augmentation pipeline
        # Random flips
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_flip_up_down(image)
        
        # Color augmentations
        image = tf.image.random_brightness(image, max_delta=0.3)
        image = tf.image.random_contrast(image, 0.7, 1.3)
        image = tf.image.random_saturation(image, 0.8, 1.2)
        
        # Random rotation with various angles
        angle = tf.random.uniform([], -0.2, 0.2)  # Random rotation between -11.5 and 11.5 degrees
        image = tfa.image.rotate(image, angles=angle, fill_mode='reflect')
        
        # Random zoom
        zoom_factor = tf.random.uniform([], 0.8, 1.2)
        image_shape = tf.shape(image)
        crop_size = tf.cast(tf.cast(image_shape[:-1], tf.float32) * zoom_factor, tf.int32)
        if crop_size[0] > 0 and crop_size[1] > 0 and crop_size[0] <= image_shape[0] and crop_size[1] <= image_shape[1]:
            image = tf.image.random_crop(image, [crop_size[0], crop_size[1], 3])
            image = tf.image.resize(image, [img_size, img_size])
        
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

def create_resnet_model():
    # Use ResNet50 with improved architecture
    base_model = ResNet50(
        include_top=False, 
        weights='imagenet', 
        input_shape=(img_size, img_size, 3), 
        pooling='avg'
    )
    
    # Freeze fewer layers to allow more fine-tuning
    for layer in base_model.layers[:-10]:  # Unfreeze more layers (50 instead of 30)
        layer.trainable = False
        
    inputs = base_model.input
    x = base_model.output
    
    # More complex classification head with batch normalization
    x = layers.Dense(1024, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.5)(x)
    
    x = layers.Dense(512, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.3)(x)
    
    outputs = layers.Dense(1, activation='sigmoid')(x)
    return tf.keras.Model(inputs, outputs)

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
    plt.savefig(f'history_fold{fold+1}.png')
    plt.close()

def evaluate_model(model, test_ds, y_test):
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
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels, cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig('final_confusion_matrix.png')
    plt.show()
    
    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob_flat)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig('roc_curve.png')
    plt.show()
    
    return y_pred, y_pred_prob_flat

def train_model():
    # Load and preprocess data
    print("Loading data...")
    data, labels_data = loading_data(data_path)
    X, y = preprocess_data(data, labels_data)
    
    # Check class balance and calculate class weights if needed
    class_counts = check_class_balance(y)
    if len(np.unique(y)) == 2:  # Binary classification
        class_weight = {
            0: len(y) / (2.0 * np.sum(y == 0)),
            1: len(y) / (2.0 * np.sum(y == 1))
        }
        print(f"Using class weights: {class_weight}")
    else:
        class_weight = None

    # Split data into train/validation/test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, 
        y, 
        test_size=0.15,  # Increased test size for better evaluation
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
        model = create_resnet_model()
        
        # Learning rate schedule
        initial_learning_rate = 1e-5
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate,
            decay_steps=1000,
            decay_rate=0.9,
            staircase=True
        )
        
        optimizer = Adam(learning_rate=lr_schedule)
        
        # Compile model with additional metrics
        model.compile(
            optimizer=optimizer, 
            loss='binary_crossentropy', 
            metrics=[
                'accuracy', 
                tf.keras.metrics.AUC(name='auc'),
                tf.keras.metrics.Precision(name='precision'), 
                tf.keras.metrics.Recall(name='recall')
            ]
        )
        
        # Setup callbacks
        log_dir = f"logs/Resnet_fold{fold+1}_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        
        callbacks = [
            EarlyStopping(
                monitor='val_auc', 
                patience=10,
                mode='max',
                restore_best_weights=True,
                verbose=1
            ),
            ModelCheckpoint(
                f"ResNet_fold{fold + 1}.h5", 
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
        
        # Train model
        print("Training model...")
        history = model.fit(
            train_ds, 
            validation_data=val_ds, 
            epochs=epochs, 
            callbacks=callbacks,
            class_weight=class_weight
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
        
        # Keep track of best model
        if val_acc > best_acc:
            best_acc = val_acc
            best_model = model

    # Print summary of cross-validation
    print("\n======= Cross-Validation Results =======")
    metrics = ['val_loss', 'val_acc', 'val_auc', 'val_precision', 'val_recall']
    for metric in metrics:
        values = [result[metric] for result in fold_results]
        print(f"Mean {metric}: {np.mean(values):.4f} (±{np.std(values):.4f})")

    # Create test dataset
    test_ds = create_dataset(X_test, y_test, augment=False)
    
    # Evaluate best model on test set
    print("\n======= Final Evaluation on Test Set =======")
    test_loss, test_acc, test_auc, test_precision, test_recall = best_model.evaluate(test_ds)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test AUC: {test_auc:.4f}")
    print(f"Test Precision: {test_precision:.4f}")
    print(f"Test Recall: {test_recall:.4f}")
    
    # Detailed evaluation with confusion matrix and ROC curve
    y_pred, y_pred_prob = evaluate_model(best_model, test_ds, y_test)
    
    # Save best model
    best_model.save('best_histopathology_model.h5')
    print("Best model saved as 'best_histopathology_model.h5'")

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

