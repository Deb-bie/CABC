import os
import cv2 # type: ignore
import io
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import seaborn as sns
import datetime
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras import layers # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau # type: ignore
import tensorflow_addons as tfa # type: ignore
import tensorflow_hub as hub # type: ignore


# Constants
data_path = "../../../data/BreaKHis_Total_dataset"
labels = ['benign', 'malignant']
img_size = 224
batch_size = 10
epochs = 10


class GPUMemoryCallback(tf.keras.callbacks.Callback):
    """
    Callback to monitor GPU memory usage during training
    """
    def __init__(self, log_dir=None, log_freq=1):
        """
        Initialize the callback
        
        Args:
            log_dir: Directory to write TensorBoard logs (if None, no logs written)
            log_freq: Log frequency (in epochs)
        """
        super(GPUMemoryCallback, self).__init__()
        self.log_dir = log_dir
        self.log_freq = log_freq
        self.writer = None
        
        if log_dir:
            self.writer = tf.summary.create_file_writer(log_dir + '/gpu_memory')
    
    def on_epoch_end(self, epoch, logs=None):
        """Log at the end of each epoch"""
        if epoch % self.log_freq != 0:
            return
        
        # Get GPU memory info
        physical_devices = tf.config.list_physical_devices('GPU')
        if not physical_devices:
            print("No GPU available. Skipping memory monitoring.")
            return
        
        try:
            # Initialize containers for metrics
            memory_info = []
            
            # Get memory info for each GPU
            for i, device in enumerate(physical_devices):
                # Get memory info
                memory = tf.config.experimental.get_memory_info(device.name)
                current_memory_mb = memory['current'] / (1024 * 1024)
                peak_memory_mb = memory['peak'] / (1024 * 1024)
                
                # Print info
                print(f"Epoch {epoch}: GPU-{i} Memory - "
                      f"Current: {current_memory_mb:.2f} MB, "
                      f"Peak: {peak_memory_mb:.2f} MB")
                
                # Store for TensorBoard
                memory_info.append((current_memory_mb, peak_memory_mb))
            
            # Log to TensorBoard if writer exists
            if self.writer:
                with self.writer.as_default():
                    for i, (current, peak) in enumerate(memory_info):
                        tf.summary.scalar(f'gpu_{i}/current_memory_mb', current, step=epoch)
                        tf.summary.scalar(f'gpu_{i}/peak_memory_mb', peak, step=epoch)
        
        except Exception as e:
            print(f"Error in GPU memory monitoring: {e}")


def memory_efficient_loading_data(data_dir):
    """
    Load image paths instead of actual images to save memory
    """
    image_paths = []
    labels_list = []
    
    for label in labels:
        path = os.path.join(data_dir, label)
        class_num = labels.index(label)
        files = os.listdir(path)
        total_files = len(files)

        print(f"Processing {label} image paths ({total_files} files)")

        for img in files:
            img_path = os.path.join(path, img)
            image_paths.append(img_path)
            labels_list.append(class_num)
    
    return np.array(image_paths), np.array(labels_list)


def create_path_dataset(image_paths, labels, batch_size=32, is_training=False):
    """
    Create a dataset that loads and processes images on-the-fly
    """
    # Combine paths and labels
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    
    # Function to load and process a single image
    def load_and_preprocess(path, label):
        # Read the image file
        img = tf.io.read_file(path)
        # Decode the image - use decode_jpeg for both JPEG and PNG (it's more versatile)
        img = tf.io.decode_image(img, channels=1, expand_animations=False)
        # Resize the image
        img = tf.image.resize(img, [img_size, img_size])
        # Normalize pixel values
        img = tf.cast(img, tf.float32) / 255.0
        # Convert to RGB
        img = tf.image.grayscale_to_rgb(img)
        
        # Data augmentation (only if training)
        if is_training:
            img = tf.image.random_flip_left_right(img)
            img = tf.image.random_flip_up_down(img)
            img = tf.image.rot90(
                img, 
                k=tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)
            )
            img = tf.image.random_brightness(img, max_delta=0.1)
            img = tf.image.random_contrast(img, lower=0.9, upper=1.1)
            
        return img, label
    
    # Apply the function to each element
    dataset = dataset.map(
        load_and_preprocess, 
        num_parallel_calls=tf.data.AUTOTUNE
    )
    
    # Optimize performance
    if is_training:
        dataset = dataset.shuffle(buffer_size=2000)
    
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset


def create_improved_dataset(X, y, batch_size=32, is_training=False):
    """
    Create an optimized TensorFlow dataset that processes images efficiently
    
    Args:
        X: Image data array
        y: Labels array
        batch_size: Batch size for training
        is_training: Whether this dataset is for training (enables augmentation)
        
    Returns:
        A tf.data.Dataset object
    """
    # Create the base dataset from numpy arrays
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    
    # Define preprocessing function
    def preprocess(image, label):
        # Convert to float and normalize
        image = tf.cast(image, tf.float32) / 255.0
        
        # Convert grayscale to RGB (expanding channels from 1 to 3)
        image = tf.image.grayscale_to_rgb(image)
        
        # Data augmentation (only for training)
        if is_training:
            # Random flip
            image = tf.image.random_flip_left_right(image)
            image = tf.image.random_flip_up_down(image)
            
            # Random rotation (using TensorFlow 2.x)
            image = tf.image.rot90(
                image, 
                k=tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)
            )
            
            # Random brightness and contrast
            image = tf.image.random_brightness(image, max_delta=0.1)
            image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
        
        return image, label
    
    # Apply preprocessing to each element
    dataset = dataset.map(
        preprocess, 
        num_parallel_calls=tf.data.AUTOTUNE
    )
    
    # Optimize dataset performance
    if is_training:
        dataset = dataset.shuffle(buffer_size=2000)
    
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset


# def loading_data(data_dir):
#     data = []
#     labels_list = []
#     for label in labels:
#         path = os.path.join(data_dir, label)
#         class_num = labels.index(label)
#         files = os.listdir(path)
#         total_files = len(files)

#         print(f"Loading {label} images ({total_files} files)")

#         for i, img in enumerate(files):
#             if i % 100 == 0:
#                 print(f" Progress: {i}/{total_files}")

#             img_path = os.path.join(path, img)
#             img_arr = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

#             if img_arr is not None:
#                 resized_arr = cv2.resize(img_arr, (img_size, img_size))
#                 data.append(resized_arr)
#                 labels_list.append(class_num)
#             else:
#                 print(f"Warning: Unable to read image {img_path}")

#     return np.array(data), np.array(labels_list)

# def preprocess_data(data, labels):
#     X_data = np.array(data).astype('float32')
#     X_data = (X_data - X_data.mean()) / (X_data.std() + 1e-7) 
#     X_data = X_data.reshape(-1, img_size, img_size, 1)
#     print(f"Data shape after preprocessing: {X_data.shape}")
    
#     y_data = np.array(labels)
    
#     return X_data, y_data

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

# def rgb_generator(X, y):
#     for i in range(len(X)):
#         rgb = tf.image.grayscale_to_rgb(tf.convert_to_tensor(X[i]))
#         yield rgb.numpy(), y[i]

# def create_dataset(X, y):
#     return tf.data.Dataset.from_generator(
#         lambda: rgb_generator(X, y),
#         output_types=(tf.float32, tf.int32),
#         output_shapes=((img_size, img_size, 3), ())
#     ).shuffle(2000).batch(batch_size).prefetch(tf.data.AUTOTUNE)


def create_deit_model():
    """
    Create a model using the DeiT (Data-efficient image Transformer) architecture.
    DeiT is a Vision Transformer that performs well even with limited data.
    """
    # Load DeiT base model from TensorFlow Hub
    deit_url = "https://tfhub.dev/sayakpaul/deit_base_patch16_224/1"
    
    # Create input layer
    inputs = layers.Input(shape=(img_size, img_size, 3))
    
    # Preprocessing as required by DeiT
    # Normalize the images to [0, 1] range
    x = tf.keras.layers.Rescaling(1./255)(inputs)
    
    # Load the DeiT model with the pretrained weights
    deit_model = hub.KerasLayer(deit_url, trainable=True)

    # Get outputs (returns multiple tensors)
    deit_outputs = deit_model(x)

    # Extract the class token (first output) for classification
    x = deit_outputs[0]  # Shape: (batch_size, 1000)
    
    # Add classification head
    x = layers.Dense(1024, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.5)(x)
    
    x = layers.Dense(512, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.3)(x)
    
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    # Create and return model
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    return model


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
    plt.savefig(f'history_fold{fold+1}.png')
    plt.close()

def evaluate_model(model, test_ds, y_test, log_dir, epoch=0):
    # Create image writer for this evaluation
    tb_image_writer = log_images_to_tensorboard(log_dir)

    # Get predictions
    y_pred_prob = model.predict(test_ds)
    
    # Extract probabilities and convert to flat array
    y_pred_prob_flat = []
    for prob in y_pred_prob:
        y_pred_prob_flat.append(prob[0])
    y_pred_prob_flat = np.array(y_pred_prob_flat)[:len(y_test)]
    
    # Convert probabilities to class predictions
    y_pred = (y_pred_prob_flat > 0.5).astype(int)
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=labels))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
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
    
    return y_pred, y_pred_prob_flat



# def log_images_to_tensorboard(log_dir):
#     """Create a TensorBoard image logger"""
#     file_writer = tf.summary.create_file_writer(log_dir + '/images')
#     return file_writer

# def log_image(file_writer, name, figure, step=0):
#     """Log a matplotlib figure to TensorBoard"""
#     with file_writer.as_default():
#         # Convert figure to PNG image
#         buffer = io.BytesIO()
#         figure.savefig(buffer, format='png')
#         buffer.seek(0)
        
#         # Convert PNG buffer to TF image
#         image = tf.image.decode_png(buffer.getvalue(), channels=4)
        
#         # Add batch dimension and log
#         image = tf.expand_dims(image, 0)
#         tf.summary.image(name, image, step=step)
        
#     plt.close(figure)


def train_model():

    print("Loading image paths...")
    image_paths, image_labels = memory_efficient_loading_data(data_path)


    # print("Loading data...")
    # data, labels = loading_data(data_path)
    # X, y = preprocess_data(data, labels)

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

    # Split data - stratify to maintain class distribution
    paths_temp, paths_test, labels_temp, labels_test = train_test_split(
        image_paths,
        image_labels,
        test_size=0.2,
        random_state=42,
        stratify=image_labels
    )

    # X_temp, X_test, y_temp, y_test = train_test_split(
    #     X, 
    #     y, 
    #     test_size=0.2,
    #     random_state=42,
    #     stratify=y  # Ensure balanced split
    # )

    kf = KFold(
        n_splits=5, 
        shuffle=True, 
        random_state=42
    )

    # Tracking metrics
    fold_results = []
    best_model = None
    best_acc = 0

    accs, losses = [], []

    for fold, (train_idx, val_idx) in enumerate(kf.split(paths_temp, labels_temp)):
        print(f"\n=========== Fold {fold + 1}/5 ===========")

        # X_train, X_val = X_temp[train_idx], X_temp[val_idx]
        # y_train, y_val = y_temp[train_idx], y_temp[val_idx]

        # print(f"Train set: {X_train.shape}, {y_train.shape}")
        # print(f"Validation set: {X_val.shape}, {y_val.shape}")

        # # train_ds = create_dataset(X_train, y_train)
        # # val_ds = create_dataset(X_val, y_val)

        # train_ds = create_improved_dataset(X_train, y_train, batch_size=batch_size, is_training=True)
        # val_ds = create_improved_dataset(X_val, y_val, batch_size=batch_size, is_training=False)
        # test_ds = create_improved_dataset(X_test, y_test, batch_size=batch_size, is_training=False)



        train_paths, val_paths = paths_temp[train_idx], paths_temp[val_idx]
        train_labels, val_labels = labels_temp[train_idx], labels_temp[val_idx]

        print(f"Train set: {train_paths.shape}, {train_labels.shape}")
        print(f"Validation set: {val_paths.shape}, {val_labels.shape}")

        # Create datasets
        train_ds = create_path_dataset(train_paths, train_labels, batch_size=batch_size, is_training=True)
        val_ds = create_path_dataset(val_paths, val_labels, batch_size=batch_size, is_training=False)


        # Create model
        print("Creating model...")
        model = create_deit_model()
        model.summary()

        initial_learning_rate = 1e-5
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate,
            decay_steps=1000,
            decay_rate=0.9,
            staircase=True
        )
        
        optimizer = Adam(learning_rate=lr_schedule)

        model.compile(
            optimizer=optimizer, 
            loss=focal_loss(gamma=2.0, alpha=0.75),
            metrics=[
                'accuracy',
                tf.keras.metrics.AUC(name='auc'),
                tf.keras.metrics.Precision(name='precision'), 
                tf.keras.metrics.Recall(name='recall')
            ]
        )

        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        log_dir = f"logs/DeiT_fold{fold+1}_{timestamp}"
        
        callbacks = [
            EarlyStopping(
                monitor='val_auc', 
                patience=10,
                mode='max',
                restore_best_weights=True,
                verbose=1
            ),
            ModelCheckpoint(
                f"DeiT_fold{fold + 1}_{timestamp}.h5", 
                monitor='val_auc',
                mode='max',
                save_best_only=True,
                verbose=1, 
                append_hash=True
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1
            ),
            tf.keras.callbacks.TensorBoard(log_dir=log_dir),
            # Add GPU memory monitoring
            GPUMemoryCallback(log_dir=log_dir, log_freq=1)
        ]


        # logging
        tb_image_writer = log_images_to_tensorboard(log_dir)


        # Train model
        print("Training model...")
        history = model.fit(
            train_ds, 
            validation_data=val_ds, 
            epochs=epochs, 
            callbacks=callbacks,
            class_weight=class_weight
        )

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
    test_ds = create_path_dataset(paths_test, labels_test, batch_size=batch_size, is_training=False)
    
    # Evaluate best model on test set
    print("\n======= Final Evaluation on Test Set =======")
    test_loss, test_acc, test_auc, test_precision, test_recall = best_model.evaluate(test_ds)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test AUC: {test_auc:.4f}")
    print(f"Test Precision: {test_precision:.4f}")
    print(f"Test Recall: {test_recall:.4f}")
    
    # Detailed evaluation with confusion matrix and ROC curve
    final_timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    final_log_dir = f"logs/final_evaluation_{final_timestamp}"
    y_pred, y_pred_prob = evaluate_model(best_model, test_ds, labels_test, final_log_dir)
    
    # Save best model with timestamp to avoid conflicts
    best_model.save(f'best_histopathology_model_{final_timestamp}.h5', overwrite=True)
    print(f"Best model saved as 'best_histopathology_model_{final_timestamp}.h5'")



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



