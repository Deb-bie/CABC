import os
import numpy as np
import tensorflow as tf
import datetime
import uuid
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, Callback
import tensorflow_hub as hub


# Constants
data_path = "../../../data/BreaKHis_Total_dataset"
labels = ['benign', 'malignant']
img_size = 224  
batch_size = 48  
epochs = 30    
mixed_precision = True  


# Configure mixed precision globally but handle hub models specially
if mixed_precision:
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    tf.keras.mixed_precision.set_global_policy(policy)
    print('Mixed precision enabled (with special handling for hub models)')


# Memory growth for GPU
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
        print(f"Found {len(physical_devices)} GPU(s). Memory growth enabled.")
    except Exception as e:
        print(f"Error setting memory growth: {e}")


class GPUMemoryCallback(Callback):
    """Callback to monitor GPU memory usage during training"""
    def on_epoch_end(self, epoch, logs=None):
        physical_devices = tf.config.list_physical_devices('GPU')
        if not physical_devices:
            print("No GPU available. Skipping memory monitoring.")
            return
        
        try:
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
    """Callback to clean up memory after each epoch"""
    def on_epoch_end(self, epoch, logs=None):
        import gc
        gc.collect()
        try:
            # Don't clear the session during training as it can cause issues
            # Just focus on garbage collection
            pass
        except:
            pass
        print("Memory cleanup performed")


def memory_efficient_loading_data(data_dir):
    """Load image paths instead of actual images to save memory"""
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
    """Create a dataset that loads and processes images on-the-fly"""
    # Combine paths and labels
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    
    # Function to load and process a single image
    def load_and_preprocess(path, label):
        # Read the image file
        img = tf.io.read_file(path)
        # Decode the image
        img = tf.io.decode_image(img, channels=1, expand_animations=False)
        # Resize the image
        img = tf.image.resize(img, [img_size, img_size])
        # Normalize pixel values
        img = tf.cast(img, tf.float32) / 255.0  # Keep as float32 for hub models
        # Convert to RGB (DeiT expects 3 channels)
        img = tf.image.grayscale_to_rgb(img)
        
        # Minimal data augmentation if training to reduce computation
        if is_training:
            img = tf.image.random_flip_left_right(img)
        
        return img, label
    
    # Apply the function to each element
    dataset = dataset.map(
        load_and_preprocess, 
        num_parallel_calls=tf.data.AUTOTUNE
    )
    
    # Optimize performance
    if is_training:
        dataset = dataset.shuffle(buffer_size=500)  # Reduced buffer size
    
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(1)  # Reduced prefetch to save memory
    
    return dataset


def check_class_balance(y):
    unique, counts = np.unique(y, return_counts=True)
    print(f"Class distribution: {dict(zip([labels[i] for i in unique], counts))}")
    return counts


def create_deit_model():
    """
    Create DeiT model with proper type handling
    """
    print("Loading DeiT model from TensorFlow Hub...")
    
    # Use the smaller DeiT-Tiny variant to save memory
    deit_url = "https://tfhub.dev/sayakpaul/deit_tiny_patch16_224/1"
    
    # Create model with explicit float32 input for the hub model
    # Hub models typically expect float32 even when mixed precision is enabled
    inputs = layers.Input(shape=(img_size, img_size, 3), dtype=tf.float32, name='model_input')
    
    # Apply simple preprocessing to match DeiT requirements
    x = tf.keras.applications.imagenet_utils.preprocess_input(inputs, mode='tf')
    
    # Load and apply DeiT model
    deit_layer = hub.KerasLayer(deit_url, trainable=True)
    x = deit_layer(x)
    
    # Add classification head - after this point mixed precision can be used
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    model = models.Model(inputs, outputs)
    
    # Print model summary
    model.summary()
    
    return model


def focal_loss(gamma=2.0, alpha=0.75):
    """
    Focal Loss for addressing class imbalance.
    alpha: weighs the importance of positive class (set higher for the minority class)
    gamma: focuses more on hard examples
    """
    def focal_loss_with_logits(logits, targets, alpha, gamma, y_pred):
        # Cast both targets and predictions to the same dtype
        targets = tf.cast(targets, dtype=tf.float32)
        y_pred = tf.cast(y_pred, dtype=tf.float32)
        
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


def create_custom_deit_model():
    """
    Alternative approach: Create a custom DeiT-inspired model instead of loading from Hub
    This avoids Hub compatibility issues with mixed precision
    """
    print("Creating custom ViT/DeiT-like model (not using Hub)...")
    
    # Custom Vision Transformer implementation
    def mlp(x, hidden_units, dropout_rate):
        for units in hidden_units:
            x = layers.Dense(units, activation=tf.nn.gelu)(x)
            x = layers.Dropout(dropout_rate)(x)
        return x
    
    def create_vit_classifier():
        # Parameters
        input_shape = (img_size, img_size, 3)
        patch_size = 16  # Size of the patches to be extracted from the input images
        num_patches = (img_size // patch_size) ** 2
        projection_dim = 192  # Smaller than typical DeiT-Base for memory efficiency
        num_heads = 3  # Multi-head attention
        transformer_units = [projection_dim * 2, projection_dim]  # MLP units
        transformer_layers = 4  # Number of transformer blocks (reduced)
        
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
        
        # Add MLP head
        features = mlp(representation, hidden_units=[projection_dim, projection_dim // 2], dropout_rate=0.3)
        
        # Classification layer
        outputs = layers.Dense(1, activation="sigmoid")(features)
        
        # Create the model
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        return model
    
    return create_vit_classifier()


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


def evaluate_model(model, test_ds, y_test, epoch=0):

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
    train_ds = create_path_dataset(train_paths, train_labels, batch_size=batch_size, is_training=True)
    val_ds = create_path_dataset(val_paths, val_labels, batch_size=batch_size, is_training=False)
    test_ds = create_path_dataset(test_paths, test_labels, batch_size=batch_size, is_training=False)

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
            loss=focal_loss(gamma=2.0, alpha=0.75),
            metrics=[
                'accuracy',
                tf.keras.metrics.AUC(name='auc'),
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall')
            ]
        )

        # Create unique model filename
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        model_filename = f"DeiT_{timestamp}_{unique_id}.keras"
        
        # Define callbacks with memory optimization
        callbacks = [
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
            GPUMemoryCallback(),
            MemoryCleanupCallback()  # Clean up memory after each epoch
        ]

        # Calculate steps per epoch to limit iterations and save memory
        # This allows you to see progress faster and potentially stop if there are memory issues
        steps_per_epoch = min(len(train_paths) // batch_size, 50)  # Cap at 50 steps
        validation_steps = min(len(val_paths) // batch_size, 20)   # Cap at 20 steps
        
        print(f"Training with {steps_per_epoch} steps per epoch and {validation_steps} validation steps")
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

        y_pred, y_pred_prob = evaluate_model(model, test_ds, test_labels)
        
        # Save final model
        final_model_filename = f'final_DeiT_model_{timestamp}_{unique_id}.keras'
        model.save(final_model_filename)
        print(f"Model successfully saved as '{final_model_filename}'")
        
        
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


if __name__ == "__main__":
    # Clear any existing TF sessions
    tf.keras.backend.clear_session()
    
    # Train with memory optimizations
    train_model_with_memory_optimizations()


