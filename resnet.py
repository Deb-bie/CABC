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
from tensorflow.keras.applications import ResNet50 # type: ignore
import tensorflow_addons as tfa # type: ignore


# Constants
data_path = "../../../data/BreaKHis_Total_dataset"
labels = ['benign', 'malignant']
img_size = 224
batch_size = 48
epochs = 30


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
    X_data = np.array(data).astype('float32')
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

def rgb_generator(X, y):
    for i in range(len(X)):
        rgb = tf.image.grayscale_to_rgb(tf.convert_to_tensor(X[i]))
        yield rgb.numpy(), y[i]

def create_dataset(X, y):
    return tf.data.Dataset.from_generator(
        lambda: rgb_generator(X, y),
        output_types=(tf.float32, tf.int32),
        output_shapes=((img_size, img_size, 3), ())
    ).shuffle(2000).batch(batch_size).prefetch(tf.data.AUTOTUNE)


def create_resnet_model():
    # input_layer = layers.Input(shape=(img_size, img_size, 3))
    # base_model = ResNet50(include_top=False, input_tensor=input_layer, weights='imagenet', pooling='avg')
    # base_model.trainable = False
    # x = base_model.output
    # x = layers.Dense(128, activation='relu')(x)
    # output = layers.Dense(1, activation='sigmoid')(x)
    # return tf.keras.Model(inputs=input_layer, outputs=output)


    # Use ResNet50 with improved architecture
    base_model = ResNet50(
        include_top=False, 
        weights='imagenet', 
        input_shape=(img_size, img_size, 3), 
        pooling='avg'
    )
    
    # Freeze fewer layers to allow more fine-tuning
    for layer in base_model.layers[:-20]:  
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


def plot_training_history(history, fold, log_dir):
    # Create figure for accuracy plot
    fig_acc = plt.figure(figsize=(10, 6))
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title(f'Model Accuracy - Fold {fold + 1}')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.tight_layout()
    
    # Create figure for loss plot
    fig_loss = plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title(f'Model Loss - Fold {fold + 1}')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.tight_layout()
    
    # Log to TensorBoard
    with tf.summary.create_file_writer(log_dir + '/training_plots').as_default():
        # Log accuracy plot
        buf = io.BytesIO()
        fig_acc.savefig(buf, format='png')
        buf.seek(0)
        img = tf.image.decode_png(buf.getvalue(), channels=4)
        img = tf.expand_dims(img, 0)
        tf.summary.image("Accuracy Plot", img, step=0)
        
        # Log loss plot
        buf = io.BytesIO()
        fig_loss.savefig(buf, format='png')
        buf.seek(0)
        img = tf.image.decode_png(buf.getvalue(), channels=4)
        img = tf.expand_dims(img, 0)
        tf.summary.image("Loss Plot", img, step=0)
    
    # Save locally too
    fig_acc.savefig(f'history_accuracy_fold{fold+1}.png')
    fig_loss.savefig(f'history_loss_fold{fold+1}.png')
    
    plt.close(fig_acc)
    plt.close(fig_loss)


def log_confusion_matrix_to_tensorboard(y_true, y_pred, log_dir, epoch=0, class_names=None):
    """Log a confusion matrix to TensorBoard"""
    if class_names is None:
        class_names = labels
        
    cm = confusion_matrix(y_true, y_pred)
    
    # Create figure
    fig = plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    
    # Log to TensorBoard
    with tf.summary.create_file_writer(log_dir + '/confusion_matrix').as_default():
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        img = tf.image.decode_png(buf.getvalue(), channels=4)
        img = tf.expand_dims(img, 0)
        tf.summary.image("Confusion Matrix", img, step=epoch)
    
    plt.close(fig)
    return fig


def log_roc_curve_to_tensorboard(y_true, y_pred_prob, log_dir, epoch=0):
    """Log a ROC curve to TensorBoard"""
    fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    
    # Create figure
    fig = plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.tight_layout()
    
    # Log to TensorBoard
    with tf.summary.create_file_writer(log_dir + '/roc_curve').as_default():
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        img = tf.image.decode_png(buf.getvalue(), channels=4)
        img = tf.expand_dims(img, 0)
        tf.summary.image("ROC Curve", img, step=epoch)
    
    plt.close(fig)
    return fig


def log_classification_report_to_tensorboard(y_true, y_pred, log_dir, epoch=0):
    """Log classification report as text to TensorBoard"""
    report = classification_report(y_true, y_pred, target_names=labels)
    
    with tf.summary.create_file_writer(log_dir + '/metrics').as_default():
        tf.summary.text("Classification Report", report, step=epoch)


def log_metric_to_tensorboard(log_dir, metric_name, value, epoch=0):
    """Log a scalar metric to TensorBoard"""
    with tf.summary.create_file_writer(log_dir + '/metrics').as_default():
        tf.summary.scalar(metric_name, value, step=epoch)


def log_sample_predictions_to_tensorboard(X_test, y_test, y_pred, y_pred_prob, log_dir, num_samples=10, epoch=0):
    """Log sample predictions with images to TensorBoard"""
    # Get random indices (but ensure we have some of each class)
    indices_class0 = np.where(y_test == 0)[0]
    indices_class1 = np.where(y_test == 1)[0]
    
    # Choose samples from each class
    n_class0 = min(num_samples // 2, len(indices_class0))
    n_class1 = min(num_samples // 2, len(indices_class1))
    
    sample_indices_class0 = np.random.choice(indices_class0, n_class0, replace=False)
    sample_indices_class1 = np.random.choice(indices_class1, n_class1, replace=False)
    
    sample_indices = np.concatenate([sample_indices_class0, sample_indices_class1])
    
    # Create figure with subplots
    n_cols = min(5, num_samples)
    n_rows = (num_samples + n_cols - 1) // n_cols
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(15, 3 * n_rows))
    axs = axs.flatten() if isinstance(axs, np.ndarray) else [axs]
    
    for i, idx in enumerate(sample_indices):
        # Get the image
        img = X_test[idx].reshape(img_size, img_size)
        
        # Plot
        axs[i].imshow(img, cmap='gray')
        true_label = labels[y_test[idx]]
        pred_label = labels[y_pred[idx]]
        prob = y_pred_prob[idx]
        color = 'green' if y_test[idx] == y_pred[idx] else 'red'
        axs[i].set_title(f"True: {true_label}\nPred: {pred_label}\nProb: {prob:.2f}", color=color)
        axs[i].axis('off')
    
    # Hide unused subplots
    for i in range(len(sample_indices), len(axs)):
        axs[i].axis('off')
    
    plt.tight_layout()
    
    # Log to TensorBoard
    with tf.summary.create_file_writer(log_dir + '/sample_predictions').as_default():
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        img = tf.image.decode_png(buf.getvalue(), channels=4)
        img = tf.expand_dims(img, 0)
        tf.summary.image("Sample Predictions", img, step=epoch)
    
    plt.close(fig)


def evaluate_model(model, test_ds, X_test, y_test, log_dir, epoch=0):

    """Comprehensive model evaluation with TensorBoard logging"""
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


    # Log classification report to TensorBoard
    log_classification_report_to_tensorboard(y_test, y_pred, log_dir, epoch)
    
    # Create and log confusion matrix
    cm_fig = log_confusion_matrix_to_tensorboard(y_test, y_pred, log_dir, epoch)
    cm_fig.savefig('final_confusion_matrix.png')
    
    # Create and log ROC curve
    roc_fig = log_roc_curve_to_tensorboard(y_test, y_pred_prob_flat, log_dir, epoch)
    roc_fig.savefig('roc_curve.png')
    
    # Calculate and log additional metrics
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    log_metric_to_tensorboard(log_dir, "Final Accuracy", accuracy, epoch)
    log_metric_to_tensorboard(log_dir, "Final Precision", precision, epoch)
    log_metric_to_tensorboard(log_dir, "Final Recall", recall, epoch)
    log_metric_to_tensorboard(log_dir, "Final F1 Score", f1, epoch)
    
    # Log sample predictions
    log_sample_predictions_to_tensorboard(X_test, y_test, y_pred, y_pred_prob_flat, log_dir, epoch=epoch)
    
    return y_pred, y_pred_prob_flat
    

class EvaluationCallback(tf.keras.callbacks.Callback):
    """Custom callback to evaluate model and log to TensorBoard during training"""
    def __init__(self, val_data, X_val, y_val, log_dir, evaluation_frequency=5):
        super().__init__()
        self.val_data = val_data
        self.X_val = X_val
        self.y_val = y_val
        self.log_dir = log_dir
        self.evaluation_frequency = evaluation_frequency
    
    def on_epoch_end(self, epoch, logs=None):
        # Only evaluate every N epochs to save time
        if (epoch + 1) % self.evaluation_frequency == 0 or epoch == 0:
            print(f"\nEvaluating and logging metrics to TensorBoard at epoch {epoch+1}...")
            # Evaluate and log
            evaluate_model(
                self.model, 
                self.val_data, 
                self.X_val, 
                self.y_val, 
                self.log_dir, 
                epoch=epoch
            )


### 4. TRAINING FUNCTION ###
def train_model():
    print("Loading data...")
    data, labels = loading_data(data_path)
    X, y = preprocess_data(data, labels)

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


    X_temp, X_test, y_temp, y_test = train_test_split(
        X, 
        y, 
        test_size=0.2,
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

    # Tracking metrics
    fold_results = []
    best_model = None
    best_acc = 0

    accs, losses = [], []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_temp, y_temp)):
        print(f"\n=========== Fold {fold + 1}/5 ===========")

        X_train, X_val = X_temp[train_idx], X_temp[val_idx]
        y_train, y_val = y_temp[train_idx], y_temp[val_idx]

        print(f"Train set: {X_train.shape}, {y_train.shape}")
        print(f"Validation set: {X_val.shape}, {y_val.shape}")

        train_ds = create_dataset(X_train, y_train)
        val_ds = create_dataset(X_val, y_val)

        # Create model
        print("Creating model...")
        model = create_resnet_model()
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
            tf.keras.callbacks.TensorBoard(log_dir=log_dir),
            # Add our custom evaluation callback
            EvaluationCallback(
                val_ds, 
                X_val, 
                y_val, 
                log_dir, 
                evaluation_frequency=5  # Evaluate every 5 epochs
            )
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

        # Plot and log training history
        plot_training_history(history, fold, log_dir)

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
    test_ds = create_dataset(X_test, y_test)
    
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
    y_pred, y_pred_prob = evaluate_model(best_model, test_ds, X_test, y_test, final_log_dir)


    # Log final cross-validation summary to TensorBoard
    with tf.summary.create_file_writer(final_log_dir + '/cross_validation').as_default():
        for metric in metrics:
            values = [result[metric] for result in fold_results]
            tf.summary.scalar(f"mean_{metric}", np.mean(values), step=0)
            tf.summary.scalar(f"std_{metric}", np.std(values), step=0)
    
    # Save best model
    best_model.save('best_histopathology_model.h5')
    print("Best model saved as 'best_histopathology_model.h5'")
    print(f"TensorBoard logs saved to {final_log_dir}")
    print("Run 'tensorboard --logdir logs/' to view the results")

    


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





