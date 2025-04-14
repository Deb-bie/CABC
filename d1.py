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
from transformers import TFAutoModel
import tensorflow_addons as tfa

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress warnings
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# Constants
data_path = "../../../data/BreaKHis_Total_dataset"
labels = ['benign', 'malignant']
img_size = 224
batch_size = 32  # Reduced for Transformer memory requirements
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
    X_data = X_data / 255.0
    X_data = (X_data - 0.5) / 0.5  # Normalize to [-1, 1]
    X_data = np.repeat(X_data[..., np.newaxis], 3, axis=-1)  # Convert to 3-channel
    return X_data, np.array(labels)

def check_class_balance(y):
    counts = np.bincount(y)
    class_weight = {0: 1/counts[0], 1: 1/counts[1]}
    print(f"Class weights: {class_weight}")
    return class_weight

def focal_loss(gamma=2.0, alpha=0.75):
    def focal_loss_with_logits(logits, targets, alpha, gamma, y_pred):
        targets = tf.cast(targets, dtype=tf.float32)
        BCE = tf.keras.losses.binary_crossentropy(targets, y_pred)
        alpha_t = targets * alpha + (1 - targets) * (1 - alpha)
        p_t = targets * y_pred + (1 - targets) * (1 - y_pred)
        FL = alpha_t * tf.pow(1 - p_t, gamma) * BCE
        return tf.reduce_mean(FL)
    
    def loss(y_true, y_pred):
        return focal_loss_with_logits(y_pred, y_true, alpha, gamma, y_pred)
    
    return loss

def create_dataset(X, y, augment=False):
    def _generator():
        for i in range(len(X)):
            rgb = tf.image.grayscale_to_rgb(tf.convert_to_tensor(X[i]))
            label = y[i]
            yield rgb, label

    def _augment(image, label):
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_flip_up_down(image)
        image = tf.image.random_brightness(image, max_delta=0.3)
        image = tf.image.random_contrast(image, 0.7, 1.3)
        image = tf.image.random_saturation(image, 0.8, 1.2)
        
        angle = tf.random.uniform([], -0.2, 0.2)
        image = tfa.image.rotate(image, angles=angle, fill_mode='reflect')
        
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

def create_deit_model():
    # Load model with proper architecture
    from transformers import TFViTForImageClassification
    
    base_model = TFViTForImageClassification.from_pretrained(
        "facebook/deit-base-patch16-224",
        num_labels=1,  # For binary classification
        ignore_mismatched_sizes=True
    )
    
    # Freeze encoder layers
    for layer in base_model.layers[:-4]:
        layer.trainable = False
    
    # Custom classification head
    x = base_model.layers[-2].output
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    return tf.keras.Model(inputs=base_model.input, outputs=outputs)

def plot_training_history(history, fold):
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title(f'Model Accuracy - Fold {fold + 1}')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
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
    tb_image_writer = log_images_to_tensorboard(log_dir)

    y_pred_prob = model.predict(test_ds)
    y_pred_prob_flat = np.array([prob[0] for prob in y_pred_prob])
    
    y_pred = (y_pred_prob_flat > 0.5).astype(int)
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=labels))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    fig = plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels, cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    log_image(tb_image_writer, 'confusion_matrix', fig, step=epoch)
    plt.close()
    
    # ROC Curve
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
    log_image(tb_image_writer, 'roc_curve', fig, step=epoch)
    plt.close()
    
    return y_pred, y_pred_prob_flat

def log_images_to_tensorboard(log_dir):
    return tf.summary.create_file_writer(log_dir + '/images')

def log_image(file_writer, name, figure, step=0):
    with file_writer.as_default():
        buffer = io.BytesIO()
        figure.savefig(buffer, format='png')
        buffer.seek(0)
        image = tf.image.decode_png(buffer.getvalue(), channels=4)
        image = tf.expand_dims(image, 0)
        tf.summary.image(name, image, step=step)

def train_model():
    print("Loading data...")
    data, labels_data = loading_data(data_path)
    X, y = preprocess_data(data, labels_data)
    
    class_weights = check_class_balance(y)
    
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    fold_results = []
    best_model = None
    best_acc = 0
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_temp, y_temp)):
        print(f"\n=========== Fold {fold + 1}/5 ===========")

        X_train, X_val = X_temp[train_idx], X_temp[val_idx]
        y_train, y_val = y_temp[train_idx], y_temp[val_idx]
        
        train_ds = create_dataset(X_train, y_train, augment=True)
        val_ds = create_dataset(X_val, y_val, augment=False)
        
        print("Creating DeiT model...")
        model = create_deit_model()
        
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            1e-5,  # Lower initial LR for Transformer
            decay_steps=2000,
            decay_rate=0.95,
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
        
        log_dir = f"logs/DeiT_fold{fold+1}_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        
        callbacks = [
            EarlyStopping(
                monitor='val_auc',
                patience=8,  # Increased patience for Transformer
                mode='max',
                restore_best_weights=True,
                verbose=1
            ),
            ModelCheckpoint(
                f"DeiT_fold{fold + 1}.h5",
                monitor='val_auc',
                mode='max',
                save_best_only=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-6,
                verbose=1
            ),
            tf.keras.callbacks.TensorBoard(log_dir=log_dir)
        ]

        print("Training model...")
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            callbacks=callbacks,
            class_weight=class_weights
        )
        
        plot_training_history(history, fold)
        
        val_loss, val_acc, val_auc, val_precision, val_recall = model.evaluate(val_ds)
        print(f"Fold {fold + 1} - Validation Metrics:")
        print(f"  Loss: {val_loss:.4f}")
        print(f"  Accuracy: {val_acc:.4f}")
        print(f"  AUC: {val_auc:.4f}")
        
        fold_results.append({
            'fold': fold + 1,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'val_auc': val_auc
        })
        
        if val_acc > best_acc:
            best_acc = val_acc
            best_model = model

    print("\n======= Cross-Validation Results =======")
    metrics = ['val_loss', 'val_acc', 'val_auc']
    for metric in metrics:
        values = [result[metric] for result in fold_results]
        print(f"Mean {metric}: {np.mean(values):.4f} (±{np.std(values):.4f})")

    test_ds = create_dataset(X_test, y_test, augment=False)
    
    print("\n======= Final Evaluation =======")
    test_loss, test_acc, test_auc, _, _ = best_model.evaluate(test_ds)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test AUC: {test_auc:.4f}")
    
    final_log_dir = "logs/final_evaluation_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    evaluate_model(best_model, test_ds, y_test, final_log_dir)
    
    best_model.save('best_deit_model.h5')
    print("Best model saved as 'best_deit_model.h5'")

if __name__ == "__main__":
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        try:
            for device in physical_devices:
                tf.config.experimental.set_memory_growth(device, True)
            print(f"Found {len(physical_devices)} GPU(s). Memory growth enabled.")
        except Exception as e:
            print(f"Error setting memory growth: {e}")
    
    np.random.seed(42)
    tf.random.set_seed(42)
    
    train_model()