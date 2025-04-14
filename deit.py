import os
import cv2 # type: ignore
import io
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_accuracy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.applications import EfficientNetB7
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
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
    # Enhanced preprocessing with standardization
    X_data = np.array(data).astype('float32')
    X_data = (X_data - X_data.mean()) / (X_data.std() + 1e-7)  # Adding epsilon to avoid division by zero
    X_data = X_data.reshape(-1, img_size, img_size, 1)
    print(f"Data shape after preprocessing: {X_data.shape}")
    
    y_data = np.array(labels)
    
    return X_data, y_data


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


def create_deit_model():
    deit = tf.keras.models.load_model('deit_model.h5')

    # Freeze the first few layers of the model
    for layer in deit.layers[:-10]:
        layer.trainable = False

    x = deit.layers[-2].output
    outputs = Dense(1, activation='sigmoid')(x)
    
    return Model(deit.input, outputs)


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



def train_model():

    # Load and preprocess data
    print("Loading data...")
    data, labels_data = loading_data(data_path)
    X, y = preprocess_data(data, labels_data)

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
        model = create_deit_model()
        
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
            loss=focal_loss(gamma=2.0, alpha=0.75),  # Focus more on benign class
            # loss='binary_crossentropy', 
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
    final_log_dir = "logs/final_evaluation_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    y_pred, y_pred_prob = evaluate_model(best_model, test_ds, y_test, final_log_dir)
    
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






# Compile the model with an appropriate optimizer and loss function
# model.compile(optimizer=Adam(lr=0.001),
#               loss='binary_crossentropy',
#               metrics=[categorical_accuracy])


# Prepare the data for training and validation
# train_data_dir = 'train_data'
# validation_data_dir = 'validation_data'



# train_datagen = ImageDataGenerator(
#     rescale=1. / 255,
#     shear_range=0.2,
#     zoom_range=0.2,
#     horizontal_flip=True)


# validation_datagen = ImageDataGenerator(rescale=1. / 255)


# train_generator = train_datagen.flow_from_directory(
#     train_data_dir,
#     target_size=(224, 224),
#     batch_size=32,
#     class_mode='binary')


# validation_generator = validation_datagen.flow_from_directory(
#     validation_data_dir,
#     target_size=(224, 224),
#     batch_size=32,
#     class_mode='categorical')


# Define callbacks for saving the best model and early stopping
# filepath = "best_model.h5"

# checkpoint = ModelCheckpoint(filepath, monitor='val_categorical_accuracy', verbose=1,
#                              save_best_only=True, mode='max')

# early_stop = EarlyStopping(monitor='val_categorical_accuracy', patience=5, mode='max')


# Train the model on the data and validate


# history = model.fit(
#     train_generator,
#     steps_per_epoch=train_generator.samples // train_generator.batch_size,
#     epochs=20,
#     validation_data=validation_generator,
#     validation_steps=validation_generator.samples // validation_generator.batch_size,
#     callbacks=[checkpoint, early_stop])


# # Evaluate the model on a test set
# test_data_dir = 'test_data'
# test_datagen = ImageDataGenerator(rescale=1. / 255)
# test_generator = test_datagen.flow_from_directory(
#     test_data_dir,
#     target_size=(224, 224),
#     batch_size=32,
#     class_mode='categorical')


# model.load_weights('best_model.h5')


# test_loss, test_acc = model.evaluate_generator(test_generator, steps=test_generator.samples // test_generator.batch_size)

# print('Test accuracy:', test_acc)
# print('Test loss:', test_loss)


# # Plot the training and validation loss and accuracy
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('Model loss')
# plt.ylabel('Loss')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Validation'], loc='upper left')
# plt.show()
# plt.plot(history.history['categorical_accuracy'])
# plt.plot(history.history['val_categorical_accuracy'])
# plt.title('Model accuracy')
# plt.ylabel('Accuracy')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Validation'], loc='upper left')
# plt.show()


