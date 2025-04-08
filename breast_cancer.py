### IMPORTS ###
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import datetime
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers, Model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.applications import ResNet50
import tensorflow_addons as tfa

from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import confusion_matrix, classification_report


# Constants
data_path = "../../../data/BreaKHis_Total_dataset"
labels = ['benign', 'malignant']
img_size = 224
batch_size = 16
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
    X_data = np.array(data) / 255
    X_data = X_data.reshape(-1, img_size, img_size, 1)
    print(X_data.shape)
    y_data = np.array(labels)

    return X_data, y_data


def create_dataset(X, y, augment=False):
    def _generator():
        for i in range(len(X)):
            rgb = tf.image.grayscale_to_rgb(tf.convert_to_tensor(X[i]))
            label = y[i]
            yield rgb, label

    def _augment(image, label):
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_brightness(image, max_delta=0.2)
        image = tf.image.random_contrast(image, 0.8, 1.2)

        # Rotate by exactly 0.1 radians (≈5.73 degrees)
        image = tfa.image.rotate(image, angles=0.1, fill_mode='nearest')
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
    base_model = ResNet50(
        include_top=False, 
        weights='imagenet', 
        input_shape=(img_size, img_size, 3), 
        pooling='avg'
    )
    for layer in base_model.layers[:-30]:
        layer.trainable = False
    inputs = base_model.input
    x = base_model.output
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(1, activation='sigmoid')(x)
    return Model(inputs, outputs)


def plot_history(history, fold):
    plt.plot(history.history['accuracy'], label='train acc')
    plt.plot(history.history['val_accuracy'], label='val acc')
    plt.title(f'Resnet Accuracy Fold {fold + 1}')
    plt.legend()
    plt.savefig(f'Resnet_accuracy_fold{fold+1}.png')
    plt.clf()


def train_model():
    data, labels = loading_data(data_path)
    X, y = preprocess_data(data, labels)

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, 
        y, 
        test_size=0.1, 
        random_state=42
    )

    kf = KFold(
        n_splits=5, 
        shuffle=True, 
        random_state=42
    )
    accs, losses = [], []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_temp, y_temp)):
        print(f"Train index: {train_idx}")
        print(f"\n Fold {fold + 1}/5")

        X_train, X_val = X_temp[train_idx], X_temp[val_idx]
        y_train, y_val = y_temp[train_idx], y_temp[val_idx]

        train_ds = create_dataset(X_train, y_train, augment=True)
        val_ds = create_dataset(X_val, y_val, augment=False)
        test_ds = create_dataset(X_test, y_test, augment=False)

        model = create_resnet_model()
        
        model.compile(
            optimizer='adam', 
            loss='binary_crossentropy', 
            metrics=['accuracy']
        )
        
        log_dir = f"logs/Resnet_fold{fold+1}_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

        callbacks = [
            EarlyStopping(
                monitor='val_accuracy', 
                patience=5, 
                restore_best_weights=True
            ),
            
            ModelCheckpoint(
                f"ResNet_fold{fold + 1}.h5", 
                save_best_only=True
            ),
            tf.keras.callbacks.TensorBoard(log_dir=log_dir)
        ]

        history = model.fit(
            train_ds, 
            validation_data=val_ds, 
            epochs=epochs, 
            callbacks=callbacks
        )

        plot_history(history, fold)

        loss, acc = model.evaluate(test_ds)
        accs.append(acc)
        losses.append(loss)

    print(f"\n Final Results for ResNet: ")
    print(f"Mean Accuracy: {np.mean(accs):.4f}")
    print(f"Std Dev Accuracy: {np.std(accs):.4f}")
    print(f"Mean Loss: {np.mean(losses):.4f}")
    print(f"Std Dev Loss: {np.std(losses):.4f}")


    # Evaluate final model on test set
    final_loss, final_acc = model.evaluate(test_ds)
    print(f"\nTest Accuracy: {final_acc:.4f}")
    print(f"Test Loss: {final_loss:.4f}")


# Train ResNet50
train_model()