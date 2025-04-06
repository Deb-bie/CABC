import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import KFold
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Constants
data_path = "../../../data/BreaKHis_Total_dataset"
labels = ['benign', 'malignant']
img_size = 224
batch_size = 32
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
    # Normalize and reshape
    data = data.astype('float32') / 255.0
    data = np.expand_dims(data, axis=-1)  # (N, H, W, 1)
    return data, labels

data, data_labels = loading_data(data_path)
X_data, y_data = preprocess_data(data, data_labels)

num_folds = 5
kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
all_fold_metrics = []

for fold, (train_index, val_index) in enumerate(kf.split(X_data, y_data)):
    print(f"Fold {fold + 1}/{num_folds}")

    # Use TensorFlow's graph ops for efficient grayscale to RGB conversion
    X_train, y_train = X_data[train_index], y_data[train_index]
    X_val, y_val = X_data[val_index], y_data[val_index]

    def rgb_generator(X, y):
        for i in range(len(X)):
            rgb_img = tf.image.grayscale_to_rgb(tf.convert_to_tensor(X[i]))
            yield rgb_img.numpy(), y[i]

    train_dataset = tf.data.Dataset.from_generator(
        lambda: rgb_generator(X_train, y_train),
        output_types=(tf.float32, tf.int32),
        output_shapes=((img_size, img_size, 3), ())
    ).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    val_dataset = tf.data.Dataset.from_generator(
        lambda: rgb_generator(X_val, y_val),
        output_types=(tf.float32, tf.int32),
        output_shapes=((img_size, img_size, 3), ())
    ).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    base_model = ResNet50(
        include_top=False,
        input_shape=(img_size, img_size, 3),
        weights='imagenet',
        pooling='avg'
    )
    base_model.trainable = False

    model = models.Sequential([
        base_model,
        layers.Dense(512, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    checkpoint = ModelCheckpoint(
        "resnet_fold_{}.h5".format(fold + 1),
        monitor='val_accuracy',
        save_best_only=True,
        mode='max'
    )

    early = EarlyStopping(
        monitor='val_accuracy',
        patience=10,
        restore_best_weights=True
    )

    model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        callbacks=[checkpoint, early],
        verbose=1
    )

    loss, accuracy = model.evaluate(val_dataset, verbose=0)
    print(f"Fold {fold + 1} Validation Accuracy: {accuracy:.4f}")
    all_fold_metrics.append(accuracy)

print("\nCross-Validation Results:")
print(f"Mean Accuracy: {np.mean(all_fold_metrics):.4f}")
print(f"Std Dev: {np.std(all_fold_metrics):.4f}")