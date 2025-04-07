### IMPORTS ###
import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import KFold
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.applications import ResNet50

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
    # Normalize and reshape
    data = data.astype('float32') / 255.0
    data = np.expand_dims(data, axis=-1)  # (N, H, W, 1)
    return data, labels


### 2. DATASET GENERATOR ###
def rgb_generator(X, y):
    for i in range(len(X)):
        rgb = tf.image.grayscale_to_rgb(tf.convert_to_tensor(X[i]))
        yield rgb.numpy(), y[i]


def create_dataset(X, y):
    return tf.data.Dataset.from_generator(
        lambda: rgb_generator(X, y),
        output_types=(tf.float32, tf.int32),
        output_shapes=((img_size, img_size, 3), ())
    ).batch(batch_size).prefetch(tf.data.AUTOTUNE)



### 3. MODEL DEFINITIONS ###
def create_resnet_model():
    input_layer = layers.Input(shape=(img_size, img_size, 3))
    base_model = ResNet50(include_top=False, input_tensor=input_layer, weights='imagenet', pooling='avg')
    base_model.trainable = False
    x = base_model.output
    x = layers.Dense(128, activation='relu')(x)
    output = layers.Dense(1, activation='sigmoid')(x)
    return tf.keras.Model(inputs=input_layer, outputs=output)



### 4. TRAINING FUNCTION ###
def train_model():
    data, labels = loading_data(data_path)
    X, y = preprocess_data(data, labels)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    accs, losses = [], []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        print(f"\n Fold {fold + 1}/5")

        train_ds = create_dataset(X[train_idx], y[train_idx])
        val_ds = create_dataset(X[val_idx], y[val_idx])

        model = create_resnet_model()
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        callbacks = [
            EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True),
            ModelCheckpoint(f"ResNet_fold{fold + 1}.h5", save_best_only=True)
        ]

        model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks)

        loss, acc = model.evaluate(val_ds)
        accs.append(acc)
        losses.append(loss)

        print(f"Fold {fold + 1} - Accuracy: {acc:.4f} | Loss: {loss:.4f}")

    print(f"\n Final Results for ResNet: ")
    print(f"Mean Accuracy: {np.mean(accs):.4f}")
    print(f"Std Dev Accuracy: {np.std(accs):.4f}")
    print(f"Mean Loss: {np.mean(losses):.4f}")
    print(f"Std Dev Loss: {np.std(losses):.4f}")


# Train ResNet50
train_model()