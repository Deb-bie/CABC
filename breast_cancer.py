### IMPORTS ###
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import datetime
from tensorflow.keras.optimizers import Adam
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

    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    for label in labels:
        path = os.path.join(data_dir, label)
        class_num = labels.index(label)
        files = os.listdir(path)
        total_files = len(files)

        print(f"Loading {label} images ({total_files} files)")

        # load only 2000 images

        for i, img in enumerate(files):
            if i % 100 == 0:
                print(f" Progress: {i}/{total_files}")
            
            if i <= 2000:

                img_path = os.path.join(path, img)
                img_arr = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

                if img_arr is not None:
                    resized_arr = cv2.resize(img_arr, (img_size, img_size))
                    data.append(resized_arr)
                    labels_list.append(class_num)
                else:
                    print(f"Warning: Unable to read image {img_path}")

        # for i, img in enumerate(files):
        #     if i % 100 == 0:
        #         print(f" Progress: {i}/{total_files}")

        #     img_path = os.path.join(path, img)
        #     img_arr = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        #     if img_arr is not None:
        #         resized_arr = cv2.resize(img_arr, (img_size, img_size))
        #         data.append(resized_arr)
        #         labels_list.append(class_num)
        #     else:
        #         print(f"Warning: Unable to read image {img_path}")

    return np.array(data), np.array(labels_list)

def preprocess_data(data, labels):
    # Normalize and reshape
    # data = data.astype('float32') / 255.0
    # data = np.expand_dims(data, axis=-1)  # (N, H, W, 1)
    # return data, labels

    X_data = np.array(data) / 255
    X_data = X_data.reshape(-1, img_size, img_size, 1)
    print(X_data.shape)

    y_data = np.array(labels)

    return X_data, y_data


### 2. DATASET GENERATOR ###
def rgb_generator(X, y):
    for i in range(len(X)):
        rgb = tf.image.grayscale_to_rgb(tf.convert_to_tensor(X[i]))
        yield rgb.numpy(), y[i]


# def create_dataset(X, y):
#     return tf.data.Dataset.from_generator(
#         lambda: rgb_generator(X, y),
#         output_types=(tf.float32, tf.int32),
#         output_shapes=((img_size, img_size, 3), ())
#     ).from_tensor_slices((X, y))
# .shuffle(1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)


def create_dataset(X, y):
    return tf.data.Dataset.from_generator(
        lambda: rgb_generator(X, y),
        output_signature=(
            tf.TensorSpec(shape=(img_size, img_size, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.int32)
        )
    ).shuffle(1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)



### 3. MODEL DEFINITIONS ###
def create_resnet_model():
    # input_layer = layers.Input(shape=(img_size, img_size, 3))
    # base_model = ResNet50(include_top=False, input_tensor=input_layer, weights='imagenet', pooling='avg')
    # base_model.trainable = False
    # x = base_model.output
    # x = layers.Dense(128, activation='relu')(x)
    # output = layers.Dense(1, activation='sigmoid')(x)
    # return tf.keras.Model(inputs=input_layer, outputs=output)


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
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    return tf.keras.Model(inputs, outputs)



def plot_history(history, fold):
    plt.plot(history.history['accuracy'], label='train acc')
    plt.plot(history.history['val_accuracy'], label='val acc')
    plt.title(f'Resnet Accuracy Fold {fold + 1}')
    plt.legend()
    plt.savefig(f'Resnet_accuracy_fold{fold+1}.png')
    plt.clf()



### 4. TRAINING FUNCTION ###
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

        train_ds = create_dataset(X_train, y_train)
        val_ds = create_dataset(X_val, y_val)
        test_ds = create_dataset(X_test, y_test)

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

        # # Ensure X_test is 3-channel for prediction
        # if X_test.shape[-1] == 1:
        #     print("test-rgb")
        #     X_test_rgb = np.concatenate([X_test]*3, axis=-1)
        # else:
        #     X_test_rgb = X_test

        # y_pred = model.predict(X_test_rgb)
        # y_pred_bin = (y_pred > 0.5).astype(int)

        # print(classification_report(y_test, y_pred_bin, target_names=labels))
        # cm = confusion_matrix(y_test, y_pred_bin)
        # sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels, cmap='Blues')
        # plt.title(f'Resnet Confusion Matrix - Fold {fold + 1}')
        # plt.xlabel('Predicted')
        # plt.ylabel('Actual')
        # plt.tight_layout()
        # plt.savefig(f'resnet_fold{fold+1}_conf_matrix.png')
        # plt.clf()

        # print(f"Fold {fold + 1} - Accuracy: {acc:.4f} | Loss: {loss:.4f}")

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