### IMPORTS ###

import os
import cv2
import numpy as np 
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D, MaxPool2D, GlobalAveragePooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.applications import ResNet50
from sklearn.metrics import accuracy_score
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import classification_report 
import seaborn as sns
from sklearn.metrics import confusion_matrix



data_path = "./new_data/BreaKHis_Total_dataset"

# Image labels
labels = ['benign', 'malignant']
img_size = 224


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

# Load data
data, data_labels = loading_data(data_path)

print(f" Data loaded")


def data_preprocessing(data, data_labels):
    # Data Normalizations
    X_data = np.array(data) / 255
    print(f"X_data")

    # Reshape the graysclae images to 128x128x1
    X_data = X_data.reshape(-1, img_size, img_size, 1)
    print(f"X_data reshape")

    # Convert labels to numpy arrays
    y_data = np.array(data_labels)
    print(f"y_data labels")
    print(f" X_data shape {X_data.shape}")  # This should now show (num_samples, 128, 128, 3)

    return X_data, y_data

X_data, y_data = data_preprocessing(data, data_labels)




num_folds = 5
kf = KFold(n_splits=num_folds, shuffle=True, random_state=42) # Adjust random_state for reproducibility

all_fold_metrics = []

for fold, (train_index, val_index) in enumerate(kf.split(X_data, y_data)):
    print(f"Fold {fold + 1}/{num_folds}")

    X_train_fold, X_val_fold = X_data[train_index], X_data[val_index]
    y_train_fold, y_val_fold = y_data[train_index], y_data[val_index]

    # Concatenate for the current fold
    X_train_rgb = np.concatenate([X_train_fold, X_train_fold, X_train_fold], axis=-1)
    X_val_rgb = np.concatenate([X_val_fold, X_val_fold, X_val_fold], axis=-1)



    # Create data generators for this fold
    train_datagen = ImageDataGenerator(rescale=1./255)
    val_datagen = ImageDataGenerator(rescale=1./255)
    train_generator = train_datagen.flow(X_train_rgb, y_train_fold, batch_size=32)
    validation_generator = val_datagen.flow(X_val_rgb, y_val_fold, batch_size=32)




    resnet_model = Sequential()
    pretrained_model= ResNet50(
        include_top=False, 
        input_shape=(img_size, img_size, 3), 
        pooling='avg', 
        weights='imagenet'
    )

    for layer in pretrained_model.layers:
        layer.trainable=False
    
    resnet_model.add(pretrained_model)

    # Fully connected layers for classification
    resnet_model.add(layers.Flatten())
    resnet_model.add(layers.Dense(512, activation='relu'))
    resnet_model.add(layers.Dense(1, activation='sigmoid'))

    # compile and train the model
    resnet_model.compile(
        optimizer=Adam(learning_rate=0.0001), 
        loss='binary_crossentropy', 
        metrics=['accuracy']
    )

    checkpoint = ModelCheckpoint(
        "resnet_1.h5", 
        monitor='val_accuracy', 
        verbose=1, 
        save_best_only=True, 
        save_weights_only=False, 
        mode='max', 
        save_freq=10
    )
    
    early = EarlyStopping(
        monitor='val_accuracy', 
        min_delta=0, 
        patience=20, 
        verbose=1, 
        mode='max'
    )

    resnet_model.fit(
        train_generator,
        batch_size=32,
        validation_data= validation_generator, 
        validation_steps=32,
        epochs=10,
        callbacks=[checkpoint,early]
    )


    # Evaluate on the validation set for this fold
    loss, accuracy = model.evaluate(validation_generator, verbose=0)
    print(f"Fold {fold + 1} Validation Accuracy: {accuracy:.4f}, Loss: {loss:.4f}")
    all_fold_metrics.append(accuracy)

print("\nCross-Validation Results:")
print(f"Mean Validation Accuracy: {np.mean(all_fold_metrics):.4f}")
print(f"Standard Deviation of Validation Accuracy: {np.std(all_fold_metrics):.4f}")

# After cross-validation, you would train your final model (potentially on the entire training set)
# and evaluate it on the separate test set (converted to RGB).




















