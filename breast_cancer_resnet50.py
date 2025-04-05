
### IMPORTS ###

import os
import cv2
import numpy as np 
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D, MaxPool2D
from tensorflow.keras.layers import Flatten
from sklearn.metrics import accuracy_score
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import classification_report 
import seaborn as sns
from sklearn.metrics import confusion_matrix


data_path = "../../../data/BreaKHis_Total_dataset"

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


# # load random data
# random_indices = np.random.choice(len(data), 6, replace=False)

# #Set up the figure
# plt.figure(figsize=(8,8))

# # Plot the images
# for i, index in enumerate(random_indices):
#     plt.subplot(3, 3, i+1)
#     plt.imshow(data[index], cmap='gray')
#     plt.title(
#         'Benign' if data_labels[index] == 0 else 'Malignant cancer',
#         fontsize=14,
#         fontweight='bold',
#         color='blue' if data_labels[index] == 0 else 'green'
#     )
#     plt.axis('off')

# # Add a main title
# plt.suptitle(
#     "Random Sample of Breast tumors images",
#     fontsize=18,
#     fontweight='bold',
#     y=1.02
# )

# # Adjust layout for better spacing
# plt.tight_layout()
# # plt.show(block=False)
# plt.savefig("img1")

# Data Normalizations
X_data = np.array(data) / 255

print(f"X_data")

# Reshape the graysclae images to 128x128x1
X_data = X_data.reshape(-1, img_size, img_size, 1)

print(f"X_data reshape")

# Convert grayscale to RGB by duplicating the single channel 3 times
X_data = np.repeat(X_data, 3, axis=-1)

print(f"X_data rgb")

# Convert labels to numpy arrays
y_data = np.array(data_labels)

print(f"y_data labels")

print(f" X_data shape {X_data.shape}")  # This should now show (num_samples, 128, 128, 3)

# train-validation split on the data
# val_size = 0.2
# X_train, X_val, y_train, y_val = train_test_split(
#     X_data, 
#     y_data, 
#     test_size=val_size, 
#     random_state=42
# )

X_train, X_temp, y_train, y_temp = train_test_split(
    X_data, 
    y_data, 
    test_size=0.3, 
    random_state=42
)

# split temp into val and test
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, 
    y_temp, 
    test_size=0.5, 
    random_state=42
)

# Check the shapes
print("Training data shape:", X_train.shape)
print("Validation data shape:", X_val.shape)
print("Test data shape:", X_test.shape)
print("Training labels shape:", y_train.shape)
print("Validation labels shape:", y_val.shape)
print("Test labels shape:", y_test.shape)



# Data Augmentation
data_generator = ImageDataGenerator(  
                    rotation_range = 30,
                    zoom_range = 0.2, 
                    width_shift_range=0.1,  
                    height_shift_range=0.1,  
                    horizontal_flip = True,  
                    shear_range=0.2,
                    fill_mode='nearest',
                 )

val_data = ImageDataGenerator()

data_generator.fit(X_train)
val_data.fit(X_val)


train_gen = data_generator.flow(X_train, y_train, batch_size=32)
val_gen = val_data.flow(X_val, y_val, batch_size=32)


#ResNet code
resnet_model = Sequential()
pretrained_model= tf.keras.applications.ResNet50(include_top=False, 
                                                 input_shape=(img_size, img_size, 3), 
                                                 pooling='avg', 
                                                 weights='imagenet')
for layer in pretrained_model.layers:
        layer.trainable=False
resnet_model.add(pretrained_model)


# Fully connected layers for classification
resnet_model.add(layers.Flatten())
resnet_model.add(layers.Dense(512, activation='relu'))
resnet_model.add(layers.Dense(1, activation='sigmoid'))


# compile and train the model


resnet_model.compile(optimizer='adam', 
                     loss='binary_crossentropy', 
                     metrics=['accuracy'])



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
    train_gen,
    batch_size=32,
    validation_data= val_gen, 
    validation_steps=32,
    epochs=10,
    callbacks=[checkpoint,early]
)


# history = resnet_model.fit(X_train, 
#                            validation_data=X_val, 
#                            epochs=10, 
#                            batch_size=32)



# Retrieve metrics from the training history
history = resnet_model.history.history  # Access the 'history' dictionary

train_acc = history['accuracy']
train_loss = history['loss']
val_acc = history['val_accuracy']
val_loss = history['val_loss']

# Epochs
epochs = range(1, len(train_acc) + 1)


# Create a figure and axes for the plots
# fig, ax = plt.subplots(1, 2, figsize=(18, 6))

# # Plot training and validation accuracy
# ax[0].plot(epochs, train_acc, 'o-', color='darkgreen', label='Training Accuracy', markersize=8)
# ax[0].plot(epochs, val_acc, 's--', color='darkred', label='Validation Accuracy', markersize=8)
# ax[0].set_title('Training vs. Validation Accuracy', fontsize=16)
# ax[0].set_xlabel('Epochs', fontsize=14)
# ax[0].set_ylabel('Accuracy', fontsize=14)
# ax[0].legend()
# ax[0].grid(True)

# # Plot training and validation loss
# ax[1].plot(epochs, train_loss, 'o-', color='darkblue', label='Training Loss', markersize=8)
# ax[1].plot(epochs, val_loss, 's--', color='orange', label='Validation Loss', markersize=8)
# ax[1].set_title('Training vs. Validation Loss', fontsize=16)
# ax[1].set_xlabel('Epochs', fontsize=14)
# ax[1].set_ylabel('Loss', fontsize=14)
# ax[1].legend()
# ax[1].grid(True)

# # Display the plots
# plt.tight_layout()
# # plt.show(block=False)
# plt.savefig("img2")

evaluation = resnet_model.evaluate(X_test,y_test)
print("=="*20)
print(evaluation)
print(f"Accuracy - {evaluation[1]*100}%")
print(f"Loss - {evaluation[0]}")
print("=="*20)



predictions = resnet_model.predict(X_test)
predictions = (predictions > 0.5).astype(int)  # Convert probabilities to binary labels

# Print the classification report
print("Classification Report:")
print(classification_report(y_test, predictions, target_names=['Benign', 'Malignant cancer']))


# # Function to plot confusion matrix with percentages
# def plot_confusion_matrix_with_percentages(cm, model_name):
#     plt.figure(figsize=(5, 5))
#     cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100  # Convert to percentages

#     # Annotate with both raw numbers and percentages
#     annot = np.empty_like(cm).astype(str)
#     for i in range(cm.shape[0]):
#         for j in range(cm.shape[1]):
#             annot[i, j] = f"{cm[i, j]}\n({cm_percentage[i, j]:.1f}%)"

#     sns.heatmap(cm, annot=annot, fmt="", cmap="Blues", xticklabels=["Benign", "Malignant cancer"], yticklabels=["Non-malignant", "Malignant cancer"])
#     plt.title(f"Confusion Matrix - {model_name}", fontsize=16)
#     plt.xlabel("Predicted", fontsize=14)
#     plt.ylabel("True", fontsize=14)
#     # plt.show(block=False)
#     plt.savefig("img3")


# Generate the confusion matrix
cm = confusion_matrix(y_test, predictions)

# Plot the confusion matrix with percentages
# plot_confusion_matrix_with_percentages(cm, "Resnet50")



# Randomly select 8 indices from the test set
# random_indices = np.random.choice(len(X_test), 8, replace=False)

# # Define the figure size
# plt.figure(figsize=(15, 5))

# # Iterate through the selected indices
# for i, idx in enumerate(random_indices):
#     plt.subplot(2, 4, i + 1)

#     # Display the image
#     plt.imshow(X_test[idx].reshape(224, 224, 3), cmap='magma', interpolation='none')

#     # Set the title with predicted and actual classes
#     title_color = 'red' if predictions[idx] != y_test[idx] else 'green'  # Red if incorrect, green if correct
#     plt.title(f"Predicted: {predictions[idx]}   Actual: {y_test[idx]}", fontsize=10, color=title_color)

#     # Remove x and y ticks
#     plt.axis('off')

# # Set the main title for the figure
# plt.suptitle("Sample Test Images with Predictions", size=18)

# # Adjust layout to prevent overlapping
# plt.tight_layout()

# # Show the plot
# # plt.show(block=False)
# plt.savefig("img4")
