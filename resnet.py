### IMPORTS ###
import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split, KFold
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
        
    return y_pred, y_pred_prob_flat


### 4. TRAINING FUNCTION ###
def train_model():
    print("Loading data...")
    data, labels = loading_data(data_path)
    X, y = preprocess_data(data, labels)

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

        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        callbacks = [
            EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True),
            ModelCheckpoint(f"ResNet_fold{fold + 1}.h5", save_best_only=True)
        ]


        # Train model
        print("Training model...")
        history = model.fit(
            train_ds, 
            validation_data=val_ds, 
            epochs=epochs, 
            callbacks=callbacks
        )

        loss, acc = model.evaluate(val_ds)
        accs.append(acc)
        losses.append(loss)

        print(f"Fold {fold + 1} - Accuracy: {acc:.4f} | Loss: {loss:.4f}")


         # Keep track of best model
        if val_acc > best_acc:
            best_acc = val_acc
            best_model = model

    print(f"\n Final Results for ResNet: ")
    print(f"Mean Accuracy: {np.mean(accs):.4f}")
    print(f"Std Dev Accuracy: {np.std(accs):.4f}")
    print(f"Mean Loss: {np.mean(losses):.4f}")
    print(f"Std Dev Loss: {np.std(losses):.4f}")


    # Create test dataset
    test_ds = create_dataset(X_test, y_test)

    # Evaluate best model on test set
    print("\n======= Final Evaluation on Test Set =======")
    test_loss, test_acc = best_model.evaluate(test_ds)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")

    # Save best model
    best_model.save('best_resnet_model.h5')
    print("Best model saved as 'best_resnet_model.h5'")



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





