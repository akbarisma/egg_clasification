# ======================================
# CNN Model untuk Klasifikasi Gambar Telur
# ======================================

import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ======================================
# Fungsi Memuat Dataset Gambar
# ======================================
def load_image_dataset(base_dir, label_names, img_size=(128, 128)):
    X = []
    y = []
    for idx, label in enumerate(label_names):
        folder = os.path.join(base_dir, label)
        for fname in os.listdir(folder):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                path = os.path.join(folder, fname)
                img = cv2.imread(path)
                img = cv2.resize(img, img_size)
                img = img.astype(np.float32) / 255.0
                X.append(img)
                y.append(idx)
    return np.array(X), np.array(y)

# ======================================
# Bangun Model CNN Sederhana
# ======================================
def build_cnn_model(input_shape, num_classes):
    inputs = Input(shape=input_shape)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# ======================================
# Evaluasi Model CNN
# ======================================
def evaluate_cnn_model(model, X_test, y_test, label_names):
    y_pred_prob = model.predict(X_test)
    y_pred = np.argmax(y_pred_prob, axis=1)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=label_names))

    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(cm)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_names, yticklabels=label_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("CNN Confusion Matrix")
    plt.tight_layout()
    plt.show()

# ======================================
# Program Utama
# ======================================
if __name__ == "__main__":
    DATASET_DIR = "C:/prak pengenalan pola/machine_learning/Eggs_Split_Fixed_new/train"
    VAL_DIR = "C:/prak pengenalan pola/machine_learning/Eggs_Split_Fixed_new/val"
    TEST_DIR = "C:/prak pengenalan pola/machine_learning/Eggs_Split_Fixed_new/test"
    LABELS = ["Damaged", "Not_Damaged"]
    IMG_SIZE = (128, 128)

    # Load Data Train & Validasi
    X_train, y_train = load_image_dataset(DATASET_DIR, LABELS, IMG_SIZE)
    X_val, y_val = load_image_dataset(VAL_DIR, LABELS, IMG_SIZE)
    X_test, y_test = load_image_dataset(TEST_DIR, LABELS, IMG_SIZE)

    y_train_cat = to_categorical(y_train, num_classes=len(LABELS))
    y_val_cat = to_categorical(y_val, num_classes=len(LABELS))

    # Bangun & Latih CNN
    model = build_cnn_model((IMG_SIZE[0], IMG_SIZE[1], 3), len(LABELS))
    early_stop = EarlyStopping(patience=5, restore_best_weights=True)
    history = model.fit(
        X_train, y_train_cat,
        validation_data=(X_val, y_val_cat),
        epochs=30,
        batch_size=32,
        callbacks=[early_stop]
    )

    # Simpan Model
    model.save("cnn_egg_model.keras")
    print("✅ Model CNN disimpan sebagai cnn_egg_model.keras")

    # Evaluasi pada Test Set
    evaluate_cnn_model(model, X_test, y_test, LABELS)

    # Plot Loss & Accuracy
plt.plot(history.history['loss'])
plt.plot(history.history['accuracy'])
plt.title('Training Loss dan Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Nilai')
plt.legend(['Loss', 'Accuracy'])
plt.grid(True)
plt.show()

from tensorflow.keras.models import load_model

# Muat model CNN yang telah disimpan
model = load_model("cnn_egg_model.keras")
# Evaluasi pada Validation Set
print("\n📊 Evaluasi pada Validation Set:")
evaluate_cnn_model(model, X_val, y_val, LABELS)
# Evaluasi pada Test Set
print("\n📊 Evaluasi pada Test Set:")
evaluate_cnn_model(model, X_test, y_test, LABELS)
