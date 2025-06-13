'''
import os

folder_path = "C:/prak pengenalan pola/rusak"
files = os.listdir(folder_path)

for i, filename in enumerate(files):
    old_file = os.path.join(folder_path, filename)
    
    if not os.path.isfile(old_file):
        continue
       
    # Ganti ekstensi jika perlu
    ext = os.path.splitext(filename)[1]
    new_filename = f"damaged_{i+634}{ext}"
    new_file = os.path.join(folder_path, new_filename)

    os.rename(old_file, new_file)
    print(f"{filename} => {new_filename}")
'''
try:
    import tensorflow as tf
    print("✅ TensorFlow:", tf.__version__)
    
    from tensorflow.keras.applications import ResNet50
    from tensorflow.keras.applications.resnet50 import preprocess_input
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, Dense, Dropout, GlobalAveragePooling2D
    from tensorflow.keras.callbacks import EarlyStopping
except ImportError:
    print("❌ TensorFlow belum terinstal.")

try:
    from sklearn.utils.class_weight import compute_class_weight
    import sklearn
    print("✅ scikit-learn:", sklearn.__version__)
except ImportError:
    print("❌ scikit-learn belum terinstal.")

try:
    import matplotlib.pyplot as plt
    print("✅ Matplotlib:", plt.__name__)
except ImportError:
    print("❌ matplotlib belum terinstal.")
