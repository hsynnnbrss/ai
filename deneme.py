import os
import numpy as np
import pydicom
import cv2
from tqdm import tqdm

# Veri seti yolu
dataset_path =r"C:\Users\Huseyin\Desktop\data_set\veriseti"

# Klasörleri ve etiketleri belirleme
categories = {
    "inme_yok": 0,
    "iskemik": 1,
    "kanama": 2
}

# Verileri ve etiketleri saklamak için listeler
data = []
labels = []

# Görselleri belirlenen boyuta getirmek için
IMG_SIZE = 128  # Görüntüler 128x128 olarak yeniden boyutlandırılacak

# Verileri yükleme
for category, label in categories.items():
    category_path = os.path.join(dataset_path, category)
    
    for filename in tqdm(os.listdir(category_path), desc=f"Loading {category}"):
        file_path = os.path.join(category_path, filename)
        
        try:
            # DICOM dosyasını aç
            dicom_data = pydicom.dcmread(file_path)
            img = dicom_data.pixel_array  # Görüntü verisini al

            # Görüntüyü normalize edip boyutlandır
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = img.astype(np.float32) / 255.0  # 0-1 aralığına getir
            
            data.append(img)
            labels.append(label)
        
        except Exception as e:
            print(f"Hata oluştu: {file_path} -> {e}")

# NumPy dizilerine çevirme
data = np.array(data).reshape(-1, IMG_SIZE, IMG_SIZE, 1)  # CNN için uygun hale getiriyoruz
labels = np.array(labels)

print(f"Toplam veri sayısı: {len(data)}")
print(f"Veri şekli: {data.shape}")
print(f"Etiketlerin dağılımı: {np.unique(labels, return_counts=True)}")
print(data)
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Etiketleri one-hot encoding formatına çevirme
num_classes = 3  # 3 farklı sınıfımız var
labels = to_categorical(labels, num_classes)

# Veriyi eğitim ve test setlerine ayırma (80% eğitim, 20% test)
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# CNN Modelini oluşturma
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(128, 128, 1)),
    MaxPooling2D((2,2)),
    
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),  # Overfitting'i önlemek için
    Dense(num_classes, activation='softmax')
])

# Modeli derleme
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Modeli eğitme
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

# Modeli değerlendirme
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test doğruluğu: {test_acc:.4f}")

# Eğitilmiş modeli kaydetme
model.save("brain_stroke_cnn.h5")

    