import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os


class EMNISTPreprocessor:
    def __init__(self, data_path='data/raw'):
        self.data_path = data_path

    def check_preprocessing(self, image):
        """Görüntü ön işleme kontrolü"""
        plt.figure(figsize=(15, 3))

        # Orijinal görüntü
        plt.subplot(141)
        plt.imshow(image, cmap='gray')
        plt.title('Original')

        # Normalize edilmiş
        normalized = image.astype('float32') / 255.0
        plt.subplot(142)
        plt.imshow(normalized, cmap='gray')
        plt.title('Normalized')

        # Threshold uygulanmış
        _, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        plt.subplot(143)
        plt.imshow(thresh, cmap='gray')
        plt.title('Thresholded')

        # Final (model inputu)
        final = self.enhanced_preprocess(image)
        plt.subplot(144)
        plt.imshow(final.reshape(28, 28), cmap='gray')
        plt.title('Model Input')

        plt.show()

    def enhanced_preprocess(self, image):
        """Geliştirilmiş ön işleme"""
        # Gri tonlamaya çevir
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Görüntüyü düzelt
        image = cv2.resize(image, (28, 28))
        image = np.rot90(image, k=3)  # 270 derece döndür
        image = np.flip(image, axis=1)  # Yatay eksende çevir

        # Contrast artır
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        image = clahe.apply(image)

        # Adaptive threshold
        image = cv2.adaptiveThreshold(
            image, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )

        # Noise reduction
        image = cv2.medianBlur(image, 3)

        # Normalize
        image = image.astype('float32') / 255.0

        return image.reshape(1, 28, 28, 1)

    def visualize_emnist_samples(self, train_data, mapping, letter='F', num_samples=5):
        """EMNIST örneklerini görselleştir"""
        # EMNIST'ten örnekler
        samples = train_data[train_data.labels == mapping[letter]][:num_samples]

        pixels = samples.iloc[:, 1:].values
        for i in range(len(pixels)):
            plt.subplot(1, num_samples, i + 1)
            plt.imshow(pixels[i].reshape(28, 28), cmap='gray')
            plt.title(f'EMNIST {letter}-{i + 1}')
        plt.show()
    def load_emnist_data(self, train_file='emnist-balanced-train.csv',
                         test_file='emnist-balanced-test.csv'):
        """EMNIST CSV dosyalarını yükle"""
        print("EMNIST verisi yükleniyor...")

        # Train data
        train_path = os.path.join(self.data_path, train_file)
        if os.path.exists(train_path):
            train_data = pd.read_csv(train_path)
            print(f"Train data yüklendi: {train_data.shape}")
        else:
            raise FileNotFoundError(f"Train dosyası bulunamadı: {train_path}")

        # Test data (eğer varsa)
        test_path = os.path.join(self.data_path, test_file)
        if os.path.exists(test_path):
            test_data = pd.read_csv(test_path)
            print(f"Test data yüklendi: {test_data.shape}")
        else:
            print("Test dosyası bulunamadı, train'den ayırılacak.")
            test_data = None

        return train_data, test_data

    def extract_images_labels(self, data):
        """CSV'den görüntü ve etiketleri ayır"""
        labels = data.iloc[:, 0].values
        pixels = data.iloc[:, 1:].values

        # 28x28 görüntülere reshape
        images = pixels.reshape(-1, 28, 28)

        return images, labels

    def preprocess_image(self, image):
        """Tek görüntü ön işleme"""
        # Zaten 0-255 aralığında, float32'ye çevir
        if len(image.shape) == 3:
            # Renkli ise gri tonlamaya çevir
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Normalize et (0-1 arası)
        image = image.astype('float32') / 255.0

        # Gürültü azaltma

        ##kernel size ve sigma araştırılacak
        image = cv2.GaussianBlur(image, (3, 3), 0)

        # Contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        image_uint8 = (image * 255).astype('uint8')
        image = clahe.apply(image_uint8) / 255.0

        return image

    def preprocess_batch(self, images):
        """Batch görüntü ön işleme"""
        print(f"Batch preprocessing: {images.shape}")
        processed_images = []

        for i, image in enumerate(images):
            if i % 10000 == 0:
                print(f"İşlenen: {i}/{len(images)}")

            processed_img = self.preprocess_image(image)
            processed_images.append(processed_img)

        return np.array(processed_images)


    def prepare_for_model(self, images, labels):
        """Model için veriyi hazırla"""
        # Images: (batch_size, 28, 28, 1) şeklinde olmalı
        images = images.reshape(-1, 28, 28, 1)

        print(f"Final shape - Images: {images.shape}, Labels: {labels.shape}")
        return images, labels

    def split_data(self, images, labels, test_size=0.2, val_size=0.1):
        """Veriyi train/val/test olarak böl"""
        # İlk train/temp split
        X_train, X_temp, y_train, y_temp = train_test_split(
            images, labels, test_size=test_size + val_size,
            random_state=42, stratify=labels
        )

        # Temp'i val/test olarak böl
        val_ratio = val_size / (test_size + val_size)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=1 - val_ratio,
            random_state=42, stratify=y_temp
        )

        print(f"Data split:")
        print(f"Train: {X_train.shape[0]} samples")
        print(f"Validation: {X_val.shape[0]} samples")
        print(f"Test: {X_test.shape[0]} samples")

        return (X_train, y_train), (X_val, y_val), (X_test, y_test)

    def save_processed_data(self, data_splits, output_dir='data/processed'):
        """İşlenmiş veriyi kaydet"""
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = data_splits

        # Train data
        np.save(os.path.join(output_dir, 'train/images.npy'), X_train)
        np.save(os.path.join(output_dir, 'train/labels.npy'), y_train)

        # Validation data
        np.save(os.path.join(output_dir, 'val/images.npy'), X_val)
        np.save(os.path.join(output_dir, 'val/labels.npy'), y_val)

        # Test data
        np.save(os.path.join(output_dir, 'test/images.npy'), X_test)
        np.save(os.path.join(output_dir, 'test/labels.npy'), y_test)

        print("İşlenmiş veri kaydedildi.")


    def visualize_samples(self, images, labels, utils, num_samples=16):
        """Veri örneklerini görselleştir"""
        fig, axes = plt.subplots(4, 4, figsize=(12, 12))
        fig.suptitle('EMNIST Dataset Samples', fontsize=16)

        for i in range(num_samples):
            row = i // 4
            col = i % 4

            image = images[i].reshape(28, 28)
            label = labels[i]
            character = utils.get_character(label)

            axes[row, col].imshow(image, cmap='gray')
            axes[row, col].set_title(f'Label: {label} ({character})')
            axes[row, col].axis('off')

        plt.tight_layout()
        plt.savefig('results/data_samples.png', dpi=150, bbox_inches='tight')
        plt.show()


def preprocess_input_image(image_path):
    """Kullanıcı inputu için ön işleme"""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise ValueError(f"Görüntü yüklenemedi: {image_path}")

    # 28x28'e resize
    resized = cv2.resize(image, (28, 28))

    # Preprocessing
    preprocessor = EMNISTPreprocessor()
    processed = preprocessor.preprocess_image(resized)

    # Model için shape ayarla
    processed = processed.reshape(1, 28, 28, 1)

    return processed