import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight
import os
from src.model import EMNISTModel, create_callbacks
from src.preprocessing import EMNISTPreprocessor
from src.utils import EMNISTUtils, create_directories


class EMNISTTrainer:
    def __init__(self):
        self.model_handler = EMNISTModel()
        self.history = None

    def load_data(self):
        """İşlenmiş veriyi yükle"""
        try:
            print("İşlenmiş veri yükleniyor...")
            X_train = np.load('data/processed/train/images.npy')
            y_train = np.load('data/processed/train/labels.npy')
            X_val = np.load('data/processed/val/images.npy')
            y_val = np.load('data/processed/val/labels.npy')
            X_test = np.load('data/processed/test/images.npy')
            y_test = np.load('data/processed/test/labels.npy')

            print(f"Train: {X_train.shape}, Validation: {X_val.shape}, Test: {X_test.shape}")
            return (X_train, y_train), (X_val, y_val), (X_test, y_test)

        except FileNotFoundError:
            print("İşlenmiş veri bulunamadı. Önce preprocessing yapılmalı.")
            return None

    def compute_class_weights(self, y_train):
        """Sınıf ağırlıklarını hesapla (dengesiz veri için)"""
        classes = np.unique(y_train)
        weights = compute_class_weight('balanced', classes=classes, y=y_train)
        class_weight_dict = dict(zip(classes, weights))

        print(f"Sınıf ağırlıkları hesaplandı. Örnek: {list(class_weight_dict.items())[:5]}")
        return class_weight_dict

    def train_model(self, model_type='simple', epochs=20, batch_size=32,
                    use_class_weights=False):
        """Model eğitimi"""
        # Veriyi yükle
        data = self.load_data()
        if data is None:
            return None

        (X_train, y_train), (X_val, y_val), (X_test, y_test) = data

        # Model oluştur
        if model_type == 'simple':
            model = self.model_handler.create_simple_cnn()
        else:
            model = self.model_handler.create_improved_cnn()

        # Compile et
        self.model_handler.compile_model()

        # Model özetini göster
        self.model_handler.get_model_summary()

        # Callbacks
        callbacks_list = create_callbacks(f'emnist_{model_type}')

        # Sınıf ağırlıkları
        class_weight = None
        if use_class_weights:
            class_weight = self.compute_class_weights(y_train)

        print("Eğitim başlıyor...")

        # Eğitim
        self.history = model.fit(
            X_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(X_val, y_val),
            callbacks=callbacks_list,
            class_weight=class_weight,
            verbose=1
        )

        # En iyi modeli kaydet
        self.model_handler.save_model(f'models/emnist_{model_type}_final.h5')

        return self.history

    def plot_training_history(self, save_path='results/training_history.png'):
        """Eğitim geçmişini görselleştir"""
        if self.history is None:
            print("Eğitim geçmişi bulunamadı!")
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # Accuracy
        ax1.plot(self.history.history['accuracy'], label='Train Accuracy')
        ax1.plot(self.history.history['val_accuracy'], label='Val Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)

        # Loss
        ax2.plot(self.history.history['loss'], label='Train Loss')
        ax2.plot(self.history.history['val_loss'], label='Val Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()

        print(f"Eğitim grafiği kaydedildi: {save_path}")


def main():
    """Ana eğitim fonksiyonu"""
    # Klasörleri oluştur
    create_directories()

    # Önce preprocessing yapılmalı mı kontrol et
    if not os.path.exists('data/processed/train/images.npy'):
        print("Veri preprocessing'i yapılıyor...")

        # Preprocessing
        preprocessor = EMNISTPreprocessor()
        utils = EMNISTUtils()

        # Veriyi yükle
        train_data, test_data = preprocessor.load_emnist_data()

        # Images ve labels'ı ayır
        train_images, train_labels = preprocessor.extract_images_labels(train_data)

        # Preprocessing
        processed_images = preprocessor.preprocess_batch(train_images)

        # Model için hazırla
        processed_images, train_labels = preprocessor.prepare_for_model(
            processed_images, train_labels
        )

        # Split data
        data_splits = preprocessor.split_data(processed_images, train_labels)

        # Kaydet
        # Validation klasörünü oluştur
        os.makedirs('data/processed/val', exist_ok=True)
        preprocessor.save_processed_data(data_splits)

        # Örnekleri görselleştir
        preprocessor.visualize_samples(
            processed_images[:16], train_labels[:16], utils
        )

    # Eğitim
    trainer = EMNISTTrainer()

    print("\n=== Basit CNN Eğitimi ===")
    history = trainer.train_model(
        model_type='simple',
        epochs=15,
        batch_size=64,
        use_class_weights=True
    )

    if history:
        trainer.plot_training_history('results/simple_cnn_history.png')

    print("\n=== Gelişmiş CNN Eğitimi ===")
    trainer_improved = EMNISTTrainer()
    history_improved = trainer_improved.train_model(
        model_type='improved',
        epochs=20,
        batch_size=32,
        use_class_weights=True
    )

    if history_improved:
        trainer_improved.plot_training_history('results/improved_cnn_history.png')


if __name__ == "__main__":
    main()