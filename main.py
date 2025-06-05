import os
import sys
import argparse
import numpy as np
from datetime import datetime

# Kendi modüllerimizi import et
from src.utils import EMNISTUtils, create_directories
from src.preprocessing import EMNISTPreprocessor, preprocess_input_image
from src.model import EMNISTModel
from src.train import EMNISTTrainer
from src.evaluate import EMNISTEvaluator


class OCRApp:
    def __init__(self):
        print("🚀 EMNIST OCR Uygulaması Başlatılıyor...")
        create_directories()
        self.utils = EMNISTUtils()

    def show_menu(self):
        print("\n" + "=" * 50)
        print("📋 EMNIST OCR - Ana Menü")
        print("=" * 50)
        print("1. 📂 Veri Ön İşleme")
        print("2. 🎯 Model Eğitimi")
        print("3. 📊 Model Değerlendirme")
        print("4. 🔍 Tek Görüntü Tahmin")
        print("5. 🧪 Harf Testi")  # Yeni eklenen
        print("6. 📈 Sonuçları Görüntüle")
        print("7. ❌ Çıkış")

    def preprocessing_menu(self):
        """Preprocessing menüsü"""
        print("\n📂 Veri Ön İşleme")
        print("-" * 30)

        if os.path.exists('data/processed/train/images.npy'):
            print("⚠️  İşlenmiş veri zaten mevcut!")
            choice = input("Yeniden işlemek istiyor musunuz? (y/n): ").lower()
            if choice != 'y':
                return

        print("Preprocessing başlıyor...")

        try:
            # Preprocessing
            preprocessor = EMNISTPreprocessor()

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
            os.makedirs('data/processed/val', exist_ok=True)
            preprocessor.save_processed_data(data_splits)

            # Örnekleri görselleştir
            preprocessor.visualize_samples(
                processed_images[:16], train_labels[:16], self.utils
            )

            print("✅ Preprocessing tamamlandı!")

        except Exception as e:
            print(f"❌ Preprocessing hatası: {e}")

    def training_menu(self):
        """Eğitim menüsü"""
        print("\n🎯 Model Eğitimi")
        print("-" * 30)

        # Veri kontrolü
        if not os.path.exists('data/processed/train/images.npy'):
            print("❌ İşlenmiş veri bulunamadı! Önce preprocessing yapın.")
            return

        print("Model tipini seçin:")
        print("1. Basit CNN (Hızlı)")
        print("2. Gelişmiş CNN (Daha iyi performans)")

        choice = input("Seçiminiz (1-2): ").strip()

        if choice == "1":
            model_type = "simple"
            epochs = 15
            batch_size = 64
        elif choice == "2":
            model_type = "improved"
            epochs = 20
            batch_size = 32
        else:
            print("❌ Geçersiz seçim!")
            return

        print(f"🚀 {model_type.title()} CNN eğitimi başlıyor...")

        try:
            trainer = EMNISTTrainer()
            history = trainer.train_model(
                model_type=model_type,
                epochs=epochs,
                batch_size=batch_size,
                use_class_weights=True
            )

            if history:
                trainer.plot_training_history(f'results/{model_type}_training_history.png')
                print("✅ Eğitim tamamlandı!")
            else:
                print("❌ Eğitim başarısız!")

        except Exception as e:
            print(f"❌ Eğitim hatası: {e}")

    def test_letter(self):
        """Harf test menüsü"""
        print("\n🔍 Harf Testi")
        print("-" * 30)

        # Model seç
        model_files = [f for f in os.listdir('models') if f.endswith('.h5')]
        if not model_files:
            print("❌ Eğitilmiş model bulunamadı!")
            return

        print("Modeli seçin:")
        for i, model_file in enumerate(model_files, 1):
            print(f"{i}. {model_file}")

        try:
            choice = int(input("Seçiminiz: ")) - 1
            model_path = os.path.join('models', model_files[choice])

            # Görüntü yolu
            image_path = input("Test edilecek görüntü yolu: ").strip()
            true_label = input("Gerçek harf (örn: F): ").strip().upper()

            # Test yap
            model = EMNISTModel()
            model.load_model(model_path)

            preprocessor = EMNISTPreprocessor()
            evaluator = EMNISTEvaluator(model.model, preprocessor, self.utils.mapping)

            # Test et
            evaluator.test_model_on_letter(image_path, true_label)

        except Exception as e:
            print(f"❌ Test hatası: {e}")
    def evaluation_menu(self):
        """Değerlendirme menüsü"""
        print("\n📊 Model Değerlendirme")
        print("-" * 30)

        # Model dosyalarını listele
        model_files = []
        if os.path.exists('models'):
            for file in os.listdir('models'):
                if file.endswith('.h5'):
                    model_files.append(file)

        if not model_files:
            print("❌ Eğitilmiş model bulunamadı! Önce model eğitimi yapın.")
            return

        print("Değerlendirilecek modeli seçin:")
        for i, model_file in enumerate(model_files, 1):
            print(f"{i}. {model_file}")

        try:
            choice = int(input("Seçiminiz: ")) - 1
            selected_model = model_files[choice]
            model_path = os.path.join('models', selected_model)

            print(f"🔍 {selected_model} değerlendiriliyor...")

            evaluator = EMNISTEvaluator(model_path)
            evaluator.full_evaluation()

        except (ValueError, IndexError):
            print("❌ Geçersiz seçim!")
        except Exception as e:
            print(f"❌ Değerlendirme hatası: {e}")

    def prediction_menu(self):
        """Tek görüntü tahmin menüsü"""
        print("\n🔍 Tek Görüntü Tahmin")
        print("-" * 30)

        # Model dosyalarını listele
        model_files = []
        if os.path.exists('models'):
            for file in os.listdir('models'):
                if file.endswith('.h5'):
                    model_files.append(file)

        if not model_files:
            print("❌ Eğitilmiş model bulunamadı!")
            return

        print("Tahmin için modeli seçin:")
        for i, model_file in enumerate(model_files, 1):
            print(f"{i}. {model_file}")

        try:
            choice = int(input("Seçiminiz: ")) - 1
            selected_model = model_files[choice]
            model_path = os.path.join('models', selected_model)

            # Görüntü yolu
            image_path = input("Görüntü dosyası yolu: ").strip()

            if not os.path.exists(image_path):
                print("❌ Görüntü dosyası bulunamadı!")
                return

            print("🔍 Tahmin yapılıyor...")

            # Model yükle
            model_handler = EMNISTModel()
            model_handler.load_model(model_path)

            # Görüntüyü işle
            processed_image = preprocess_input_image(image_path)

            # Tahmin yap
            predicted_class, confidence, all_probabilities = model_handler.predict_single(processed_image)
            predicted_char = self.utils.get_character(predicted_class)

            print("\n📋 Tahmin Sonucu:")
            print(f"🎯 Karakter: {predicted_char}")
            print(f"📊 Güven: %{confidence * 100:.2f}")
            print(f"🏷️  Sınıf ID: {predicted_class}")

            # Top-5 tahminleri göster
            top_5_indices = np.argsort(all_probabilities)[-5:][::-1]
            print("\n🏆 En Olası 5 Tahmin:")
            for i, idx in enumerate(top_5_indices, 1):
                char = self.utils.get_character(idx)
                prob = all_probabilities[idx] * 100
                print(f"{i}. {char} (%{prob:.2f})")

        except (ValueError, IndexError):
            print("❌ Geçersiz seçim!")
        except Exception as e:
            print(f"❌ Tahmin hatası: {e}")

    def view_results_menu(self):
        """Sonuçları görüntüle"""
        print("\n📈 Sonuçlar")
        print("-" * 30)

        if os.path.exists('results'):
            result_files = os.listdir('results')
            if result_files:
                print("Mevcut sonuç dosyaları:")
                for file in result_files:
                    file_path = os.path.join('results', file)
                    print(f"📄 {file} - {os.path.getsize(file_path)} bytes")

                print(f"\n📁 Sonuçlar klasörü: {os.path.abspath('results')}")
            else:
                print("❌ Henüz sonuç dosyası yok.")
        else:
            print("❌ Results klasörü bulunamadı.")

    def run(self):
        """Ana uygulama döngüsü"""
        while True:
            self.show_menu()
            choice = input("\nSeçiminizi yapın (1-6): ").strip()

            if choice == "1":
                self.preprocessing_menu()
            elif choice == "2":
                self.training_menu()
            elif choice == "3":
                self.evaluation_menu()
            elif choice == "4":
                self.prediction_menu()
            elif choice == "5":
                self.view_results_menu()
            elif choice == "6":
                print("👋 Görüşürüz!")
                break
            else:
                print("❌ Geçersiz seçim! Lütfen 1-6 arası bir sayı girin.")

            input("\nDevam etmek için Enter'a basın...")


def main():
    """Ana fonksiyon - Command line arguments ile de çalışabilir"""
    parser = argparse.ArgumentParser(description='EMNIST OCR Uygulaması')
    parser.add_argument('--mode', choices=['preprocess', 'train', 'evaluate', 'predict'],
                        help='Direkt mod seçimi')
    parser.add_argument('--model-type', choices=['simple', 'improved'], default='simple',
                        help='Model tipi')
    parser.add_argument('--image', help='Tahmin için görüntü dosyası')
    parser.add_argument('--model', help='Kullanılacak model dosyası')

    args = parser.parse_args()

    if args.mode:
        # Command line mode
        create_directories()

        if args.mode == 'preprocess':
            print("🔄 Preprocessing mode...")
            # Preprocessing kodları buraya

        elif args.mode == 'train':
            print(f"🎯 Training mode - {args.model_type}")
            # Training kodları buraya

        elif args.mode == 'evaluate':
            print("📊 Evaluation mode...")
            # Evaluation kodları buraya

        elif args.mode == 'predict' and args.image:
            print(f"🔍 Prediction mode - {args.image}")
            # Prediction kodları buraya

    else:
        # Interactive mode
        app = OCRApp()
        app.run()


if __name__ == "__main__":
    main()