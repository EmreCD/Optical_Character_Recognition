import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import datetime
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             confusion_matrix, classification_report)
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import euclidean, cityblock
import pandas as pd
from src.model import EMNISTModel
from src.utils import EMNISTUtils


class EMNISTEvaluator:
    def __init__(self, model_path=None):
        self.model_handler = EMNISTModel()
        self.utils = EMNISTUtils()

        if model_path:
            self.model_handler.load_model(model_path)

    def test_model_on_letter(self, image_path, true_label='F'):
        """Model test fonksiyonu"""
        # Görüntüyü yükle
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        # Ön işleme kontrolü
        self.preprocessor.check_preprocessing(image)

        # Model tahmini
        processed = self.preprocessor.enhanced_preprocess(image)
        predictions = self.model.predict(processed)[0]

        # Top 5 tahmin
        top_5_idx = np.argsort(predictions)[-5:][::-1]

        print(f"\nTahminler için '{true_label}':")
        print("-" * 30)
        for idx in top_5_idx:
            char = self.mapping[idx]
            conf = predictions[idx] * 100
            print(f"{char}: %{conf:.2f}")

        # Confusion matrix için kaydet
        if predictions[self.mapping_rev[true_label]] < 0.5:
            print(f"\n⚠️ Düşük güven! Görüntüyü kaydediyorum...")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            cv2.imwrite(f"debug/low_conf_{true_label}_{timestamp}.png", image)
    def load_test_data(self):
        """Test verisini yükle"""
        try:
            X_test = np.load('data/processed/test/images.npy')
            y_test = np.load('data/processed/test/labels.npy')
            print(f"Test verisi yüklendi: {X_test.shape}")
            return X_test, y_test
        except FileNotFoundError:
            print("Test verisi bulunamadı!")
            return None, None

    def analyze_errors(self, test_data):
        """Hata analizi"""
        # Test setinden F örnekleri
        f_samples = test_data[test_data.labels == self.mapping_rev['F']]

        errors = []
        for idx, sample in f_samples.iterrows():
            pred = self.model.predict(sample[1:].reshape(1, 28, 28, 1))[0]
            pred_class = np.argmax(pred)

            if pred_class != self.mapping_rev['F']:
                errors.append({
                    'true': 'F',
                    'pred': self.mapping[pred_class],
                    'conf': pred[pred_class],
                    'image': sample[1:].reshape(28, 28)
                })

        # Hatalı tahminleri görselleştir
        if errors:
            plt.figure(figsize=(15, 3))
            for i, error in enumerate(errors[:5]):
                plt.subplot(1, 5, i + 1)
                plt.imshow(error['image'], cmap='gray')
                plt.title(f"Pred: {error['pred']}\nConf: {error['conf']:.2f}")
            plt.show()
    def calculate_cosine_similarity(self, pred_features, true_features):
        """Cosine similarity hesaplama"""
        # Feature vektörlerini normalize et

        pred_norm = pred_features / np.linalg.norm(pred_features, axis=1, keepdims=True)
        true_norm = true_features / np.linalg.norm(true_features, axis=1, keepdims=True)

        # Cosine similarity hesapla
        similarities = []
        for i in range(len(pred_norm)):
            sim = cosine_similarity(
                pred_norm[i].reshape(1, -1),
                true_norm[i].reshape(1, -1)
            )[0][0]
            similarities.append(sim)

        return np.array(similarities)

    def extract_features(self, X, layer_name=None):
        """Model'den feature çıkar"""
        if layer_name is None:
            # Son dense layer'dan önceki features
            feature_extractor = tf.keras.Model(
                inputs=self.model_handler.model.input,
                outputs=self.model_handler.model.layers[-3].output
            )
        else:
            # Belirli bir layer'dan features
            feature_extractor = tf.keras.Model(
                inputs=self.model_handler.model.input,
                outputs=self.model_handler.model.get_layer(layer_name).output
            )

        features = feature_extractor.predict(X, verbose=0)
        return features

    def comprehensive_evaluation(self, X_test, y_test):
        """Kapsamlı değerlendirme"""
        print("Kapsamlı değerlendirme başlıyor...")

        # Tahminler
        predictions = self.model_handler.predict(X_test)
        y_pred = np.argmax(predictions, axis=1)

        # Feature extraction
        print("Feature extraction...")
        pred_features = self.extract_features(X_test)

        # True features (one-hot encoding)
        true_features_onehot = np.eye(47)[y_test]

        # Metrics hesaplama
        metrics = {}

        # 1. Basic Classification Metrics
        metrics['accuracy'] = accuracy_score(y_test, y_pred)

        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average='macro'
        )
        metrics['precision_macro'] = precision
        metrics['recall_macro'] = recall
        metrics['f1_score_macro'] = f1

        # Weighted averages
        precision_w, recall_w, f1_w, _ = precision_recall_fscore_support(
            y_test, y_pred, average='weighted'
        )
        metrics['precision_weighted'] = precision_w
        metrics['recall_weighted'] = recall_w
        metrics['f1_score_weighted'] = f1_w

        # 2. Top-k Accuracy
        top_k_acc = self.calculate_top_k_accuracy(predictions, y_test, k=5)
        metrics['top_5_accuracy'] = top_k_acc

        # 3. Distance Metrics
        print("Distance metrics hesaplanıyor...")

        # Euclidean distances
        euclidean_distances = []
        manhattan_distances = []

        sample_size = min(1000, len(y_test))  # Performance için sample
        indices = np.random.choice(len(y_test), sample_size, replace=False)

        for i in indices:
            euclidean_dist = euclidean(predictions[i], true_features_onehot[i])
            manhattan_dist = cityblock(predictions[i], true_features_onehot[i])

            euclidean_distances.append(euclidean_dist)
            manhattan_distances.append(manhattan_dist)

        metrics['mean_euclidean_distance'] = np.mean(euclidean_distances)
        metrics['mean_manhattan_distance'] = np.mean(manhattan_distances)

        # 4. Cosine Similarity
        print("Cosine similarity hesaplanıyor...")
        cosine_similarities = self.calculate_cosine_similarity(
            predictions[indices], true_features_onehot[indices]
        )
        metrics['mean_cosine_similarity'] = np.mean(cosine_similarities)