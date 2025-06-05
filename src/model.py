import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
import numpy as np


class EMNISTModel:
    def __init__(self, num_classes=47, input_shape=(28, 28, 1)):
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.model = None

    def create_simple_cnn(self):
        """Basit CNN modeli"""
        model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),

            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(self.num_classes, activation='softmax')
        ])

        self.model = model
        return model

    def create_improved_cnn(self):
        """Geliştirilmiş CNN modeli"""
        model = models.Sequential([
            # Input layer
            layers.Conv2D(32, (3, 3), padding='same', input_shape=self.input_shape),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.MaxPooling2D((2, 2)),

            # First conv block
            layers.Conv2D(64, (3, 3), padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Conv2D(64, (3, 3), padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),

            # Second conv block
            layers.Conv2D(128, (3, 3), padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Conv2D(128, (3, 3), padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),

            # Dense layers
            layers.Flatten(),
            layers.Dense(512),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Dropout(0.5),
            layers.Dense(256),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Dropout(0.5),
            layers.Dense(self.num_classes, activation='softmax')
        ])

        self.model = model
        return model

    def compile_model(self, learning_rate=0.001):
        """Modeli compile et"""
        if self.model is None:
            raise ValueError("Önce model oluşturulmalı!")

        optimizer = optimizers.Adam(learning_rate=learning_rate)

        self.model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        print("Model compile edildi.")
        return self.model

    def create_data_generator(self):
        """Veri çoğaltma için generator"""
        return tf.keras.preprocessing.image.ImageDataGenerator(
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.1,
            fill_mode='nearest'
        )

    def get_model_summary(self):
        """Model özetini göster"""
        if self.model is None:
            print("Model henüz oluşturulmamış!")
        else:
            self.model.summary()

        print(self.model.summary())
        return self.model.summary()

    def save_model(self, filepath):
        """Modeli kaydet"""
        if self.model is None:
            raise ValueError("Kaydedilecek model yok!")

        self.model.save(filepath)
        print(f"Model kaydedildi: {filepath}")

    def load_model(self, filepath):
        """Modeli yükle"""
        self.model = tf.keras.models.load_model(filepath)
        print(f"Model yüklendi: {filepath}")
        return self.model

    def predict(self, X):
        """Tahmin yap"""
        if self.model is None:
            raise ValueError("Model yüklenmemiş!")

        return self.model.predict(X)

    def predict_single(self, image):
        """Tek görüntü için tahmin"""
        if len(image.shape) == 3:
            image = image.reshape(1, *image.shape)

        prediction = self.model.predict(image, verbose=0)
        predicted_class = np.argmax(prediction[0])
        confidence = prediction[0][predicted_class]

        return predicted_class, confidence, prediction[0]


def create_callbacks(model_name='emnist_model'):
    """Training callbacks oluştur"""
    callbacks_list = [
        callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        callbacks.ModelCheckpoint(
            filepath=f'models/{model_name}_best.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        ),
        callbacks.CSVLogger(
            filename=f'logs/{model_name}_training.log'
        )
    ]

    return callbacks_list