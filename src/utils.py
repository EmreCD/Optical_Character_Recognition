import numpy as np
import pandas as pd
import os


class EMNISTUtils:
    def __init__(self, mapping_file_path='data/raw/emnist-balanced-mapping.txt'):
        self.mapping = self.load_mapping(mapping_file_path)
        self.reverse_mapping = {v: k for k, v in self.mapping.items()}

    def load_mapping(self, mapping_file_path):
        """EMNIST mapping dosyasını yükle"""
        mapping = {}
        if os.path.exists(mapping_file_path):
            with open(mapping_file_path, 'r') as f:
                for line in f:
                    class_id, ascii_code = line.strip().split()
                    mapping[int(class_id)] = chr(int(ascii_code))
        else:
            # Fallback mapping if file doesn't exist
            print("Mapping file not found. Using default mapping.")
            mapping = {
                **{i: str(i) for i in range(10)},  # 0-9
                **{i + 10: chr(65 + i) for i in range(26)},  # A-Z
                **{i + 36: chr(97 + i) for i in range(11)}  # a-k (47 total)
            }
        return mapping

    def get_character(self, class_id):
        """Sınıf ID'sinden karakteri al"""
        return self.mapping.get(class_id, '?')

    def get_class_id(self, character):
        """Karakterden sınıf ID'sini al"""
        return self.reverse_mapping.get(character, -1)

    def get_all_classes(self):
        """Tüm sınıfları listele"""
        return list(self.mapping.values())


def create_directories():
    """Proje klasörlerini oluştur"""
    directories = [
        'data/raw',
        'data/processed/train',
        'data/processed/test',
        'data/processed/input_images/original',
        'data/processed/input_images/preprocessed',
        'models',
        'results',
        'logs'
    ]

    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    print("Proje klasörleri oluşturuldu.")