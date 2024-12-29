#!/usr/bin/env python
# coding: utf-8

# Importowanie wymaganych bibliotek
import cv2  # OpenCV - biblioteka do przetwarzania obrazów
import numpy as np  # NumPy - biblioteka do operacji na tablicach i macierzach
import os  # Biblioteka do operacji na systemie plików
import yaml  # Biblioteka do obsługi plików YAML
from yaml.loader import SafeLoader  # Bezpieczny loader dla plików YAML


# Klasa do obsługi predykcji YOLO
class YOLO_Pred():
    def __init__(self, onnx_model, data_yaml):
        # Inicjalizator klasy
        # Ładowanie konfiguracji z pliku YAML (np. nazwy klas, liczba klas)
        with open(data_yaml, mode='r') as f:
            data_yaml = yaml.load(f, Loader=SafeLoader)  # Otwiera plik YAML i ładuje dane

        self.labels = data_yaml['names']  # Nazwy klas 
        self.nc = data_yaml['nc']  # Liczba klas

        # Wczytywanie modelu YOLO w formacie ONNX
        self.yolo = cv2.dnn.readNetFromONNX(onnx_model)  # Ładowanie modelu ONNX
        # Konfiguracja backendu OpenCV dla modelu YOLO
        self.yolo.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)  # Ustawienie backendu OpenCV
        self.yolo.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)  # Używanie CPU jako celu do obliczeń

    # Metoda do wykonywania predykcji na obrazie
    def predictions(self, image):
        # Rozmiary obrazu wejściowego
        row, col, d = image.shape

        # Krok 1: Konwersja obrazu na kwadratowy
        max_rc = max(row, col)  # Znalezienie większego wymiaru (szerokość lub wysokość)
        input_image = np.zeros((max_rc, max_rc, 3), dtype=np.uint8)  # Tworzenie kwadratowego obrazu
        input_image[0:row, 0:col] = image  # Kopiowanie pikseli z obrazu wejściowego do kwadratu

        # Krok 2: Przygotowanie obrazu dla YOLO
        INPUT_WH_YOLO = 640  # Wymagany rozmiar wejścia dla YOLO
        # Tworzenie blobu z obrazu (normalizacja pikseli, zmiana rozmiaru, zamiana RGB na BGR)
        blob = cv2.dnn.blobFromImage(input_image, 1/255, (INPUT_WH_YOLO, INPUT_WH_YOLO), swapRB=True, crop=False)
        self.yolo.setInput(blob)  # Ustawienie blobu jako wejścia do modelu
        preds = self.yolo.forward()  # Uzyskanie wyników predykcji z modelu YOLO

        # Krok 3: Filtrowanie wyników na podstawie pewności i progu prawdopodobieństwa
        detections = preds[0]  # Uzyskanie wszystkich detekcji
        boxes = []  # Lista na ramki ograniczające (bounding boxes)
        confidences = []  # Lista na poziomy pewności detekcji
        classes = []  # Lista na identyfikatory klas

        # Rozmiary obrazu wejściowego
        image_w, image_h = input_image.shape[:2]
        x_factor = image_w / INPUT_WH_YOLO  # Skalowanie szerokości
        y_factor = image_h / INPUT_WH_YOLO  # Skalowanie wysokości

        for i in range(len(detections)):  # Iteracja przez wszystkie detekcje
            row = detections[i]
            confidence = row[4]  # Pewność detekcji obiektu
            if confidence > 0.4:  # Filtrowanie na podstawie progu pewności
                class_score = row[5:].max()  # Maksymalne prawdopodobieństwo klasy
                class_id = row[5:].argmax()  # Indeks klasy z maksymalnym prawdopodobieństwem

                if class_score > 0.25:  # Filtrowanie na podstawie progu prawdopodobieństwa klasy
                    # Współrzędne ramki ograniczającej (środek, szerokość, wysokość)
                    cx, cy, w, h = row[0:4]
                    # Przeliczenie współrzędnych na lewy górny róg, szerokość i wysokość
                    left = int((cx - 0.5 * w) * x_factor)
                    top = int((cy - 0.5 * h) * y_factor)
                    width = int(w * x_factor)
                    height = int(h * y_factor)

                    box = np.array([left, top, width, height])  # Konstrukcja ramki ograniczającej

                    # Dodanie wyników do list
                    confidences.append(confidence)
                    boxes.append(box)
                    classes.append(class_id)

        # Krok 4: Usuwanie zduplikowanych detekcji (Non-Maximum Suppression - NMS)
        boxes_np = np.array(boxes).tolist()  # Konwersja ramek na listę
        confidences_np = np.array(confidences).tolist()  # Konwersja pewności na listę

        # Wykonanie NMS z wykorzystaniem OpenCV
        index = np.array(cv2.dnn.NMSBoxes(boxes_np, confidences_np, 0.25, 0.45)).flatten()

        # Krok 5: Rysowanie ramek ograniczających na obrazie
        for ind in index:  # Iteracja przez indeksy po NMS
            x, y, w, h = boxes_np[ind]  # Współrzędne ramki
            bb_conf = int(confidences_np[ind] * 100)  # Pewność w procentach
            classes_id = classes[ind]  # ID klasy
            class_name = self.labels[classes_id]  # Nazwa klasy
            colors = self.generate_colors(classes_id)  # Generowanie kolorów dla klasy

            text = f'{class_name}: {bb_conf}%'  # Tekst z nazwą klasy i pewnością

            # Rysowanie ramki ograniczającej
            cv2.rectangle(image, (x, y), (x + w, y + h), colors, 2)
            # Rysowanie prostokąta wypełnionego pod tekstem
            cv2.rectangle(image, (x, y - 30), (x + w, y), colors, -1)

            # Dodanie tekstu na obrazie
            cv2.putText(image, text, (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 0.7, (0, 0, 0), 1)

        return image  # Zwrócenie obrazu z narysowanymi ramkami

    # Metoda generująca kolory dla klas
    def generate_colors(self, ID):
        np.random.seed(10)  # Ustawienie ziarna dla generatora pseudolosowego
        # Generowanie losowych kolorów dla klas
        colors = np.random.randint(100, 255, size=(self.nc, 3)).tolist()
        return tuple(colors[ID])  # Zwrócenie koloru dla danej klasy
