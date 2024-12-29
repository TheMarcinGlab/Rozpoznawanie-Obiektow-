# Importowanie wymaganych bibliotek
import streamlit as st  # Biblioteka do tworzenia interaktywnych aplikacji webowych
from yolo_predictions import YOLO_Pred  # Klasa do predykcji obiektów przy użyciu YOLO
from PIL import Image  # Biblioteka do obsługi obrazów
import numpy as np  # Biblioteka do operacji na macierzach

# Ustawienia strony Streamlit
st.set_page_config(page_title="Marcin Głąb",  # Tytuł aplikacji
                   layout='wide',  # Układ szeroki (lepsza widoczność elementów)
                   page_icon='./images/object.png')  # Ikona strony

# Nagłówek aplikacji
st.header('Rozpoznawanie Obrazów przy użyciu biblioteki YOLO')  # Wyświetlenie nagłówka
st.write('Proszę prześlij zdjęcie')  # Krótki opis funkcji aplikacji

# Pokazanie spinnera (animacji ładowania) podczas inicjalizacji modelu YOLO
with st.spinner('Proszę czekać model jest ładowany'):
    # Inicjalizacja klasy YOLO do predykcji, podanie ścieżek do modelu ONNX i pliku YAML
    # yolo = YOLO_Pred(onnx_model='C:/Users/marci/OneDrive/Pulpit/RozpoznawanieObrazow/models/best.onnx',
    #                  data_yaml='C:/Users/marci/OneDrive/Pulpit/RozpoznawanieObrazow/models/data.yaml')
    yolo = YOLO_Pred(onnx_model='./models/best.onnx', data_yaml='./models/data.yaml')


# Funkcja obsługująca przesyłanie obrazów
def upload_image():
    # Widget do przesyłania plików
    image_file = st.file_uploader(label='Prześlij obraz')
    if image_file is not None:  # Sprawdzenie, czy plik został przesłany
        # Obliczenie rozmiaru pliku w MB
        size_mb = image_file.size / (1024**2)
        # Szczegóły przesłanego pliku
        file_details = {"filename": image_file.name,  # Nazwa pliku
                        "filetype": image_file.type,  # Typ pliku (np. image/jpeg)
                        "filesize": "{:,.2f} MB".format(size_mb)}  # Rozmiar pliku w MB
        # Walidacja typu pliku
        if file_details['filetype'] in ('image/png', 'image/jpeg'):  # Akceptowane formaty: PNG, JPEG
            st.success('Walidacja ok (png or jpeg')  # Wyświetlenie komunikatu o poprawnym formacie
            return {"file": image_file,  # Zwrócenie pliku
                    "details": file_details}  # Zwrócenie szczegółów pliku
        else:
            # Komunikaty o błędzie dla nieprawidłowego formatu
            st.error('Niepoprawny plik')  # Wyświetlenie błędu dla nieprawidłowego pliku
            st.error('Prześlij w rozszerzeniu png, jpg, jpeg')  # Akceptowane formaty to png, jpg, jpeg
            return None  # Zwrócenie None, gdy plik jest nieprawidłowy

# Główna funkcja aplikacji
def main():
    # Wywołanie funkcji obsługującej przesyłanie obrazów
    object = upload_image()
    
    if object:  # Sprawdzenie, czy przesłano poprawny obraz
        prediction = False  # Flaga informująca, czy predykcja została wykonana
        image_obj = Image.open(object['file'])  # Wczytanie obrazu przy użyciu PIL
        
        # Podział interfejsu na dwie kolumny
        col1, col2 = st.columns(2)
        
        with col1:
            st.info('Podgląd obrazu')  # Informacja o podglądzie obrazu
            st.image(image_obj)  # Wyświetlenie przesłanego obrazu w pierwszej kolumnie
            
        with col2:
            st.subheader('Szczegóły obrazu')  # Podtytuł dla szczegółów pliku
            st.json(object['details'])  # Wyświetlenie szczegółów pliku w formacie JSON
            button = st.button('Wykonaj detekcję YOLO')  # Przycisk do wywołania predykcji
            if button:  # Sprawdzenie, czy przycisk został kliknięty
                with st.spinner("""Getting Objects from image. Please wait"""):  # Wyświetlenie spinnera
                    # Konwersja obrazu na tablicę NumPy
                    image_array = np.array(image_obj)
                    # Wywołanie predykcji przy użyciu modelu YOLO
                    pred_img = yolo.predictions(image_array)
                    # Konwersja przewidywanego obrazu z powrotem na obiekt obrazu
                    pred_img_obj = Image.fromarray(pred_img)
                    prediction = True  # Ustawienie flagi predykcji na True
                
        if prediction:  # Jeśli predykcja została wykonana
            st.subheader("Przewidywany obraz")  # Podtytuł dla przewidywanego obrazu
            st.caption("Obiekt po detekcji YOLO V5 model")  # Opis wyniku
            st.image(pred_img_obj)  # Wyświetlenie przewidywanego obrazu
    
# Uruchomienie funkcji głównej, jeśli skrypt jest wywołany bezpośrednio
if __name__ == "__main__":
    main()  # Wywołanie głównej funkcji aplikacji
