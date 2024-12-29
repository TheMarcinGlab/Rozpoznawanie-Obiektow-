import streamlit as st 

# Ustawienia strony Streamlit
st.set_page_config(page_title="Marcin Głąb",
                   layout='wide',
                   page_icon='./images/home.png')

# Tytuł aplikacji
st.title("Aplikacja do wykrywania obiektów YOLO V5")
st.caption('Ta aplikacja webowa demonstruje wykrywanie obiektów')

# Treść aplikacji
st.markdown("""
### Ta aplikacja wykrywa obiekty na obrazach
- Automatycznie wykrywa 20 obiektów na obrazie
- [Kliknij tutaj, aby przejść do aplikacji](/YOLO_for_image/)  

Poniżej znajduje się lista obiektów, które nasz model potrafi wykryć:
1. Osoba
2. Samochód
3. Krzesło
4. Butelka
5. Sofa
6. Rower
7. Koń
8. Łódź
9. Motocykl
10. Kot
11. TV
12. Krowa
13. Owca
14. Samolot
15. Pociąg
16. Stół jadalny
17. Autobus
18. Roślina doniczkowa
19. Ptak
20. Pies
           
            """)
