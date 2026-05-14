import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
import pandas as pd

st.title("📈 Twój Osobisty Aproksymator")

# 1. Przycisk do wgrywania pliku
uploaded_file = st.file_uploader("Wgraj plik TXT (kolumny oddzielone spacją lub przecinkiem)", type="txt")

if uploaded_file is not None:
    # Wczytanie danych z pliku
    try:
        # Zakładamy, że plik ma dwie kolumny: X i Y
        df = pd.read_csv(uploaded_file, sep=None, engine='python', header=None)
        x_points = df[0].values
        y_points = df[1].values
        
        # Sortowanie danych (ważne dla splajnów!)
        idx = x_points.argsort()
        x_points, y_points = x_points[idx], y_points[idx]

        # 2. Interfejs suwaka
        s_val = st.sidebar.slider("Płynność (s)", 0.0, 1.0, 0.005, format="%.4f")
        st.sidebar.write("Im mniejsze S, tym linia bliżej punktów.")

        # 3. Modelowanie
        model = UnivariateSpline(x_points, y_points, k=3, s=s_val)
        x_smooth = np.linspace(x_points.min(), x_points.max(), 500)
        y_smooth = model(x_smooth)

        # 4. Wykres
        fig, ax = plt.subplots()
        ax.plot(x_smooth, y_smooth, color='#e67e22', linewidth=3, label='Twój Model')
        ax.scatter(x_points, y_points, color='red', alpha=0.6, label='Dane z pliku')
        ax.set_title("Wynik aproksymacji")
        ax.legend()
        
        st.pyplot(fig)
        
        # Opcja: pokazanie tabeli z danymi
        if st.checkbox("Pokaż surowe dane"):
            st.write(df)

    except Exception as e:
        st.error(f"Błąd podczas czytania pliku: {e}")
else:
    st.info("Czekam na plik... Wgraj dokument .txt, w którym są dwie kolumny liczb.")