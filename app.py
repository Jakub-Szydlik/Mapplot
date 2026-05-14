import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
import pandas as pd

# ==========================================
# KONFIGURACJA
# ==========================================
st.set_page_config(page_title="Natural Curve Pro", layout="wide")
st.title(" Uniwersalny Aproksymator Wykresów")

# ==========================================
# PANEL BOCZNY
# ==========================================
st.sidebar.header(" Ustawienia")
uploaded_file = st.sidebar.file_uploader("Wgraj plik TXT", type="txt")

# Przywrócony oryginalny, czuły zakres S
s_final = st.sidebar.number_input(
    "Wartość S (Gładkość):",
    min_value=0.0000,
    max_value=1.0000,
    value=0.0050,
    step=0.0001,
    format="%.4f"
)

# ==========================================
# LOGIKA
# ==========================================
if uploaded_file is not None:
    try:
        # Wczytywanie ignorujące nadmiarowe spacje
        df = pd.read_csv(uploaded_file, sep=r'\s+', engine='python', header=None)
        
        if df.shape[1] >= 2:
            x_pts = df[0].values
            y_pts = df[1].values
            
            # Sortowanie danych
            sort_idx = x_pts.argsort()
            x_pts, y_pts = x_pts[sort_idx], y_pts[sort_idx]

            # Obsługa wag - jeśli są 3 kolumny
            w_array = None
            y_err = None
            if df.shape[1] >= 3:
                y_err = df[2].values[sort_idx]
                # Używamy wag bezpośrednio, ale ograniczamy ich wpływ, by nie robiły "garbów"
                # Jeśli błąd jest mały, waga jest duża, ale bez przesady
                w_array = 1.0 / np.clip(y_err, 0.01, None) 

            # MODEL - dokładnie tak jak w Twoim pierwszym kodzie
            model = UnivariateSpline(x_pts, y_pts, w=w_array, k=3, s=s_final)
            
            x_smooth = np.linspace(x_pts.min(), x_pts.max(), 1000)
            y_smooth = model(x_smooth)

            # WYKRES
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(x_smooth, y_smooth, color='#1e90ff', linewidth=3, label='Krzywa Aproksymowana')
            
            if y_err is not None:
                ax.errorbar(x_pts, y_pts, yerr=y_err, fmt='o', color='red', ecolor='black', capsize=2, label='Dane + błąd')
            else:
                ax.scatter(x_pts, y_pts, color='red', edgecolor='black', s=60, label='Punkty')
            
            ax.grid(True, linestyle=':', alpha=0.6)
            ax.legend()
            st.pyplot(fig)
            
    except Exception as e:
        st.error(f"Coś poszło nie tak: {e}")