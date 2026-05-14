import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
import pandas as pd

# ==========================================
# KONFIGURACJA STRONY
# ==========================================
st.set_page_config(page_title="Natural Curve Pro", layout="wide")

st.title(" Uniwersalny Aproksymator Wykresów")
st.markdown("""
Ta aplikacja dopasowuje gładką krzywą do Twoich danych przy użyciu **splajnów sześciennych**.
""")

# ==========================================
# PANEL BOCZNY (USTAWIENIA)
# ==========================================
st.sidebar.header(" Ustawienia i Dane")

uploaded_file = st.sidebar.file_uploader("Wgraj plik TXT z danymi", type="txt")

st.sidebar.markdown("---")
st.sidebar.subheader("Kontrola Gładkości (S)")

# Przywracamy precyzyjny suwak do 1.0, jak w oryginale
s_final = st.sidebar.number_input(
    "Wpisz dokładną wartość S:",
    min_value=0.0000,
    max_value=1.0000,
    value=0.0050,
    step=0.0001,
    format="%.4f"
)

st.sidebar.info("""
**Instrukcja:**
- **S = 0**: Interpolacja (linia przez punkty).
- **S > 0**: Aproksymacja (wygładzanie).
""")

# ==========================================
# LOGIKA GŁÓWNA
# ==========================================
if uploaded_file is not None:
    try:
        # Pancerne wczytywanie (obsługuje dowolną liczbę spacji/tabulatorów)
        df = pd.read_csv(uploaded_file, sep=r'\s+', engine='python', header=None)
        
        if df.shape[1] < 2:
            st.error("Plik musi mieć przynajmniej dwie kolumny.")
        else:
            x_pts = df[0].values
            y_pts = df[1].values
            
            # Sortowanie
            sort_idx = x_pts.argsort()
            x_pts, y_pts = x_pts[sort_idx], y_pts[sort_idx]

            # OBSŁUGA NIEPEWNOŚCI BEZ PSUUCIA S
            weights = None
            y_err = None
            if df.shape[1] >= 3:
                y_err = df[2].values[sort_idx]
                # Normalizujemy wagi, aby ich średnia wynosiła 1.
                # Dzięki temu S zachowuje się tak samo jak w Twoim oryginalnym kodzie!
                raw_weights = 1.0 / np.where(y_err <= 0, 1e-10, y_err)
                weights = raw_weights / np.mean(raw_weights) 

            # Obliczenia modelu (Logika identyczna z oryginałem)
            model = UnivariateSpline(x_pts, y_pts, w=weights, k=3, s=s_final)
            
            x_smooth = np.linspace(x_pts.min(), x_pts.max(), 1000)
            y_smooth = model(x_smooth)

            # Wykres
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(x_smooth, y_smooth, color='#1e90ff', linewidth=3, label=f'Splajn (s={s_final:.4f})')
            
            if y_err is not None:
                ax.errorbar(x_pts, y_pts, yerr=y_err, fmt='o', color='red', ecolor='black', capsize=3, label='Dane + Niepewność')
            else:
                ax.scatter(x_pts, y_pts, color='red', edgecolor='black', s=60, label='Punkty')
            
            ax.grid(True, linestyle=':', alpha=0.7)
            ax.legend()
            st.pyplot(fig)

    except Exception as e:
        st.error(f"Błąd: {e}")