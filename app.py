import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
import pandas as pd

# ==========================================
# KONFIGURACJA STRONY
# ==========================================
st.set_page_config(page_title="Natural Curve Pro", layout="wide")

st.title("🚀 Uniwersalny Aproksymator Wykresów")
st.markdown("""
Ta aplikacja dopasowuje gładką krzywą do danych przy użyciu **splajnów sześciennych**, uwzględniając **niepewności pomiarowe**.
""")

# ==========================================
# PANEL BOCZNY (USTAWIENIA)
# ==========================================
st.sidebar.header("📂 Ustawienia i Dane")

# 1. Wgrywanie pliku
uploaded_file = st.sidebar.file_uploader("Wgraj plik TXT z danymi", type="txt")

st.sidebar.markdown("---")
st.sidebar.subheader("📐 Niepewności pomiarowe (Y)")

uncertainty_type = st.sidebar.radio(
    "Źródło niepewności:",
    ["Stała wartość", "Z pliku (3. kolumna)"]
)

error_val = 0.0
if uncertainty_type == "Stała wartość":
    error_val = st.sidebar.number_input("Wpisz niepewność (±σ):", min_value=0.0, value=0.1, step=0.01, format="%.3f")

st.sidebar.markdown("---")
st.sidebar.subheader("📈 Kontrola Gładkości (S)")

s_final = st.sidebar.number_input(
    "Wartość wygładzania S:", 
    min_value=0.0000, 
    max_value=100.0, 
    value=0.0500, 
    step=0.001, 
    format="%.4f"
)

st.sidebar.info("""
- **S = 0**: Interpolacja (przechodzi przez punkty).
- **S > 0**: Aproksymacja (wygładza szum).
""")

# ==========================================
# LOGIKA GŁÓWNA I WYKRES
# ==========================================
if uploaded_file is not None:
    try:
        # Wczytanie danych
        df = pd.read_csv(uploaded_file, sep=None, engine='python', header=None)
        
        if df.shape[1] < 2:
            st.error("Plik musi mieć przynajmniej dwie kolumny (X i Y).")
        else:
            x_pts = df[0].values
            y_pts = df[1].values
            
            # Obsługa niepewności
            if uncertainty_type == "Z pliku (3. kolumna)" and df.shape[1] >= 3:
                y_err = df[2].values
            else:
                if uncertainty_type == "Z pliku (3. kolumna)":
                    st.warning("Nie znaleziono 3. kolumny. Używam domyślnej wartości 0.1")
                    y_err = np.full_like(y_pts, 0.1)
                else:
                    y_err = np.full_like(y_pts, error_val)

            # Sortowanie danych (wymóg UnivariateSpline)
            sort_idx = x_pts.argsort()
            x_pts, y_pts, y_err = x_pts[sort_idx], y_pts[sort_idx], y_err[sort_idx]

            # Obliczenia modelu
            # Wagi w spline to zazwyczaj 1/sigma
            weights = 1.0 / (y_err + 1e-9) # unikamy dzielenia przez zero
            model = UnivariateSpline(x_pts, y_pts, w=weights, k=3, s=s_final)
            
            x_smooth = np.linspace(x_pts.min(), x_pts.max(), 1000)
            y_smooth = model(x_smooth)

            # Rysowanie wykresu
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Słupki błędów i punkty
            ax.errorbar(x_pts, y_pts, yerr=y_err, fmt='o', color='red', 
                        ecolor='black', capsize=3, elinewidth=1, 
                        markeredgecolor='black', label='Punkty pomiarowe z ±σ')
            
            # Linia modelu
            ax.plot(x_smooth, y_smooth, color='#1e90ff', linewidth=3, 
                    label=f'Naturalny Splajn (s={s_final:.4f})')
            
            ax.set_xlabel("Oś X")
            ax.set_ylabel("Oś Y")
            ax.set_title("Aproksymacja z uwzględnieniem niepewności")
            ax.grid(True, linestyle=':', alpha=0.7)
            ax.legend()

            st.pyplot(fig)
            
            # Podgląd danych
            with st.expander("Zobacz wczytane dane"):
                display_df = df.copy()
                cols = {0: "Oś X", 1: "Oś Y", 2: "Niepewność (σ)"}
                st.dataframe(display_df.rename(columns=cols))

    except Exception as e:
        st.error(f"Wystąpił błąd: {e}")
else:
    st.info("👈 Wgraj plik .txt, aby rozpocząć.")
    
    st.write("### Format pliku z niepewnościami:")
    st.code("""
# Przykład dla 3 kolumn (X, Y, Sigma):
119  4.53  0.05
129  4.72  0.08
139  4.87  0.04
    """, language="text")

# Stopka
st.markdown("---")
st.caption("Aplikacja do analizy danych z uwzględnieniem wag wagowych (1/σ).")