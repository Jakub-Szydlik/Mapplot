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
Ta aplikacja dopasowuje gładką krzywą do danych przy użyciu **splajnów sześciennych**.
Model uwzględnia niepewności pomiarowe (wagi) i pozwala na pełną kontrolę nad gładkością linii.
""")

# ==========================================
# PANEL BOCZNY (USTAWIENIA)
# ==========================================
st.sidebar.header("📂 Ustawienia i Dane")

# 1. Wgrywanie pliku
uploaded_file = st.sidebar.file_uploader("Wgraj plik TXT z danymi", type="txt")

st.sidebar.markdown("---")
st.sidebar.subheader("📈 Kontrola Gładkości (S)")

# 2. Mechanizm sterowania parametrem S
st.sidebar.markdown("""
**Parametr S (Smoothing):**
*   **S = 0**: Ścisła interpolacja (linia "pędzi" od punktu do punktu).
*   **S = 1-10**: Naturalne wygładzenie (złoty środek).
*   **S > 10**: Silna aproksymacja (krzywa staje się "sztywna" i ignoruje lokalne skoki).
""")

s_manual = st.sidebar.number_input(
    "Wpisz dokładną wartość S (0 - 100):",
    min_value=0.0000,
    max_value=100.0000,
    value=0.5000,
    step=0.01,
    format="%.4f"
)

s_slider = st.sidebar.slider(
    "Szybki suwak (zakres 0-20):",
    min_value=0.0000,
    max_value=20.0000,
    value=s_manual if s_manual <= 20.0 else 20.0,
    step=0.01,
    format="%.4f"
)

# Priorytet ma wpis ręczny, jeśli użytkownik go zmienił
s_final = s_manual

# ==========================================
# LOGIKA GŁÓWNA I WYKRES
# ==========================================
if uploaded_file is not None:
    try:
        # Wczytanie danych z TXT (automatyczne wykrywanie separatora)
        df = pd.read_csv(uploaded_file, sep=None, engine='python', header=None)
        
        if df.shape[1] < 2:
            st.error("❌ Plik musi mieć przynajmniej dwie kolumny (X i Y).")
        else:
            x_pts = df[0].values
            y_pts = df[1].values
            
            # Obsługa niepewności (kolumna 3)
            y_err = None
            weights = None
            if df.shape[1] >= 3:
                y_err = df[2].values
                # Wagi w UnivariateSpline to 1/sigma. Unikamy dzielenia przez zero.
                weights = 1.0 / np.where(y_err <= 0, 1e-10, y_err)
                st.sidebar.success("✅ Wykryto kolumnę niepewności!")
            else:
                st.sidebar.info("ℹ️ Brak kolumny niepewności - używam wag równych 1.")

            # Sortowanie danych (wymóg konieczny dla UnivariateSpline)
            sort_idx = x_pts.argsort()
            x_pts = x_pts[sort_idx]
            y_pts = y_pts[sort_idx]
            if weights is not None:
                weights = weights[sort_idx]
                y_err = y_err[sort_idx]

            # Obliczenia modelu
            # w=weights sprawia, że punkty z dużą niepewnością mają mniejszy wpływ na krzywą
            model = UnivariateSpline(x_pts, y_pts, w=weights, k=3, s=s_final)
            
            # Generowanie gęstej siatki dla gładkiego rysunku
            x_smooth = np.linspace(x_pts.min(), x_pts.max(), 1000)
            y_smooth = model(x_smooth)

            # Rysowanie wykresu
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Linia aproksymacji
            ax.plot(x_smooth, y_smooth, color='#1e90ff', linewidth=3, 
                    label=f'Splajn (S={s_final:.4f})', zorder=4)
            
            # Punkty pomiarowe
            if y_err is not None:
                ax.errorbar(x_pts, y_pts, yerr=y_err, fmt='o', color='#e63946', 
                            ecolor='#1d3557', elinewidth=1, capsize=3, 
                            markeredgecolor='black', ms=7, label='Dane z niepewnością', zorder=3)
            else:
                ax.scatter(x_pts, y_pts, color='#e63946', edgecolor='black', 
                           s=60, zorder=3, label='Punkty pomiarowe')
            
            # Estetyka
            ax.set_xlabel("Oś X", fontsize=12)
            ax.set_ylabel("Oś Y", fontsize=12)
            ax.set_title("Analiza Aproksymacji Krzywej", fontsize=14, fontweight='bold')
            ax.grid(True, linestyle='--', alpha=0.5)
            ax.legend()

            # Wyświetlenie wykresu
            st.pyplot(fig)
            
            # Podgląd danych
            with st.expander("🔍 Zobacz tabelę wczytanych danych"):
                cols = {0: "X (Wartość)", 1: "Y (Wynik)", 2: "Niepewność (Błąd Y)"}
                st.dataframe(df.rename(columns=cols), use_container_width=True)

    except Exception as e:
        st.error(f"❌ Błąd krytyczny: {e}")
else:
    # Ekran powitalny
    st.info("👈 Wgraj plik tekstowy w panelu bocznym, aby rozpocząć analizę.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("### Format pliku (2 kolumny):")
        st.code("1.0  10.5\n2.0  15.2\n3.0  12.8", language="text")
    with col2:
        st.write("### Format pliku (3 kolumny):")
        st.code("1.0  10.5  0.5\n2.0  15.2  0.1\n3.0  12.8  0.9", language="text")

st.markdown("---")
st.caption("Aplikacja do profesjonalnego modelowania trendów. Parametr S pozwala na płynne przejście od interpolacji do luźnej aproksymacji trendu.")