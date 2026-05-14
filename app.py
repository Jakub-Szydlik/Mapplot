import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
import pandas as pd

# ==========================================
# KONFIGURACJA
# ==========================================
st.set_page_config(page_title="Natural Curve Pro", layout="wide")
st.title("🚀 Uniwersalny Aproksymator Wykresów")

# ==========================================
# PANEL BOCZNY
# ==========================================
st.sidebar.header("📂 Ustawienia i Dane")
uploaded_file = st.sidebar.file_uploader("Wgraj plik TXT", type="txt")

st.sidebar.markdown("---")
st.sidebar.subheader("📈 Kontrola Gładkości (S)")

s_final = st.sidebar.number_input(
    "Wartość S (Gładkość):",
    min_value=0.0000,
    max_value=1.0000,
    value=0.0050,
    step=0.0001,
    format="%.4f"
)

# PRZYWRÓCONA INSTRUKCJA DOTYCZĄCA NIEPEWNOŚCI
st.sidebar.info("""
**💡 Jak dodać niepewności?**
Aplikacja obsługuje dwa formaty plików tekstowych:
1. **2 kolumny:** (X, Y) – standardowa aproksymacja.
2. **3 kolumny:** (X, Y, Niepewność) – model uwzględni słupki błędów. 

**Wskazówka:** Punkty z większą niepewnością (3. kolumna) mają mniejszy wpływ na przebieg linii, co pozwala uzyskać gładszy trend.
""")

# ==========================================
# LOGIKA GŁÓWNA
# ==========================================
if uploaded_file is not None:
    try:
        # Pancerne wczytywanie (dowolna ilość spacji/tabów)
        df = pd.read_csv(uploaded_file, sep=r'\s+', engine='python', header=None)
        
        if df.shape[1] >= 2:
            x_pts = df[0].values
            y_pts = df[1].values
            
            # Sortowanie danych
            sort_idx = x_pts.argsort()
            x_pts, y_pts = x_pts[sort_idx], y_pts[sort_idx]

            # Obsługa wag (3. kolumna)
            w_array = None
            y_err = None
            if df.shape[1] >= 3:
                y_err = df[2].values[sort_idx]
                # Wagi = 1/niepewność (z ograniczeniem, by nie tworzyć "garbów")
                w_array = 1.0 / np.clip(y_err, 0.01, None) 

            # Model splajnu sześciennego
            model = UnivariateSpline(x_pts, y_pts, w=w_array, k=3, s=s_final)
            
            x_smooth = np.linspace(x_pts.min(), x_pts.max(), 1000)
            y_smooth = model(x_smooth)

            # Rysowanie wykresu
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(x_smooth, y_smooth, color='#1e90ff', linewidth=3, label=f'Krzywa (S={s_final:.4f})')
            
            if y_err is not None:
                ax.errorbar(x_pts, y_pts, yerr=y_err, fmt='o', color='red', ecolor='black', capsize=2, label='Dane + Niepewność')
            else:
                ax.scatter(x_pts, y_pts, color='red', edgecolor='black', s=60, label='Punkty pomiarowe')
            
            ax.set_xlabel("Oś X")
            ax.set_ylabel("Oś Y")
            ax.grid(True, linestyle=':', alpha=0.6)
            ax.legend()
            st.pyplot(fig)
            
            with st.expander("🔍 Podgląd danych"):
                st.dataframe(df)
            
    except Exception as e:
        st.error(f"Błąd podczas przetwarzania: {e}")
else:
    st.info("👈 Wgraj plik .txt, aby zobaczyć wykres.")
    st.write("### Przykład pliku z niepewnością:")
    st.code("0.0  0.5  0.1\n1.0  2.2  0.05\n2.0  3.8  0.2", language="text")

st.markdown("---")
st.caption("Aplikacja do modelowania gładkich trendów z opcjonalną analizą niepewności pomiarowej.")