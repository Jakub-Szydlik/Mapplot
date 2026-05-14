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
Model dba o ciągłość pochodnych, co eliminuce nienaturalne 'falowanie' wykresu.
""")

# ==========================================
# PANEL BOCZNY (USTAWIENIA)
# ==========================================
st.sidebar.header(" Ustawienia i Dane")

# 1. Wgrywanie pliku
uploaded_file = st.sidebar.file_uploader("Wgraj plik TXT z danymi", type="txt")

st.sidebar.markdown("---")
st.sidebar.subheader("Kontrola Gładkości (S)")

# 2. Mechanizm precyzyjnego wpisywania i suwaka
s_manual = st.sidebar.number_input(
    "Wpisz dokładną wartość S:",
    min_value=0.0000,
    max_value=10.0000, # Zwiększono zakres dla danych z niepewnościami
    value=0.0050,
    step=0.0001,
    format="%.4f"
)

s_slider = st.sidebar.slider(
    "Lub przesuń suwak:",
    min_value=0.0000,
    max_value=1.0000,
    value=s_manual if s_manual <= 1.0 else 1.0,
    step=0.0001,
    format="%.4f"
)

s_final = s_manual

st.sidebar.info("""
**Instrukcja:**
- **S = 0**: Linia przechodzi przez każdy punkt (Interpolacja).
- **S > 0**: Linia wygładza szum i tworzy naturalne łuki (Aproksymacja).
- Jeśli podasz 3 kolumny, model uwzględni **niepewności** przy dopasowaniu.
""")

# ==========================================
# LOGIKA GŁÓWNA I WYKRES
# ==========================================
if uploaded_file is not None:
    try:
        # Wczytanie danych z TXT
        df = pd.read_csv(uploaded_file, sep=None, engine='python', header=None)
        
        if df.shape[1] < 2:
            st.error("Plik musi mieć przynajmniej dwie kolumny danych (X i Y).")
        else:
            x_pts = df[0].values
            y_pts = df[1].values
            
            # Obsługa niepewności (kolumna 3)
            y_err = None
            weights = None
            if df.shape[1] >= 3:
                y_err = df[2].values
                # Wagi dla splajnu (zazwyczaj 1/sigma)
                # Unikamy dzielenia przez zero
                weights = 1.0 / np.where(y_err == 0, 1e-10, y_err)

            # Sortowanie danych
            sort_idx = x_pts.argsort()
            x_pts, y_pts = x_pts[sort_idx], y_pts[sort_idx]
            if y_err is not None:
                y_err = y_err[sort_idx]
                weights = weights[sort_idx]

            # Obliczenia modelu: UnivariateSpline (w=weights uwzględnia niepewności)
            model = UnivariateSpline(x_pts, y_pts, w=weights, k=3, s=s_final)
            
            x_smooth = np.linspace(x_pts.min(), x_pts.max(), 1000)
            y_smooth = model(x_smooth)

            # Rysowanie wykresu
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Linia modelu
            ax.plot(x_smooth, y_smooth, color='#1e90ff', linewidth=3, label=f'Naturalny Splajn (s={s_final:.4f})', zorder=4)
            
            # Oryginalne punkty z niepewnościami
            if y_err is not None:
                ax.errorbar(x_pts, y_pts, yerr=y_err, fmt='o', color='red', 
                            ecolor='black', elinewidth=1, capsize=3, 
                            markeredgecolor='black', ms=8, label='Punkty pomiarowe + Niepewność', zorder=3)
            else:
                ax.scatter(x_pts, y_pts, color='red', edgecolor='black', s=60, zorder=3, label='Punkty pomiarowe')
            
            # Estetyka wykresu
            ax.set_xlabel("Oś X")
            ax.set_ylabel("Oś Y")
            ax.set_title("Analiza aproksymacji z uwzględnieniem niepewności")
            ax.grid(True, linestyle=':', alpha=0.7)
            ax.legend()

            st.pyplot(fig)
            
            with st.expander("Zobacz wczytane dane"):
                cols = {0: "Oś X", 1: "Oś Y", 2: "Niepewność (Y error)"}
                st.dataframe(df.rename(columns=cols))

    except Exception as e:
        st.error(f"Wystąpił błąd podczas przetwarzania pliku: {e}")
else:
    st.info("👈 Zacznij od wgrania pliku .txt w panelu bocznym.")
    
    st.write("### Jak powinien wyglądać plik .txt?")
    st.write("Możesz podać 2 kolumny (X, Y) lub 3 kolumny (X, Y, Niepewność):")
    st.code("""
119 4.53 0.10
129 4.72 0.05
139 4.87 0.12
149 4.98 0.08
159 5.03 0.15
    """, language="text")

st.markdown("---")
st.caption("Aplikacja stworzona do modelowania gładkich trendów danych pomiarowych.")