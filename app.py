import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
import pandas as pd

# ==========================================
# KONFIGURACJA
# ==========================================
st.set_page_config(page_title="Natural Curve Pro", layout="wide")
st.title("  Uniwersalny Aproksymator Wykresów")

# ==========================================
# PANEL BOCZNY
# ==========================================
st.sidebar.header("  Ustawienia i Dane")
uploaded_file = st.sidebar.file_uploader("Wgraj plik TXT", type="txt")

st.sidebar.markdown("---")
st.sidebar.subheader("  Opisy Osi Wykresu")

# Nowe pola do wpisywania nazw osi
x_label_input = st.sidebar.text_input("Nazwa osi X:", value="Oś X")
y_label_input = st.sidebar.text_input("Nazwa osi Y:", value="Oś Y")

st.sidebar.markdown("---")
st.sidebar.subheader("  Kontrola Gładkości (S)")

# Wyjaśnienie mechanizmu S
st.sidebar.warning("""
**Dlaczego S musi być duże?**
Parametr S to suma kwadratów odchyleń. Jeśli Twoje dane Y są rzędu 30-40, musisz ustawić S na poziomie 100-500, aby zobaczyć wygładzenie.
""")

# Pole do wpisywania wartości S
s_final = st.sidebar.number_input(
    "Wpisz wartość S (Gładkość):",
    min_value=0.0,
    max_value=100000.0, # Bardzo duży zakres dla każdej skali danych
    value=0.0,
    step=0.001,
    format="%.2f"
)

# ==========================================
# LOGIKA GŁÓWNA
# ==========================================
if uploaded_file is not None:
    try:
        # Wczytywanie z obsługą dowolnych odstępów
        df = pd.read_csv(uploaded_file, sep=r'\s+', engine='python', header=None)
        
        if df.shape[1] >= 2:
            x_pts = df[0].values
            y_pts = df[1].values
            
            # Sortowanie (kluczowe dla splajnów)
            sort_idx = x_pts.argsort()
            x_pts, y_pts = x_pts[sort_idx], y_pts[sort_idx]

            # Obsługa 3 kolumny (Niepewność)
            w_array = None
            y_err = None
            if df.shape[1] >= 3:
                y_err = df[2].values[sort_idx]
                # Wagi zapobiegające "wariowaniu" splajnu
                w_array = 1.0 / np.clip(y_err, 0.001, None)

            # MODEL - UnivariateSpline
            # k=3 (sześcienny), s=s_final (klucz do gładkości)
            model = UnivariateSpline(x_pts, y_pts, w=w_array, k=3, s=s_final)
            
            x_smooth = np.linspace(x_pts.min(), x_pts.max(), 1000)
            y_smooth = model(x_smooth)

            # WYKRES
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(x_smooth, y_smooth, color='#1e90ff', linewidth=3, label=f'Krzywa Aproksymowana (S={s_final})')
            
            if y_err is not None:
                ax.errorbar(x_pts, y_pts, yerr=y_err, fmt='o', color='red', ecolor='black', capsize=2, label='Dane + Niepewność')
            else:
                ax.scatter(x_pts, y_pts, color='red', edgecolor='black', s=60, label='Punkty pomiarowe')
            
            # Zastosowanie dynamicznych nazw osi z panelu bocznego
            ax.set_xlabel(x_label_input if x_label_input else "Oś X")
            ax.set_ylabel(y_label_input if y_label_input else "Oś Y")
            
            ax.grid(True, linestyle=':', alpha=0.6)
            ax.legend()
            st.pyplot(fig)
            
            # Statystyki dla użytkownika
            st.write(f"**Aktualny poziom dopasowania:** Przy S={s_final}, model dopuszcza błąd kwadratowy o tej wartości.")
            
    except Exception as e:
        st.error(f"Błąd: {e}")
else:
    st.info("  Wgraj plik .txt, aby zacząć.")