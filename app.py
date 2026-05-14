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

# Używamy st.number_input do ręcznego wpisania wartości

s_manual = st.sidebar.number_input(

    "Wpisz dokładną wartość S:",

    min_value=0.0000,

    max_value=1.0000,

    value=0.0050,

    step=0.0001,

    format="%.4f"

)



# Suwak pomocniczy (można nim sterować, jeśli nie wpisujemy ręcznie)

s_slider = st.sidebar.slider(

    "Lub przesuń suwak:",

    min_value=0.0000,

    max_value=0.1000,

    value=s_manual if s_manual <= 0.1 else 0.1,

    step=0.0001,

    format="%.4f"

)



# Decydujemy, którą wartość przyjąć (priorytet ma wpis ręczny, jeśli różni się od suwaka)

s_final = s_manual



st.sidebar.info("""

**Instrukcja:**

- **S = 0**: Linia przechodzi przez każdy punkt (Interpolacja).

- **S > 0**: Linia wygładza szum i tworzy naturalne łuki (Aproksymacja).

""")



# ==========================================

# LOGIKA GŁÓWNA I WYKRES

# ==========================================

if uploaded_file is not None:

    try:

        # Wczytanie danych z TXT (kolumny rozdzielone spacją/przecinkiem)

        df = pd.read_csv(uploaded_file, sep=None, engine='python', header=None)

       

        # Sprawdzamy czy mamy przynajmniej 2 kolumny

        if df.shape[1] < 2:

            st.error("Plik musi mieć dwie kolumny danych (X i Y).")

        else:

            x_pts = df[0].values

            y_pts = df[1].values

           

            # Sortowanie danych (wymóg matematyczny splajnów)

            sort_idx = x_pts.argsort()

            x_pts, y_pts = x_pts[sort_idx], y_pts[sort_idx]



            # Obliczenia modelu: UnivariateSpline (k=3 oznacza stopień sześcienny)

            model = UnivariateSpline(x_pts, y_pts, k=3, s=s_final)

           

            # Generowanie gęstej siatki punktów dla gładkiej linii

            x_smooth = np.linspace(x_pts.min(), x_pts.max(), 1000)

            y_smooth = model(x_smooth)



            # Rysowanie wykresu

            fig, ax = plt.subplots(figsize=(12, 6))

           

            # Linia modelu (Gładka)

            ax.plot(x_smooth, y_smooth, color='#1e90ff', linewidth=3, label=f'Naturalny Splajn (s={s_final:.4f})')

           

            # Oryginalne punkty

            ax.scatter(x_pts, y_pts, color='red', edgecolor='black', s=60, zorder=3, label='Punkty pomiarowe')

           

            # Estetyka wykresu

            ax.set_xlabel("Oś X")

            ax.set_ylabel("Oś Y")

            ax.set_title("Analiza aproksymacji najmniejszych kwadratów")

            ax.grid(True, linestyle=':', alpha=0.7)

            ax.legend()



            # Wyświetlenie wykresu w Streamlit

            st.pyplot(fig)

           

            # Dodatkowa opcja podejrzenia danych

            with st.expander("Zobacz wczytane dane"):

                st.dataframe(df.rename(columns={0: "Oś X", 1: "Oś Y"}))



    except Exception as e:

        st.error(f"Wystąpił błąd podczas przetwarzania pliku: {e}")

else:

    # Komunikat startowy

    st.info("👈 Zacznij od wgrania pliku .txt w panelu bocznym.")

   

    # Przykładowy widok jak powinien wyglądać plik

    st.write("### Jak powinien wyglądać plik .txt?")

    st.code("""

119 4.53

129 4.72

139 4.87

149 4.98

159 5.03

    """, language="text")



# Stopka

st.markdown("---")

st.caption("Aplikacja stworzona do modelowania gładkich trendów danych pomiarowych.")