import streamlit as st
import pickle
from datetime import datetime

startTime = datetime.now()

filename = "model.h5"
model = pickle.load(open(filename, 'rb'))

# print(f"Model expects {model.n_features_in_} features")
# if hasattr(model, 'feature_names_in_'):
#     print(f"Feature names: {model.feature_names_in_}")

sex_d = {0: "Kobieta", 1: "Mezczycna"}
pclass_d = {0: "Pierwsza", 1: "Druga", 2: "Trzecia"}
embark_d = {0: "Cherbourg", 1: "Queenstown", 2: "Southapton"}

def main() -> None:
    st.set_page_config(page_title="Czy przezylbys katastrofe?")
    overview = st.container()
    left, right = st.columns(2)
    prediction = st.container()
    st.image("https://media1.popsugar-assets.com/files/thumbor/7CwCuGAKxTrQ4wPyOBpKjSsd1JI/fit-in/2048xorig/filters:format_auto-!!-:strip_icc-!!-/2017/04/19/743/n/41542884/5429b59c8e78fbc4_MCDTITA_FE014_H_1_.JPG")

    with overview:
        st.title("Czy przezylbys katastrofe?")

    with left:
        sex_radio = st.radio("Plec", list(sex_d.keys()), format_func=lambda x: sex_d[x])
        pclass_radio = st.radio("Klasa", list(pclass_d.keys()), format_func=lambda x: pclass_d[x])
        embark_radio = st.radio("Port zaokretowania", list(embark_d.keys()), format_func=lambda x: embark_d[x])

    with right:
        age_slider = st.slider("wiek", value=50, min_value=1, max_value=100)
        sibsp_slider = st.slider("# Licba rodzenstwa i/lub partnera", min_value=0, max_value=8)
        parch_slider = st.slider("# Liczba rodzicow i/lub dzieci", min_value=0, max_value=6)
        fare_slider = st.slider("Cena biletu", min_value=0, max_value=500, step=10)

    # data = [] # to jest zadanie domowe
    # data = [[pclass_radio, sex_radio, age_slider, sibsp_slider, parch_slider, fare_slider, embark_radio, 0]]
    # data = [[pclass_radio, sex_radio, age_slider, sibsp_slider, parch_slider, fare_slider, embark_radio, sibsp_slider + parch_slider]]
    data = [[pclass_radio, age_slider, sibsp_slider, parch_slider, fare_slider, embark_radio, sex_radio, sex_radio]]

    survival = model.predict(data)
    s_confidance = model.predict_proba(data)

    with prediction:
        st.header("Czy dana osoba przezyje? {0}".format("Tak" if survival[0] == 1 else "Nie"))
        st.subheader("Pewnosc predykcji {0:.2f}%".format(s_confidance[0][survival[0] * 100]))

    
if __name__ == "__main__":
    main()