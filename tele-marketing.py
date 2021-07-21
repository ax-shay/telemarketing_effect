import streamlit as st
import home
import eda
import model
import prediction
import defSessionState as ss

st.set_page_config(
    layout="centered",
    initial_sidebar_state="expanded",
    page_title="Team Jacks",
    page_icon=None,  # String, anything supported by st.image, or None.
)

PAGES = {
    "Home": home,
    "EDA": eda,
    "Model": model,
    "Prediction": prediction,
}


def main():
    state = ss._get_state()

    st.sidebar.title("Navigation")
    selection = st.sidebar.radio("Go to", list(PAGES.keys()))

    page = PAGES[selection].write(state)

    st.sidebar.title("Team-JACKS")
    st.sidebar.info(
        """
        **J**onathan Littleton, **A**kshay Shembekar, **C**ourtney Golding, **K**omal Handa, **S**ambhavi Parajuli 
        """
    )

    state.sync()


if __name__ == "__main__":
    main()
