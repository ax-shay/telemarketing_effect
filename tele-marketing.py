import streamlit as st
import home
import eda
import model
import defSessionState as ss

st.set_page_config(
    layout="centered",
    initial_sidebar_state="expanded",
    page_title="UDCDSA Captsone Project: Predicting Effect of Bank Telemarketing (Term Deposit Sale)",
    page_icon=None,  # String, anything supported by st.image, or None.
)

PAGES = {
    "Home": home,
    "EDA": eda,
    "Model": model,
    "Prediction": None,
}


def main():
    state = ss._get_state()

    st.sidebar.title("Navigation")
    selection = st.sidebar.radio("Go to", list(PAGES.keys()))

    page = PAGES[selection].write(state)

    st.sidebar.title("Team-2")
    st.sidebar.info(
        """
        Akshay Shembekar, Courtney Golding, Jonathan Littleton, Komal Handa, Sambhavi Parajuli 
        """
    )

    state.sync()


if __name__ == "__main__":
    main()
