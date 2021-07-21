import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image

# Data Processing
import pandas as pd

image1 = Image.open('images/Feature_Imp.JPG')

st.set_option('deprecation.showPyplotGlobalUse', False)
template_dict = {'Decision Tree': 0, 'Logistic Regression': 1, 'Random Forest': 2, 'XGBoost': 3}
inputs = {}

data = {'Model': ['Random Forest', 'Random Forest', 'Random Forest', 'Random Forest',
                  'Decision Tree', 'Decision Tree', 'Decision Tree', 'Decision Tree',
                  'Logistic Regression', 'Logistic Regression', 'Logistic Regression', 'Logistic Regression',
                  'XGBoost', 'XGBoost', 'XGBoost', 'XGBoost'],
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1',
                   'Accuracy', 'Precision', 'Recall', 'F1',
                   'Accuracy', 'Precision', 'Recall', 'F1',
                   'Accuracy', 'Precision', 'Recall', 'F1'],
        'Score': [93.1, 89.9, 96.9, 93.3,
                  91.2, 87.3, 96.1, 91.5,
                  73.1, 74.8, 68.3,71.4,
                  76.1, 82.4, 65.6, 73.0]
        }
df = pd.DataFrame(data, columns=['Model', 'Metric', 'Score'])


def write(state):
    st.title('Model')
    st.markdown('This section provides the ability to select different models and analyze the importance of different features.')
    col1, col2 = st.beta_columns(2)
    with col1:
        inputs["model1"] = st.selectbox(
            "select Model-1", list(template_dict.keys())
        )

        # if inputs["model1"] == 'Decision Tree':
        df1 = df.loc[df['Model'] == inputs["model1"]]
        df1.plot.barh(x='Metric', y='Score', title=inputs["model1"], color='green')
        st.pyplot()
        if inputs["model1"] == 'Random Forest':
            st.image(image1)

    with col2:
        inputs["model2"] = st.selectbox(
            "select Model-2", list(template_dict.keys())
        )
        df2 = df.loc[df['Model'] == inputs["model2"]]
        df2.plot.barh(x='Metric', y='Score', title=inputs["model2"], color='blue')
        st.pyplot()
        if inputs["model2"] == 'Random Forest':
            st.image(image1)

    st.write('')
    sp1, new_row, sp2 = st.beta_columns((0.1, 1, 0.1))
    df_disp = pd.pivot_table(df, values='Score', index=['Model'], columns='Metric').reset_index()
    with new_row:
        st.header("**Data**")
        st.dataframe(df_disp)
