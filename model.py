import numpy as np
import pandas as pd
import io
# from PIL import Image
# image1 = Image.open('images/IHM Pandora.jpeg')

import streamlit as st

# Linear Algebra
import math
import numpy as np

# Data Processing
import pandas as pd

# Data Visualization
from matplotlib import style
import matplotlib.pyplot as plt

# Modelling Algorithm
import seaborn as sns
from sklearn import tree
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

template_dict = {'Decision Tree': 0, 'Logistic Regression': 1, 'Random Forest': 2}
inputs = {}


def write(state):
    st.title('Model')
    st.markdown('This section provides the ability to select different models and analyze the importance of different hyperparameters.')
    inputs["model"] = st.selectbox(
        "Choose your Model", list(template_dict.keys())
    )

    if inputs["model"] == 'Random Forest':
        st.markdown('Work in Progress..')

    elif inputs["model"] == 'Logistic Regression':
        st.markdown('Logistic Regression..')

    elif inputs["model"] == 'Decision Tree':
        num_features = ['age', 'campaign', 'previous', 'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed', 'has_loan']
        cat_features = ['job', 'marital', 'education', 'default', 'month', 'poutcome']

        num_pipeline = Pipeline(
            steps=[
                ('scaler', StandardScaler()),
            ]
        )

        cat_pipeline = Pipeline(
            steps=[
                ('onehot', OneHotEncoder()),
            ]
        )

        from sklearn.compose import ColumnTransformer

        preprocessor = ColumnTransformer(
            transformers=[
                ('num_pipeline', num_pipeline, num_features),
                ('cat_pipeline', cat_pipeline, cat_features),
            ]
        )

        from sklearn.tree import DecisionTreeClassifier

        pipeline_dt = Pipeline(
            steps=[
                ('preprocessor', preprocessor),
                ('clf_dt', DecisionTreeClassifier()),
            ]
        )

        st.subheader('Train - Test Split:')
        st.markdown('* Stratifying the data and allocating 80-20% for Train-Test, respectively.')

        from sklearn.model_selection import train_test_split
        df_data_clean = pd.read_csv('/Users/akshayshembekar/Documents/Projects/JupyterLabInstall/UDCDSA/BUAD625/EDA/AS/df_data_clean.csv')
        X = df_data_clean.drop(['subscribed'], axis=1)
        y = df_data_clean['subscribed']

        # Clean this up, later. This is redundant code from eda.py
        class_count_0, class_count_1 = df_data_clean['subscribed'].value_counts()
        class_0 = df_data_clean[df_data_clean['subscribed'] == 0]
        class_1 = df_data_clean[df_data_clean['subscribed'] == 1]
        class_1_over = class_1.sample(class_count_0, replace=True)
        test_over = pd.concat([class_1_over, class_0], axis=0).reset_index()


        # X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=test_over['subscribed'], test_size=0.2)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        from sklearn.model_selection import GridSearchCV

        param_grid_dt = [
            {
                #         'preprocessor__num_pipeline__num_imputer__strategy': ['mean', 'median'],
                'clf_dt__criterion': ['gini', 'entropy'],
                'clf_dt__max_depth': [3, 4, 5, 6, 7, 9, 8, 10, 11, 12, 20, 30],
                'clf_dt__min_samples_split': [10, 20, 30, 40],
                'clf_dt__class_weight': ['balanced', None]

            }
        ]

        # # set up the grid search
        grid_search_dt = GridSearchCV(pipeline_dt, param_grid_dt, cv=10, scoring='accuracy')
        grid_search_dt.fit(X_train, y_train)
        st.write('Best score is: ', grid_search_dt.best_score_)

        clf_best = grid_search_dt.best_estimator_
        y_pred = clf_best.predict(X_test)
        from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

        st.write(f'Accuracy Score : {accuracy_score(y_test, y_pred)}')
        st.write(f'Precision Score : {precision_score(y_test, y_pred)}')
        st.write(f'Recall Score : {recall_score(y_test, y_pred)}')
        st.write(f'F1 Score : {f1_score(y_test, y_pred)}')

        # grid_search_dt.best_params_
        # sorted(grid_search_dt.cv_results_.keys())


