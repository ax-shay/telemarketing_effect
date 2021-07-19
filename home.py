import numpy as np
import pandas as pd
# from PIL import Image
# image1 = Image.open('images/IHM Pandora.jpeg')

import streamlit as st


def write(state):

    st.title('UDCDSA Captsone Project: Predicting Effect of Bank Telemarketing (Term Deposit Sale)')

    st.markdown('')
    st.subheader("Goal: To predict if the banking clients will subscribe to a term deposit")
    st.markdown(' ')
    st.markdown(' ')
    st.subheader('**Overview**')
    st.markdown('This research project focuses on targeting through telemarketing phone calls to sell long-term deposits. Within a campaign, the human agents execute phone calls to a list of clients to sell the deposit (outbound) or, if meanwhile the client calls the contact-center for any other reason, he is asked to subscribe the deposit (inbound). Thus, the result is a binary unsuccessful or successful contact.')

    st.markdown('')
    st.subheader('**Data**')
    st.markdown('Dataset Link: https://archive.ics.uci.edu/ml/datasets/bank+marketing')
    st.markdown(''' There is one input dataset: 
    >1. **bank-additional-full.csv**: 
    - It has 41188 x 20 inputs, ordered by date (from May 2008 to November 2010)
    ''')
    st.markdown('The data is related with direct marketing campaigns of a Portuguese banking institution. The marketing campaigns were based on phone calls. Often, more than one contact to the same client was required, in order to access if the product (bank term deposit) would be ("yes") or not ("no") subscribed. ')

    st.markdown('')
    st.markdown('The smallest datasets are provided to test more computationally demanding machine learning algorithms (e.g., SVM). The classification goal is to predict if the client will subscribe (yes/no) a term deposit (variable y).')

    st.subheader('**Importance of EDA:**')

    st.markdown('''
    
    >1. Exploratory Data Analysis (EDA) helps understand the number of data attributes, their meaning and data types.

    >2. It prompts us to look for anomalies and identify potentially missing/null values, or defaulted values in some specific cases

    >3. It also helps ensure the data attributes are coherent i.e. have the same scale/grain so that the data is relevant  
    
    >4. A thorough analysis of attributes & their correlation can help identify potentially important features that influence the outcome class

    >5. Trends in data emerge and we get a better perspective of how different attributes interlace to create new features adding different perspective for modelling 
    
    ''')

    st.markdown('')
    st.subheader('Findings & Insights:')

    st.markdown('''
    
    >* There are quite a few [categorical](#categorical) attributes (10 to be precise), which will have to be converted into numeric ahead of using them in model(s). We'll be using Label Encoding & one-hot encoding for these purposes

    >* The [missing values](#missing) are few (max. extent being 21% for field default) and so these will be retained in the dataset and assumed to be defaulted as & when required

    >* Certain attributes (namely, emp.var.rate, cons.price.idx, cons.conf.idx, euribor3m and nr.employed) are at a different grain and will have to be [scaled](#scaled) to make them coherent  
    
    >* The outcome variable is vastly [imbalanced](#imbalanced) and stratification of data will be needed to improve accuracy of model

    >* Although the data attributes are weakly [correlated](#correlated) when taken independently, they are showing potential of combining to form composite new attributes (feature engineering) that will help in building better predictive models. 
    
    ''')

    # st.image(image1,use_column_width=True)#, caption='Arquitetura')

