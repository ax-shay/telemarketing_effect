import streamlit as st
import joblib
import pandas as pd
import zipfile


def write(state):

    st.title('Making Predictions')
    st.markdown('')
    st.markdown('**Please provide client information**:')  # you can use markdown like this

    # load models
    pkl_zp = zipfile.ZipFile('clf-best.pickle.zip')
    pkl_zp.extractall()
    tree_clf = joblib.load('clf-best.pickle')


    # get inputs

    age = int(st.number_input('Age:', 0, 100, 20))
    job = st.selectbox('Job', ['admin.', 'blue-collar', 'technician', 'services', 'management',
                               'retired', 'entrepreneur', 'self-employed', 'housemaid',
                               'unemployed', 'student'])
    marital = st.selectbox('Marital', ['married', 'single', 'divorced'])
    education = st.selectbox('Education', ['university.degree', 'high.school', 'professional.course',
                                           'unknown', 'illiterate', 'basic'])
    default = st.selectbox('Has Defaulted in the past', ['yes', 'no'])
    campaign = int(st.number_input('# of times potential client was contacted:', 0, 100, 0))
    previous = int(st.number_input('# of times potential client was contacted, previously:', 0, 10, 0))
    poutcome = st.selectbox('Outcome of the previous marketing campaign', ['success', 'failure', 'nonexistent'])
    has_loan = int(st.number_input('Has Housing or Personal Loan (1 = Yes, 0 = No):', 0, 1, 0))
    # sib_sp = int(st.number_input('# of siblings / spouses aboard:', 0, 10, 0))
    #par_ch = int(st.number_input('# of parents / children aboard:', 0, 10, 0))
    # pclass = st.selectbox('Ticket class (1 = 1st, 2 = 2nd, 3 = 3rd)', [1, 2, 3])
    # fare = int(st.number_input('# of parents / children aboard:', 0, 100, 0))
    #embarked = st.selectbox('Port of Embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)', ['C', 'Q', 'S'])

    # this is how to dynamically change text
    prediction_state = st.markdown('calculating...')

    client = pd.DataFrame(
        {
            'age': [age],
            'job': [job],
            'marital': [marital],
            'education': [education],
            'default': [default],
            'campaign': [campaign],
            'previous': [previous],
            'poutcome': [poutcome],
            'emp.var.rate': [1.1],
            'cons.price.idx': [93.918],
            'cons.conf.idx': [-42.7],
            'euribor3m': [4.857],
            'nr.employed': [5191],
            'has_loan': [has_loan]
        }
    )

    y_pred = tree_clf.predict(client)

    if y_pred[0] == 0:
        msg = 'This client is predicted to : **Not Subscribe** to Term Deposit'
    else:
        msg = 'This client is predicted to : **Subscribe** to Term Deposit'

    prediction_state.markdown(msg)
