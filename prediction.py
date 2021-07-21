import streamlit as st
import joblib
import pandas as pd
import zipfile


def write(state):

    st.title('Making Predictions')
    st.markdown('')
    st.markdown('**Please provide Socio-Economic information**:')

    # load models
    pkl_zp = zipfile.ZipFile('clf-best.pickle.zip')
    pkl_zp.extractall()
    tree_clf = joblib.load('clf-best.pickle')


    # get inputs
    col1, col2 = st.beta_columns(2)
    st.markdown('**Please provide Client information**:')
    with col1:

        euribor3m = st.slider('Euribor 3M Rate (daily)', -1.0, 5.0, value=0.755)
        emp_var_rate = st.slider('Employment Variation Rate (quarterly)', -5.0, 10.0, value=-3.4)
        employed = st.slider('# of Employees (quarterly)', 100, 10000, value=5018)

    with col2:
        conf_idx = st.slider('Consumer Confidence Index (monthly)', -70.0, 50.0, value=-29.8)
        price_idx = st.slider('Consumer Price Index (monthly)', 50.0, 150.0, value=92.379)

    st.write('')
    new_row, sp2 = st.beta_columns((0.7, 0.8))
    with new_row:
        age = int(st.number_input('Age:', 0, 100, 39))
        job = st.selectbox('Job', ['self-employed', 'admin.', 'blue-collar', 'technician', 'services',
                                   'retired', 'entrepreneur', 'housemaid', 'management',
                                   'unemployed', 'student'])
        marital = st.selectbox('Marital', ['married', 'single', 'divorced'])
        education = st.selectbox('Education', ['university.degree', 'high.school', 'professional.course',
                                               'unknown', 'illiterate', 'basic'])
        default = st.selectbox('Has Defaulted in the past', ['no', 'yes'], )
    with sp2:
        campaign = int(st.number_input('# of times potential client was contacted:', 0, 100, 1))
        previous = int(st.number_input('# of times potential client was contacted, previously:', 0, 10, 1))
        poutcome = st.selectbox('Outcome of the previous marketing campaign', ['success', 'failure', 'nonexistent'])
        has_loan = int(st.number_input('Has Housing or Personal Loan (1 = Yes, 0 = No):', 0, 1, 0))

    # this is how to dynamically change text
    st.markdown("""---""")
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
            'emp.var.rate': [emp_var_rate],
            'cons.price.idx': [price_idx],
            'cons.conf.idx': [conf_idx],
            'euribor3m': [euribor3m],
            'nr.employed': [employed],
            'has_loan': [has_loan]
        }
    )

    y_pred = tree_clf.predict(client)

    if y_pred[0] == 0:
        msg = 'This client is predicted to : **Not Subscribe** to Term Deposit'
    else:
        msg = 'This client is predicted to : **Subscribe** to Term Deposit'

    prediction_state.markdown(msg)
