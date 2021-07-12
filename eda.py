
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
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


def write(state):

    st.title('Exploratory Data Analysis')

    st.markdown('')
    st.subheader("Getting Data")

    # data_load_state = st.text('Loading data...')
    df_data = load_data()
    # data_load_state.text("Done!")

    st.write(df_data)

    st.markdown(' ')
    st.subheader("Data Exploration / Analysis")
    # info = get_info(df_data)
    # st.write(info)
    st.markdown(' ')
    st.markdown('''
    * The Dataset has 41, 188 records with 20 features + 1 target variable 'y' (subscribed to term deposit)
    * Data Types are:
        - 5 Integers
        - 5 Floats
        - 11 Objects 
    * Below are the 21 features <a name="categorical"></a> listed with short description for each
        1. **age**           : Integer value for Age of the potential client
        2. **job**           : categorical column describing Job of the potential client
        3. **marital status**: categorical column describing Marital status of the potential client
        4. **education**     : categorical column describing Education of the potential client
        5. **default**       : categorical column representing whether the potential client has credit in default
        6. **housing**       : categorical column representing whether the potential client has housing loan
        7. **loan**          : categorical column representing whether the potential client has personal loan
        8. **contact**       : categorical column representing type of communication channel used
        9. **month**         : categorical column indicating the month when the potential client was last contacted
        10. **day_of_week**  : categorical column indicating the last contact day of the week of the month when contacted
        11. **duration**     : numerical columns indicating last contact duration (in seconds)
        12. **campaign**     : numerical columns indicating the # of times potential client was contacted
        13. **pdays**        : numerical columns indicating number of days since last contacted from a previous campaign
        14. **previous**     : numerical columns indicating the # of times potential client was contacted, previously
        15. **poutcome**     : categorical column indicating the outcome of the previous marketing campaign
        16. **emp.var.rate** : numerical column showing the 'quarterly' employment variation rate
        17. **cons.price.idx** : numerical column showing the 'monthly' consumer price index
        18. **cons.conf.idx**  : numerical column showing the 'monthly' consumer confidence index
        19. **euribor3m**    : numerical column showing the 'daily' euribor 3 month rate
        20. **nr.employed**  : numerical column showing the 'quarterly' number of employees employeed
        21. **y**            : binary column indication whether has the 'potential' client subscribed a term deposit
    ''')

    df_data = rationalize_data(df_data)
    st.markdown(' ')
    st.subheader('General Stats for Numerical Fields')

    desc = df_data.describe()
    st.write(desc)

    st.markdown(' ')
    st.markdown('''
    * From above we observe a few things:
        - The **effectiveness of telemarketing is 11.26%** (as that is the mean of 'subscribed' attribute)
        - Surprisingly, none of the numerical data columns have null values. Need to deep dive to understand if the nulls are defaulted in the underlying data
        - The age range that telemarketers covered was 17 Yrs - 98 Years
    * There is a sporadic presence of 'unknown' value which indicates that nulls have been substituted by 'unknown' in the underlying data.
    ''')

    st.subheader('''Missing Values''')
    df_data_clean = df_data.replace('unknown', np.nan)
    missing_data_pct = missing_value(df_data_clean)
    st.write(missing_data_pct)

    st.markdown(' ')
    st.subheader('Null Values')
    st.markdown('''
    * 6 of the 21 features have null values
    * For all of the features with null values, less than 21% of the total values are null, so we will still consider all features for the analysis
    ''')

    st.markdown(' ')
    st.subheader('Duplicate Records')
    df_data_clean_dup = df_data_clean[df_data_clean.duplicated(keep='last')].reset_index()
    st.write(df_data_clean_dup)
    st.markdown('''* 12 duplicate records found in the dataset. Dropping the duplicates''')
    df_data_clean = df_data_clean.drop_duplicates()
    shape = df_data_clean.shape
    st.write(shape)

    st.markdown(' ')
    st.subheader('Better understanding the outcome variable')
    st.markdown('Checking distribution of class variable to see if it is a balanced data set or not')

    def without_hue(plot, feature):
        total = len(feature)
        for p in ax.patches:
            percentage = '{:.2f}%'.format(100 * p.get_height()/total)
            x = p.get_x() + p.get_width() / 2 - 0.05
            y = p.get_y() + p.get_height()
            ax.annotate(percentage, (x, y), size = 12)
        st.pyplot(plt)

    plt.figure(figsize=(7, 5))
    plt.title("Fig:1 - % Subscribership")
    ax = sns.countplot(x='subscribed', data=df_data_clean)
    plt.xticks(size=12)
    plt.xlabel('subscribed', size=12)
    plt.yticks(size=12)
    plt.ylabel('count', size=12)
    without_hue(ax, df_data_clean.subscribed)

    st.markdown(' ')
    st.markdown('''
    This data set is an imbalanced data set which has 88.73% of class variable 0 and 11.27% of 1.  
    * Please note:   
        - 0 indicates the term deposit is not subscribed  
        - 1 means the term deposit is subscribed. 
    ''')

    st.subheader('Understanding Correlation')
    corrMatrix = df_data_clean.corr()
    corrFig = corrMatrix.style.background_gradient(cmap='coolwarm')
    st.markdown(corrFig)
    st.markdown('''
    * **Observations**
    1. None of the features have a very high correlation with whether a consumer opened a deposit account or not  
        1.1 Duration has the highest correlation with y of all the features (0.4)  
    2. The social and economic features have a high correlation with one another. However, these are over different time intervals (daily, monthly, quarterly). These will have to be converted to a **common scale** later on before feeding into model(s).  
        - 2.1 euribor3m (Euro Inter Bank Offered Rate), emp.var.rate (employement variation rate), and nr.employed (number of employees) all have a high, positive correlation with one another  
        - 2.2 cons.price.idx (consumer price index) has a high, positive correlation with emp.var.rate
    ''')

    st.subheader('Effect of Age on Subscription')

    df_age_sub = df_data_clean[df_data_clean['subscribed']==1].age.value_counts()
    df_age_notsub = df_data_clean[df_data_clean['subscribed']==0].age.value_counts()

    df_age_sub = df_age_sub.to_frame().reset_index()
    df_age_sub.columns = ['age', 'cnt_subscribed']

    df_age_notsub = df_age_notsub.to_frame().reset_index()
    df_age_notsub.columns = ['age', 'cnt_notsubscribed']

    df_age_range = pd.merge(df_age_sub, df_age_notsub, how='outer', on='age')
    df_age_range['ratio'] = df_age_range['cnt_subscribed']/df_age_range['cnt_notsubscribed']

    st.write(df_age_range.plot(x='age', y='ratio', style='o', grid=True, title='Effect of Age on subscribership'))
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()
    st.write(df_data_clean.age.plot(kind='kde'))
    st.pyplot()

    st.markdown('''
    * **Observations**
        1. Before the age of 20 and after the age of 60, there is a 50% + probablity of potential clients subscribing for term deposit
        2. But again - the large density of data is in the range from 20-60 years so the above observation is on a small set of data and hence not very conclusive.
    ''')

    st.subheader('Label Encoding')
    st.markdown('**Understanding relation between Job Types and Subscribership**')

    labelencoder = LabelEncoder()
    df_data_clean['Job_Types_Cat'] = labelencoder.fit_transform(df_data_clean['job'])

    plt.figure(figsize=(7, 5))
    plt.xticks(size=12)
    plt.xlabel('Job_Types_Cat', size=12)
    plt.yticks(size=12)
    plt.ylabel('subscribed', size=12)
    ax = sns.histplot(data=df_data_clean, x=df_data_clean['Job_Types_Cat'], hue='subscribed')
    ax.legend(labels=['subscribed', 'not subscribed'])
    _ = ax.set_title('Fig: 3 - % Subscribership by Job Type')

    le_name_mapping = dict(zip(labelencoder.classes_, labelencoder.transform(labelencoder.classes_)))
    st.markdown(le_name_mapping)
    st.pyplot()

    st.markdown('''
    * **Observations**
        1. Students and retired people seem to be more likely to subscribe to term deposit compared to other job workers
        ''')

    st.subheader('Effect of Loan')
    df_data_clean['has_loan'] = np.where((df_data_clean['housing']=='yes') | (df_data_clean['loan']=='yes'), 1, 0)
    df_data_clean['has_default'] = np.where(df_data_clean['default'] == 'yes', 1, 0)

    grid = sns.FacetGrid(df_data_clean, col='subscribed', row='has_loan')
    grid.map(plt.hist, 'age', alpha=0.5, bins=20)
    grid.set_titles('Fig: 4 - Effect of loan on subscribership')
    grid.add_legend()
    st.pyplot()

    st.markdown('''
    * **Observations**
    1. Surprisingly, people having having a loan (housing or personal) are marginally more likely to subscribe to term deposit
    ''')

    st.subheader('Categorical Data Analysis')
    df_data_categorical = df_data_clean[['job','marital','education','default','housing','loan','contact','month','day_of_week','poutcome','subscribed']]
    ct_job = pd.crosstab(df_data_categorical.subscribed,df_data_categorical.job).apply(lambda r: r/r.sum())
    st.write(ct_job)
    st.markdown('''
    * **Observations**
    As previously observed in Fig: 3, certain employment types seem to have a slightly higher success rate (retired and student) than others. Unemployed doesn't seem to have a higher impact on "no"'s than some of the other job types. 
    ''')
    ct_edu = pd.crosstab(df_data_categorical.subscribed,df_data_categorical.education).apply(lambda r: r/r.sum())
    st.write(ct_edu)
    st.markdown('''
    * **Observation**
    Education level doesn't seem to have a significant impact on yes vs. no's. ''')

    sns.barplot(x='poutcome', y='subscribed', data=df_data_categorical).set_title('Fig: 5 - Effect of Previous Campaign')
    st.pyplot()
    st.markdown('''
    * **Observation**
    The outcome of previous marketing campaigns contribute significantly towards a client subscribing; a previously subscribed client is likely to subscribe again.
    ''')

    data = df_data_clean.groupby("month")["subscribed"].sum()
    data.plot.pie(autopct="%.1f%%").set_title('Fig: 6 - Subscribership by month')
    st.pyplot()
    st.markdown('''
    * **Observation**
    The months of May-Jun-Aug have more clients subscribing compared to other months''')

    st.subheader('Resampling')
    # class count
    class_count_0, class_count_1 = df_data_clean['subscribed'].value_counts()

    # Separate class
    class_0 = df_data_clean[df_data_clean['subscribed'] == 0]
    class_1 = df_data_clean[df_data_clean['subscribed'] == 1]

    st.write('class 0:', class_0.shape)
    st.write('class 1:', class_1.shape)

    st.subheader('Random Over-Sampling')
    class_1_over = class_1.sample(class_count_0, replace=True)

    test_over = pd.concat([class_1_over, class_0], axis=0)

    st.write("total class of 1 and 0:", test_over['subscribed'].value_counts())

    # plot the count after under-sampeling
    test_over['subscribed'].value_counts().plot(kind='bar', title='count (target)')
    st.pyplot()

    st.markdown('''
    Moving ahead with **random over-sampling** because undersampling can cause overfitting and poor generalization and we don't have a lot of data so there's a good chance that valuable information is being removed.
    ''')

    st.subheader('Missing Value % after Over-Sampling')
    total = test_over.isnull().sum().sort_values(ascending=False)
    percent_1 = test_over.isnull().sum()/test_over.isnull().count()*100
    percent_2 = percent_1.sort_values(ascending=False)
    missing_data = pd.concat([total, percent_2], axis=1, keys=['Total', '%'])
    st.write(total.head(6))
    st.write(missing_data.head(6))

    st.subheader('Feature Retention')
    st.markdown('''
    ### Variables to be removed
    * **Housing and Loan** - When the Chi-Square test was performed to compare each of these variables to the outcome (y), it was determined that both variables are independent of y. Removing them will not have a large impact on the accuracy of the model (see below for Chi-sqaure results).
    * **Duration** - Duration of the call can not be used in the model for prediction outcomes, because the duration of the call will not be known until after the call is complete
    * **pdays** - The value "999" was used in the pdays columns for all scenarios wher a customer was not previously contacted, which will impact the model. Other features such as "Campaign" and "Previous" indicate if a customer was previously contacted, so pdays is not necessary''')

    st.markdown('''
    ### New Variables Created
    * **Has Loan** - Housing and loan will be removed, and a new variable will be created to combine them. The new variable will be "Has Loan" which will have a value of 1 if  the customer has either a Housing loan or Personal loan, and a value of 0 if the customer has no loans
    * **Education** Compression - The education levels of basic.4y, basic.6y and basic.9y have been converged into one-single category called basic''')

    st.markdown('''
    ### Handling Null Variables
    * The null variables will "job", "marital", "default", and "education" will be handled by keeping an "unknown" category for each of the variables.  After performing the chi-square there appears to be a relationship between the unknown values and the outcome (y), so the rows with null values for those attributes should not be deleted. The "unknown" category for "Has Loan" will be kept as well.''')

    # grouping different basic education into single group
    df_data_clean['education'] = np.where((df_data_clean["education"].str.startswith('basic')),'basic',df_data_clean['education'])
    df_data_clean['education'].unique()
    df_data_clean['education'].value_counts()
    st.dataframe(df_data_clean['education'].groupby(df_data_clean['subscribed']).value_counts().reset_index(name='count'))
    df_data_clean.to_csv('df_data_clean.csv', index=False)


# @st.cache
def load_data():
    df_data = pd.read_csv('bank-additional-full.csv', sep=';')
    return df_data


def get_info(df):
    buffer = io.StringIO()
    df.info(verbose=True, null_counts=False, buf=buffer)
    info = buffer.getvalue()
    print(info)
    return info


def rationalize_data(df_data):
    # if df_data.y:
    df_data['subscribed'] = np.where(df_data.y=='yes', 1, 0)
    df_data.drop('y', axis=1, inplace=True)
    return df_data


def missing_value(df_data_clean):
    total = df_data_clean.isnull().sum().sort_values(ascending=False)
    percent_1 = df_data_clean.isnull().sum()/df_data_clean.isnull().count()*100
    percent_2 = percent_1.sort_values(ascending=False)
    missing_data = pd.concat([total, percent_2], axis=1, keys=['Total', '%'])
    return missing_data.head(6)
