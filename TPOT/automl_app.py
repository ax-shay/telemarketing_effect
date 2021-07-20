from sklearn.datasets import make_classification
from sklearn.model_selection import RepeatedStratifiedKFold
from tpot import TPOTClassifier
import numpy as np
import pandas as pd
from pandas_profiling import ProfileReport
from sklearn.model_selection import train_test_split


# df_data = pd.read_csv('bank-additional-full.csv', sep=';')
# profile = ProfileReport(df_data, title="Telemarketing Data Profiling Report - Pre")
# profile.to_file("Telemarketing_Data_Profile_Report_pre.html")


df_data_post = pd.read_csv('bank-additional-full.csv', sep=';')
df_data_post['subscribed'] = np.where(df_data_post.y=='yes', 1, 0)
df_data_post.drop('y', axis=1, inplace=True)
df_data_post = df_data_post.replace('unknown', np.nan)
df_data_post['has_loan'] = np.where((df_data_post['housing']=='yes') | (df_data_post['loan']=='yes'), 1, 0)
df_data_post['has_default'] = np.where(df_data_post['default'] == 'yes', 1, 0)
df_data_post.drop(['housing', 'loan', 'has_default'], axis=1, inplace=True)
df_data_post.drop_duplicates(inplace=True)
df_data_post['education'] = np.where((df_data_post["education"].str.startswith('basic')),'basic',df_data_post['education'])
df_data_post['marital'].fillna('married', inplace=True)
df_data_post['job'].fillna('admin.', inplace=True)
df_data_post['default'].fillna('no', inplace=True)
df_data_post_clean = pd.get_dummies(df_data_post, prefix=['job', 'marital', 'education', 'contact',
                                                          'month', 'day_of_week', 'poutcome', 'default'],
                                    columns=['job', 'marital', 'education', 'contact', 'month', 'day_of_week',
                                             'poutcome', 'default']).reset_index()
# profile_post = ProfileReport(df_data_post_clean, title="Telemarketing Data Profiling Report - POST")
# profile_post.to_file("Telemarketing_Data_Profile_Report_post.html")

X = df_data_post_clean.drop(['subscribed'], axis=1)
y = df_data_post_clean['subscribed']
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=df_data_post_clean['subscribed'], test_size=0.2)


cv = RepeatedStratifiedKFold(n_splits=3, n_repeats=3, random_state=1)
pipeline_optimizer = TPOTClassifier(generations=5, population_size=50, cv=cv,  scoring='accuracy',
                                    random_state=42, verbosity=2)
pipeline_optimizer.fit(X_train, y_train)

print(pipeline_optimizer.score(X_test, y_test))
pipeline_optimizer.export('tpot_exported_pipeline.py')

