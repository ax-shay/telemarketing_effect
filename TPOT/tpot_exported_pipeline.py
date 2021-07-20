import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from tpot.builtins import StackingEstimator
from tpot.export_utils import set_param_recursive
from sklearn.preprocessing import FunctionTransformer
from copy import copy

# NOTE: Make sure that the outcome column is labeled 'target' in the data file

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

tpot_data = df_data_post_clean.copy()
tpot_data.rename(columns={'subscribed': 'target'}, inplace=True)
features = tpot_data.drop('target', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'], random_state=42)

# Average CV score on the training set was: 0.9152473022636232
exported_pipeline = make_pipeline(
    make_union(
        FunctionTransformer(copy),
        FunctionTransformer(copy)
    ),
    RandomForestClassifier(bootstrap=True, criterion="entropy", max_features=0.25, min_samples_leaf=13, min_samples_split=19, n_estimators=100)
)
# Fix random state for all the steps in exported pipeline
set_param_recursive(exported_pipeline.steps, 'random_state', 42)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
