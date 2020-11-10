# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from config import CONFIG
import pandas as pd
import numpy as np
from pathlib import Path
from logzero import setup_logger
# trainning
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import recall_score, make_scorer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# %%

logger = setup_logger(name=__file__, logfile=CONFIG.reports / 'logs' / '01_train_models.log')
k_fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=CONFIG.random_state)


# %%
# load data
logger.info('Load Train Data')
dataX = pd.read_csv(CONFIG.data_path / 'interim' / 'train_data.csv')

# %% [markdown]
# ## Model Training

# %%
X, y = dataX.iloc[:, :-1], dataX.iloc[:, -1]
print(f'Data shapes: {X.shape}, {y.shape}')


# %%
# penalty = 'l2'
# C = 1.0
# class_weight = 'balanced'
# random_state = 2018
# solver = 'liblinear'
logReg = LogisticRegression(n_jobs=-1, class_weight='balanced')
scorer = make_scorer(recall_score)

logger.info('Train LogisticRegression:')
scores = cross_val_score(logReg, X, y, scoring=scorer, cv=k_fold, n_jobs=-1)

logger.info(f'CV recall: {np.mean(scores): .3f} +/- {np.std(scores): .3f}')
logger.info('Done!')

# %%
# n_estimators = 10
# max_features = 'auto'
# max_depth = None
# min_samples_split = 2
# min_samples_leaf = 1
# min_weight_fraction_leaf = 0.0
# max_leaf_nodes = None
# bootstrap = True
# oob_score = False
# n_jobs = -1
# random_state = 2018
# class_weight = 'balanced'

RFC = RandomForestClassifier(n_jobs=-1, class_weight='balanced')

logger.info('Train RandomForest:')
scores = cross_val_score(RFC, X, y, scoring=scorer, cv=k_fold, n_jobs=-1)
logger.info(f'CV recall: {np.mean(scores): .3f} +/- {np.std(scores): .3f}')
logger.info('Done!')

# %%
from lightgbm import LGBMClassifier

lgbm = LGBMClassifier(n_jobs=-1, scale_pos_weight=578)

logger.info('Train Lgbm:')
scores = cross_val_score(lgbm, X, y, cv=k_fold, scoring=scorer, n_jobs=-1)
logger.info(f'CV recall: {np.mean(scores): .3f} +/- {np.std(scores): .3f}')
logger.info('Done!')

# %%
from xgboost import XGBClassifier

xgbm = XGBClassifier(n_jobs=-1, scale_pos_weight=578)

logger.info('Train Xgbm:')
scores = cross_val_score(xgbm, X, y, cv=k_fold, scoring=scorer, n_jobs=-1)
logger.info(f'CV recall: {np.mean(scores): .3f} +/- {np.std(scores): .3f}')
logger.info('Done!')
