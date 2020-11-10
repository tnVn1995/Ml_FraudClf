from config import CONFIG
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import recall_score, make_scorer
from logzero import setup_logger

import numpy as np
import pandas as pd

logger = setup_logger(name=__file__, logfile=CONFIG.reports / 'logs' / '02_modelFinetune.log')
logger.info(f'Load data from {CONFIG.data_path / "interim"}')


df_train = pd.read_csv(CONFIG.data_path / 'interim' / 'train_data.csv')
df_test = pd.read_csv(CONFIG.data_path / 'interim' / 'test_data.csv')
X_train, y_train = df_train.iloc[:, :-1], df_train.iloc[:, -1]
X_test, y_test = df_test.iloc[:, :-1], df_test.iloc[:, -1]

# Logistic Regression Tuning
param_grid = {
    'penalty': ['l1', 'l2'],
    'C': np.logspace(-4, 4, 20),
    'class_weight': ['balanced'],
    'solver':['liblinear']
}

# LRgs = GridSearchCV(logReg, param_grid=param_grid, cv=k_fold, n_jobs=-1, scoring=scorer, refit=True)
# best_lr = gs.fit(X, y)

k_fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=CONFIG.random_state)
logger.info('Finetune model using recall score...')
scorer = make_scorer(recall_score)
logReg = LogisticRegression(n_jobs=-1, class_weight='balanced')
logger.info('Finetune logistic regression:')
rs = RandomizedSearchCV(logReg, param_distributions=param_grid, cv=k_fold, n_jobs=-1, scoring=scorer, refit=True)
rs.fit(X_train, y_train)
logger.info(f'LR finetuned best scores: {rs.best_score_}')
logger.info(f'LR finetuned best params: {rs.best_params_}')


from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform
param_test ={'num_leaves': sp_randint(6, 50), 
             'min_child_samples': sp_randint(100, 500), 
             'min_child_weight': [1e-5, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4],
             'subsample': sp_uniform(loc=0.2, scale=0.8), 
             'colsample_bytree': sp_uniform(loc=0.4, scale=0.6),
             'reg_alpha': [0, 1e-1, 1, 2, 5, 7, 10, 50, 100],
             'reg_lambda': [0, 1e-1, 1, 5, 10, 20, 50, 100],
             'scale_pos_weight': [100, 200, 300, 400, 500, 600]}


fit_params={"early_stopping_rounds":30, 
            "eval_metric" : 'auc', 
            "eval_set" : [(X_test,y_test)],
            'eval_names': ['valid'],
            #'callbacks': [lgb.reset_parameter(learning_rate=learning_rate_010_decay_power_099)],
            'verbose': 100,
            'categorical_feature': 'auto'}

from lightgbm import LGBMClassifier

lgbm = LGBMClassifier(n_jobs=-1, scale_pos_weight=578)
lgbm_gs = RandomizedSearchCV(
    lgbm, param_distributions=param_test, 
    n_iter=100, scoring=scorer, cv=k_fold, refit=True, n_jobs=-1,

)
lgbm_gs.fit(X_train, y_train)
logger.info(f'LGBM finetuned best scores: {lgbm_gs.best_score_}')
logger.info(f'LGBM finetuned best params: {lgbm_gs.best_params_}')

logger.info('Evaluating models on test set...')

logger.info('LR score:')
logger.info(f'{rs.score(X_test, y_test)}')
logger.info('LGBN score:')
logger.info(f'{lgbm_gs.score(X_test, y_test)}')

from joblib import dump

if rs.score(X_test, y_test) >= lgbm_gs.score(X_test, y_test):
    logger.info(f'Save model to {CONFIG.model_path}')
    dump(rs, CONFIG.model_path / 'LR_fintuned.joblib')

else:
    dump(lgbm_gs, CONFIG.model_path / 'LGBM_finetuned.joblib')

logger.info('Done!')
