# Main 
import logging
import numpy as np
import pandas as pd
from pathlib import Path
import sys
from logzero import setup_logger

# Own module
from config import CONFIG

# Data Prep
from sklearn.model_selection import train_test_split
from sklearn import preprocessing as pp
from sklearn.preprocessing import StandardScaler


logger = setup_logger(name = __file__, logfile=CONFIG.data_path / '01_makeTrainTestSet.log', fileLoglevel=logging.INFO)

# data path
data_path = CONFIG.data_path / 'raw' / 'creditcard.csv'


def main():
        # load data
    data = pd.read_csv(str(data_path))

    # split attribute and label data
    dataX = data.copy().drop(['Class'], axis=1)
    dataY = data['Class'].copy()

    # standardize data
    logger.info('Scaling numerical data ...')
    featureToScale = dataX.drop(['Time'], axis=1).columns
    sX = pp.StandardScaler(copy=True)
    dataX.loc[:, featureToScale] = sX.fit_transform(dataX[featureToScale])

    # split train and test data
    X_train, X_test, y_train, y_test = train_test_split(dataX, dataY, test_size = 0.3, 
                                        random_state=CONFIG.random_state, stratify=dataY)


    train_data = pd.concat([X_train, y_train], axis=1)
    test_data = pd.concat([X_test, y_test], axis=1)

    logger.info(f'Save data to {CONFIG.data_path / "interim"}')
    train_data.to_csv(CONFIG.data_path / 'interim' / 'train_data.csv', index=None)
    test_data.to_csv(CONFIG.data_path / 'interim' / 'test_data.csv', index=None)
    logger.info('Done!')
if __name__ == "__main__":
    main()






