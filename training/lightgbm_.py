import numpy as np
import pandas as pd
import lightgbm as lgb
import os

from baseclass import BaseClass

class lightgbm_(BaseClass):

    def __init__(self):
        self.rand = self.rand
        self.params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'random_state': self.rand,
            'verbose': -1,

            'bagging_fraction': 0.7,
            'bagging_freq': 1,
            'feature_fraction': 1,
            'min_data_in_leaf': 5,
            'lambda_l1': 0.5,
            'lambda_l2': 0.5
        }

    def train(self, tr_x, tr_y, va_x, va_y, random_state:int =None):
        '''
        lightgbmでトレーニングを行う

        Parameters
        ----------
        tr_x : dataframe
        tr_y : dataframe
        va_x : dataframe
        va_y : dataframe

        Returns
        -------
        model : Booster model
        '''
        lgb_train = lgb.Dataset(tr_x, tr_y)
        lgb_val = lgb.Dataset(va_x, va_y)
        
        model = lgb.train(
            params=self.params,
            train_set=lgb_train,
            num_boost_round=100,
            valid_sets=lgb_val,
            valid_names=['train', 'val'],
            callbacks=[lgb.early_stopping(stopping_rounds=3, verbose=True)]
        )

        return model
    
    def predict(self, model, df):
        '''
        lightgbmモデルでの予測

        Parameters
        ----------
        model : Booster model
        df : dataframe
            予測用データフレーム、Id列は削除済み
        
        Returns
        -------
        result : ndarray
        '''
        return np.array(model.predict(df))

if __name__ == '__main__':
    lgb_ = lightgbm_()
    lgb_.main()
