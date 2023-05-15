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
        }

    def train(self, tr_x, tr_y, va_x, va_y, random_state:int =None):
        lgb_train = lgb.Dataset(tr_x, tr_y)
        lgb_val = lgb.Dataset(va_x, va_y)
        
        model = lgb.train(
            self.params,
            lgb_train,
            1000,
            lgb_val,
            ['train', 'val'],
            callbacks=[lgb.early_stopping(stopping_rounds=3, verbose=True)]
        )

        return model
    
    def predict(self, model, df):
        return model.predict(df, predict_disable_shape_check=True)

if __name__ == '__main__':
    lgb_ = lightgbm_()
    lgb_.main()
