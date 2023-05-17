import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
import os

from baseclass import BaseClass

class mlpclassifier_(BaseClass):
    st = True
    rand = 22

    def train(self, tr_x, tr_y, va_x=None, va_y=None, random_state:int =None):
        '''
        mlpclassifierでトレーニングを行う

        Parameters
        ----------
        tr_x : dataframe
        tr_y : dataframe
        va_x : dataframe
        va_y : dataframe

        Returns
        -------
        model : mlpclassifier model
        '''
        model = MLPClassifier(
            hidden_layer_sizes=(30, 10, 4),
            max_iter=1000,
            shuffle=True,
            random_state=self.rand,
            early_stopping=True,
            validation_fraction=0.1
        )
        model.fit(tr_x, tr_y)

        print('training finished.')
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
        return model.predict_proba(df)[:,1]

if __name__ == '__main__':
    mlpc_ = mlpclassifier_()
    mlpc_.main()
