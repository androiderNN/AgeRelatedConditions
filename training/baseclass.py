import pandas as pd
import os
from sklearn.model_selection import KFold

class BaseClass():
    source_dir = '../sources/processed'
    rand = 0

    def main(self):
        pass

    def load_dataset(self):
        return pd.read_csv(os.path.join(source_dir, 'train.csv'))

    def fix_dataset(self, train: DataFrame, target:str =None):
        if target == None:
            target = 'Class'
        dlist = ['Class', 'B', 'C', 'D']

        y = train[target]
        x = train.drop(columns=dlist, errors='ignore')

        return x, y

    def train(self, tr_x, tr_y, va_x=None, va_y=None, random_state:int=rand):
        model = self.train_(tr_x, tr_y, va_x, va_y, random_state)
        return model

    def training_cv(self, x, y):
        kf = KFold(n_splits=5, shuffle=True, random_state=rand)
        model = []
        
        for tr_ind, va_ind in kf.split(x):
            tr_x = x[tr_ind]
            tr_y = y[tr_ind]
            va_x = x[va_ind]
            va_y = y[va_ind]

            model.append(self.train(tr_x, tr_y, va_x, va_y, random_state=self.rand))
            
        return model
    
    def predict(self):
        df = self.predict_()
        return df
