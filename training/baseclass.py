import numpy as np
import pandas as pd
import os
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import log_loss

class BaseClass():
    source_dir = '../sources/processed'
    export_dir = '../export'
    rand = 0

    def main(self):
        train = self.load_dataset()
        x, y = self.fix_dataset(train)
        x, validation_x, y, validation_y = train_test_split(x, y, test_size=0.2, random_state=self.rand)

        model = self.train_cv(x, y)
        pred_mean, _, _ = self.predict_cv(model, validation_x)
        print(log_loss(validation_y, pred_mean))

        pass

    def load_dataset(self):
        '''
        前処理済みのトレーニングデータをロードする関数
        引数なしで実行
        '''
        return pd.read_csv(os.path.join(self.source_dir, 'train.csv'))

    def fix_dataset(self, train, target:str =None):
        '''
        トレーニングデータを学習データとクラスデータに分割する関数
        
        Parameters
        ----------
        train : dataframe
            分割したいトレーニングデータ 特に'Class'列が必要
        target : str
            クラスデータに利用したいDataframeの列名 Class, B, C, Dを想定
        
        Returns
        -------
        x : dataframe
            教師データ
        y : dataframe
            クラスデータ
        '''
        if target == None:
            target = 'Class'
        dlist = ['Class', 'B', 'C', 'D']

        y = train[target]
        x = train.drop(columns=dlist, errors='ignore')

        return x, y

    def split_dataset(self, x, y):
        train_x, train_y, validation_x, validation_y = train_test_split(x, y, test_size=0.2, random_state=self.rand)
        return train_x, train_y, validation_x, validation_y

    # def train(self, tr_x, tr_y, va_x=None, va_y=None, random_state:int=rand):
    #     '''
    #     各機械学習アルゴリズム
    #     '''
    #     model = self.train_(tr_x, tr_y, va_x, va_y, random_state)
    #     return model

    def train_cv(self, x, y, rand: int =rand) ->list:
        '''
        クロスバリデーションでトレーニングデータを分割し学習を行う関数

        Parameters
        ----------
        x : dataframe
            学習データ
        y : dataframe
            クラスデータ
        rand : int
            random_state
        
        Returns
        -------
        model : list
            cvを行った回数分のmodelを格納したリスト
        '''
        # x.reset_index(inplace=True)
        # y.reset_index(inplace=True)
        kf = KFold(n_splits=5, shuffle=True, random_state=rand)
        model = []
        x = x.drop(columns='Id', errors='ignore')
        y = y.drop(columns='Id', errors='ignore')
        
        for tr_ind, va_ind in kf.split(x):
            tr_x = x.iloc[tr_ind]
            tr_y = y.iloc[tr_ind]
            va_x = x.iloc[va_ind]
            va_y = y.iloc[va_ind]

            model.append(self.train(tr_x, tr_y, va_x, va_y, random_state=self.rand))

        return model
    
    # def predict(self, model, df):
    #     pred = np.array(self.predict_(model, df))
    #     return pred

    def predict_cv(self, model, df):
        '''
        cvの回数予測を行い予測値を出力する関数

        Parameters
        ----------
        model : list[model]
            cvの回数分のモデルが格納されたリスト
        df : dataframe
            予測値を得たいdataframe
        
        Returns
        -------
        pred_mean : ndarray
            予測値について列ごとに平均値をとった結果
        pred : ndarray[ndarray]
            予測値を格納したndarray
            cv数*サンプル数
        '''
        pred = list()
        for mod in model:
            pred.append(self.predict(mod, df.drop(columns='Id')))
        pred = np.array(pred).T
        pred_mean = pred.mean(axis=1)
        try:
            submission = pd.DataFrame({'Id':df['Id'], 'class_0':1-pred_mean, 'class_1':pred_mean})
        except KeyError:
            submission = None

        return pred_mean, pred, submission
