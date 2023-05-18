import numpy as np
import pandas as pd
import os
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import log_loss

class BaseClass():
    source_dir = '../sources/processed'
    export_dir = '../export'
    st = False
    rand = 0

    def main(self):
        train = self.load_dataset(self.st)
        x, y = self.fix_dataset(train)
        x, validation_x, y, validation_y = train_test_split(x, y, test_size=0.2, random_state=self.rand)

        model = self.train_cv(x, y)
        pred_mean, pred, submission = self.predict_cv(model, x)
        print('\ntraining data   :', log_loss(y, pred_mean))
        pred_mean, pred, submission = self.predict_cv(model, validation_x)
        print('validation data :', log_loss(validation_y, pred_mean))
        # print(pd.DataFrame({'true':validation_y, 'pred':pred_mean}).head(30))
        
        self.col = x.columns.tolist()
        self.col.remove('Id')
        self.model = model

    def load_dataset(self, st: bool =False):
        '''
        前処理済みのトレーニングデータをロードする関数
        
        Parameters
        ----------
        st : bool
            標準化されたトレーニングデータを読み込むか
        
        Returns
        -------
        train : dataframe
            トレーニングデータ
        '''
        if st:
            return pd.read_csv(os.path.join(self.source_dir, 'train_st.csv'))
        else:            
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
        print('-----------------------------------------------------------\n')
        kf = KFold(n_splits=5, shuffle=True, random_state=rand)
        model = list()

        x = x.drop(columns='Id', errors='ignore')
        y = y.drop(columns='Id', errors='ignore')
        
        for tr_ind, va_ind in kf.split(x):
            tr_x = x.iloc[tr_ind]
            tr_y = y.iloc[tr_ind]
            va_x = x.iloc[va_ind]
            va_y = y.iloc[va_ind]

            mod = self.train(tr_x, tr_y, va_x, va_y, random_state=self.rand)
            model.append(mod)

            print("train's log_loss :", log_loss(tr_y, self.predict(mod, tr_x)))
            print("val's log_loss   :", log_loss(va_y, self.predict(mod, va_x)), '\n')

        print('-----------------------------------------------------------')
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
