import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from recommender_model.config.config import BUST_SIZE, CUP_MAP


class TransformBustSize(BaseEstimator,TransformerMixin):
    def __init__(self, bust_size=BUST_SIZE):
        self.variable=bust_size
    
    def fit(self, user_df, y=None):
        return self
    
    def transform(self, user_df):
        user_df = user_df.copy()
        if self.variable in user_df.columns:
            user_df['cup'] = user_df[self.variable].apply(
                lambda x: ''.join(filter(str.isalpha,x)) if type(x) == str else x)
            user_df['cup'] = user_df['cup'].map(CUP_MAP)
            user_df[self.variable] = user_df[self.variable].apply(
                lambda x: np.float(''.join(filter(str.isdigit,x))) if type(x) == str else x)
    
        return user_df


class OneHotEncode(BaseEstimator, TransformerMixin):
    def __init__(self,variable=None, mapper=None):
        self.variable = variable
        self.mapper = mapper
        
    def fit(self, user_df, y=None):
        return self
    
    def transform(self, user_df):
        user_df = user_df.copy()
        if self.variable in user_df.columns:
            tmp_df = user_df.apply(
                lambda x: self.mapper[x[self.variable]], axis=1, result_type='expand')

            tmp_df.columns = [''.join(self.variable.split())+'_{:02d}'.format(ii) for ii in range(tmp_df.shape[1])]
            
            return user_df.drop(self.variable,axis=1).join(tmp_df).drop_duplicates()
        else:
            return user_df