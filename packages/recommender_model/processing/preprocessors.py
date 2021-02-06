import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class TransformHeight(BaseEstimator, TransformerMixin):
    '''
    convert height info provided as string (feet' inch") to float (inches)
    '''
    def __init__(self, height=None):
        self.height=height
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        def convert_height_to_inch(string):
            feet, inch = string.split()

            feet = np.float32(feet.strip("'"))
            inch = np.float32(inch.strip('"'))

            height_in_inch = feet*12 + inch

            return height_in_inch
        
        X = X.copy()
        X[self.height+'_inch'] = X[self.height].apply(
            lambda x: convert_height_to_inch(x) if type(x)==str else np.nan)
        
        return X
    

class DropFeatures(BaseEstimator,TransformerMixin):
    
    '''
    Drop unecessary features
    '''
    
    def __init__(self, variables_to_drop=None):
        if not isinstance(variables_to_drop,list):
            self.variables = [variables_to_drop]
        else:
            self.variables = variables_to_drop
            
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # drop variables
        X = X.copy()
        X = X.drop(self.variables, axis=1)

        return X
    
class ConvertToFloat(BaseEstimator,TransformerMixin):
    '''
    convert numerical variables to float
    '''
    def __init__(self, variables_to_float=None):
        if not isinstance(variables_to_float,list):
            self.variables = [variables_to_float]
        else:
            self.variables = variables_to_float
            
    def fit(self, X,y=None):
        return self
    
    def transform(self,X):
        X = X.copy()
        for var in self.variables:
            if var == 'weight':
                X[var] = X[var].apply(lambda x: np.float32(x.strip("lbs")) if type(x) == str else x)
            else:
                X[var] = X[var].astype(np.float32)
        return X