import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings("ignore")

from recommender_model.config import config
from recommender_model.processing.data_management import load_datasets

# import preproc and feature engineering pipelines
from recommender_model import pipeline as pp

# sklearn
from sklearn.preprocessing import normalize

class Recommender:
    def __init__(self):
        self.df = load_datasets(file_name=config.DATA_FILE)
        self.proc_df = pp.proc_pipe.transform(self.df)
        self.proc_df.rename(columns={"rented for":"occasion"},inplace=True)
        self.user_df = None
        self.target_user_df = None
        self.recommendations = None
        
    def get_user_info(self,user_info=None):
        '''
        Pass user information as a list.
        Items in the list should be in following order:
            Bust size, weight, body type, size, age, height
        '''
        user_attribs = ["bust size","weight","body type","size","age","height_inch"]
        user_info = dict(zip(user_attribs, user_info))
        
        for attrib, val in user_info.items():
            if attrib in ["bust size","body type"]:
                user_info[attrib] = val if val else np.nan
            else:
                user_info[attrib] = np.float64(val) if val else np.nan
                
        self.target_user_df = pd.Series(user_info).to_frame().T.dropna(axis=1)
        
        '''
        Filter out attibutes from the processed dataframe for 
        which no information is provided in the target_user_df
        and create a new processed df called user_df
        '''
        
        user_attribs = self.target_user_df.columns

        self.user_df = self.proc_df[["user_id"]+list(user_attribs)]
        # Set user id as the index
        self.user_df.set_index("user_id",inplace=True)
        # Drop repeated row
        self.user_df.drop_duplicates(inplace=True)
        
        
    def transform(self):
        # transform both target and user dfs
        self.target_user_df = pp.user_pipe.transform(self.target_user_df)
        self.user_df = pp.user_pipe.transform(self.user_df)
        self.user_df.dropna(inplace=True)
        
    def get_recommendations(self):
        
        '''
        Computes cosine similarity between the new customer and 
        existing customers
        '''

        user_IDs = np.array(self.user_df.index)

        norm_target = np.squeeze(normalize(self.target_user_df))
        norm_users = normalize(self.user_df)

        similarities = pd.Series(norm_users.dot(norm_target),index=user_IDs)
        top_10_similar_users = similarities.sort_values(ascending=False)[:10].index

        self.recommendations = self.proc_df[self.proc_df['user_id'].isin(top_10_similar_users)][
    ["item_id","occasion","category","body type"]].drop_duplicates().reset_index(drop=True)
        
        
        return self.recommendations
            