import json
from pandas.io.json import json_normalize
import pandas as pd
from recommender_model.config import config

def load_datasets(*, file_name: str) -> pd.DataFrame:
    
    if 'txt' in file_name:
        df = pd.read_csv(f"{config.DATASET_DIR}/{file_name}",sep=',')
    else:
        with open(f"{config.DATASET_DIR}/{file_name}") as f:
            _data = json.loads("[" + 
            f.read().replace("}\n{", "},\n{") + "]")

        df = json_normalize(_data)
    
    return df