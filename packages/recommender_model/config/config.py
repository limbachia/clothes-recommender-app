import pathlib
import recommender_model
import numpy as np

PACKAGE_ROOT = pathlib.Path(recommender_model.__file__).resolve().parent
DATASET_DIR = PACKAGE_ROOT / "datasets"

DATA_FILE = 'renttherunway_final_data.txt' # 'renttherunway_final_data.json' 


HEIGHT = 'height'
BUST_SIZE = 'bust size'
VARS_TO_DROP = ["review_text","review_summary","review_date","height"]
VARS_TO_NUM = ['rating','size','age','weight']

CUP_MAP = {
    'a': 1, 'aa': 2, 
    'b': 3,
    'c': 4,
    'd': 5, 'dd': 6, 'ddde': 7,
    'f': 8,
    'g': 9,
    'h': 10,
    'i': 11,
    'j': 12
}

BODY_TYPE_MAP = {
    'apple': [0, 0, 0, 0, 0, 0, 1],
    'athletic': [0, 0, 0, 1, 0, 0, 0],
    'full bust': [0, 0, 0, 0, 1, 0, 0],
    'hourglass': [1, 0, 0, 0, 0, 0, 0],
    'pear': [0, 0, 1, 0, 0, 0, 0],
    'petite': [0, 0, 0, 0, 0, 1, 0],
    'straight & narrow': [0, 1, 0, 0, 0, 0, 0],
    np.nan:[np.nan]*7,
}

FIT_MAP = {
    'fit':[0, 0, 1],
    'small':[0, 1, 0],
    'large':[1, 0, 0],
    np.nan:[np.nan]*3,
}

RENTED_FOR_MAP = {
    'date':[0, 0, 0, 0, 0, 1, 0, 0],
    'everyday':[0, 0, 0, 0, 0, 0, 1, 0],
    'formal affair':[0, 0, 0, 1, 0, 0, 0, 0],
    'other':[0, 1, 0, 0, 0, 0, 0, 0],
    'party':[0, 0, 1, 0, 0, 0, 0, 0],
    'vacation':[1, 0, 0, 0, 0, 0, 0, 0],
    'wedding':[0, 0, 0, 0, 1, 0, 0, 0],
    'work':[0, 0, 0, 0, 0, 0, 0, 1],
    np.nan: [np.nan]*8
}