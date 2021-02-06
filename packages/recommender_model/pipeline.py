from sklearn.pipeline import Pipeline

from recommender_model.config import config
from recommender_model.processing import preprocessors as pp
from recommender_model.processing import features as fe

proc_pipe = Pipeline(
    [
        ("Process Height", pp.TransformHeight(config.HEIGHT)),
        ("Drop Features", pp.DropFeatures(config.VARS_TO_DROP)),
        ("Convert To Float", pp.ConvertToFloat(config.VARS_TO_NUM)),
    ]
)

user_pipe = Pipeline(
    [
        ('TrasformBustSize', fe.TransformBustSize(bust_size=config.BUST_SIZE)),
        ('OneHotEncode', fe.OneHotEncode(variable='body type',
                                      mapper=config.BODY_TYPE_MAP)),
    ]
)
