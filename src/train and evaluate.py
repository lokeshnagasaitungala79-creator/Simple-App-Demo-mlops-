#load the train and test
#train algo
#save the metrics,params

import os
import argparse
import pandas as pd
import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error , mean_absolute_error , r2_score
from get_data import get_data
from sklearn.model_selection import train_test_split
from urllib.parse import urlparse
import joblib
import json


def eval_metrics(actual , pred):
    rmse  = np.sqrt(mean_squared_error(actual , pred))
    mae = mean_absolute_error(actual , pred)
    r2 = r2_score(actual , pred)
def evaluate_metrics(actual , pred):
    config = get_data(config_path)
    test_data_path = config["test_path"]["split data"]
    train_data_path = config["train_path"]["split data"]
    random_state = config["random_state"]["base"]
    model_dir = config["model_dir"]
    alpha = config["estimators"]["ElasticNet"]["params"]["alpha"]
    l1_ratio = config["estimators"]["ElasticNet"]["params"]["l1_ratio"]
    target = config["target_col"]["base"]

    train = pd.read_csv(train_data_path , sep = ',')
    test  =pd.read_csv(test_data_path , sep = ',')

    train_y = train[target]
    test_y = test[target]

    train_x = train.drop(columns=[target] , axis = 1)
    test_x = test.drop(columns=[target] , axis = 1)

    lr = ElasticNet(alpha=alpha , L1_ratio=l1_ratio , random_state = random_state)
    lr.fit(train_x, train_y)
    pred = lr.predict(test_x)
    (rmse , mae , r2) = evaluate_metrics(test_y , pred=pred)
    print("ElasticNet Model(alpha=%f , l1_ratio = %f):" %(alpha , l1_ratio))
    print("RMSE : %f" %rmse)
    print("MAE : %f" %mae)
    print("R2 : %f" %r2)

    source_file = config["reports"]["scores"]
    params_file = config["reports"]["params"]
    
    with open(scores_file , "w") as f:
        scores = {
            "rmse" : rmse,
            "mae" : mae,
            "r2" : r2
        }
        json.dump(scores , f , indent  = 4)

    with open(params_file , "w") as f:
            params = {
                "rmse" : rmse,
                "mae" : mae,
                "r2" : r2
            }
            json.dump(params , f , indent = 4)

    os.makedirs(model_dir , exist_ok = True)
    model_path  = os.path.join(model_dir , "model.joblib")

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config")
    parsed_args = args.parse_args()
    train_and_evaluate(config_path=parsed_args.config)