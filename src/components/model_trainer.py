import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import (
    GridSearchCV,RandomizedSearchCV
    )
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    classification_report
)  
from sklearn.multiclass import OneVsRestClassifier
from src.exception import CustomException
from src.logger import logging
from src.utils import evaluate_models
from src.utils import save_object


@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split data into train and test input data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models={
                "Random Forest": RandomForestClassifier(),
                "Decision Trees": DecisionTreeClassifier(),
                "Naive Bayes" : GaussianNB(),
                "Gradient Boosting": GradientBoostingClassifier(),
                "K-Nearest Neighbor": KNeighborsClassifier(),
                "Logistic Regression": LogisticRegression(),
                "Support Vector Machine":SVC()
                
            }

            params={
                "Random Forest":{ 
                    'n_estimators': [50, 100, 200],
                    'max_features': ['auto', 'sqrt', 'log2'],
                    'max_depth' : [2,4,6,8],
                    'criterion' :['gini', 'entropy']
                },
                "Decision Trees":{
                    'max_depth': range(1,20), 
                    'min_samples_split': range(2,10), 
                    'min_samples_leaf': range(2,10), 
                    'criterion':['gini','entropy']
                },
                "Naive Bayes":{
                    'var_smoothing': np.logspace(0,-9, num=100)
                },
                    "Gradient Boosting":{
                    'learning_rate': np.arange(000.1,000.5,0.009),
                    'n_estimators': np.arange(50,500,50),
                    'max_depth': np.arange(1,10,1),
                    'min_samples_split': np.arange(2,10,1),
                    'min_samples_leaf': np.arange(1,10,1)
                },
                "Logistic Regression":{
                    'penalty': ['l1', 'l2'],
                    'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]
                },
                "K-Nearest Neighbors":{
                    'n_neighbors': range(1,31),
                    'weights': ['uniform', 'distance'], 
                    'p': [1, 2]
                },
                "Support Vector Machine":{
                    'C': [0.1, 1, 10, 100], 
                    'gamma': [1, 0.1, 0.01, 0.001], 
                    'kernel': ['rbf', 'linear']
                }

            }                
                
            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,
                                             y_test=y_test,models=models, param=params)
            
            ## Get Best Model Score from dict
            best_model_score = max(sorted(model_report.values()))

            ## Get Best Model Name from dict
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            if best_model_score<0.85:
                raise CustomException("Best Model Not Found")
            logging.info(f"Best found model on training and testing data")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted=best_model.predict(X_test)

            class_report = classification_report(y_test, predicted)
            return class_report


        except Exception as e:
            raise CustomException(e,sys)
