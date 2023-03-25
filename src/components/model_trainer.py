import os
import sys
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

            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,
                                             y_test=y_test,models=models)
            
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
