import os
import sys
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_model
from dataclasses import dataclass

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

@dataclass
class ModelTrainerConfig:
    '''
        Desc: This class will join artifact folder with model trainer under artifact
    '''
    train_model_file_path = os.path.join('artifact/model_trainer','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config =ModelTrainerConfig()

    def initiate_model_trainer(self,train_arr,test_arr):
        try:
            X_train,y_train,X_test,y_test = (
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1],
            )

            models = {
                'logistic':LogisticRegression(),
                'svc':SVC(),
                'dt':DecisionTreeClassifier(),
                'random':RandomForestClassifier(criterion="entropy") 
            }

            model_report:dict = evaluate_model(X_train=X_train,X_test=X_test,
                                               y_train=y_train,y_test=y_test,
                                               models=models)
            # To get best model from model_report
            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(models.keys())[
                list(model_report.values()).index(best_model_score)
                ]
            best_model = models[best_model_name]
            logging.info(f'Best model found: {best_model_name} with accuracy score: {best_model_score}')
            logging.info('Saving model.....')
            save_object(
                file_path=self.model_trainer_config.train_model_file_path,
                obj=best_model
            )
        except Exception as e:
            raise CustomException(e,sys)
        