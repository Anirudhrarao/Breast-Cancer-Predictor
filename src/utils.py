import os
import sys
import pickle
from src.exception import CustomException
from src.logger import logging

from sklearn.metrics import (accuracy_score, 
                             confusion_matrix, 
                             precision_recall_curve, 
                             f1_score, precision_score ,recall_score)


def save_object(file_path,obj):
    '''
        Desc: This function will responsible for saving our model and preprocessor in pickle
    '''
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path,"wb") as file_obj:
            pickle.dump(obj,file_obj)
    except Exception as e:
        raise CustomException(e,sys)
    

def evaluate_model(X_train,y_train,X_test,y_test,models):
    '''
        Desc: This function will evaluate our performance of models and than return dict
              of performance as report
    '''
    try:
        report = {}
        for i in range(len(list(models))):
            model = list(models.values())[i]
            model.fit(X_train,y_train)
            # make prediction
            y_pred = model.predict(X_test)
            test_model_accuracy = accuracy_score(y_test,y_pred)
            report[list(models.values())[i]] = test_model_accuracy
        return report
    
    except Exception as e:
        raise CustomException(e,sys)


def load_object(file_path):
    '''
        Desc: This function will read or load our pickled file
    '''
    try:
        with open(file_path,'rb') as file_obj:
            return pickle.load(file_obj)
        
    except Exception as e:
        raise CustomException(e,sys)