# 1. we import some dependecies 
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import AdaBoostClassifier
import lightgbm as lgb
import catboost as cb
import warnings
warnings.filterwarnings('ignore')

# 2. create a function to load data ad split it into train-test
def load_data() ->  np.ndarray:
    "load and split dataset"

    data = load_iris() # It is a default datset in scikit-learn

    #split the item 
    X = data.data # create X for the features
    y = data.target # create y for target column

    # split the datset into train and test 
    X_train , y_train , X_test, y_test = train_test_split(X, y, test_size= 0.2, random_state= 42) # test_size = 0.2-> 80 percent train and 20 percent test ; random_state = 42 -> shuffle the dataset before appyling split

    # return the items 
    return X_train, y_train, X_test, y_test

def train_lightgbm(X_train, y_train):
    """Train LightGBM with GridSearchCV"""
    # model initialization
    model = lgb.LGBMClassifier(random_state=42, verbose=-1) # verbose -1 -> standard output 
    
    # different value for different parameter present in the classifier
    param_grid = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7]
    }
    
    # given model= classifier , param_grid = parameter_dictionary , cv = batch_size that means we put data into model by batch of 3 [ first 3 will go -> second 3 will go], scoring= "accuracy" -> maintain accuracy all the time
    grid_search = GridSearchCV(model, param_grid, cv=3, scoring='accuracy')

    # fit the model with training dataset
    grid_search.fit(X_train, y_train)
    
    # returning 1. the model that gain accuray on high perfomance metric 2. the bestparameter 
    return grid_search.best_estimator_, grid_search.best_params_

def train_adaboost(X_train, y_train):
    """Train AdaBoost with GridSearchCV"""
    # model initialization
    model = AdaBoostClassifier(random_state=42)
    
    # different value for different parameter present in the classifier
    param_grid = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 1.0]
    }
    
    # given model= classifier , param_grid = parameter_dictionary , cv = batch_size that means we put data into model by batch of 3 [ first 3 will go -> second 3 will go], scoring= "accuracy" -> maintain accuracy all the time
    grid_search = GridSearchCV(model, param_grid, cv=3, scoring='accuracy')

    # fit the model with training dataset
    grid_search.fit(X_train, y_train)

    # returning 1. the model that gain accuray on high perfomance metric 2. the bestparameter 
    return grid_search.best_estimator_, grid_search.best_params_

def evaluate_model(model, X_test, y_test, model_name):
    """Evaluate a single model"""
    # predict model on the test set 
    y_pred = model.predict(X_test)

    # get the accuracy by given test and prediction on 'accuracy_score' function
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n{model_name} Results:")
    print(f"Accuracy: {accuracy:.4f}") # get the 4 values after (.)
    print(f"Classification Report:\n{classification_report(y_test, y_pred)}")
    
    return accuracy

def main():
    """Main function to run all models"""
    print("Loading data...")
    X_train, X_test, y_train, y_test = load_data()
    
    # Train models
    print("Training LightGBM...")
    lgb_model, lgb_params = train_lightgbm(X_train, y_train)
    print(f"Best LightGBM params: {lgb_params}")
    
    print("Training AdaBoost...")
    ada_model, ada_params = train_adaboost(X_train, y_train)
    print(f"Best AdaBoost params: {ada_params}")

    # Evaluate models
    print("\n" + "="*50)
    print("MODEL EVALUATION")
    print("="*50)
    
    lgb_acc = evaluate_model(lgb_model, X_test, y_test, "LightGBM")
    ada_acc = evaluate_model(ada_model, X_test, y_test, "AdaBoost")
    
    # Compare results
    results = {
        'LightGBM': lgb_acc,
        'AdaBoost': ada_acc
    }

    best_model = max(results, key=results.get)
    print(f"\nBest Model: {best_model} with accuracy: {results[best_model]:.4f}")

    return results

if __name__ == "__main__":
    main()
