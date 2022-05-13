#!/usr/bin/env python3

from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import argparse
import joblib
import pandas as pd
import os
import warnings
from joblib import dump, load
warnings.filterwarnings("ignore")

# First of all, let's define some global variables
MODEL_DIR = os.environ["MODEL_DIR"]
STAND_DIR = os.environ["STAND_DIR"]
REGRES_DIR = os.environ["REGRES_DIR"]

# These variables will contain the path to the files where the model and the standarizer tool will be saved
STAND_PATH = os.path.join(MODEL_DIR, STAND_DIR)
REGRES_PATH = os.path.join(MODEL_DIR, REGRES_DIR)



def train(rand=516):
    # First we load the dataset
    data = load_boston()
    df = pd.DataFrame(data['data'], columns = data['feature_names'])
    df = df.drop(['B'], axis = 1) # This feature measures the proportion of black people by town and for ethical reasons we will not use it. We will use the rest of the features.

    new_data = pd.DataFrame() # I have looked at the correlation coeficient of all the other features and all of them have a correlation coefficient over 0.2, so we will use all of them for our model
    variables_to_standarize = ['CRIM', 'ZN', 'INDUS', 'NOX', 'RM', 'AGE', 'TAX', 'PTRATIO', 'LSTAT', 'DIS']

    # now I am going to standarize all the features that are not integeers
    st = StandardScaler()
    new_data[variables_to_standarize] = st.fit_transform(df[variables_to_standarize])
    for i in df.columns:
        if i not in variables_to_standarize:
            new_data[i] = df[i]
    X = np.array(new_data)
    y = np.array(data['target'])
    # now let's divide our data in two sets, one of them for training our model and the other for testing it
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=rand)
    reg = LinearRegression() # Let's create our linear regressor
    reg.fit(x_train, y_train) # Here we train our linear regressor
    result = reg.score(x_test, y_test) # Then we obtain the accuracy of our model
    
    # We save our model and our standarizer into the corresponding files
    dump(reg, REGRES_PATH)
    dump(st, STAND_PATH)
    print('Accuracy of the model: {}\nMAE of the model: {}\n'.format(result, mean_absolute_error(y_test, reg.predict(x_test))))
    print('Parameters of the model: {} and intercept: {}\n'.format(reg.coef_, reg.intercept_))
    return result, reg, st

def predict(X, model):
    return model.predict(X)

# We define the arguments of our script. One of them for training or predict and the rest for the values used to predict
parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, help='train or predict', default='train')
parser.add_argument('--crim', type=float, help='per capita crime rate', default=0.)
parser.add_argument('--zn', type=float, help='proportion of residential land zoned for lots over 25000 sq. ft.', default=0.)
parser.add_argument('--indus', type=float, help='proportion of non-retail business acres per town', default=0.)
parser.add_argument('--chas', type=float, help='1 if tract bounds river, 0 otherwise', default=0.)
parser.add_argument('--nox', type=float, help='nitric oxides concentration (parts per 10 million)', default=0.)
parser.add_argument('--rm', type=float, help='average number of rooms per dwelling', default=0.)
parser.add_argument('--age', type=float, help='proportion of owner-occupied units built prior to 1940', default=0.)
parser.add_argument('--dis', type=float, help='weighted distances to five Boston employment centres', default=0.)
parser.add_argument('--rad', type=float, help='index of accessibility to radial highways', default=0.)
parser.add_argument('--tax', type=float, help='full-value property-tax rate per $10,000', default=0.)
parser.add_argument('--ptratio', type=float, help='pupil-teacher ratio by town', default=0.)
parser.add_argument('--lstat', type=float, help='percentage of lower status of the population', default=0.)

args = parser.parse_args()

if args.mode == 'train':
    train()
    
elif args.mode == 'predict':
    # We load the standarizer and the model
    st = load(STAND_PATH)
    reg = load(REGRES_PATH)
    # Collect all input variables into a dataframe
    values = np.array([args.crim, args.zn, args.indus, args.nox, args.rm, args.age, args.tax, args.ptratio, args.lstat, args.dis, args.chas, args.rad]).reshape(1, -1)
    orig_values = np.array([args.crim, args.zn, args.indus, args.nox, args.rm, args.age, args.tax, args.ptratio, args.lstat, args.dis, args.chas, args.rad]).reshape(1, -1)
    df = pd.DataFrame(values, columns=['CRIM', 'ZN', 'INDUS', 'NOX', 'RM', 'AGE', 'TAX', 'PTRATIO', 'LSTAT', 'DIS', 'CHAS', 'RAD'])
    variables_to_standarize = ['CRIM', 'ZN', 'INDUS', 'NOX', 'RM', 'AGE', 'TAX', 'PTRATIO', 'LSTAT', 'DIS']
    # Then standarize the needed ones
    df[variables_to_standarize] = st.transform(df[variables_to_standarize])
    # Finally, predict the output
    pred = predict(np.array(df), reg)
    print('Given the following values: \n {} \n the expected value is: {}$'.format(pd.DataFrame(orig_values, columns=['CRIM', 'ZN', 'INDUS', 'NOX', 'RM', 'AGE', 'TAX', 'PTRATIO', 'LSTAT', 'DIS', 'CHAS', 'RAD']), pred[0]*1000))
    
    
else:
    print('Enter a valid mode: train or predict. {} is not valid'.format(args.mode))
