## Loading Data

from types import new_class
from typing import Tuple, Union, List
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from keras.utils import to_categorical

## Data Preprocessing
XY = Tuple[np.ndarray, np.ndarray]
Dataset = Tuple[XY, XY]
LogRegParams = Union[XY, Tuple[np.ndarray]]
XYList = List[XY]

def get_keras_model_parameters(model: tf.keras.Model) -> List[np.ndarray]:
    return [param.numpy() for param in model.weights]

def set_keras_model_params(model: tf.keras.Model, params: List[np.ndarray]) -> None:
    model.set_weights(params)

def create_keras_model():
    model = Sequential()
    model.add(Dense(50, activation='relu', input_shape=(8,)))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(3, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def load_data_client1() -> Dataset:
    ## Load Dataset
    fdf = pd.read_csv('data/Train Data/Train Data Zip/frequency_domain_features_train.csv')
    hrn = pd.read_csv('data/Train Data/Train Data Zip/heart_rate_non_linear_features_train.csv')
    tdf = pd.read_csv('data/Train Data/Train Data Zip/time_domain_features_train.csv')

    train_df = pd.merge(fdf, hrn, on='uuid')
    train_df = pd.merge(train_df, tdf, on='uuid')

    ## Test Data
    tfdf = pd.read_csv('data/Test Data/Test Zip/frequency_domain_features_test.csv')
    thrn = pd.read_csv('data/Test Data/Test Zip/heart_rate_non_linear_features_test.csv')
    ttdf = pd.read_csv('data/Test Data/Test Zip/time_domain_features_test.csv')

    test_df = pd.merge(tfdf, thrn, on='uuid')
    test_df = pd.merge(test_df, ttdf, on='uuid')

    df = pd.concat([train_df, test_df])
    df = df.drop(['uuid', 'HR'], axis=1)
    #df = df[0:3000]

    ## Label Encoder
    lb = LabelEncoder()  ## Encoder that convert and store all the information
    df['condition'] = lb.fit_transform(df['condition'])

    x = df.drop('condition', axis=1)
    y = df.condition

    ## Model
    lrr = LogisticRegression(penalty="l2",)
    lrr.fit(x, y)
    result = permutation_importance(lrr, x, y, n_repeats=3, random_state=0)
    sorted_idx = result.importances_mean.argsort()
    index = sorted_idx[25:33]
    x = x.iloc[:, index]

    # Standardizing the features
    sc = StandardScaler()
    x = sc.fit_transform(x)

    y = to_categorical(y)

    """ Select the 80% of the data as Training data and 20% as test data """
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=85, shuffle=True, stratify=y)
    return (x_train, y_train), (x_test, y_test)


def load_data_client2() -> Dataset:
    ## Load Dataset
    fdf = pd.read_csv('data/Train Data/Train Data Zip/frequency_domain_features_train.csv')
    hrn = pd.read_csv('data/Train Data/Train Data Zip/heart_rate_non_linear_features_train.csv')
    tdf = pd.read_csv('data/Train Data/Train Data Zip/time_domain_features_train.csv')

    train_df = pd.merge(fdf, hrn, on='uuid')
    train_df = pd.merge(train_df, tdf, on='uuid')

    ## Test Data
    tfdf = pd.read_csv('data/Test Data/Test Zip/frequency_domain_features_test.csv')
    thrn = pd.read_csv('data/Test Data/Test Zip/heart_rate_non_linear_features_test.csv')
    ttdf = pd.read_csv('data/Test Data/Test Zip/time_domain_features_test.csv')

    test_df = pd.merge(tfdf, thrn, on='uuid')
    test_df = pd.merge(test_df, ttdf, on='uuid')

    df = pd.concat([train_df, test_df])
    df = df.drop(['uuid', 'HR'], axis=1)
   # df = df[0:3000]

    ## Label Encoder
    lb = LabelEncoder()  ## Encoder that convert and store all the information
    df['condition'] = lb.fit_transform(df['condition'])

    x = df.drop('condition', axis=1)
    y = df.condition

    ## Same
    lrr = LogisticRegression(penalty="l2", )
    lrr.fit(x, y)
    result = permutation_importance(lrr, x, y, n_repeats=3, random_state=0)
    sorted_idx = result.importances_mean.argsort()
    index = sorted_idx[17:25]
    x = x.iloc[:, index]

    # Standardizing the features
    sc = StandardScaler()
    x = sc.fit_transform(x)


    y = to_categorical(y)

    """ Select the 80% of the data as Training data and 20% as test data """
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=77, shuffle=True, stratify=y)
    return (x_train, y_train), (x_test, y_test)

def shuffle(X: np.ndarray, y: np.ndarray) -> XY:
    """Shuffle X and y Datasets"""
    randon_gen = np.random.default_rng()
    perm = randon_gen.permutation(len(X))
    return X[perm], y[perm]


def partition(X: np.ndarray, y: np.ndarray, num_partitions: int) -> XYList:
    """Split X and y Datasets into a variety of partitions."""
    return list(
        zip(np.array_split(X, num_partitions), np.array_split(y, num_partitions))
    )
