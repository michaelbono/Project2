import sys
from sqlalchemy import create_engine
import sqlite3

import nltk
nltk.download(['punkt', 'wordnet'])

import re
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

import pandas as pd
from sqlalchemy import create_engine
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import warnings
import re
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
nltk.download('punkt')
nltk.download('stopwords')
warnings.simplefilter('ignore')


def load_data(database_filepath):
    '''
    Load dataset and return independent and dependent variables and column names
    input:
        database name and filepath
    output: 
        X: full messages
        Y: remaining fields in the data set
        cat_names: list of column names of Y
    '''
    full_path = 'sqlite:///' + database_filepath
    engine = create_engine(full_path)
    df = pd.read_sql_table('DisasterResponse',con=engine)

    X = df['message']
    Y = df.drop(['message','original','genre','id'], axis = 1)
    cat_names = Y.columns.tolist()
    
    return X, Y, cat_names


def tokenize(text):
    '''
    Normalize and tokenize input text
    input:
        string. text to be tokenized
    output:
        strings returned in tokenized format
    '''
    words = word_tokenize(text)
    return words


def build_model():
    '''
    Construct a machine learning model
    input: 
        none
    output:
        constructed ML pipeline
    '''
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    return pipeline

def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Score and evaluate effectiveness of ML model
    input:
        model: machine learning pipeline model
        X_test: testing message to evaluate
        Y_test: testing classification
        category_names: list of column names of Y
    output:
        prints report of scores
    '''
    y_test_pred = model.predict(X_test)
    cols = list(Y_test.columns.values)
    for i in range(len(cols)):
        report = classification_report(np.array(Y_test)[:, i],y_test_pred[:, i])
        print(cols[i])
        print(report)

def save_model(model, model_filepath):
    '''
    Exports ML model to a pickle file
    '''
    pickle.dump(model, open('models/classifier.pkl', 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
