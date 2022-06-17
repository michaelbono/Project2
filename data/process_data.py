import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    Load messages and categories datasets
    input:
        messages_filepath: string containing messages dataset filepath
        categories_filepath: string containing messages dataset filepath
    output:
        df: merged dataframe of messages and categories
    '''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, on = ['id'])
    return df


def clean_data(df):
    '''
    Clean and prepare data for modeling by splitting columns and turning categories into binary values
    input:
        df: dataframe containing the data to be cleaned
    output:
        df: cleaned dataframe
    '''
    categories = df['categories'].str.split(';',expand=True)
    row = categories.iloc[0]
    category_colnames = row.str.split('-', expand=True)[0]
    categories.columns = category_colnames
    
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]

        # convert column from string to numeric
        categories[column] = categories[column].astype(np.float64)
    
    df = df.drop(['categories'], axis = 1)
    df = df.join(categories)
    df = df.drop_duplicates()
    
    return df


def save_data(df, database_filename):
    '''
    Save the data to database
    input:
        df: dataframe to be saved
        database_filename: filepath of where the database should be saved to
    '''
    full_path = 'sqlite:///' + database_filename
    engine = create_engine(full_path)
    df.to_sql('DisasterResponse', engine, index=False, if_exists = 'replace')


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
