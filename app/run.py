import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('DisasterResponse', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    #adding related breakdown
    rel = df[df['related']==1].groupby('genre').count()['message']
    unrel = df[df['related']==0].groupby('genre').count()['message']
    genre_names = list(rel.index)
    
    #adding class breakdown
    class_counts = df.drop(['id', 'message', 'original', 'genre'], axis = 1).sum()
    class_counts = class_counts.sort_values(ascending = False)
    class_names = list(class_counts.index)
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
            'data': [
                Bar(
                    x = genre_names,
                    y = rel,
                    name = 'Related'
                ),
                Bar(
                    x = genre_names,
                    y = unrel,
                    name = 'Unrelated'
                )
            ],
            'layout': {
                'title': 'Messages by Genre and Relation',
                'yaxis': {
                    'title': 'Count of Messages'
                },
                'xaxis': {
                    'title': 'Genre'
                },
                'barmode': 'group'
            }
        },
        {
            'data': [
                Bar(
                    x=class_names,
                    y=class_counts,
                    name = 'Class'
                )
            ],
            'layout': {
                'title': 'Messages by Class Type',
                'yaxis': {
                    'title': 'Count of Messages'
                },
                'xaxis': {
                    'title': 'Class'
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3000, debug=True)


if __name__ == '__main__':
    main()
