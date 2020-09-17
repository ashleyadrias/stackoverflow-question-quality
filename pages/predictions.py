import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
from joblib import load
from app import app
import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer



column1 = dbc.Col(
    [
        dcc.Markdown(
            """
        
            #### Enter your Stackoverflow Question Description: 
            - High-Quality signifies a high quality question
            - Low-Quality-Edit signifies that some improvements should be made to your question
            - Low-Quality-Closed signifies a low quality question and the risk of Stackoverflow closing your question
            """,style={'width': '90%', 'display': 'inline-block'}, className='mb-4'
        ),
        dcc.Textarea(id='tokens',placeholder='Example: What is the best way to add/remove stop words with spacy? I am using token.is_stop function and would like to make some custom changes to the set. I was looking at the documentation but could not find anything regarding of stop words. Thanks!',style={'height':100,'width': '90%', 'display': 'inline-block'},value='',className='mb-4'),
        dcc.Markdown(
            """
        
            #### Question Rating: 
            """,style={'width': '90%', 'display': 'inline-block'}, className='mb-4'
        ),
        html.Div(id='prediction-content', className='lead'),
        html.Img(src='assets/happy-coder-3.jpeg',style={'width': '90%', 'display': 'inline-block'}, className='img-fluid'),
        # dbc.FormText("Type something in the box above"),
               
        # for _ in ALLOWED_TYPES
    ],style={'display': 'inline-block'}
    # md=7,
)

column2 = dbc.Col(
    [   #et tu auras une recette selon les ingrédients que tu as

        # html.H2('Sandwich Recommender Marmiton', className='mb-5'), 
        # html.Div(id='prediction-content', className='lead'),
        # dcc.Link(id='url', href='', children="Lien De La Recette ICI!!!", target="_blank"),
        
        # dcc.Link(dbc.Button('Clique ICI pour voir la recette !!!', color='warning'), id='url', href='', target="_blank"),


        # html.A(html.Img(src='assets/Netflix_people.jpeg', className='img-fluid'), href="http://www.google.com/search?q='prediction-content',
        # html.Img(src='assets/Sandwich2.jpeg', className='img-fluid')
    ]
)

layout = dbc.Row([column1])


@app.callback([
    Output('prediction-content', 'children'),
    ], 
    [Input('tokens','value')]
)


def predict(tokens):

    pipeline = pickle.load(open("./notebooks/pipe_01.pkl", "rb"))

    y_pred = pipeline.predict([tokens])

    return list(y_pred)