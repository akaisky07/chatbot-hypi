from flask import Flask, render_template, request

app = Flask(__name__)

import numpy as np
import string
from nltk.corpus import stopwords
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
from sklearn.pipeline import Pipeline

# Assuming the provided dataset path is correct, read the CSV file
df = pd.read_csv('dialogs.txt', sep='\t')

# Rename columns for clarity
df.columns = ['Questions', 'Answers']

# Add some sample data to the dataframe
b = {'Questions': 'Hi', 'Answers': 'hello'}
c = {'Questions': 'Hello', 'Answers': 'hi'}
d = {'Questions': 'how are you', 'Answers': "i'm fine. how about yourself?"}
e = {'Questions': 'how are you doing', 'Answers': "i'm fine. how about yourself?"}

df = pd.concat([df, pd.DataFrame([b, c, d, e])], ignore_index=True)

# Define a function for text cleaning
def cleaner(x):
    return [a for a in (''.join([a for a in x if a not in string.punctuation])).lower().split()]

# Create a pipeline with CountVectorizer, TfidfTransformer, and DecisionTreeClassifier
Pipe = Pipeline([
    ('bow', CountVectorizer(analyzer=cleaner)),
    ('tfidf', TfidfTransformer()),
    ('classifier', DecisionTreeClassifier())
])

# Fit the pipeline with the training data
Pipe.fit(df['Questions'], df['Answers'])

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        user_input = request.form['user_input']
        prediction = Pipe.predict([user_input])[0]
        return render_template('index.html', user_input=user_input, prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)

