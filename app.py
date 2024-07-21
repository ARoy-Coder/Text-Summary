from flask import Flask, render_template, request
import spacy
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tokenize import sent_tokenize
from heapq import nlargest
import string
from collections import Counter



app = Flask(__name__)

# Load SpaCy English model
nlp = spacy.load('en_core_web_sm')

# Home page
@app.route('/')
def home():
    return render_template('index.html')

# Summarization function
def summarize_text(text, num_sentences=2):
    # Tokenize the text into sentences
    sentences = sent_tokenize(text)
    
    # Join sentences into a single string for SpaCy to process
    doc = nlp(" ".join(sentences))
    
    # Extract sentences with highest rank and return as summary
    summary = sorted(doc.sents, key=lambda x: x.rank, reverse=True)[:num_sentences]
    
    return " ".join(str(sentence) for sentence in summary)

# Summarization route
@app.route('/summarize', methods=['POST'])
def summarize():
    if request.method == 'POST':
        text = request.form['text']
        summary = summarize_text(text)
        return render_template('result.html', text=text, summary=summary)

if __name__ == '__main__':
    app.run(debug=True)
