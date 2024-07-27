from flask import Flask, request, render_template, jsonify
import spacy
from collections import defaultdict
from heapq import nlargest

# Initialize spaCy model
nlp = spacy.load('en_core_web_sm')

app = Flask(__name__)

def summarize_text(text, num_sentences=3):
    doc = nlp(text)

    # Tokenize sentences
    sentences = [sent.text for sent in doc.sents]
    
    # Calculate term frequency
    word_freq = defaultdict(int)
    for token in doc:
        if token.is_stop or token.is_punct:
            continue
        word_freq[token.text.lower()] += 1
    
    # Normalize term frequency
    max_freq = max(word_freq.values())
    word_freq = {word: freq / max_freq for word, freq in word_freq.items()}
    
    # Score sentences
    sentence_scores = defaultdict(int)
    for sent in doc.sents:
        for token in sent:
            if token.text.lower() in word_freq:
                sentence_scores[sent.text] += word_freq[token.text.lower()]
    
    # Select the top sentences
    top_sentences = nlargest(num_sentences, sentence_scores, key=sentence_scores.get)
    return ' '.join(top_sentences)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process_text():
    # Get text from the form
    text = request.form['text']
    
    # Summarize text
    summary = summarize_text(text)
    
    response = {
        'summary': summary
    }
    
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
