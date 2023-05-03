import spacy
from spacy.lang.fr.stop_words import STOP_WORDS
from collections import Counter
import json

# Load the model
nlp = spacy.load('fr_dep_news_trf')

# Load the JSON data
text_file="../resources/transcriptions/output.json"
with open(text_file, mode='r') as f:
        transcript = json.load(f)['transcription']

# Process the text with spaCy
doc = nlp(transcript)

# Define the list of stopwords to filter out
stopwords = list(STOP_WORDS)

# Define the list of parts of speech to include as keywords
pos_to_include = ['NOUN', 'PROPN', 'ADJ']

# Extract the keywords
keywords = []
for token in doc:
    if token.pos_ in pos_to_include and token.text.lower() not in stopwords:
        keywords.append(token.text)

# Count the frequency of each keyword
keyword_counts = Counter(keywords)

# Get the top 10 keywords
top_keywords = keyword_counts.most_common(10)

# Print the top keywords
keywords_list = []
print('Top Keywords:')
for keyword, count in top_keywords:
    # print(f'{keyword}: {count}')
    keywords_list.append(keyword)

print(keywords_list)