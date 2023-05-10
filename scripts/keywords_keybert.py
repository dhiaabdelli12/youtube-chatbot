import json
from keybert import KeyBERT

# Load the JSON data
text_file="../resources/transcriptions/output.json"
with open(text_file, mode='r') as f:
        transcript = json.load(f)['transcription']

# Initialize the KeyBERT model
model = KeyBERT( "distilbert-base-multilingual-cased",)

# Extract the keywords
keywords = model.extract_keywords(transcript)

# Print the top 10 keywords
keywords_list = []
print("Top Keywords:")
for keyword, score in keywords[:10]:
    print(keyword)
    keywords_list.append(keyword)
print(keywords_list)
