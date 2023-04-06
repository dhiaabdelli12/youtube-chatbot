import sys
from settings import *
from transformers import pipeline
import abc
import argparse
import json
args = abc.abstractproperty()

nlp = pipeline('question-answering', model='etalab-ia/camembert-base-squadFR-fquad-piaf', tokenizer='etalab-ia/camembert-base-squadFR-fquad-piaf')


def get_answer(query, text_file):
    with open(text_file, mode='r') as f:
        document = json.load(f)['transcription']

    result = nlp({
    'question': query,
    'context': document
    })

    return result['answer']
    


def parse_args():
    parser = argparse.ArgumentParser(
        description='Video to text')

    parser.add_argument('--from-transcript', action = 'store', nargs=1, type=str)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    global_args = parse_args()
    args.from_transcript = global_args.from_transcript
    if args.from_transcript[0] is not None:
        text_file = args.from_transcript[0]
        text_file_path = f'{os.path.join(TRANSCRIPTIONS_DIR,text_file)}.json'
        while True:
            query = input('Enter your query: ')
            answer = get_answer(query,text_file_path)
            print(answer)
            print('\n')




    



        

    