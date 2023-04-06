import speech_recognition as sr 
import sys
from settings import *
from transformers import pipeline
import abc
import argparse
from convert_mp4_wav import convert
from download_yt_vid import download_video
args = abc.abstractproperty()

nlp = pipeline('question-answering', model='etalab-ia/camembert-base-squadFR-fquad-piaf', tokenizer='etalab-ia/camembert-base-squadFR-fquad-piaf')

def transcribe_audio(audio_file): 
    audio_file_path = f'{os.path.join(AUDIO_DIR, audio_file)}.wav'
    r = sr.Recognizer()
    audio = sr.AudioFile(audio_file_path)

    with audio as source:
        audio_file = r.record(source)
        result = r.recognize_google(audio_file)
    text_file_path = os.path.join(TRANSCRIPTIONS_DIR, 'output.txt')

    with open(text_file_path, mode='w') as file:
        file.write(result)


def get_answer(query, text_file):
    with open(text_file, mode='r') as file:
        document = file.read()

    result = nlp({
    'question': query,
    'context': document
    })

    return result['answer']
    


def parse_args():
    parser = argparse.ArgumentParser(
        description='Video to text')
    parser.add_argument('--from-youtube', action = 'store_true')
    parser.add_argument('--from-audio', action = 'store_true')
    parser.add_argument('--from-transcript', action = 'store_true')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    global_args = parse_args()
    args.from_youtube = global_args.from_youtube
    args.from_audio = global_args.from_audio
    args.from_transcript = global_args.from_transcript

    if args.from_youtube:
        url = input('Youtube video url: ')
        print('Downloading Video.')
        download_video(url, 'test')
        print('Converting video to audio.')
        convert('test')
        print('Transcribing audio to text.')
        transcribe_audio('test')
        text_file_path = f'{os.path.join(TRANSCRIPTIONS_DIR,"output")}.txt'
        while True:
            query = input('Enter your query: ')
            answer = get_answer(query,text_file_path)
            print(answer)
            print('\n')
    elif args.from_audio:
        audio_file = input('Audio file name: ')
        audio_file_path = f'{os.path.join(AUDIO_DIR, audio_file)}.wav'
        text_file_path = f'{os.path.join(TRANSCRIPTIONS_DIR,"output")}.txt'
        print('Transcribing audio to text')
        transcribe_audio(audio_file)
        while True:
            query = input('Enter your query: ')
            answer = get_answer(query,text_file_path)
            print(answer)
            print('\n')

    elif args.from_transcript:
        text_file = input('Text file name: ')
        text_file_path = f'{os.path.join(TRANSCRIPTIONS_DIR,text_file)}.txt'
        while True:
            query = input('Enter your query: ')
            answer = get_answer(query,text_file_path)
            print(answer)
            print('\n')




    



        

    