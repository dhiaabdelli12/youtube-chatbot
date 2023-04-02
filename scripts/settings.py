import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODELS_DIR = os.path.join(ROOT_DIR, 'models')

RESOURCES_DIR = os.path.join(ROOT_DIR, 'resources')

VIDEOS_DIR = os.path.join(RESOURCES_DIR, 'videos')
AUDIO_DIR = os.path.join(RESOURCES_DIR, 'audio')

TRANSCRIPTIONS_DIR = os.path.join(RESOURCES_DIR, 'transcriptions')