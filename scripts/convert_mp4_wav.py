from moviepy.editor import *
import sys
from settings import *

def convert(video_name):
    video_path = f'{os.path.join(VIDEOS_DIR,video_name)}.mp4'

    audio_path = f'{os.path.join(AUDIO_DIR,video_name)}.wav'

    clip = VideoFileClip(video_path)

    clip.audio.write_audiofile(audio_path)



if __name__=='__main__':

    video_name = sys.argv[1]
    convert(video_name)

    