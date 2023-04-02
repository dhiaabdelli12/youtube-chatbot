import sys
from pytube import YouTube
from settings import *


def download_video(url, vid_name):
    yt = YouTube(url)
    video_title = f'{vid_name}.mp4'
    streams = yt.streams
    stream = yt.streams.filter(file_extension='mp4').first()
    stream.download(output_path=VIDEOS_DIR, filename=video_title)



if __name__ == '__main__':
    url = sys.argv[1]
    vid_name = sys.argv[2]
    download_video(url,vid_name)
    
