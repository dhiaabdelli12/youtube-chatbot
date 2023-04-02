# Video To Text

This script allows you to input a video url and then ask questions about the content that's in the video.
The way this is achieved is by automating the following process:
- Downloading the youtube video
- Converting it to an audio file (wav)
- Transcribing the audio file into text
- Using camemBERT language model for question answering

## Setup
### Virtual environement
```
conda create --name video-to-text python=3.9
conda activate video-to-text 
```
### Dependencies
```
pip -r requirements.txt
cd scripts/
```

## Usage
The script is fairly simple to use. You need just to specify the source of your content.

To specify the youtube url as your source run the following command
```SHELL
python main.py --from-youtube
``` 
You will be asked to provide the youtube url.

If you hava an audio file. Place it in resources/audio/ directory and then run this command
```SHELL
python main.py --from-audio
``` 
You will be asked to provide the audio file name.