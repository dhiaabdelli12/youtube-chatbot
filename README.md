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
Install FFmpeg:
- On Ubuntu: 
    ```bash 
    sudo apt-get install ffmpeg
    ```
- On Windows: 
      - Download the latest static build of FFmpeg from the official website: https://ffmpeg.org/download.html#build-windows
      - Extract the downloaded ZIP file to a folder on your system.
      - Add the path to the bin folder of the extracted FFmpeg to your system's PATH environment variable

## Usage

### Transcription

- Transcribe videos from the urls JSON file in data folder using the following command:
```bash 
python transcribe.py
```
- Transcribe videos that have already been downloaded locally and stored in the folder data/videos using the following command:
```bash 
python transcribe.py --locally
```
- Transcribe a Youtube playlist using the following command:
```bash 
python transcribe.py --playlist YT_PLAYLIST_URL
```

- Transcribe a single Youtube Video using the following command:
```bash 
python transcribe.py --url YT_VIDEO_URL
```


#### Additional Options

- `--res`: The resolution of the video(s) to download (default: 360).
- `--no-save`: Add this to delete the video(s) after transcription.

#### Configuration

The tool uses the following paths:

- `input_path`: The path to the input file (default: `data/urls.json`).
- `videos_path`: The path to the folder where the videos are saved (default: `data/videos`).
- `output_path`: The path to the output file (default: `data/output.json`).

The tool also uses the Whisper's small model. The size of the small model is ~461M. You can change it in the code to use the base or another model.

- `model_name`: The name of the Whisper model to use (default: `small`).

