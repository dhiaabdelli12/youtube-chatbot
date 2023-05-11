# Video To Text
## Results

- Pre-fine-tuning CamemBERT on Sentences Similarity task using PAWS-C french dataset. 


### Summarization

| Model | Rouge-1 (Recall) | Loss
| --- | --- | --- |
| Pegasus - 0-shot| 0.43 | - |
| Pegasus - 0-shot quantized | 0.42 | - |
| Pegasus - finetuned | 0.7 | - |
| Pegasus - finetuned quantized | 0.65 | - |
| ProphetNet | - | 1.9 |
| ProphetNet quantized | - | 2.1 |
| T5 | - | 2.02 |
| T5 instruction-tuned | - | 1.13 |
| T5 instruction-tuned + prompt engineering | - | 0.87 |


### Question Answering
| Model | EM (Exact Match) | F1 | Loss
| --- | --- | --- | ---
| Camembert | 66.2 | 63 | 2.34
| Camembert quantized | 34 | 30 | 20
| DistilBert finetuned fquad | 29 | 28 | 12.4
| Distilbert finteuned fquad + pre-finetuned | 29.3 | 28 | 11


### Sentiment Analysis
| Model | Accuracy |
| --- | --- 
| Bert for Sequence Classification | 0.97 |
| Bert for Sequence Classification quantized | 0.9 |
| DistilBert for Sequence Classification | 0.95 |
| DistilBert for Sequence Classification quantized | 0.88 |


### Keywords Extraction

Epochs: 1 , Learning Rate: 3e-5, Batch Size: 16, Optimizer: AdamW

| Model | Validation F1-score | Validation Accuracy | Params | Size(Mb) |
| --- | --- | --- | --- | --- | 
| Prefinetuned CamemBERT | 0.9056 | 0.906 | 110M | 442 Mb |
| + Dynamic Quantization | 0.3253 | 0.453 | 110M | 186 Mb |

- Fine-tuning CamemBERT for French keywords extraction.


Epochs: 20, Learning Rate: 5e-5, Batch Size: 8, Optimizer: AdamW

| Model | Training Loss | Training Accuracy | Validation Loss | Validation Accuracy | Params | Size(Mb) |
| --- | --- | --- | --- | --- | --- | --- |
| Finetuned CamemBERT | 0.0016 | 0.9996 | 0.09359 | 0.9859 | 110M | 419 Mb |
| + Dynamic Quantization | - | - | 0.2880 | 0.9240 | 110M | 176 Mb |





## Setup
### Virtual environement
```
conda create --name video-to-text python=3.9
conda activate video-to-text 
```
### Dependencies
Install the required Python packages:
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

- `input_path`: The path to the input file (default: `resources/urls.json`).
- `videos_path`: The path to the folder where the videos are saved (default: `resources/videos`).
- `output_path`: The path to the output file (default: `resources/output.json`).

The tool also uses the Whisper's small model. The size of the small model is ~461M. You can change it in the code to use the base or another model.

- `model_name`: The name of the Whisper model to use (default: `small`).

