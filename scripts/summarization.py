from transformers import PegasusForConditionalGeneration, PegasusTokenizer
import torch
import os


if torch.cuda.is_available():
   device = torch.device("cuda")
else:
   device = torch.device("cuda")


pegasus_model = PegasusForConditionalGeneration.from_pretrained("google/Pegasus-cnn_dailymail").to(device)
pegasus_tokenizer = PegasusTokenizer.from_pretrained("google/Pegasus-cnn_dailymail")

file_path = '../resources/transcriptions/output.txt'

# Check if file exists
if os.path.exists(file_path):
    # Check if file is not empty
    if os.path.getsize(file_path) > 0:
        with open(file_path, 'r') as file:
            file_contents = file.read()
            input_text = ' '.join(file_contents.split())
            batch = pegasus_tokenizer.prepare_seq2seq_batch(input_text, truncation=True, padding='longest', return_tensors="pt").to(device)
                    
            summary_ids = pegasus_model.generate(**batch,
                                                        num_beams=6,
                                                        num_return_sequences=1,
                                                        no_repeat_ngram_size = 2,
                                                        length_penalty = 1,
                                                        min_length = 30,
                                                        max_length = 128,
                                                        early_stopping = True)
                    
            output = [pegasus_tokenizer.batch_decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)]

            print(output)

    else:
        print(f"Error: {file_path} is empty")
else:
    print(f"Error: {file_path} not found")

