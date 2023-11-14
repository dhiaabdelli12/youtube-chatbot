import pandas as pd
from simplet5 import SimpleT5
from sklearn.model_selection import train_test_split


df=pd.read_csv('./resources/train/train.csv')
df = df.rename(columns={"summary":"target_text", "dialogue":"source_text"})
df = df[['source_text', 'target_text']]
df['source_text'] = "abstractive summarization: " + df['source_text']
train_df, test_df = train_test_split(df, test_size=0.2)

model = SimpleT5()
model.from_pretrained(model_type="t5", model_name="t5-base")

# defining the different prompting methods
template_prompt = {"method":"template prompting","text":"Given the following text, give an abstractive summary : "}
conditional_prompt = {"method":"conditional prompting","text":"If the text contains famous names use it in summary "}

prompts = [template_prompt,conditional_prompt]

for prp in prompts:
    print(prp['method'])

    model.train(train_df=train_df[:5000],
            eval_df=test_df[:100], 
            source_max_token_len=180, 
            target_max_token_len=100, 
            batch_size=8, max_epochs=4)
model.load_model("/content/outputs/simplet5-epoch-3-train-loss-1.2929-val-loss-1.8517", use_gpu=True)

text_to_summarize="summarize: L’intelligence artificielle (IA) désigne une technologie produite par des êtres humains, qui traite systématiquement de grands ensembles de données selon un modèle itératif. Cela lui permet de prédire des résultats, en générant des réponses mathématiques ou linguistiques aux demandes de l’utilisateur. Elle détecte des formes ou « patterns » dans d’énormes volumes de données brutes, appelées données d’apprentissage, pour créer un modèle. Elle teste ensuite son modèle en posant une question dont elle connaît déjà la réponse et en analysant la précision de sa réponse. Les données générées par l’IA sont appelées données de test. Au fil du temps, à mesure qu’elle multiplie les entrées et les données de test, elle apprend et itère de mieux en mieux sur ce modèle"
model.predict(text_to_summarize)



"""
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

"""
