import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

from transformers import DistilBertTokenizer
from transformers import DistilBertModel, DistilBertForSequenceClassification
from transformers import DistilBertTokenizerFast
from tqdm import tqdm

import json
from settings import *

import pandas as pd
from sklearn.preprocessing import OneHotEncoder

from transformers import logging

logging.set_verbosity_warning()
logging.set_verbosity_error()


import argparse

parser = argparse.ArgumentParser(description='Question answering Training Script')

parser.add_argument('-t', '--task', type=str, help='Training task')
parser.add_argument('-e', '--epochs', type=int, help='Number of training epochs')
parser.add_argument('-q', '--quantize',  action='store_true', help='Quantizing model')

args = parser.parse_args()



device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')





class QAModel(nn.Module):
    def __init__(self):
        super(QAModel, self).__init__()
        self.bert = DistilBertModel.from_pretrained(os.path.join(MODELS_DIR,'intent_classification_model'))
        #self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.fc = nn.Linear(self.bert.config.hidden_size, 2) 

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state  # Last-layer hidden states of the input sequence

        logits = self.fc(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        return start_logits, end_logits
 


class ProcessedDataset(Dataset):
    def __init__(self, raw_dataset):
        self.raw_dataset = raw_dataset
        self.processed = preprocess(self.raw_dataset, max_length=512, stride=128)

        self.input_ids = self.processed['input_ids']
        self.token_type_ids = self.processed['token_type_ids']
        self.attention_mask = self.processed['attention_mask']
        self.start_positions = self.processed['start_positions']
        self.end_positions = self.processed['end_positions']

    def __len__(self):
        return len(self.raw_dataset)

    def __getitem__(self, idx):
       return {
           'input_ids': torch.tensor(self.input_ids[idx],dtype=torch.long),
           'token_type_ids': torch.tensor(self.token_type_ids[idx],dtype=torch.long),
           'attention_mask': torch.tensor(self.attention_mask[idx],dtype=torch.long),
           'start_position': torch.tensor(self.start_positions[idx],dtype=torch.long),
           'end_position': torch.tensor(self.end_positions[idx],dtype=torch.long)
       }




class FquadDataset(Dataset):
    def __init__(self,split, tokenizer):
        self.data = {}
        self.questions = []
        self.contexts = []
        self.answers = []

        with open(os.path.join(RESOURCES_DIR,'data','qa',f'{split}.json'), 'r', encoding='UTF-8') as file:
            data = json.load(file)

        for data in data['data']:
            for p in data['paragraphs']:
                context = p['context']
                for qas in p['qas']:
                    question= qas['question']
                    context=context
                    answer=qas['answers'][0]
                    self.questions.append(question)
                    self.contexts.append(context)
                    self.answers.append(answer)

        self.data['questions'] = self.questions
        self.data['contexts'] = self.contexts
        self.data['answers'] = self.answers

        self.data = preprocess(self.data, tokenizer, max_length=512, stride=128)
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self,idx):
        return {
            'input_ids': torch.tensor(self.data['input_ids'][idx], dtype=torch.long),
            'attention_mask': torch.tensor(self.data['attention_mask'][idx], dtype=torch.long),
            'start_position': torch.tensor(self.data['start_positions'][idx], dtype=torch.long),
            'end_position': torch.tensor(self.data['end_positions'][idx], dtype=torch.long)
        }




def preprocess(data, tokenizer, max_length, stride):
    questions = [q.strip() for q in data["questions"][1:]]
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    inputs = tokenizer(
        questions,
        data["contexts"][1:],
        max_length=max_length,
        truncation="only_second",
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    offset_mapping = inputs.pop("offset_mapping")
    sample_map = inputs.pop("overflow_to_sample_mapping")
    answers = data["answers"][1:]
    start_positions = []
    end_positions = []

    for i, offset in enumerate(offset_mapping):
        sample_idx = sample_map[i]
        answer = answers[sample_idx]
        start_char = answer["answer_start"]
        end_char = answer["answer_start"] + len(answer["text"])
        sequence_ids = inputs.sequence_ids(i)

        # Find the start and end of the context
        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        while sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1

        # If the answer is not fully inside the context, label is (0, 0)
        if offset[context_start][0] > start_char or offset[context_end][1] < end_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            # Otherwise it's the start and end token positions
            idx = context_start
            while idx <= context_end and offset[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx - 1)

            idx = context_end
            while idx >= context_start and offset[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx + 1)


    
    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    
    return inputs

   


def finetune_bert(model,train_loader,num_epochs, learning_rate):
    optimizer = AdamW(model.parameters(), lr = learning_rate)
    loss_fn = nn.CrossEntropyLoss()
      
    for epoch in range(num_epochs):

        model.train()
        loop = tqdm(train_loader, leave=True)

        for batch in loop:
            optimizer.zero_grad()

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            start_positions = batch['start_position'].to(device)
            
            end_positions = batch['end_position'].to(device)

        

            start_logits, end_logits = model(input_ids, attention_mask=attention_mask)
            
            loss = loss_fn(start_logits, start_positions) + loss_fn(end_logits, end_positions)
            # calculate loss for every parameter that needs grad update
            loss.backward()
            # update parameters
            optimizer.step()
            # print relevant info to progress bar
            loop.set_description(f'Epoch {epoch}')
            loop.set_postfix(loss=loss.item())

    return model



def load_intent_classification_dataset():
    file_path = os.path.join(RESOURCES_DIR, 'data','intent_classification','intent_classification.csv')
    data = pd.read_csv(file_path)

    return data.head()


class IntentClassificationDataset(Dataset):
    def __init__(self):
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        file_path = file_path = os.path.join(RESOURCES_DIR, 'data','intent_classification','intent_classification.csv')
        self.data = pd.read_csv(file_path)
        self.texts = self.data['text-fr'].tolist()

        labels = self.data['intent-fr'].tolist()
        label_to_id = {label: idx for idx, label in enumerate(set(labels))} 
        self.labels = [label_to_id[label] for label in labels]

    

    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self,idx):
        text = self.texts[idx] 
        label = self.labels[idx] 

        encoded_inputs = self.tokenizer(text, padding='max_length', truncation=True, max_length=128, return_tensors='pt')
        input_ids = encoded_inputs['input_ids'].squeeze()
        attention_mask = encoded_inputs['attention_mask'].squeeze()

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'label': torch.tensor(label)
        }


def pre_finetune(model, dataset,num_epochs, learning_rate):
    optimizer = AdamW(model.parameters(), lr = learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):

        model.train()
        loop = tqdm(dataset, leave=True)

        for batch in loop:


            optimizer.zero_grad()
            input_ids, attention_mask, labels = batch['input_ids'].to(device), batch['attention_mask'].to(device), batch['label'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()

            loop.set_description(f'Epoch {epoch}')
            loop.set_postfix(loss=loss.item())

    return model


def save_inputs(inputs):
    with open('inputs.txt','w',encoding='UTF-8') as f:
        for ids in inputs['input_ids']:
            f.write(f'{tokenizer.decode(ids)}\n')



if __name__ == '__main__':

    task = args.task
    epochs = args.epochs
    quantize = args.quantize


    if epochs is None:
        print('Provide number of epochs')
        exit()


    if task == 'finetune':
        model = QAModel().to(device)
        # Move quantized model to the device
        model.to(device)

        train_data = FquadDataset('train', tokenizer)
        eval_data = FquadDataset('valid', tokenizer)


        train_loader = DataLoader(train_data, batch_size=4, shuffle=True)
        eval_loader = DataLoader(train_data, batch_size=4, shuffle=True)



        model = finetune_bert(model,train_loader,num_epochs=epochs, learning_rate=2e-5)

        if quantize == True:
            quantized_model = torch.quantization.quantize_dynamic(
            model , {torch.nn.Linear}, dtype=torch.qint8
                )
            torch.save(quantized_model.state_dict(), os.path.join(MODELS_DIR,'quantized_custom_model_fquad_distilbert.pth'))
        else:
            torch.save(model.state_dict(), os.path.join(MODELS_DIR,'custom_model_fquad_distilbert.pth'))

        
    elif task == 'pre-finetune':
        num_classes = 150
        model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')
        model.classifier = torch.nn.Linear(model.config.hidden_size,num_classes)

        model.to(device)

        ic_data = IntentClassificationDataset()
        ic_dataloader = DataLoader(ic_data, batch_size=4, shuffle=True)


        model = pre_finetune(model,ic_dataloader,num_epochs=epochs,learning_rate=2e-5)
        model.save_pretrained(os.path.join(MODELS_DIR,'intent_classification_model'))
    else:
        print('Not a valid task')



    
    



    
    



    
    