import pandas as pd
from transformers import DistilBertTokenizer
import torch
from train_utils import *

import os
os.environ['HF_HOME'] = '../cache'

pretrained_model_name = 'distilbert-base-uncased'

# Defining some key variables that will be used later on in the training
MAX_LEN = 512
BATCH_SIZE = 32
base_fp = '../models/ta_model'
EPOCHS = 4
steps = 20000
include_valid = False
test_set_during_training = 'valid' if include_valid else 'test'

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Read the data
df = pd.read_pickle('../data/submissions.parquet')

# remove URLs
df['clean_body'] = df['clean_body'].apply(lambda x: remove_urls(x))

# remove tifu, cmv
df['clean_body'] = df['clean_body'].apply(lambda x: x.replace('TIFU by', '').replace('tifu ', '').replace('tifu: ', '').replace('TIFU: ', '').replace('TIFU ', '').replace('Cmv: ', "").replace('cmv: ', "").replace('CMV ', "").replace('CMV: ', ''))
print('Dataframe loaded and shuffled...')

# keep only the text and the target value
df = df[['clean_body', 'mean_openai', 'train_valid_test']]

# rename the columns
df.columns = ['text', 'label', 'train_valid_test']

# call the tokenizer
tokenizer = DistilBertTokenizer.from_pretrained(pretrained_model_name, truncation=True)

data = create_dataset(df, tokenizer, MAX_LEN, include_valid = include_valid)

# define the parameter sets to be used for training
param_sets = [
            {"lr" : 2e-05, "folder":"lr2e-05_wd0_final", 'weight_decay':0}, 
            {"lr" : 1e-04, "folder":"lr1e-04_wd0", 'weight_decay':0},
              
            {"lr" : 2e-05, "folder":"lr2e-05_wd001", 'weight_decay':0.01}, 
            {"lr" : 1e-04, "folder":"lr1e-04_wd001", 'weight_decay':0.01},
              
            {"lr" : 2e-05, "folder":"lr2e-05_wd005", 'weight_decay':0.05}, 
            {"lr" : 1e-04, "folder":"lr1e-04_wd005", 'weight_decay':0.05},
              ]

data_params = {'batch_size': BATCH_SIZE,
                'num_workers': 0
                }

for ps in param_sets:
    lr = ps['lr']
    folder = ps['folder']
    weight_decay = ps['weight_decay']

    model = DistilBERTClass(pretrained_model_name=pretrained_model_name)
    model.to(device)

    print(f"Training with learning rate: {lr}, weight decay: {weight_decay}...\n")
    train(
        EPOCHS=EPOCHS,
        model=model,
        train_set=data['train'],
        test_set=data[test_set_during_training],
        data_params=data_params,
        best_model_folder=f"{base_fp}/{folder}",
        lr=lr,
        steps=steps,
        weight_decay=weight_decay,
        )
    
    print('Training finished, clearing cache...\n')
    torch.cuda.empty_cache()