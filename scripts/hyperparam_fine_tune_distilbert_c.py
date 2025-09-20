import pandas as pd
from transformers import DistilBertTokenizer
import torch
from train_utils import *

import os
os.environ['HF_HOME'] = '../cache'

base_fp = "../models/c_model"
pretrained_model_name = 'distilbert-base-uncased'

# Defining some key variables that will be used later on in the training
MAX_LEN = 512
BATCH_SIZE = 16
EPOCHS = 10

device = 'cuda' if torch.cuda.is_available() else 'cpu'

#seed_everything(seed = 42)

# Read the data
df = pd.read_csv('../data/c_training/dataset_III.csv', encoding='cp1252')

# transform label scores into 
df['Label Score (Ratio of positive labels)'] = df['Label Score (Ratio of positive labels)'].astype('float')
df['text'] = df['Wiki Title'] + ". " + df['DbPedia description']

df = df.dropna(subset=['text'])
df = df[df['text'].apply(lambda x: len(x)>50)]

# shuffle the data
df = df.sample(frac=1, random_state=89)

# reset index
df = df.reset_index(drop=True)

# train-validation-test split
N = round(len(df)*.7)
N_valid = round(len(df)*.85)

df['train_valid_test'] = 'train'
df.loc[N:N_valid,'train_valid_test'] = 'valid'
df.loc[N_valid:,'train_valid_test'] = 'test'

# keep only the necessary columns
df = df[['text', 'Label Score (Ratio of positive labels)', 'train_valid_test']]

# rename
df.columns = ['text', 'label', 'train_valid_test']

# define the tokenizer
tokenizer = DistilBertTokenizer.from_pretrained(pretrained_model_name, truncation=True)

# define the parameter sets to explore
param_sets = [{"lr" : 1e-05, "folder":"lr1e-05_wd0", 'weight_decay':0},
              {"lr" : 2e-05, "folder":"lr2e-05_wd0", 'weight_decay':0}, 
              {"lr" : 1e-04, "folder":"lr1e-04_wd0", 'weight_decay':0},
              
              {"lr" : 1e-05, "folder":"lr1e-05_wd001", 'weight_decay':0.01},
              {"lr" : 2e-05, "folder":"lr2e-05_wd001", 'weight_decay':0.01}, 
              {"lr" : 1e-04, "folder":"lr1e-04_wd001", 'weight_decay':0.01},
              
              {"lr" : 1e-05, "folder":"lr1e-05_wd005", 'weight_decay':0.05},
              {"lr" : 2e-05, "folder":"lr2e-05_wd005", 'weight_decay':0.05}, 
              {"lr" : 1e-04, "folder":"lr1e-04_wd005", 'weight_decay':0.05},
              ]

for ps in param_sets:
    lr = ps['lr']
    folder = ps['folder']
    weight_decay = ps['weight_decay']

    data_dict = {}

    for subpart in ['train', 'valid', 'test']:
        
        # take only the relevant part
        temp = df[df['train_valid_test']==subpart]
        
        # reset_index
        temp.reset_index(drop=True, inplace=True)

        print(f"{subpart} Dataset Shape: {temp.shape}")

        # push into the dictionary
        data_dict[subpart] = RegressionDataset(temp, tokenizer, MAX_LEN)

    data_params = {'batch_size': BATCH_SIZE,
                    'num_workers': 0
                    }

    model = DistilBERTClass(pretrained_model_name = pretrained_model_name)
    model.to(device)

    train(EPOCHS,
            model,
            data_dict['train'],
            data_dict['valid'],
            data_params,
            best_model_folder=f"{base_fp}/{folder}",
            lr=lr,
            steps=150,
            weight_decay=weight_decay,
        )

    print('Finished!')