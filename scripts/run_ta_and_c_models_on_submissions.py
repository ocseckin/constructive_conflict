import pandas as pd
from torch.utils.data import DataLoader
from transformers import DistilBertTokenizer
from train_utils import *
import torch

# Initialize the models
ta_PATH = "../models/ta_model/lr2e-05_wd0_final/distilbert_best_model.bin"
ta_model = torch.load(ta_PATH, weights_only=False)
ta_model.eval()

c_PATH = "../models/c_model/lr2e-05_wd005_original/distilbert_best_model.bin"
c_model = torch.load(c_PATH, weights_only=False)
c_model.eval()

# Initialize the data
MAX_LEN = 512
VALID_BATCH_SIZE = 16

# Read the data
df = pd.read_pickle('../data/submissions.parquet')

# keep only the text and the target value
test_set = df[['clean_body', 'mean_openai']]

# rename the columns
test_set.columns = ['text', 'label']

test_set = test_set.reset_index(drop=True)

print("TEST Dataset: {}".format(test_set.shape))

# call the tokenizer
pretrained_model_name = 'distilbert-base-uncased'
tokenizer = DistilBertTokenizer.from_pretrained(pretrained_model_name, truncation=True)

test_set = RegressionDataset(test_set, tokenizer, MAX_LEN)

test_params = {
                'batch_size': VALID_BATCH_SIZE,
                'num_workers': 0
                }

test_loader = DataLoader(test_set, **test_params)


# Run the models
eval_targets, eval_outputs = validation(test_loader, ta_model)
df['ta_score'] = eval_outputs

eval_targets, eval_outputs = validation(test_loader, c_model)
df['c_score'] = eval_outputs

df.to_parquet('../data/submissions.parquet')