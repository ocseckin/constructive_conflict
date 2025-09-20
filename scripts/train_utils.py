import numpy as np
import re
from transformers import DistilBertModel
import torch
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, mean_squared_error
from torch.nn import MSELoss
from scipy.stats import spearmanr
from torch.utils.data import Dataset, DataLoader
from transformers.optimization import get_linear_schedule_with_warmup
from tqdm import tqdm

import json

import os
os.environ['HF_HOME'] = '../cache'

def remove_urls(text):
    """Removes URLs from a string."""
    return re.sub(r'http\S+', '', text)

class RegressionDataset(Dataset):

    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.text = dataframe.text
        self.targets = self.data.label
        self.max_len = max_len

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        text = str(self.text[index])
        text = " ".join(text.split())

        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            padding='max_length',
            return_token_type_ids=True,
            truncation=True,
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(self.targets[index], dtype=torch.float)
        }


def create_dataset(df, tokenizer, MAX_LEN, include_valid = True):
    # keep all data in one dictionary separately for train, valid and test
    data = {}

    # split the data into train, validation and test sets, tokenize and transform into Dataset objects
    if include_valid:
        for subpart in ['train', 'valid', 'test']:
            data[subpart] = df[df['train_valid_test'] == subpart].reset_index(drop=True)
            data[subpart].drop(columns=['train_valid_test'], inplace=True)
            data[subpart] = RegressionDataset(data[subpart], tokenizer, MAX_LEN)
    else:
        # validation set to be included in the test set
        df.loc[df['train_valid_test'] == 'valid', 'train_valid_test'] = 'train'

        for subpart in ['train', 'test']:
            data[subpart] = df[df['train_valid_test'] == subpart].reset_index(drop=True)
            data[subpart].drop(columns=['train_valid_test'], inplace=True)
            data[subpart] = RegressionDataset(data[subpart], tokenizer, MAX_LEN)
    
    return data

# re-define the model class
class DistilBERTClass(torch.nn.Module):
    def __init__(self, pretrained_model_name):
        super(DistilBERTClass, self).__init__()
        self.l1 = DistilBertModel.from_pretrained(pretrained_model_name)
        self.pre_classifier = torch.nn.Linear(768, 768)
        self.dropout = torch.nn.Dropout(0.1)
        self.regressor = torch.nn.Linear(768, 1)

    def forward(self, input_ids, attention_mask, labels=None):
        distilbert_output = self.l1(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = distilbert_output[0]
        pooler = hidden_state[:, 0]
        pooler = self.pre_classifier(pooler)
        pooler = torch.nn.ReLU()(pooler)
        pooler = self.dropout(pooler)
        logits = self.regressor(pooler)

        # clip the values to get 1 at maximum and 0 at minimum
        #logits = torch.clamp(logits, min=0.0, max=1.0)
        
        return logits

def loss_fn(outputs, targets):
    return MSELoss()(outputs, targets)

def compute_metrics(outputs, targets):
    
    rmse = root_mean_squared_error(targets, outputs)
    mse = mean_squared_error(targets, outputs)
    mae = mean_absolute_error(targets, outputs)
    stat,pval = spearmanr(targets, outputs)

    return {"rmse": rmse, "mse":mse, "mae": mae, "spearmanr_stat":stat, "spearmanr_pval":pval}

def train(EPOCHS, model, train_set, test_set, data_params, best_model_folder, lr=1e-04, steps=150, weight_decay=0.01):
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    optimizer = torch.optim.AdamW(params=model.parameters(), lr=lr, weight_decay=weight_decay)
    
    best_loss = float('inf')
    patience = 3
    patience_counter = 0

    best_model_fp = f"{best_model_folder}/distilbert_best_model.bin"

    train_loader = DataLoader(train_set, **data_params)
    test_loader = DataLoader(test_set, **data_params)

    num_training_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_training_steps*.1, num_training_steps=num_training_steps)
    
    # log the training and validation loss
    training_log = {}

    model.train()
    for epoch in range(EPOCHS):
        i = 0
        for _, data in tqdm(enumerate(train_loader, 0)):
            
            i += 1

            ids = data['ids'].to(device, dtype = torch.long)
            mask = data['mask'].to(device, dtype = torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
            targets = data['targets'].to(device, dtype = torch.float)
            
            outputs = model(ids, mask, token_type_ids)

            optimizer.zero_grad()
            loss = loss_fn(outputs.view(-1), targets)
            if _ % 150==0:
                print(f'Epoch: {epoch}, Loss:  {loss.item()}')

            if (_ != 0) and (_ % steps == 0):
                eval_targets, eval_outputs = validation(test_loader, model)
                eval_ = compute_metrics(eval_outputs, eval_targets)
                eval_['train_mse'] = loss.item()
                print(eval_)
                training_log[f"{epoch}_{i}"] = eval_

                # Early stopping
                current_loss = eval_['mse']
                if current_loss < best_loss:
                    print("current:", current_loss, "\nbest:", best_loss)
                    best_loss = current_loss
                    patience_counter = 0
                    # Save the best model
                    print('Saving the best model...')
                    torch.save(model, best_model_fp)
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    with open(f"{best_model_folder}/training_log.json", "w") as f:
                        json.dump(training_log, f)
                    print("Early stopping!")
                    return
            
            loss.backward()
            optimizer.step()
            scheduler.step()

    with open(f"{best_model_folder}/training_log.json", "w") as f:
        json.dump(training_log, f)


def validation(test_loader, model):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model.eval()
    eval_targets=[]
    eval_outputs=[]
    with torch.no_grad():
        for _, data in tqdm(enumerate(test_loader, 0)):
            ids = data['ids'].to(device, dtype = torch.long)
            mask = data['mask'].to(device, dtype = torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
            targets = data['targets'].to(device, dtype = torch.float)
            outputs = model(ids, mask, token_type_ids)
            eval_targets = np.append(eval_targets, targets.cpu().numpy())
            eval_outputs = np.append(eval_outputs, outputs.cpu().numpy())
            
    return eval_targets, eval_outputs