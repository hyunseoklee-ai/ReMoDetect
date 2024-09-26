import torch
import math
from torch.optim.lr_scheduler import _LRScheduler
from datasets import load_dataset, concatenate_datasets
import random
import pandas as pd
from tqdm import tqdm

def process_spaces(text):
    return text.replace(
        ' ,', ',').replace(
        ' .', '.').replace(
        ' ?', '?').replace(
        ' !', '!').replace(
        ' ;', ';').replace(
        ' \'', '\'').replace(
        ' ’ ', '\'').replace(
        ' :', ':').replace(
        '<newline>', '\n').replace(
        '`` ', '"').replace(
        ' \'\'', '"').replace(
        '\'\'', '"').replace(
        '.. ', '... ').replace(
        ' )', ')').replace(
        '( ', '(').replace(
        ' n\'t', 'n\'t').replace(
        ' i ', ' I ').replace(
        ' i\'', ' I\'').replace(
        '\\\'', '\'').replace(
        '\n ', '\n').strip()


class WarmupCosineDecayScheduler(_LRScheduler):
    def __init__(self, optimizer, warmup_steps, total_steps, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        super(WarmupCosineDecayScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # Linear warmup
            return [base_lr * self.last_epoch / self.warmup_steps for base_lr in self.base_lrs]
        else:
            # Cosine decay
            return [base_lr * (1 + math.cos(math.pi * (self.last_epoch - self.warmup_steps) / (self.total_steps - self.warmup_steps))) / 2
                    for base_lr in self.base_lrs]

def load_reg_data(cache_dir='.cache'):
    dataset1 = load_dataset('Anthropic/hh-rlhf',cache_dir=cache_dir)
    dataset2 = load_dataset('Dahoas/synthetic-instruct-gptj-pairwise',cache_dir=cache_dir)['train'].train_test_split(test_size=0.1)

    dataset_train = dataset1['train']['chosen']+dataset1['train']['rejected']+list(map(lambda x: x[0] + x[1] ,zip(dataset2['train']['prompt'],dataset2['train']['chosen'])))+list(map(lambda x: x[0] + x[1] ,zip(dataset2['train']['prompt'],dataset2['train']['rejected'])))
    dataset_test= dataset1['test']['chosen']+dataset1['test']['rejected']+list(map(lambda x: x[0] + x[1] ,zip(dataset2['train']['prompt'],dataset2['test']['chosen'])))+list(map(lambda x: x[0] + x[1] ,zip(dataset2['train']['prompt'],dataset2['test']['rejected'])))

    return dataset_train, dataset_test

class PerturbedDataset(torch.utils.data.Dataset):
    def __init__(self, pos,perturb, neg):
        self.pos = pos
        self.perturb = perturb
        self.neg = neg

    def __getitem__(self, idx):
        item_pos = {key: torch.tensor(val[idx]) for key, val in self.pos.items()}
        item_perturb = {key: torch.tensor(val[idx]) for key, val in self.perturb.items()}
        item_neg = {key: torch.tensor(val[idx]) for key, val in self.neg.items()}
        
        
        return item_pos,item_perturb, item_neg

    def __len__(self):
        return len(self.pos['input_ids'])    
    
class RegularDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings
        
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx])
                for key, val in self.encodings.items()}
        return item

    def __len__(self):
        # return len(self.encodings.items())
        return len(self.encodings['input_ids'])
    
def load_perturbed_data():
    import json
    f = pd.read_csv(f"data/HC3cleaned_para_LLMs.csv")

    a_human = f["human"].tolist()
    a_chat = f['ChatGPT'].fillna("").tolist()
    a_perturb = f['paraphrased'].fillna("").tolist()

    res = []
    for i in range(len(a_human)):
        if len(a_human[i].split()) > 1 and len(a_chat[i].split()) > 1:
            res.append([a_human[i],a_perturb[i], a_chat[i]])

    data_new = {
        'train': {
            'pos': [],
            'perturb': [],
            'neg': [],
        },
        'test': {
            'pos': [],
            'perturb': [],
            'neg': [],
        }
    }

    index_list = list(range(len(res)))
    random.seed(0)
    random.shuffle(index_list)

    total_num = len(res)
    for i in tqdm(range(total_num), desc="parsing data"):
        if i < total_num * 0.8:
            data_partition = 'train'
        else:
            data_partition = 'test'
        data_new[data_partition]['neg'].append(process_spaces(res[index_list[i]][0]))
        data_new[data_partition]['perturb'].append(process_spaces(res[index_list[i]][1]))
        data_new[data_partition]['pos'].append(process_spaces(res[index_list[i]][2]))



    return data_new
