import argparse
import math
from datetime import datetime

import numpy as np
import torch
import transformers
from torch.utils.data import DataLoader
from torch.nn.functional import logsigmoid
from tqdm import tqdm
from transformers import (AdamW, AutoConfig, AutoModelForSequenceClassification,
                          AutoTokenizer)

from utils import (PerturbedDataset, RegularDataset, WarmupCosineDecayScheduler,
                   load_perturbed_data, load_reg_data, process_spaces)



parser = argparse.ArgumentParser(description="Training Configuration")

# Add arguments
parser.add_argument("--model", type=str, default='OpenAssistant/reward-model-deberta-v3-large-v2', help="Pre-trained model")
parser.add_argument("--num_labels", type=int, default=1, help="Number of labels for classification")
parser.add_argument("--cache_dir", type=str, default='.cache', help="Cache directory for model data")
parser.add_argument("--device", type=str, default='cuda', help="Device for the current model")
parser.add_argument("--device_old", type=str, default='cuda', help="Device for the teacher model")
parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
parser.add_argument("--batchsize", type=int, default=1, help="Training batch size")
parser.add_argument("--l2_ratio", type=float, default=0.01, help="Regularization ratio")
parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
parser.add_argument("--subset", type=int, default=-1, help="Subset of the dataset (-1 for full dataset)")
parser.add_argument("--lambdas", type=str, default="0.3_1_0.3", help="Lambda values for RM loss")
parser.add_argument("--output", type=str, default="trained_model", help="Output path for trained model.")

# Parse arguments
args = parser.parse_args()

# Assign variables from parsed arguments
model = args.model
num_labels = args.num_labels
cache_dir = args.cache_dir
DEVICE = args.device
DEVICE_OLD = args.device_old
epochs = args.epochs
batchsize = args.batchsize
l2_ratio = args.l2_ratio
lr = args.lr
subset = args.subset
lambdas = [float(x) for x in args.lambdas.split('_')]
save_path = f'trained/{args.output}'

def rm_loss(logits_list, beta = 0.01):
    pairs = [[0,1],[0,2],[1,2]]
    losses = []
    for i, (pos, neg) in enumerate(pairs):
        pos_logits = logits_list[pos]
        neg_logits = logits_list[neg]
        l2 = 0.5 * (pos_logits**2 + neg_logits**2)
        _loss = lambdas[i]*(-logsigmoid(pos_logits - neg_logits) + beta * l2).mean()
        losses.append(_loss)
    loss = torch.stack(losses)
    return loss.mean()

torch.manual_seed(0)
config = AutoConfig.from_pretrained(model, output_hidden_states=True, cache_dir=cache_dir)
tokenizer = transformers.AutoTokenizer.from_pretrained(model, cache_dir=cache_dir)
detector = AutoModelForSequenceClassification.from_pretrained(
        model,
        ignore_mismatched_sizes=True, 
        config=config,
        cache_dir=cache_dir)

detector.to(DEVICE)

detecter_teacher = AutoModelForSequenceClassification.from_pretrained(
        model,
        ignore_mismatched_sizes=True,
        config=config,
        cache_dir=cache_dir)
detecter_teacher.to(DEVICE_OLD)
detecter_teacher.eval()

print(f'Loading dataset HC3...')
data = load_perturbed_data()

train_pos = data['train']['pos']
train_neg = data['train']['neg']
train_perturb = data['train']['perturb']

pos_encodings = tokenizer(train_pos, truncation=True,max_length=512, padding=True)
neg_encodings = tokenizer(train_neg, truncation=True,max_length=512, padding=True)
perturb_encodings = tokenizer(train_perturb, truncation=True,max_length=512, padding=True)

train_dataset = PerturbedDataset(pos_encodings,perturb_encodings,neg_encodings)
if subset != -1:
    sub_train_dataset = torch.utils.data.Subset(train_dataset, range(subset))
else:
    sub_train_dataset = train_dataset
train_loader = DataLoader(
    sub_train_dataset, batch_size=batchsize, shuffle=True)

reg_data_train, reg_data_test = load_reg_data(cache_dir=cache_dir)
reg_text = list(map(lambda x: process_spaces(x),reg_data_train))
reg_encodings = tokenizer(reg_text, truncation=True,max_length=512, padding=True, return_tensors="pt")
reg_dataset = RegularDataset(reg_encodings)
reg_loader = DataLoader(reg_dataset, batch_size=batchsize, shuffle=True)


no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in detector.named_parameters() if not any(
        nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in detector.named_parameters() if any(
        nd in n for nd in no_decay)], 'weight_decay': 0.0}
]
optimizer = AdamW(optimizer_grouped_parameters, lr=lr)
i=0

total_steps = len(train_loader)
warmup_steps = int(total_steps*0.1)
scheduler = WarmupCosineDecayScheduler(optimizer, warmup_steps, total_steps)

detector.train()
for epoch in range(epochs):
    iter_num = 0
    iter_train = iter(train_loader)
    iter_reg = iter(reg_loader)
    while True:
        iter_num+=1
        try:                
            batch_kd = next(iter_reg)
            batch = next(iter_train)
        except StopIteration:
            break
        optimizer.zero_grad()
        input_ids_kd = batch_kd['input_ids']
        attention_mask_kd = batch_kd['attention_mask']
        detecter_teacher.eval()

        with torch.no_grad():
            teacher_outputs = detecter_teacher(input_ids_kd.to(DEVICE_OLD), attention_mask=attention_mask_kd.to(DEVICE_OLD))
            teacher_logits = teacher_outputs.logits.to(DEVICE)
        
        student_outputs = detector(input_ids_kd.to(DEVICE), attention_mask=attention_mask_kd.to(DEVICE))
        student_logits = student_outputs.logits.to(DEVICE)
        loss_l2 = l2_ratio * torch.sum((teacher_logits - student_logits)**2)
        
        pos_logits = detector(batch[0]['input_ids'].to(DEVICE), attention_mask=batch[0]['attention_mask'].to(DEVICE)).logits
        perturb_logits = detector(batch[1]['input_ids'].to(DEVICE), attention_mask=batch[1]['attention_mask'].to(DEVICE)).logits
        neg_logits = detector(batch[2]['input_ids'].to(DEVICE), attention_mask=batch[2]['attention_mask'].to(DEVICE)).logits
        
            
        loss = rm_loss([pos_logits,perturb_logits,neg_logits]) + loss_l2
        loss.backward()
        optimizer.step()
        
        if iter_num % 10==0:
            print(f'[Epoch {epoch}/{epochs} - {iter_num}/{len(train_loader)} - lr:{scheduler.get_lr()[0]}] train_loss: {loss.cpu().detach().item()}, kl_loss:{loss_l2.cpu().detach().item()}')
        scheduler.step()
            
detector.eval()

detector.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)