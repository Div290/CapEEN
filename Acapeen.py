import numpy as np
import pandas as pd
from scipy.stats import entropy
import scipy.stats as st
import matplotlib.pyplot as plt
import random
import ast
import requests
import torch
from PIL import Image
from transformers import *
from tqdm import tqdm
import torch.nn as nn
import urllib.parse as parse
import os
import pickle


df = pd.read_csv("path to the stored csv file with confidence.csv")

# Assuming df is your DataFrame, and 'confidence_lis_3' is the column containing the string representation of lists
df['confidence_lis_5'] = df['confidence_lis_5'].apply(ast.literal_eval)
df['confidence_lis_3'] = df['confidence_lis_3'].apply(ast.literal_eval)
df['confidence_lis_12'] = df['confidence_lis_12'].apply(ast.literal_eval)
df['predictions_lis_5'] = df['predictions_lis_5'].apply(ast.literal_eval)
df['predictions_lis_3'] = df['predictions_lis_3'].apply(ast.literal_eval)
df['predictions_lis_12'] = df['predictions_lis_12'].apply(ast.literal_eval)
df['confidence_lis_1'] = df['confidence_lis_1'].apply(ast.literal_eval)
df['predictions_lis_1'] = df['predictions_lis_1'].apply(ast.literal_eval)
df['confidence_lis_2'] = df['confidence_lis_2'].apply(ast.literal_eval)
df['predictions_lis_2'] = df['predictions_lis2'].apply(ast.literal_eval)
df['confidence_lis_7'] = df['confidence_lis_7'].apply(ast.literal_eval)
df['predictions_lis_7'] = df['predictions_lis_7'].apply(ast.literal_eval)
df['confidence_lis_9'] = df['confidence_lis_9'].apply(ast.literal_eval)
df['predictions_lis_9'] = df['predictions_lis_9'].apply(ast.literal_eval)



predictions_lis_3 = df['predictions_lis_3']
for i in predictions_lis_3:
    while len(i) < 32:
        i.append(50256)


predictions_lis_1 = df['predictions_lis_1']
for i in predictions_lis_1:
    while len(i) < 32:
        i.append(50256)


predictions_lis_2 = df['predictions_lis_2']
for i in predictions_lis_2:
    while len(i) < 32:
        i.append(50256)

predictions_lis_5 = df['predictions_lis_5']
for i in predictions_lis_5:
    while len(i) < 32:
        i.append(50256)


predictions_lis_7 = df['predictions_lis_7']
for i in predictions_lis_7:
    while len(i) < 32:
        i.append(50256)


predictions_lis_9 = df['predictions_lis_9']
for i in predictions_lis_9:
    while len(i) < 32:
        i.append(50256)


predictions_lis_12 = df['predictions_lis_12']
for i in predictions_lis_12:
    while len(i) < 32:
        i.append(50256)
        


confidence_lis_12 = df['confidence_lis_12']
for i in confidence_lis_12:
    while len(i) < 32:
        i.append(1.0)

confidence_lis_1 = df['confidence_lis_1']
for i in confidence_lis_1:
    while len(i) < 32:
        i.append(1.0)


confidence_lis_2 = df['confidence_lis_2']
for i in confidence_lis_2:
    while len(i) < 32:
        i.append(1.0)


confidence_lis_5 = df['confidence_lis_5']
for i in confidence_lis_5:
    while len(i) < 32:
        i.append(1.0)


confidence_lis_7 = df['confidence_lis_7']
for i in confidence_lis_7:
    while len(i) < 32:
        i.append(1.0)


confidence_lis_9 = df['confidence_lis_9']
for i in confidence_lis_9:
    while len(i) < 32:
        i.append(1.0)
    
        
confidence_lis_3 = df['confidence_lis_3']
for i in confidence_lis_3:
    while len(i) < 32:
        i.append(1.0)
        
        
import numpy as np
from itertools import product

# Define the number of arms (actions) and the action set  # 10 values in each set, so 10x10 = 100 possible actions
action_set = list(product(np.linspace(0.5, 0.8, 3), np.linspace(0.5, 0.8, 3), np.linspace(0.4, 0.8, 4), np.linspace(0.3, 0.7, 4)))
num_arms = len(action_set)

# Initialize variables to track the number of times each action is selected and the total rewards for each action
num_actions = np.zeros(num_arms)
total_rewards = np.zeros(num_arms)

# Define a reward function for each action (manually set for demonstration purposes)
# In practice, you would replace these with your actual reward functions.
def get_reward(selected_action, c_1, c_2, c_3, c_4, c_l, overhead, df, step, j):
    if c_1[step][j] >= action_set[selected_action][0]:
        reward = 0
        count_1 = 1
        count_2 = 0
        count_3 = 0
        count = 0
        prediction = df['predictions_lis_2'][step][j]
        
    elif c_2[step][j] >= action_set[selected_action][1]:
        reward = c_2[step][j] - c_1[step][j] - overhead[0]
        count_1 = 0
        count_2 = 1
        count_3 = 0
        count = 0
        prediction = df['predictions_lis_5'][step][j]
        
    elif c_3[step][j] >= action_set[selected_action][2]:
        reward = c_3[step][j] - c_1[step][j] - overhead[1] 
        count_1 = 0
        count_2 = 0
        count_3 = 1
        count = 0
        prediction = df['predictions_lis_7'][step][j]
        
    elif c_4[step][j] >= action_set[selected_action][3]:
        reward = c_4[step][j] - c_1[step][j] - overhead[2] 
        count_1 = 0
        count_2 = 0
        count_3 = 0
        count = 1
        prediction = df['predictions_lis_9'][step][j]
        
    else:
        reward = c_l[step][j] - c_1[step][j] - overhead[3] 
        prediction = df['predictions_lis_12'][step][j]
        count_1 = 0
        count_2 = 0
        count_3 = 0
        count = 0
    return reward, prediction, count_1, count_2, count_3, count

# UCB parameters
ucb_parameter = 1.0  # Exploration parameter (you can adjust this)
overhead = [1/12, 4/12, 7/12, 1]

# Main UCB loop
t = 0
count_4 = 0
count_5 = 0
count_6 = 0
count_7 = 0
predictions = []
for step in range(df.shape[0]):
    print(step)
    preds = []
    prediction = 0
    j = 0
    # Calculate the upper confidence bound for each action
    # print(step)
    while prediction < 50256 and len(preds) < 32:
        # print(len(preds))
        ucb_values = [
            total_rewards[action_index] / max(1, num_actions[action_index]) +
            ucb_parameter * np.sqrt(np.log(t + 1) / max(1, num_actions[action_index]))
            for action_index in range(num_arms)
        ]

        # Choose the action with the highest UCB value
        selected_action = np.argmax(ucb_values)
        # print(selected_action)
        c_1_data = []
        for i in df["confidence_lis_1"]:
            c_1_data.append(i)
            
        c_2_data = []
        for i in df["confidence_lis_5"]:
            c_2_data.append(i)
            
        c_3_data = []
        for i in df["confidence_lis_7"]:
            c_3_data.append(i)
            
        c_4_data = []
        for i in df["confidence_lis_9"]:
            c_4_data.append(i)

        c_l_data = []
        for i in df["confidence_lis_12"]:
            c_l_data.append(i)

        # Get the reward for the selected action
        reward, prediction, count_1, count_2, count_3, count = get_reward(selected_action, c_1_data, c_2_data, c_3_data, c_4_data, c_l_data, overhead, df, step, j)
        if count_1 == 1:
            count_4+=1
        if count_2 == 1:
            count_5+=1
        if count_3 == 1:
            count_6+=1
        if count == 1:
            count_7+=1
            
        preds.append(prediction)
        # print(prediction)

        # Update the number of times the selected action is chosen and the total rewards
        selected_action_set = []
        if c_1_data[step][j] >= action_set[selected_action][0]:
            for i in range(len(action_set)):
                if action_set[i][0] <= action_set[selected_action][0]:
                    selected_action_set.append(i) 
        elif c_2_data[step][j] >= action_set[selected_action][1]:
            for i in range(len(action_set)):
                if action_set[i][0] >= action_set[selected_action][0] and action_set[i][1] < action_set[selected_action][1]:
                    selected_action_set.append(i) 
        elif c_3_data[step][j] >= action_set[selected_action][2]:
            for i in range(len(action_set)):
                if action_set[i][0] >= action_set[selected_action][0] and action_set[i][1] < action_set[selected_action][1]:
                    selected_action_set.append(i) 
        else:
            if c_1_data[step][j] < action_set[selected_action][0]:
                for i in range(len(action_set)):
                    if action_set[i][0] >= action_set[selected_action][0] and action_set[i][1] >= action_set[selected_action][1]:
                        selected_action_set.append(i) 
        selected_action_set.append(selected_action)
        # print(selected_action_set)
        # print(selected_action)
        for i in selected_action_set:
            num_actions[i] += 1
            total_rewards[i] += reward
        # print(num_actions)
        j += 1
        t += 1
    # if t > 5000:
    #     break
    predictions.append(preds)

print("The reward vector is", total_rewards)

print('the chosen action set is', action_set[np.argmax(total_rewards)])

device = "cuda:1" if torch.cuda.is_available() else "cpu"
print(device)




best_checkpoint = 705000
best_model = VisionEncoderDecoderModel.from_pretrained(f"./image-captioning/checkpoint-{best_checkpoint}").to(device)

encoder_model = "microsoft/swin-base-patch4-window7-224-in22k"
decoder_model = "gpt2"
tokenizer = GPT2TokenizerFast.from_pretrained(decoder_model)
image_processor = ViTImageProcessor.from_pretrained(encoder_model)


if "gpt2" in decoder_model:
  tokenizer.pad_token = tokenizer.eos_token # pad_token_id as eos_token_id
  best_model.config.eos_token_id = tokenizer.eos_token_id
  best_model.config.pad_token_id = tokenizer.pad_token_id
  best_model.config.decoder_start_token_id = tokenizer.bos_token_id
else:
  # set the decoder start token id to the CLS token id of the tokenizer
  best_model.config.decoder_start_token_id = tokenizer.cls_token_id
  # set the pad token id to the pad token id of the tokenizer
  best_model.config.pad_token_id = tokenizer.pad_token_id
  
  
max_length = 32

    
with open('test_ds_coco.pkl', 'rb') as f:
    test_ds = pickle.load(f)
    
    
def preprocess(items):
  pixel_values = image_processor(items["image"], return_tensors="pt").pixel_values.to(device)
  targets = tokenizer([ sentence["raw"] for sentence in items["sentences"] ], 
                      max_length=max_length, padding="max_length", truncation=True, return_tensors="pt").to(device)
  return {'pixel_values': pixel_values, 'labels': targets["input_ids"]}


# using with_transform to preprocess the dataset during testing
test_dataset  = test_ds.with_transform(preprocess)


# a function we'll use to collate the batches
def collate_fn(batch):
    return {
        'pixel_values': torch.stack([x['pixel_values'] for x in batch]),
        'labels': torch.stack([x['labels'] for x in batch])
    }
import evaluation


def compute_metrics(eval_pred):
    preds = eval_pred.label_ids
    labels = eval_pred.predictions
    # decode the predictions and labels
    pred_str = tokenizer.batch_decode(preds, skip_special_tokens=True)
    labels_str = tokenizer.batch_decode(labels, skip_special_tokens=True)
    gts = evaluation.PTBTokenizer.tokenize(labels_str)
    gen = evaluation.PTBTokenizer.tokenize(pred_str)
    scores, _ = evaluation.compute_all_scores(gts, gen)
    return scores



# alternative way of training: pytorch loop
from torch.utils.data import DataLoader

# define our data loaders
# test_dataset_loader = DataLoader(test_dataset, collate_fn=collate_fn, batch_size=8, shuffle=True)


# load the best model, change the checkpoint number to the best checkpoint
# if the last checkpoint is the best, then ignore this cell

# ... (imports and configurations)

decoder_config = best_model.decoder.config

class IntermediateHead(nn.Module):
    def __init__(self, input_size, output_size):
        super(IntermediateHead, self).__init__()
        self.fc = nn.Linear(input_size, output_size)
        
    def forward(self, x):
        return self.fc(x)
      
      
vocab_size = 50257
layers_for_exit = [0, 1, 2, 4, 6, 8, 10]

# Initialize intermediate heads with the same configuration
intermediate_heads = nn.ModuleList([IntermediateHead(decoder_config.hidden_size, vocab_size) for _ in range(len(layers_for_exit))])
# ... (load your dataset, optimizer, etc.)

# Path to the saved weights directory
saved_weights_dir = f"./multi_heads/checkpoint/intermediate_head_weights/-{120000}"
# Load saved weights into intermediate heads
# Now, your intermediate heads are loaded with the saved weights

def get_evaluation_metrics(model, dataset, preds):
  model.eval()
  # define our dataloader
  dataloader = DataLoader(dataset, collate_fn=collate_fn, batch_size=batch_size, shuffle=False)
  # number of testing steps
  n_test_steps = len(dataloader)
  # initialize our lists that store the predictions and the labels
  predictions, labels = [], []
  # initialize the test loss
  test_loss = 0.0
  s = 0
  confidence_lis = []
  for batch in tqdm(dataloader, "Evaluating"):
      # get the batch
      con_lis = []
      pred_lis = []
      pixel_values = batch["pixel_values"]
      label_ids = batch["labels"]
      
      pred = preds[s]
      while len(pred)<=max_length:
        pred.append(50256)
      pred = [pred]
      predictions.extend(pred)

      labels.extend(label_ids.tolist())
      s+=1
      if s == 24920:
        break
  # make the EvalPrediction object that the compute_metrics function expects
  eval_prediction = EvalPrediction(predictions=predictions, label_ids=labels)
  # compute the metrics
  metrics = compute_metrics(eval_prediction)
  return metrics


batch_size = 1
metrics = get_evaluation_metrics(best_model, test_dataset, predictions)
print(metrics)

        