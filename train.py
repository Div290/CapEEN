import requests
import torch
from PIL import Image
from transformers import *
from tqdm import tqdm
import pickle
import numpy as np
import utils
import evaluation
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import DataParallel
from torch.optim import AdamW
import urllib.parse as parse
import os

device = "cuda" if torch.cuda.is_available() else "cpu"
print("The device used is", device)

# a function to determine whether a string is a URL or not
def is_url(string):
    try:
        result = parse.urlparse(string)
        return all([result.scheme, result.netloc, result.path])
    except:
        return False
    
# a function to load an image
def load_image(image_path):
    if is_url(image_path):
        return Image.open(requests.get(image_path, stream=True).raw)
    elif os.path.exists(image_path):
        return Image.open(image_path)
        

# a function to perform inference
def get_caption(model, image_processor, tokenizer, image_path):
    image = load_image(image_path)
    # preprocess the image
    img = image_processor(image, return_tensors="pt").to(device)
    # generate the caption (using greedy decoding by default)
    output = model.generate(**img)
    # decode the output
    caption = tokenizer.batch_decode(output, skip_special_tokens=True)[0]
    return caption



# the encoder model
encoder_model = "microsoft/swin-base-patch4-window7-224-in22k"
# the decoder model 
decoder_model = "gpt2"
model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
    encoder_model, decoder_model, output_hidden_states = True
).to(device)
tokenizer = GPT2TokenizerFast.from_pretrained(decoder_model)
image_processor = ViTImageProcessor.from_pretrained(encoder_model)

max_length = 32 

if "gpt2" in decoder_model:
  tokenizer.pad_token = tokenizer.eos_token
  model.config.eos_token_id = tokenizer.eos_token_id
  model.config.pad_token_id = tokenizer.pad_token_id
  model.config.decoder_start_token_id = tokenizer.bos_token_id
else:
  model.config.decoder_start_token_id = tokenizer.cls_token_id
  model.config.pad_token_id = tokenizer.pad_token_id

import pickle
with open('train_ds_coco.pkl', 'rb') as f:
    train_ds = pickle.load(f)
    
with open('val_ds_coco.pkl', 'rb') as f:
    valid_ds = pickle.load(f)
    
def preprocess(items):
  # preprocess the image
  pixel_values = image_processor(items["image"], return_tensors="pt").pixel_values.to(device)
  # tokenize the caption with truncation and padding
  targets = tokenizer([ sentence["raw"] for sentence in items["sentences"] ], 
                      max_length=max_length, padding="max_length", truncation=True, return_tensors="pt").to(device)
  return {'pixel_values': pixel_values, 'labels': targets["input_ids"]}

train_dataset = train_ds.with_transform(preprocess)
valid_dataset = valid_ds.with_transform(preprocess)

def collate_fn(batch):
    return {
        'pixel_values': torch.stack([x['pixel_values'] for x in batch]),
        'labels': torch.stack([x['labels'] for x in batch])
    }
    


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
  
num_epochs = 20 # number of epochs
batch_size = 4 # the size of batches


#Dataloader for data to be fed to model
from torch.utils.data import DataLoader

# define our data loaders
train_dataset_loader = DataLoader(train_dataset, collate_fn=collate_fn, batch_size=batch_size, shuffle=True)
valid_dataset_loader = DataLoader(valid_dataset, collate_fn=collate_fn, batch_size=2, shuffle=True)

n_train_steps = num_epochs * len(train_dataset_loader)
n_valid_steps = len(valid_dataset_loader)
current_step = 0
save_steps = 5000
optimizer = AdamW(model.parameters(), lr=1e-4)

vocab_size = 50257

model = DataParallel(model)
model = nn.DataParallel(model)

for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for batch in tqdm(train_dataset_loader, "Training", total=len(train_dataset_loader), leave=False):   
      if current_step % save_steps == 0:
        print()
        print(f"Validation at step {current_step}...")
        print()
        # set the model to evaluation mode
        model.eval()
        # initialize our lists that store the predictions and the labels
        predictions, labels = [], []
        # initialize the validation loss
        valid_loss = 0
        for batch in valid_dataset_loader:
            # get the batch
            pixel_values = batch["pixel_values"]
            label_ids = batch["labels"]
            # forward pass
            outputs = model(pixel_values=pixel_values, labels=label_ids, output_hidden_states = True)
            loss = outputs.loss# + alpha * int_loss
            valid_loss += loss.item()
            logits = outputs.logits.detach().cpu()
            # add the predictions to the list
            predictions.extend(logits.argmax(dim=-1).tolist())
            # add the labels to the list
            labels.extend(label_ids.tolist())
        eval_prediction = EvalPrediction(predictions=predictions, label_ids=labels)
        metrics = compute_metrics(eval_prediction)
        print(f"Epoch: {epoch}, Step: {current_step}, Train Loss: {train_loss / save_steps:.4f}, " + 
              f"Valid Loss: {valid_loss / n_valid_steps:.4f}")
        print()
        # save the model
        model.save_pretrained(f"./image-captioning/checkpoint-{current_step}")
        tokenizer.save_pretrained(f"./image-captioning/checkpoint-{current_step}")
        image_processor.save_pretrained(f"./image-captioning/checkpoint-{current_step}")
        # get the model back to train mode
        model.train()
        # reset the train and valid loss
        train_loss, valid_loss = 0, 0
      ### training code below ###
      pixel_values = batch["pixel_values"]
      labels = batch["labels"]
      # forward pass
      outputs = model(pixel_values=pixel_values, labels=labels, output_hidden_states = True)
      loss = outputs.loss
      loss.backward()
      optimizer.step()
      optimizer.zero_grad()
      loss_v = loss.item()
      train_loss += loss_v
      current_step += 1
        
