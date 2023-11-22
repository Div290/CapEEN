import requests
import torch
from PIL import Image
from transformers import *
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import os
import pickle
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

device = "cuda:0"
best_checkpoint = 2000 #find the best checkpoint by finetuning the backbone
best_model = VisionEncoderDecoderModel.from_pretrained(f"./image-captioning/checkpoint-{best_checkpoint}").to(device)
from torch.utils.data import DataLoader

encoder_model = "microsoft/swin-base-patch4-window7-224-in22k"
decoder_model = "gpt2"
tokenizer = GPT2TokenizerFast.from_pretrained(decoder_model)
image_processor = ViTImageProcessor.from_pretrained(encoder_model)


if "gpt2" in decoder_model:
  tokenizer.pad_token = tokenizer.eos_token # pad_token_id as eos_token_id
  best_model.config.eos_token_id = tokenizer.eos_token_id
  best_model.config.pad_token_id = tokenizer.pad_token_id
  # set decoder_start_token_id as bos_token_id
  best_model.config.decoder_start_token_id = tokenizer.bos_token_id
else:
  best_model.config.decoder_start_token_id = tokenizer.cls_token_id
  best_model.config.pad_token_id = tokenizer.pad_token_id
  
  
max_length = 32
with open('train_ds_coco.pkl', 'rb') as f:
    train_ds = pickle.load(f)
    
with open('val_ds_coco.pkl', 'rb') as f:
    valid_ds = pickle.load(f)
    
def preprocess(items):
  pixel_values = image_processor(items["image"], return_tensors="pt").pixel_values.to(device)
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
    
  
num_epochs = 5 # number of epochs
batch_size = 4 # the size of batches


# alternative way of training: pytorch loop
from torch.utils.data import DataLoader

# define our data loaders
train_dataset_loader = DataLoader(train_dataset, collate_fn=collate_fn, batch_size=batch_size, shuffle=True)
valid_dataset_loader = DataLoader(valid_dataset, collate_fn=collate_fn, batch_size=8, shuffle=True)


from torch.utils.tensorboard import SummaryWriter

summary_writer = SummaryWriter(log_dir="./image-captioning/tensorboard")
n_train_steps = num_epochs * len(train_dataset_loader)
n_valid_steps = len(valid_dataset_loader)
current_step = 0
# logging, eval & save steps
save_steps = 5000
decoder_config = best_model.config.decoder
class IntermediateHead(nn.Module):
    def __init__(self, input_size, output_size):
        super(IntermediateHead, self).__init__()
        self.fc = nn.Linear(input_size, output_size)
        
    def forward(self, x):
        return self.fc(x)
    
vocab_size = 50257
# Create intermediate head modules
num_intermediate_layers = 12
layers_for_exit = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
intermediate_heads = nn.ModuleList([IntermediateHead(decoder_config.hidden_size, vocab_size) for _ in range(len(layers_for_exit))])

optimizer = AdamW(intermediate_heads.parameters(), lr=1e-4)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=1000, num_training_steps=n_train_steps)

kl_div_loss = torch.nn.KLDivLoss(reduction='batchmean')

for epoch in range(num_epochs):
    # set the model to training model
    intermediate_heads.train()
    # initialize the training loss
    train_loss = 0
    for batch in tqdm(train_dataset_loader, "Training", total=len(train_dataset_loader), leave=False):
      train_loss = 0    
      if current_step % save_steps == 0:
        # evaluate on the validation set
        print()
        print(f"Validation at step {current_step}...")
        print()
        # set the model to evaluation mode
        intermediate_heads.eval()
        # initialize our lists that store the predictions and the labels
        predictions, labels = [], []
        # initialize the validation loss
        valid_loss = 0
        for batch in valid_dataset_loader:
            # get the batch
            pixel_values = batch["pixel_values"]
            label_ids = batch["labels"]
            # forward pass
            outputs = best_model(pixel_values=pixel_values, labels=label_ids, output_hidden_states = True)
            int_loss = 0
            for exit in range(len(layers_for_exit)):
                intermediate_head = intermediate_heads[exit]
                intermediate_head = intermediate_head.to(device)
                intermediate_logits = intermediate_head(outputs.decoder_hidden_states[layers_for_exit[exit]])
                intermediate_loss = F.cross_entropy(intermediate_logits.view(-1, intermediate_logits.size(-1)), label_ids.view(-1))
                int_loss+=(layers_for_exit[exit]+1)*intermediate_loss
            # get the loss
            loss = (int_loss)
            valid_loss += loss.item()
        # print the stats
        print()
        print(f"Epoch: {epoch}, Step: {current_step}, Train Loss: {train_loss:.4f}, " + 
              f"Valid Loss: {valid_loss:.4f}")
        print()
        intermediate_head.train()
        train_loss, valid_loss = 0, 0
      #training
      pixel_values = batch["pixel_values"]
      labels = batch["labels"]
      # forward pass
      outputs = best_model(pixel_values=pixel_values, labels=labels, output_hidden_states = True)
      int_loss_train = 0
      # w = [i / num_intermediate_layers for i in range(num_intermediate_layers)]
      for exit in range(len(layers_for_exit)):
          intermediate_head = intermediate_heads[exit]
          intermediate_head = intermediate_head.to(device)
          intermediate_logits = intermediate_head(outputs.decoder_hidden_states[layers_for_exit[exit]])
          compare_head = intermediate_heads[-1]
          compare_head = compare_head.to(device)
          compare_head_logits = compare_head(outputs.decoder_hidden_states[-1])
          kl_loss = kl_div_loss(F.log_softmax(intermediate_logits, dim=-1), F.softmax(outputs.logits, dim=-1))
          intermediate_loss = (F.cross_entropy(intermediate_logits.view(-1, intermediate_logits.size(-1)), labels.view(-1)))# + 0.8*cosine_loss
          int_loss_train+=0.5*intermediate_loss + 0.5*kl_loss #+ (current_step / n_train_steps)*cosine_loss_11
          # Backpropagate the intermediate loss and accumulate gradients
      # get the loss
      int_loss_train = int_loss_train / len(layers_for_exit)
      # backward pass
      int_loss_train.backward()
      # update the weights
      optimizer.step()
      scheduler.step()
      # zero the gradients
      optimizer.zero_grad()
      loss_v = int_loss_train.item()
      train_loss += loss_v
      # increment the step
      current_step += 1
      if current_step%save_steps==0:
        print(f"Epoch: {epoch}, Step: {current_step}, Train Loss: {train_loss / save_steps:.4f} " )  
        intermediate_head_weights_dir = f"./multi_heads_only_cs/checkpoint/intermediate_head_weights/-{current_step}"
        os.makedirs(intermediate_head_weights_dir, exist_ok=True)

        # Save the weights of each intermediate head
        for layer_idx, intermediate_head in enumerate(intermediate_heads):
            head_path = os.path.join(intermediate_head_weights_dir, f"head_layer_{layers_for_exit[layer_idx]}.pt")
            torch.save(intermediate_head.state_dict(), head_path)
        