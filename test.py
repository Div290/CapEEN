import requests
import torch
from PIL import Image
from transformers import *
from tqdm import tqdm
import torch.nn as nn
import evaluation
import urllib.parse as parse
import os
import pickle

device = "cuda:1" if torch.cuda.is_available() else "cpu"
print(device)


best_checkpoint = 2000#get the best finetuned model checkpoint
best_model = VisionEncoderDecoderModel.from_pretrained(f"./image-captioning/checkpoint-{best_checkpoint}").to(device)

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

encoder_model = "microsoft/swin-base-patch4-window7-224-in22k"
decoder_model = "gpt2"

tokenizer = GPT2TokenizerFast.from_pretrained(decoder_model)
image_processor = ViTImageProcessor.from_pretrained(encoder_model)


if "gpt2" in decoder_model:
  tokenizer.pad_token = tokenizer.eos_token 
  best_model.config.eos_token_id = tokenizer.eos_token_id
  best_model.config.pad_token_id = tokenizer.pad_token_id
  best_model.config.decoder_start_token_id = tokenizer.bos_token_id
else:
  best_model.config.decoder_start_token_id = tokenizer.cls_token_id
  best_model.config.pad_token_id = tokenizer.pad_token_id
  
  
max_length = 32

    
with open('test_ds_coco.pkl', 'rb') as f:
    test_ds = pickle.load(f)


def add_gaussian_noise(image_tensor, noise_stddev=0.2):
    noise = torch.randn_like(image_tensor) * noise_stddev
    noisy_image = image_tensor + noise
    return noisy_image


def preprocess(items):
    # preprocess the image and add noise
    pixel_values = image_processor(items["image"], return_tensors="pt").pixel_values.to(device)
    noisy_pixel_values = add_gaussian_noise(pixel_values, noise_stddev = 0.4)  # Adjust noise_stddev as needed

    # tokenize the caption with truncation and padding
    targets = tokenizer([sentence["raw"] for sentence in items["sentences"]],
                        max_length=max_length, padding="max_length", truncation=True, return_tensors="pt").to(device)
    return {'pixel_values': noisy_pixel_values, 'labels': targets["input_ids"]}



# using with_transform to preprocess the dataset during testing
test_dataset  = test_ds.with_transform(preprocess)


# a function we'll use to collate the batches
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

batch_size = 1 # the size of batches
from torch.utils.data import DataLoader
test_dataset_loader = DataLoader(test_dataset, collate_fn=collate_fn, batch_size=8, shuffle=True)


decoder_config = best_model.decoder.config

class IntermediateHead(nn.Module):
    def __init__(self, input_size, output_size):
        super(IntermediateHead, self).__init__()
        self.fc = nn.Linear(input_size, output_size)
        
    def forward(self, x):
        return self.fc(x)
      
      
vocab_size = 50257
layers_for_exit = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

# Initialize intermediate heads with the same configuration
intermediate_heads = nn.ModuleList([IntermediateHead(decoder_config.hidden_size, vocab_size) for _ in range(len(layers_for_exit))])

# Path to the saved weights directory
saved_weights_dir = f"./multi_heads/checkpoint/intermediate_head_weights/-{230000}"

def get_evaluation_metrics(model, dataset):
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
  layer_list = []
  for batch in tqdm(dataloader, "Evaluating"):
      # get the batch
      pixel_values = batch["pixel_values"]
      label_ids = batch["labels"]
      # forward pass
      s+=1
      outputs = model(pixel_values=pixel_values, labels=label_ids, output_hidden_states = True)
      int_output = {}
      for layer_idx, intermediate_head in enumerate(intermediate_heads):
        head_path = os.path.join(saved_weights_dir, f"head_layer_{layers_for_exit[layer_idx]}.pt")
        state_dict = torch.load(head_path, map_location=device)
        intermediate_head.load_state_dict(state_dict)
        intermediate_head.to(device)
        output = intermediate_head(outputs.decoder_hidden_states[layers_for_exit[layer_idx]])
        logits = output.detach().cpu()
        int_output[layer_idx] = logits
      # free the GPU memory
      preds = []
      threshold = 1.5
      for i in range(max_length):
        layer_idx = 0
        agg_conf = 0
        for _ in range(len(int_output)):
          prev_logit = None
          logits = int_output[layer_idx]
          logit = logits[0][i].argmax(dim = -1).item()
          logit_2 = outputs.logits.detach().cpu()
          probabilities = torch.softmax(logits, dim=-1)
          confidence = torch.max(probabilities, dim=-1).values
          if prev_logits!= None:
              if prev_logits == logit:
                  agg_conf+=confidence
              else:
                  agg_conf==confidence
          else:
              agg_conf==confidence
          if agg_conf >= threshold:
            preds.append(logit)
            if logit != 50256:
              layer_list.append(layer_idx)
            break
          layer_idx+=1
            
          if layer_idx == len(layers_for_exit):
            preds.append(logit_2[0][i].argmax(dim = -1).item())
            if logit != 50256:
              layer_list.append(layer_idx)
            break
        if logit == 50256:
          layer_list.append(layer_idx)
          break 
      prediction = []
      prediction.append(preds)
      predictions.extend(prediction)
      labels.extend(label_ids.tolist())
  eval_prediction = EvalPrediction(predictions=predictions, label_ids=labels)
  # compute the metrics
  metrics = compute_metrics(eval_prediction)
  return metrics, layer_list


batch_size = 1
metrics, layer_list = get_evaluation_metrics(best_model, test_dataset)
print(metrics)
