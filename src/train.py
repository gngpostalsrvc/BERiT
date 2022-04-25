import torch
import time
import models
import torch.nn as nn
import matplotlib.pyplot as plt
from torch import optim
from transformers import PreTrainedTokenizerFast
from utils import EarlyStopping
from dataset import train_dataset, val_dataset, train_dataloader, val_dataloader
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

tokenizer = PreTrainedTokenizerFast(tokenizer_file = '../Input/BERiT_tokenizer.json', 
                                    pad_token = "<pad>", 
                                    bos_token = "<s>", 
                                    eos_token = "</s>", 
                                    mask_token = "<mask>") 

model = models.model.to(device)

total_params = sum(p.numel() for p in model.parameters())
print(f"{total_params:,} total parameters.")

total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"{total_trainable_params:,} total trainable parameters.")

n_epochs = 50

optimizer = optim.Adam(model.parameters(), lr = .0005)

criterion = nn.CrossEntropyLoss(label_smoothing = .2, ignore_index = tokenizer.pad_token_id) #correct?

early_stopping = EarlyStopping(patience = 10) #value for patience?

loss_plot_name = 'loss'
model_name = 'model'

def fit(model, train_iterator, train_dataset, optimizer, criterion):
    print('Training')
    model.train()
    train_running_loss = 0.0
    counter = 0

    prog_bar = tqdm(enumerate(train_iterator), total = int(len(train_dataset) / train_iterator.batch_size))
    
    for idx, batch in prog_bar:
        counter += 1
        input_ids, attn_mask, labels = batch['input_ids'].to(device), batch['attention_mask'].to(torch.bool).to(device), batch['labels'].to(device)
        optimizer.zero_grad()
        output = model(input_ids, attn_mask)
        loss = criterion(output, labels)
        train_running_loss += loss.item()
        loss.backward()
        optimizer.step()

    train_loss = train_running_loss / counter
    
    return train_loss

def validate(model, val_iterator, val_datasset, criterion):
    print('Validating')
    model.eval()
    val_running_loss = 0.0
    counter = 0

    prog_bar = tqdm(enumerate(val_iterator), total = int(len(val_dataset) / val_iterator.batch_size))

    with torch.no_grad():

        for idx, batch in prog_bar:
            counter += 1
            input_ids, attn_mask, labels = batch['input_ids'].to(device), batch['attention_mask'].to(torch.bool).to(device), batch['labels'].to(device)
            output = model(input_ids, attn_mask)
            loss = criterion(output, labels)
            val_running_loss += loss.item()
    
    val_loss = val_running_loss / counter

    return val_loss

train_losses = []
val_losses = []

start = time.time()
for epoch in range(n_epochs):

    print(f"Epoch {epoch + 1}/{n_epochs}")    
    train_epoch_loss = fit(model, train_dataloader, train_dataset, optimizer, criterion)
    train_losses.append(train_epoch_loss)

    val_epoch_loss = validate(model, val_dataloader, val_dataset, optimizer, criterion)
    val_losses.append(val_epoch_loss)

    early_stopping(val_epoch_loss)

    if early_stopping.early_stop:
        break

    print(f"Training Loss: {train_epoch_loss:.5} \n Validation Loss: {val_epoch_loss:.5}")    
end = time.time()

print(f"Training Titme: {(end - start) / 60: .3f} minutes")

print("Saving loss plot...")

plt.figure(figsize = (10, 7))
plt.plot(train_losses, color = 'orange', label = 'Training loss')
plt.plot(val_losses, color = 'red', label = 'Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig(f"../outputs/{loss_plot_name}.png")
plt.show()

print('Saving model...')
torch.save(model.state_dict(), f"../outputs/{model_name}.pth")

print("TRAINING COMPLETE")