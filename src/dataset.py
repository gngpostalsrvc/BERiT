import torch
import pandas as pd
from utils import DatasetForMLM
from transformers import PreTrainedTokenizerFast
from torch.utils.data import DataLoader, random_split

df = pd.read_csv('../Input/BERiT_training_data.csv', index_col = 'Verse') 

texts = df['Text'].to_list()

tokenizer = PreTrainedTokenizerFast(tokenizer_file = '../Input/BERiT_tokenizer.json', 
                                    pad_token = "<pad>", 
                                    bos_token = "<s>", 
                                    eos_token = "</s>", 
                                    mask_token = "<mask>") 

dataset = DatasetForMLM(texts, tokenizer)

train_size = int(.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size], 
                                          generator = torch.Generator().manual_seed(42))

train_dataloader = DataLoader(train_dataset, batch_size = 10, shuffle = True) 
val_dataloader = DataLoader(val_dataset, batch_size = 10, shuffle = True)

