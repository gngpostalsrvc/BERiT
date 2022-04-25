import torch
import torch.nn as nn
from torch.utils.data import Dataset

class DatasetForMLM(Dataset):
    
    def __init__(self, texts, tokenizer, p = .15, return_tensors = 'pt', max_length = 100, truncation = True, padding = 'max_length', return_token_type_ids = False):
        
        self.encodings = tokenizer(texts, return_tensors = return_tensors, max_length = max_length, truncation = truncation, padding = padding, return_token_type_ids = return_token_type_ids)
        self.encodings['labels'] = self.encodings.input_ids.detach().clone()
        self.p = p
        self.bos_id = tokenizer.bos_token_id
        self.eos_id = tokenizer.eos_token_id
        self.mask_id = tokenizer.mask_token_id
        self.pad_id = tokenizer.pad_token_id
        self._generate_mask()
        
    def __getitem__(self, idx):
        
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
    
    def __len__(self):
        
        return self.encodings.input_ids.shape[0]
    
    def _generate_mask(self):
        
        rand = torch.rand(self.encodings.input_ids.shape)
        input_mask = (rand < self.p) * (self.encodings.input_ids != self.bos_id) * (self.encodings.input_ids != self.eos_id) * (self.encodings.input_ids != self.pad_id)
        label_mask = (rand >= self.p) * (self.encodings.input_ids != self.bos_id) * (self.encodings.input_ids != self.eos_id) * (self.encodings.input_ids != self.pad_id)
    
        for i in range(self.encodings.input_ids.shape[0]):
        
            input_selection = torch.flatten(input_mask[i].nonzero()).tolist()
            label_selection = torch.flatten(label_mask[i].nonzero()).tolist()
            self.encodings.input_ids[i, input_selection] = self.mask_id
            self.encodings.labels[i, label_selection] = self.pad_id

class Encoder(nn.Module):
    def __init__(self, 
                input_dim, 
                embed_dim,
                n_heads,
                ff_dim, 
                droprate,
                max_leng = 100):
        super(Encoder, self).__init__()

        self.tok_embed = nn.Embedding(input_dim, embed_dim)
        self.pos_embed = nn.Embedding(max_leng, embed_dim)
        
        self.norm = nn.LayerNorm(embed_dim)

        self.encoder = nn.TransformerEncoderLayer(embed_dim,
                                                  n_heads, 
                                                  ff_dim,
                                                  droprate)

        self.dropout = nn.Dropout(droprate)
        
        self.scale = torch.sqrt(torch.FloatTensor([embed_dim]))

    def forward(self, input_ids, attention_mask):
        
        batch_size = input_ids.shape[0]
        input_len = input_ids.shape[1]

        pos = torch.arange(0, input_len).unsqueeze(0).repeat(batch_size, 1)

        input_ids = self.dropout(self.norm((self.tok_embed(input_ids) * self.scale) + self.pos_embed(pos)))

        encoding = self.encoder(input_ids, src_key_padding_mask = attention_mask)

        return encoding

class EarlyStopping():
    
    def __init__(self, patience=5, min_delta=0):

        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss == None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            print(f"Early stopping counter {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                print("Early stopping")
                self.early_stop = True
    