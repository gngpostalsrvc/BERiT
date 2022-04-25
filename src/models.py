import torch
import torch.nn as nn
from utils import Encoder
from transformers import PreTrainedTokenizerFast

tokenizer = PreTrainedTokenizerFast(tokenizer_file = '../Input/BERiT_tokenizer.json', 
                                    pad_token = "<pad>", 
                                    bos_token = "<s>", 
                                    eos_token = "</s>", 
                                    mask_token = "<mask>") 

model = Encoder(input_dim = len(tokenizer.vocab), 
                embed_dim = 256, 
                n_heads = 4, 
                ff_dim = 1024, 
                droprate = .5)