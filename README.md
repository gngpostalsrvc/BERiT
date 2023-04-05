# BERiT

This model is a fine-tuned version of [roberta-base](https://huggingface.co/roberta-base) on the [Tanakh dataset](https://huggingface.co/datasets/gngpostalsrvc/Tanakh).
It achieves the following results on the evaluation set:
- Loss: 3.9931

## Model description

BERiT is a masked-language model for Biblical Hebrew, a low-resource ancient language preserved primarily in the text of the Hebrew Bible. Building on the work of [Sennrich and Zhang (2019)](https://arxiv.org/abs/1905.11901) and [Wodiak (2021)](https://arxiv.org/abs/2110.01938) on low-resource machine translation, it employs a modified version of the encoder block from Wodiak’s Seq2Seq model. Accordingly, BERiT is much smaller than models designed for modern languages like English. It features a single attention block with four attention heads, smaller embedding and feedforward dimensions (256 and 1024), a smaller max input length (128), and an aggressive dropout rate (.5) at both the attention and feedforward layers. 

The BERiT tokenizer performs character level byte-pair encoding using a 2000 word base vocabulary, which has been enriched with common grammatical morphemes.  

## How to Use

```
from transformers import RobertaModel, RobertaTokenizerFast

BERiT_tokenizer = RobertaTokenizerFast.from_pretrained('gngpostalsrvc/BERiT')
BERiT = RobertaModel.from_pretrained('gngpostalsrvc/BERiT')
```

## Training procedure

BERiT was trained on the Tanakh dataset for 150 epochs using a Tesla T4 GPU. Further training did not yield significant improvements in performance. 

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 0.0005
- train_batch_size: 8
- eval_batch_size: 8
- seed: 42
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- num_epochs: 150
