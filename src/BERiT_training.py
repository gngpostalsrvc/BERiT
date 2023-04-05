from transformers import (
    AutoModelForMaskedLM,
    DataCollatorForLanguageModeling,
    PreTrainedTokenizerFast,
    RobertaConfig,
    Trainer,
    TrainingArguments,
)
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from datasets import load_dataset

raw_data = load_dataset('gngpostalsrvc/Tanakh')
texts = list(raw_data['train']['Text'])
texts.extend(list(raw_data['test']['Text']))
     
tokenizer = Tokenizer(BPE())
tokenizer.pre_tokenizer = Whitespace()

trainer = BpeTrainer(vocab_size = 2000, special_tokens = ["<s>", "<pad>",  "</s>", "<unk>", "<mask>"])

tokenizer.post_processor = TemplateProcessing(
    single = "<s> $A </s>",
    special_tokens = [("<s>", 0), ("</s>", 2)])

tokenizer.train_from_iterator(texts, trainer)

derivational_morphs = ['הִתְ' ,'מוֹ' ,'מְ' ,'מַ' ,'הֵ' ,'הִ' ,'הֶ' ,'הֲ' ,'הוֹ' ,' ַת' ,' ֵי' ,' ָה' ,'וֹת' ,' ִים' ,' ִי' ,' ַי' ,'נִי' ,'ךָ' ,'ךְ' ,'וֹ' ,'ו' ,'הוּ' ,'נּוּ' ,' ָהּ' ,'הָ' ,'נָּה' ,'נוּ' ,'כֶם' ,'כֶן' ,'הֶם' ,' ָם' ,' ֵם' ,'ם' ,'הֵנָּה' ,'הֶן' ,' ֵן' ,' ָן' ,' ֵן' ,'נִתְ' ,'נִּתְ' ,'יִתְ' ,'יִּתְ' ,'אֶתְ' ,'תִּתְ' ,'תִתְ' ,'תּוֹ' ,'תוֹ' ,'אוֹ' ,'הוֹ' ,'נוֹ' ,'נּוֹ' ,'יוֹ' ,'יּוֹ' ,'אֲ' ,'אַ' ,'אֹ' ,'אֶ' ,'אִ' ,'אָ' ,'אֵ' ,'תֵּ' ,'תַּ' ,'תִּ' ,'תָּ' ,'תְּ' ,'תֹּ' ,'תֶּ' ,'תֵ' ,'תַ' ,'תִ' ,'תָ' ,'תְ' ,'תֹ' ,'תֶ' ,'יָ' ,'יִ' ,'יֶ' ,'יֹ' ,'יְ' ,'יַ' ,'יֵ' ,'יָּ' ,'יִּ' ,'יֶּ' ,'יֹּ' ,'יַּ' ,'יֵּ' ,'נֹ' ,'נָ' ,'נֵ' ,'נִ' ,'נֶ' ,'נַ' ,'נְ' ,'נֹּ' ,'נָּ' ,'נֵּ' ,'נִּ' ,'נֶּ' ,'נַּ' ,'וּ' ,'נָה' ,'תִּי' ,'תֶּם' ,'תֶּן' ,'תִי' ,'תֶם' ,'תֶן']

tokenizer.add_tokens(derivational_morphs)

tokenizer.save('BERiT_tokenizer_2000_enriched.json')

tokenizer = PreTrainedTokenizerFast(tokenizer_file = 'BERiT_tokenizer_2000_enriched.json')

tokenizer.add_special_tokens({'pad_token' : '<pad>', 'mask_token' : '<mask>', 'unk_token' : '<unk>', 'bos_token' : '<s>', 'eos_token' : '</s>'})

tokenizer.pad_token = tokenizer.eos_token

data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=True, mlm_probability=.15)

def tokenize(sentence):
  return tokenizer(sentence['Text'], max_length=128, truncation=True, padding=True)

tokenized_data = raw_data.map(tokenize, batched=True, remove_columns=raw_data['train'].column_names)

tokenized_data.set_format("pt", columns=["input_ids", "attention_mask"], output_all_columns=True)

config = RobertaConfig.from_pretrained(
    "roberta-base", 
    model_type='roberta',
    attention_probs_dropout_prob=.5, 
    hidden_dropout_prob=.5, 
    hidden_size=256,
    intermediate_size=1024,
    max_position_embeddings=128, 
    num_attention_heads=4,
    num_hidden_layers=1,
    vocab_size=len(tokenizer.vocab)
    )

model = AutoModelForMaskedLM.from_pretrained("roberta-base", config=config, ignore_mismatched_sizes=True)

args = TrainingArguments(output_dir="BERiT_2000_custom_architecture_150_epochs_2", 
                         evaluation_strategy="steps",
                         save_strategy="epoch",
                         learning_rate=0.0005,
                         num_train_epochs=150,
                         per_device_train_batch_size=8, 
                         per_device_eval_batch_size=8,
                         seed=42,
                        )
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_data['train'],
    eval_dataset=tokenized_data['test'],
    tokenizer=tokenizer,
    data_collator=data_collator,
  )

trainer.train()

trainer.push_to_hub()
