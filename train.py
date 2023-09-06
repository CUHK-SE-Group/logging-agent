from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset
from read_data import get_one_data
import numpy as np
import copy
import os
os.environ["WANDB_DISABLED"] = "true"

model_id = "codellama-7b"
max_length = 2048

device_map = device_map = {"": int(os.environ.get("LOCAL_RANK"))}
print("-------------", device_map)
tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir="./", padding_side="right", use_fast=False)
model = AutoModelForCausalLM.from_pretrained(model_id, cache_dir="./", device_map=device_map)
print(model.hf_device_map)

data_files = {"train": "task_data/mixtasks_train_without_index.tsv", "eval": "task_data/mixtasks_eval_without_index.tsv", "test": "task_data/mixtasks_test_without_index.tsv"}
datasets = load_dataset("csv", data_files=data_files, sep='\t')

def _tokenize_fn(strings, tokenizer):
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            max_length=max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )

def preprocess(data):
    sources, targets = get_one_data(data)
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = -100
    return dict(input_ids=input_ids, labels=labels)

tokenized_datasets = datasets.map(
    preprocess,
    batched=True,
    num_proc=8,
    remove_columns=datasets["train"].column_names,
)
print(tokenized_datasets['train'][0])
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

def exact_match_acc(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id) # remove padding
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    result = 0
    for i in range(len(decoded_labels)):
        if decoded_preds == decoded_labels:
            result += 1
    return float(result) / len(decoded_labels)

def safe_save_model_for_hf_trainer(trainer, output_dir='./'):
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)

training_args = TrainingArguments(
    f"TrainingTest-{model_id}",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_checkpointing=True,
    gradient_accumulation_steps=8,
    auto_find_batch_size=True,
    optim="adafactor",
    logging_steps=1,
    save_strategy='step',
    save_steps=500,
    save_total_limit=3,
    bf16=True,
    learning_rate=2e-5,
    num_train_epochs=3,
    weight_decay=0,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    # evaluation_strategy='steps',
    # eval_steps=500,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets["eval"],
    compute_metrics=exact_match_acc
)

trainer.train()
trainer.save_state()
safe_save_model_for_hf_trainer(trainer=trainer, output_dir="./")