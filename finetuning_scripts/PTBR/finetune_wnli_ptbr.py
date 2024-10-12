    # Fine-tuning  using Hugging Face Transformers

    # Install required libraries
    #!pip install transformers
    #!pip install datasets
    #!pip install transformers[torch] accelerate -U
    #!pip install evaluate

import os
from datasets import load_dataset, load_from_disk, Dataset, DatasetDict
from transformers import (AutoModelForSequenceClassification, AutoTokenizer, 
DataCollatorWithPadding, TrainingArguments, Trainer, AdamW, get_constant_schedule,get_linear_schedule_with_warmup,
)
import evaluate
import numpy as np
import pandas as pd

# Links for the WNLI task
train_file = "https://huggingface.co/datasets/PORTULAN/extraglue/resolve/main/data/wnli_pt-PT/train.jsonl"
val_file = "https://huggingface.co/datasets/PORTULAN/extraglue/resolve/main/data/wnli_pt-PT/validation.jsonl"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"  # Change this depending on the GPU to use
# Load the dataset from the JSONL files
dataset = load_dataset('json', data_files={'train': train_file, 'validation':val_file})
checkpoint = "PORTULAN/albertina-900m-portuguese-ptbr-encoder-brwac"

model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
tokenizer = AutoTokenizer.from_pretrained("PORTULAN/albertina-900m-portuguese-ptbr-encoder-brwac")

optimizer = AdamW(model.parameters(), lr=1e-6)
scheduler = get_constant_schedule(optimizer)

###########################################################################################################
#Define Train, Validation and Test sets (cannot use original test set, as it has no labels)
#Original Validation Set -> Test Set
# 90% Original Train Set -> New Train Set
# 10% Original Train Set -> New Validation Set

df1_train = pd.DataFrame(dataset['train'])
df1_val = pd.DataFrame(dataset['validation'])

train_size = int(len(df1_train) * 0.9)
val_size = len(df1_train) - train_size

df1_test = df1_val
df1_val = df1_train.tail(val_size)
df1_train = df1_train.head(train_size)

train_dataset = Dataset.from_pandas(df1_train)
validation_dataset = Dataset.from_pandas(df1_val)
test_dataset = Dataset.from_pandas(df1_test)

# Make a dict from all 3 datasets
dataset = DatasetDict({
    "train": train_dataset,
    "validation": validation_dataset,
    "test": test_dataset,
})
##########################################################################################################

def tokenize_function(example):
    return tokenizer(example['sentence1'], example['sentence2'], padding = True, truncation=True)


def compute_metrics(eval_preds):
    metric = evaluate.load("glue", "wnli")
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


tokenized_datasets = dataset.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

training_args = TrainingArguments(
    output_dir ='albertina-ptbr-900M',          # output directory
    run_name = "test-trainer",                  # name of the training run
    num_train_epochs = 10,                      # total number of training epochs
    #logging_dir = './logs/albertina-ptbr-900M', # directory for storing logs
    evaluation_strategy = "epoch",              # evaluation after each epoch
    save_strategy = "epoch",                    # save checkpoint at end of epoch
    save_total_limit = 2,                       # limit the total amount of checkpoints
    metric_for_best_model = "accuracy",         # metric to use for determining the best models
    load_best_model_at_end = True,              # load the best model found during training at the end
    per_device_train_batch_size = 8,            # batch size for training set
    per_device_eval_batch_size = 8,             # batch size for validation set (should be the same as above)
    report_to = "tensorboard",                  # Generate charts on metric for each epoch
)

trainer = Trainer(
    model = model,
    args = training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    optimizers=(optimizer, scheduler),
)

print("TRAINING MODEL...")
trainer.train()

#model.save_pretrained("~/tese/models/Albertina-ptbr")

print("EVALUATING ON TEST SET...\n")

#evaluation_results = trainer.predict(tokenized_datasets["test"])  # This returns predictions
evaluation_results = trainer.evaluate(tokenized_datasets["test"])  # This returns metrics
print(evaluation_results)