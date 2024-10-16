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

dataset = load_from_disk("~/tese/datasets/WSC")      # use this if using dataset from a given directory
checkpoint = "PORTULAN/albertina-900m-portuguese-ptbr-encoder-brwac"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Change this depending on the GPU to use

model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

optimizer = AdamW(model.parameters(), lr=1e-6)
scheduler = get_constant_schedule(optimizer)

def tokenize_function(example):
    
    concatenated_examples = {
    'conc': [],
    'pronoun': []
    }
    for sentence, noun, pronoun in zip(example['sentence'], example['noun'], example['pronoun']):
        concatenated_examples['conc'].append(f"{sentence} [SEP] {noun}")
        concatenated_examples['pronoun'].append(pronoun)

    tokenized_input = tokenizer(
        text = concatenated_examples['conc'],
        text_pair = concatenated_examples['pronoun'],
        padding = True,
        truncation = True,
    )

    return tokenized_input

def compute_metrics(eval_preds):
    metric = evaluate.load("super_glue", "wsc")
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