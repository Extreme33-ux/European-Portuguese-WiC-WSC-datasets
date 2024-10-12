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

dataset = load_from_disk("~/tese/datasets/WiC")      # use this if using dataset from a given directory
checkpoint = "PORTULAN/albertina-1b5-portuguese-ptpt-encoder"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Change this depending on the GPU to use

model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

optimizer = AdamW(model.parameters(), lr=1e-6)
scheduler = get_constant_schedule(optimizer)

def tokenize_function(example):
    
    concatenated_examples = {
    'conc': [],
    'word':[]
    }
    for sentence1, sentence2, word in zip(example['sentence1'], example['sentence2'], example['word']):
        concatenated_examples['conc'].append(f"{sentence1} [SEP] {sentence2}")
        concatenated_examples['word'].append(word)

    tokenized_input = tokenizer(
        text = concatenated_examples['conc'],
        text_pair = concatenated_examples['word'],
        padding = True,
        truncation = True,
    )

    return tokenized_input

def compute_metrics(eval_preds):
    metric = evaluate.load('glue', 'mrpc')
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

tokenized_datasets = dataset.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

training_args = TrainingArguments(
    output_dir ='albertina-ptpt-1.5B',          # output directory
    run_name = "test-trainer",                  # name of the training run
    num_train_epochs = 10,                      # total number of training epochs
    logging_dir = './logs/albertina-ptpt-1.5B', # directory for storing logs
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

#model.save_pretrained("~/tese/models/Albertina-ptpt")

print("EVALUATING ON TEST SET...\n")

#evaluation_results = trainer.predict(tokenized_datasets["test"])  # This returns predictions
evaluation_results = trainer.evaluate(tokenized_datasets["test"])  # This returns metrics
print(evaluation_results)
