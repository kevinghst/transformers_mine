from transformers import BertTokenizer, RobertaTokenizer

from transformers import GlueDataset, EvalPrediction

from transformers import (
    Trainer,
    glue_compute_metrics,
    glue_output_modes,
    glue_tasks_num_labels,
    set_seed,
)

import dataclasses
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional

import numpy as np

import pdb

data_args = {
    "task_name": "cola",
    "data_dir": "../../glue_data/CoLA",
    "max_seq_length": 128,
    "overwrite_cache": False
}

training_args = {
    'output_dir':'/tmp/CoLA/', 
    'overwrite_output_dir': True, 
    'do_train': True, 
    'do_eval': True, 
    'do_predict': False, 
    'evaluate_during_training': False, 
    'per_device_train_batch_size': 8, 
    'per_device_eval_batch_size': 8, 
    'per_gpu_train_batch_size': 32, 
    'per_gpu_eval_batch_size': None, 
    'gradient_accumulation_steps': 1, 
    'learning_rate': 3e-05, 
    'weight_decay': 0.0, 
    'adam_epsilon': 1e-08, 
    'max_grad_norm': 1.0, 
    'num_train_epochs': 3.0, 
    'max_steps': -1, 
    'warmup_steps': 0, 
    'logging_dir': 'runs/Jun12_11-44-41_gpu-68', 
    'logging_first_step': False, 
    'logging_steps': 500, 
    'save_steps': 500, 
    'save_total_limit': None, 
    'no_cuda': False, 
    'seed': 42, 
    'fp16': False, 
    'fp16_opt_level': 'O1', 
    'local_rank': -1, 
    'tpu_num_cores': None, 
    'tpu_metrics_debug': False, 
    'dataloader_drop_last': False
}

model_name_or_path = 'roberta-base'
output_dir = '/tmp/CoLA/'

tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

class AttributeDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__

data_args = AttributeDict(data_args)
training_args = AttributeDict(training_args)

train_dataset = (
    GlueDataset(data_args, tokenizer=tokenizer)
)
eval_dataset = (
    GlueDataset(data_args, tokenizer=tokenizer, mode="dev")
)
test_dataset = None


def build_compute_metrics_fn(task_name: str) -> Callable[[EvalPrediction], Dict]:
    def compute_metrics_fn(p: EvalPrediction):
        if output_mode == "classification":
            preds = np.argmax(p.predictions, axis=1)
        elif output_mode == "regression":
            preds = np.squeeze(p.predictions)
        return glue_compute_metrics(task_name, preds, p.label_ids)

    return compute_metrics_fn


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=build_compute_metrics_fn(data_args.task_name),
)


trainer.train(
    model_path=model_name_or_path if os.path.isdir(model_name_or_path) else None
)
trainer.save_model()

# Evaluation
eval_results = {}

logger.info("*** Evaluate ***")

# Loop to handle MNLI double evaluation (matched, mis-matched)
eval_datasets = [eval_dataset]

for eval_dataset in eval_datasets:
    trainer.compute_metrics = build_compute_metrics_fn(eval_dataset.args.task_name)
    eval_result = trainer.evaluate(eval_dataset=eval_dataset)

    output_eval_file = os.path.join(
        output_dir, f"eval_results_{eval_dataset.args.task_name}.txt"
    )
    if trainer.is_world_master():
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results {} *****".format(eval_dataset.args.task_name))
            for key, value in eval_result.items():
                logger.info("  %s = %s", key, value)
                writer.write("%s = %s\n" % (key, value))

    eval_results.update(eval_result)


#pdb.set_trace()

#exit = "exit"

#pdb.set_trace()