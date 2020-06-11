from transformers import BertTokenizer, RobertaTokenizer

from transformers import GlueDataset
import pdb

data_args = {
    "task_name": "cola",
    "data_dir": "../../glue_data/CoLA",
    "max_seq_length": 128,
    "overwrite_cache": False
}

tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

class AttributeDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__

data_args = AttributeDict(data_args)

train_dataset = (
    GlueDataset(data_args, tokenizer=tokenizer)
)

pdb.set_trace()

exit = "exit"

pdb.set_trace()