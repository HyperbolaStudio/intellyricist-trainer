from datasets import Dataset
import pandas
from modelscope.msdatasets import MsDataset
from modelscope.metainfo import Trainers
from modelscope.trainers import build_trainer

train_dataset = MsDataset(Dataset.from_pandas(pandas.read_csv("./data/train_k.txt", '\0')))
eval_dataset = MsDataset(Dataset.from_pandas(pandas.read_csv("./data/eval_k.txt", '\0')))


num_warmup_steps = 100
def noam_lambda(current_step: int):
    current_step += 1
    return min(current_step**(-0.5),
               current_step * num_warmup_steps**(-1.5))

# 可以在代码修改 configuration 的配置
def cfg_modify_fn(cfg):
    cfg.train.lr_scheduler = {
        'type': 'LambdaLR',
        'lr_lambda': noam_lambda,
        'options': {
            'by_epoch': False
        }
    }
    cfg.train.optimizer = {
        "type": "AdamW",
        "lr": 3e-4,
        "options": {}
    }
    cfg.train.max_epochs = 48
    cfg.train.dataloader = {
        "batch_size_per_gpu": 64,
        "workers_per_gpu": 1
    }
    return cfg

kwargs = dict(
    model='damo/nlp_gpt3_text-generation_chinese-large',
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    work_dir='.tmp/',
    cfg_modify_fn=cfg_modify_fn)
trainer = build_trainer(
    name=Trainers.text_generation_trainer, default_args=kwargs)
trainer.train()
