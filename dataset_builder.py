import pandas as pd

tasks = ('k', 'p2')
MAGIC = 114514

for task in tasks:
    data = pd.read_table('./data/data_'+task+'.txt', usecols=[1], header=None)

    data = data.sample(frac=1, random_state=MAGIC)

    eval_dataset = data.sample(3000, random_state=MAGIC)
    train_dataset = data.drop(eval_dataset.index)

    eval_dataset.to_csv('./data/eval_'+task+'.txt', sep='\t', header=['src_txt'], index=None)
    train_dataset.to_csv('./data/train_'+task+'.txt', sep='\t', header=['src_txt'], index=None)
