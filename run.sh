#!/bin/bash
python -m torch.distributed.launch \
    --nnodes=1 \
    --nproc_per_node=8 \
    train.py \
    > shell.log
# The ModelArguments, DataArguments, TrainingArguments are all embedded in the train.py file.