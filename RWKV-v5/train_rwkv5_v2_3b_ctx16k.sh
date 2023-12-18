#!/bin/bash

BASE_NAME="model/QA-3B"
N_LAYER="32"
N_EMBD="2560"
CTX_LEN="16384" # "16384"
QA_MASK="1"
M_BSZ="1" # takes 16G VRAM (reduce this to save VRAM)
LR_INIT="1e-5"
LR_FINAL="1e-5"
GRAD_CP=1 # set to 1 to save VRAM (will be slower)
EPOCH_SAVE=1

nohup \
python train.py --load_model "/sata/pre_models/RWKV/RWKV-5-World-3B-v2-20231118-ctx16k.pth" --wandb "" --proj_dir $BASE_NAME \
 --data_file "/home/zidian/PycharmProjects/json2binidx_tool/my_data/conversation_text_document" --data_type "binidx" --vocab_size 65536 \
 --ctx_len $CTX_LEN --my_qa_mask $QA_MASK --epoch_steps 200 --epoch_count 100 --epoch_begin 0 --epoch_save $EPOCH_SAVE \
 --micro_bsz $M_BSZ --n_layer $N_LAYER --n_embd $N_EMBD --pre_ffn 0 --head_qk 0 \
 --lr_init $LR_INIT --lr_final $LR_FINAL --warmup_steps 0 --beta1 0.9 --beta2 0.99 --adam_eps 1e-8  \
 --weight_decay 0.001 --head_size_a 64 \
 --accelerator gpu --num_nodes 1 --devices 1 --precision bf16 --strategy deepspeed_stage_2_offload --grad_cp $GRAD_CP \
 --ds_bucket_mb 200 \
&
