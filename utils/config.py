import os
import logging 
import argparse
from tqdm import tqdm
import torch
import numpy as np


parser = argparse.ArgumentParser(description='Task-oriented Dialogue System Benchmarking')


## Training Setting
parser.add_argument(
    '--do_train', action='store_true', help="do training")
parser.add_argument(
    '-epoch','--epoch', help='number of epochs to train', required=False, default=300, type=int)
parser.add_argument(
    '-patience','--patience', help='patience for early stopping', required=False, default=10, type=int)
parser.add_argument(
    '-earlystop','--earlystop', help='metric for early stopping', required=False, default="loss", type=str)
parser.add_argument(
    '--my_model', help='my cutomized model', required=False, default="")
parser.add_argument(
    '-dr','--dropout', help='Dropout ratio', required=False, type=float, default=0.2)
parser.add_argument(
    '-lr','--learning_rate', help='Learning Rate', required=False, type=float, default=5e-5)
parser.add_argument(
    '-bsz','--batch_size', help='Batch_size', required=False, type=int, default=16)
parser.add_argument(
    '-ebsz','--eval_batch_size', help='Batch_size', required=False, type=int, default=16)
parser.add_argument(
    '-hdd','--hdd_size', help='Hidden size', required=False, type=int, default=400)
parser.add_argument(
    '-emb','--emb_size', help='Embedding size', required=False, type=int, default=400)
parser.add_argument(
    '-clip','--grad_clip', help='gradient clipping', required=False, default=1, type=int) 
parser.add_argument(
    '-tfr','--teacher_forcing_ratio', help='teacher_forcing_ratio', type=float, required=False, default=0.5)
parser.add_argument(
    '-loadEmb','--load_embedding', help='Load Pretrained Glove and Char Embeddings', required=False, default=False, type=bool)
parser.add_argument(
    '-fixEmb','--fix_embedding', help='', required=False, default=False, type=bool)
parser.add_argument(
    '--n_gpu', help='', required=False, default=1, type=int)
parser.add_argument(
    '--eval_by_step', help='', required=False, default=-1, type=int)
parser.add_argument(
    '--fix_encoder', action='store_true', help="")
parser.add_argument(
    '--model_type', help='', required=False, default="bert", type=str)
parser.add_argument(
    '--model_name_or_path', help='', required=False, default="bert", type=str)
parser.add_argument(
    '--usr_token', help='', required=False, default="[USR]", type=str)
parser.add_argument(
    '--sys_token', help='', required=False, default="[SYS]", type=str)
parser.add_argument(
    '--warmup_proportion', help='warm up training in the begining', required=False, default=0.1, type=float)
parser.add_argument(
    "--local_rank", type=int, default=-1, help="For distributed training: local_rank")
parser.add_argument(
        "--gradient_accumulation_steps", type=int, default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",)
parser.add_argument(
    "--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
parser.add_argument(
    "--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
parser.add_argument(
    "--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
parser.add_argument(
        "--fp16", action="store_true", help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",)
parser.add_argument(
        "--fp16_opt_level", type=str, default="O1", 
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']." 
        "See details at https://nvidia.github.io/apex/amp.html",)
parser.add_argument(
    "--output_mode", default="classification", type=str, help="")
parser.add_argument(
        "--max_steps", default=-1, type=int, help="If > 0: set total number of training steps to perform. Override num_train_epochs.",) 
parser.add_argument(
        "--rand_seed", default=0, type=int, help="")
parser.add_argument(
        "--fix_rand_seed", action="store_true", help="fix random seed for training",) 
parser.add_argument(
        "--nb_runs", default=1, type=int, help="number of runs to conduct during training")
parser.add_argument(
        "--nb_evals", default=1, type=int, help="number of runs to conduct during inference")
parser.add_argument(
        "--max_seq_length", default=512, type=int, help="")
parser.add_argument(
        "--input_name", default="context", type=str, help="")
    
    
## Dataset or Input/Output Setting
parser.add_argument(
    '-dpath','--data_path', help='path to dataset folder, need to change to your local folder', 
    required=False, default='/export/home/dialog_datasets', type=str)
parser.add_argument(
    '-task','--task', help='task in ["nlu", "dst", "dm", "nlg", "usdl"] to decide which dataloader to use', required=True)
parser.add_argument(
    '-task_name','--task_name', help='task in ["intent", "sysact","rs"]', required=False, default="")
parser.add_argument(
    '--example_type', help='type in ["turn", "dial"]', required=False, default="turn")
parser.add_argument(
    '-ds','--dataset', help='which dataset to be used.', required=False, default='["multiwoz"]', type=str)
parser.add_argument(
    '-load_path','--load_path', help='path of the saved model to load from', required=False)
parser.add_argument(
    '-an','--add_name', help='An added name for the save folder', required=False, default='')
parser.add_argument(
    '--max_line', help='maximum line for reading data (for quick testing)', required=False, default=None, type=int)
parser.add_argument(
    '--output_dir', help='', required=False, default="save/temp/", type=str)
parser.add_argument(
    '--overwrite', action='store_true', help="")
parser.add_argument(
    "--cache_dir", default=None, type=str,
    help="Optional directory to store the pre-trained models downloaded from s3 (instead of the default one)",)
parser.add_argument(
    "--logging_steps", default=500, type=int, help="")
parser.add_argument(
    "--save_steps", default=1000, type=int, help="")
parser.add_argument(
        "--save_total_limit", type=int, default=1,
        help="Limit the total amount of checkpoints, delete the older checkpoints in the output_dir",)
parser.add_argument(
        "--train_data_ratio", default=1.0, type=float, help="")
parser.add_argument(
        "--domain_act", action="store_true", help="",)  
parser.add_argument(
        "--only_last_turn", action="store_true", help="",)  
parser.add_argument(
        "--error_analysis", action="store_true", help="",) 
parser.add_argument(
        "--not_save_model", action="store_true", help="")
parser.add_argument(
        "--nb_shots", default=-1, type=int, help="")


## Others (May be able to delete or not used in this repo)
parser.add_argument(
    '--do_embeddings', action='store_true')
parser.add_argument(
    '--create_own_vocab', action='store_true', help="")
parser.add_argument(
    '-um','--unk_mask', help='mask out input token to UNK', type=bool, required=False, default=True)
parser.add_argument(
    '-paral','--parallel_decode', help='', required=False, default=True, type=bool)
parser.add_argument(
    '--self_supervised', help='', required=False, default="generative", type=str)
parser.add_argument(
        "--oracle_domain", action="store_true", help="",)
parser.add_argument(
        "--more_linear_mapping", action="store_true", help="",) 
parser.add_argument(
        "--gate_supervision_for_dst", action="store_true", help="",) 
parser.add_argument(
        "--sum_token_emb_for_value", action="store_true", help="",) 
parser.add_argument(
        "--nb_neg_sample_rs", default=0, type=int, help="")
parser.add_argument(
        "--sample_negative_by_kmeans", action="store_true", help="",) 
parser.add_argument(
        "--nb_kmeans", default=1000, type=int, help="")
parser.add_argument(
        "--bidirect", action="store_true", help="",)       
parser.add_argument(
    '--rnn_type', help='rnn type ["gru", "lstm"]', required=False, type=str, default="gru")
parser.add_argument(
    '--num_rnn_layers', help='rnn layers size', required=False, type=int, default=1)
parser.add_argument(
    '--zero_init_rnn',action='store_true', help="set initial hidden of rnns zero")
parser.add_argument(
        "--do_zeroshot", action="store_true", help="",)  
parser.add_argument(
        "--oos_threshold", action="store_true", help="",) 
parser.add_argument(
        "--ontology_version", default="", type=str, help="1.0 is the cleaned version but not used")
parser.add_argument(
        "--dstlm", action="store_true", help="",) 
parser.add_argument(
    '-viz','--vizualization', help='vizualization', type=int, required=False, default=0)


    
args = vars(parser.parse_args())
# args = parser.parse_args()
print(str(args))

# check output_dir
if os.path.exists(args["output_dir"]) and os.listdir(args["output_dir"]) and args["do_train"] and (not args["overwrite"]):
    raise ValueError("Output directory ({}) already exists and is not empty.".format(args["output_dir"]))
os.makedirs(args["output_dir"], exist_ok=True)

# Dictionary Predefined
SEEDS = [10, 5, 0] # np.arange(0, 100, 5)

