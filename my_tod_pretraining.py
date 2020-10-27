import argparse
import glob
import logging
import os
import pickle
import random
import re
import shutil
from typing import Tuple
import gzip
import shelve
import json
import math
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from concurrent.futures import ThreadPoolExecutor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Sampler

from utils.utils_general import *
from utils.utils_multiwoz import *
from utils.utils_camrest676 import *
from utils.utils_woz import *
from utils.utils_smd import *
from utils.utils_frames import *
from utils.utils_msre2e import *
from utils.utils_taskmaster import *
from utils.utils_metalwoz import *
from utils.utils_schema import *

from transformers import (
    WEIGHTS_NAME,
    AdamW,
    BertConfig,
    BertModel,
    BertForMaskedLM,
    BertTokenizer,
    CamembertConfig,
    CamembertForMaskedLM,
    CamembertTokenizer,
    DistilBertConfig,
    DistilBertForMaskedLM,
    DistilBertTokenizer,
    GPT2Config,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    OpenAIGPTConfig,
    OpenAIGPTLMHeadModel,
    OpenAIGPTTokenizer,
    PreTrainedTokenizer,
    RobertaConfig,
    RobertaForMaskedLM,
    RobertaTokenizer,
    get_linear_schedule_with_warmup,
)


try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter


logger = logging.getLogger(__name__)


MODEL_CLASSES = {
    "gpt2": (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer),
    "openai-gpt": (OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
    "bert": (BertConfig, BertForMaskedLM, BertTokenizer),
    "bert-seq": (BertConfig, BertModel, BertTokenizer),
    "roberta": (RobertaConfig, RobertaForMaskedLM, RobertaTokenizer),
    "distilbert": (DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer),
    "camembert": (CamembertConfig, CamembertForMaskedLM, CamembertTokenizer),
}


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def _rotate_checkpoints(args, checkpoint_prefix, use_mtime=False):
    if not args.save_total_limit:
        return
    if args.save_total_limit <= 0:
        return

    # Check if we should delete older checkpoint(s)
    glob_checkpoints = glob.glob(os.path.join(args.output_dir, "{}-*".format(checkpoint_prefix)))
    if len(glob_checkpoints) <= args.save_total_limit:
        return

    ordering_and_checkpoint_path = []
    for path in glob_checkpoints:
        if use_mtime:
            ordering_and_checkpoint_path.append((os.path.getmtime(path), path))
        else:
            regex_match = re.match(".*{}-([0-9]+)".format(checkpoint_prefix), path)
            if regex_match and regex_match.groups():
                ordering_and_checkpoint_path.append((int(regex_match.groups()[0]), path))

    checkpoints_sorted = sorted(ordering_and_checkpoint_path)
    checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]
    number_of_checkpoints_to_delete = max(0, len(checkpoints_sorted) - args.save_total_limit)
    checkpoints_to_be_deleted = checkpoints_sorted[:number_of_checkpoints_to_delete]
    for checkpoint in checkpoints_to_be_deleted:
        logger.info("Deleting older checkpoint [{}] due to args.save_total_limit".format(checkpoint))
        shutil.rmtree(checkpoint)


def mask_tokens(inputs: torch.Tensor, tokenizer: PreTrainedTokenizer, args) -> Tuple[torch.Tensor, torch.Tensor]:
    """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """
    
    inputs = inputs.to("cpu")
    labels = inputs.clone()
    
    # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
    probability_matrix = torch.full(labels.shape, args.mlm_probability)
    special_tokens_mask = [
        tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
    ]
    probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
    
    # padding position value = 0
    inputs_pad_pos = (inputs == 0).cpu()
    probability_matrix.masked_fill_(inputs_pad_pos, value=0.0)
    
    masked_indices = torch.bernoulli(probability_matrix).bool()
    try:
        labels[~masked_indices] = -100  # We only compute loss on masked tokens
    except:
        masked_indices = masked_indices.byte()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens
        
    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    try:
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    except:
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool().byte() & masked_indices
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)
    
    # 10% of the time, we replace masked input tokens with random word
    try:
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    except:
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool().byte() & masked_indices & ~indices_replaced
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
    if inputs.is_cuda:
        indices_random = indices_random.to(args.device)
        random_words = random_words.to(args.device)
    inputs[indices_random] = random_words[indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return inputs, labels


def mask_for_response_selection(batch, tokenizer, args, cand_uttr_sys_dict, others, set_max_resp_len=150):
    """ Prepare (context,response) pairs for response contrastive learning (RCL). """
    
    inputs = batch["context"]
    inputs = inputs.to("cpu")
    batch_size = inputs.size(0)
    probability_matrix = torch.full(inputs.shape, 1)
    usr_token_idx = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(args.usr_token))[0]
    sys_token_idx = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(args.sys_token))[0]
    cand_uttr_sys = list(cand_uttr_sys_dict.keys())
    cand_uttr_sys_tokens = list(cand_uttr_sys_dict.values())

    ## Find positions of user and system tokens and split them into (context, repsonse) pairs
    last_sys_position, last_usr_position = [], []
    for bsz_i, batch_sample in enumerate(inputs):
        nb_sys_token = len((batch_sample == sys_token_idx).nonzero())
        nb_usr_token = len((batch_sample == usr_token_idx).nonzero())
        
        ## Randomly select a turn to split 
        if nb_sys_token == 0 or nb_usr_token == 0: 
            last_sys_position.append(len(batch_sample)//2)
            last_usr_position.append(len(batch_sample))
        else:
            if nb_sys_token > 2 and nb_usr_token > 2:
                rand_pos = random.randint(1, min(nb_sys_token, nb_usr_token)-1)
            else:
                rand_pos = -1

            temp1 = (batch_sample == sys_token_idx).nonzero()[rand_pos][0].item()
            last_sys_position.append(temp1)
            temp2 = (batch_sample == usr_token_idx).nonzero()[rand_pos][0].item()
            if temp2 > temp1:
                last_usr_position.append(temp2)
            else:
                if temp1 + 10 < len(batch_sample):
                    last_usr_position.append(temp1 + 10)
                else:
                    last_usr_position.append(len(batch_sample))

    last_usr_position = np.array(last_usr_position)
    last_sys_position = np.array(last_sys_position)
    max_last_sys_position = max(last_sys_position)
    max_response_len = max(last_usr_position-last_sys_position) + 1
    max_response_len = max_response_len if max_response_len < set_max_resp_len else set_max_resp_len
    
    ## placeholders
    input_contexts = torch.zeros(batch_size, max_last_sys_position).long() #.to(args.device)
    input_responses = torch.zeros(batch_size, max_response_len).long() #.to(args.device)
    output_labels = torch.tensor(np.arange(batch_size)).long() #.to(args.device)
    
    ## assign response indexs by start and end position
    responses = []
    for bsz_i, (sys_pos, usr_pos) in enumerate(zip(last_sys_position, last_usr_position)):
        input_contexts[bsz_i, :sys_pos] = inputs[bsz_i, :sys_pos]
        input_responses[bsz_i, 0] = inputs[bsz_i, 0] ## CLS token  
        responses.append(tokenizer.decode(inputs[bsz_i, sys_pos+1:usr_pos]).replace(" ", ""))
        s, e = (sys_pos, usr_pos) if usr_pos-sys_pos < max_response_len else (sys_pos, sys_pos+max_response_len-1)
        input_responses[bsz_i, 1:e-s+1] = inputs[bsz_i, s:e]
    
    ## Add additional negative samples. Either randomly select from candidate pool or choose by Kmeans.
    candidates_tokens = []
    if args.negative_sampling_by_kmeans:
        for ri, resp in enumerate(responses):
            if resp in others["ToD_BERT_SYS_UTTR_KMEANS"].keys():
                cur_cluster = others["ToD_BERT_SYS_UTTR_KMEANS"][resp]
                candidates = others["KMEANS_to_SENTS"][cur_cluster]
                nb_selected = min(args.nb_addition_negresponse_per_sample, len(candidates)-1)
                start_pos = random.randint(0, len(candidates)-nb_selected-1)
                sampled_neg_resps = candidates[start_pos:start_pos+nb_selected]
                candidates_tokens += [cand_uttr_sys_dict[r] for r in sampled_neg_resps]
            else:
                start_pos = random.randint(0, len(cand_uttr_sys)-args.nb_addition_negresponse_per_sample-1)
                candidates_tokens += cand_uttr_sys_tokens[start_pos:start_pos+args.nb_addition_negresponse_per_sample]  
    else:
        for i in range(args.nb_addition_negresponse_per_sample * batch_size):
            pos = random.randint(0, len(cand_uttr_sys_tokens)-1)
            candidates_tokens.append(cand_uttr_sys_tokens[pos])
    
    ## Padding
    input_responses_neg = torch.zeros(len(candidates_tokens), max_response_len).long()
    for i in range(len(candidates_tokens)):
        if len(candidates_tokens[i]) > max_response_len:
            input_responses_neg[i] = candidates_tokens[i][:max_response_len]
        else:
            input_responses_neg[i, :len(candidates_tokens[i])] = candidates_tokens[i]
    
    ## Add those negative responses to response selection pool
    input_responses = torch.cat([input_responses, input_responses_neg], 0)
    
    return input_contexts, input_responses, output_labels


def get_candidate_embeddings(uttr_sys_dict, tokenizer, model):
    """
    obtain candidate representations by passing through model encoding,
    return a dictionary that maps sentences to embeddings
    """
    print("Start obtaining representations from model...")
    ToD_BERT_SYS_UTTR_EMB = {}
    uttr_sys = list(uttr_sys_dict.keys())
    uttr_sys_tokens = list(uttr_sys_dict.values())
    batch_size = 100
    for start in tqdm(range(0, len(uttr_sys), batch_size)):
        if start+batch_size > len(uttr_sys):
            inputs = uttr_sys[start:]
            inputs_ids = uttr_sys_tokens[start:]
        else:
            inputs = uttr_sys[start:start+batch_size]
            inputs_ids = uttr_sys_tokens[start:start+batch_size]
        inputs_ids = pad_sequence(inputs_ids, batch_first=True, padding_value=0)
        if torch.cuda.is_available(): inputs_ids = inputs_ids.cuda()
        with torch.no_grad():
            outputs = model.bert(input_ids=inputs_ids, attention_mask=inputs_ids>0)
            sequence_output = outputs[0]
            cls_rep = sequence_output[:, 0, :]
        for i in range(cls_rep.size(0)):
            ToD_BERT_SYS_UTTR_EMB[inputs[i].replace(" ", "")] = {
                "sent":inputs[i],
                "emb":cls_rep[i, :].cpu().numpy()}
    return ToD_BERT_SYS_UTTR_EMB
    
    
def get_candidate_kmeans(args, uttr_sys_dict, tokenizer, model):
    """obtain kmeans clustering results"""
    import faiss
    print("Start computing kmeans with faiss...")
    ToD_BERT_SYS_UTTR_EMB = get_candidate_embeddings(uttr_sys_dict, tokenizer, model)
    ToD_BERT_SYS_UTTR_KMEANS = {}
    KMEANS_to_SENTS = {i:[] for i in range(args.nb_kmeans)}
    data = [v["emb"] for v in ToD_BERT_SYS_UTTR_EMB.values()]
    data = np.array(data)
    kmeans_1k = faiss.Kmeans(data.shape[1], args.nb_kmeans, niter=20, nredo=5, verbose=True)
    kmeans_1k.train(data)
    D, I = kmeans_1k.index.search(data, 1)
    for i, key in enumerate(ToD_BERT_SYS_UTTR_EMB.keys()):
        ToD_BERT_SYS_UTTR_KMEANS[key] = I[i][0]
        KMEANS_to_SENTS[I[i][0]].append(ToD_BERT_SYS_UTTR_EMB[key]["sent"])
        
    return ToD_BERT_SYS_UTTR_KMEANS, KMEANS_to_SENTS
    
    
def train(args, trn_loader, dev_loader, model, tokenizer, cand_uttr_sys_dict, others):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter("runs/"+args.output_dir.replace("/","-"))

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(trn_loader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(trn_loader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    # Check if saved optimizer or scheduler states exist
    if os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt")) and os.path.isfile(
        os.path.join(args.model_name_or_path, "scheduler.pt")
    ):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
        )

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Num batches = %d", len(trn_loader))
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0

    tr_loss, logging_loss = 0.0, 0.0
    loss_mlm, loss_rs = 0, 0
    patience, best_loss = 0, 1e10
    xeloss = torch.nn.CrossEntropyLoss()

    model_to_resize = model.module if hasattr(model, "module") else model  # Take care of distributed/parallel training
    model_to_resize.resize_token_embeddings(len(tokenizer))

    model.zero_grad()
    train_iterator = trange(
        epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0]
    )
    set_seed(args)  # Added here for reproducibility
    
    for _ in train_iterator:
        
        ## Calculate kmeans results in the beginning of each epoch
        if args.negative_sampling_by_kmeans:
            ToD_BERT_SYS_UTTR_KMEANS, KMEANS_to_SENTS = get_candidate_kmeans(args, cand_uttr_sys_dict, tokenizer, model)
            trn_loader = get_loader(vars(args), "train", tokenizer, others["datasets"], others["unified_meta"], "train")
        
        loss_arr, loss_mlm_arr, loss_rs_arr = [], [], []
        epoch_iterator = tqdm(trn_loader, disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            model.train()
            
            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue
            
            ## add response selection into pretraining
            if args.add_rs_loss: 
                kmeans_others = {"ToD_BERT_SYS_UTTR_KMEANS":ToD_BERT_SYS_UTTR_KMEANS, 
                                 "KMEANS_to_SENTS":KMEANS_to_SENTS} if args.negative_sampling_by_kmeans else {}
                
                ## Split dialogue into (context, response) pairs
                input_cont, input_resp, resp_label = mask_for_response_selection(batch, 
                                                                                 tokenizer, 
                                                                                 args, 
                                                                                 cand_uttr_sys_dict, 
                                                                                 kmeans_others)
                ## Mask context part for MLM loss
                input_cont, labels = mask_tokens(input_cont, tokenizer, args) if args.mlm else (input_cont, input_cont)
                
                ## Allocate tensors to (gpu) devices
                input_cont = input_cont.to(args.device)
                input_resp = input_resp.to(args.device)
                resp_label = resp_label.to(args.device)
                labels = labels.to(args.device)
                
                ## Encode the context part with BERT
                outputs = model.bert(
                    input_cont,
                    attention_mask=input_cont>0,
                )
                sequence_output = outputs[0]
                hid_cont = sequence_output[:, 0, :] ## CLS token
                
                ## Calculate MLM loss for the context
                prediction_scores = model.cls(sequence_output)
                loss = xeloss(prediction_scores.view(-1, model.config.vocab_size), labels.view(-1))
                loss_mlm = loss.item()
                
                ## Encode the response part with BERT
                outputs = model.bert(
                    input_resp,
                    attention_mask=input_resp>0,
                )
                sequence_output = outputs[0]
                hid_resp = sequence_output[:, 0, :]
                
                ## Calculate RCL loss
                scores = torch.matmul(hid_cont, hid_resp.transpose(1, 0))
                loss_rs = xeloss(scores, resp_label)
                loss += loss_rs
                loss_rs = loss_rs.item()
            
            ## with only MLM loss
            else: 
                inputs = batch["context"].clone()
                if args.mlm:
                    inputs, labels = mask_tokens(inputs, tokenizer, args)
                    inputs = inputs.to(args.device)
                    labels = labels.to(args.device)
                    outputs = model(inputs, 
                                    masked_lm_labels=labels,
                                    attention_mask=inputs>0)
                else:
                    labels = inputs.clone()
                    masked_indices = (labels == 0)
                    labels[masked_indices] = -100 
                    outputs = model(inputs, labels=labels)
                loss = outputs[0]  # model outputs are always tuple in transformers (see doc)
                loss_mlm = loss.item()

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            loss_arr.append(loss.item())
            loss_mlm_arr.append(loss_mlm)
            loss_rs_arr.append(loss_rs)
            
            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            
            ## Print loss
            epoch_iterator.set_description("Loss:{:.4f} MLM:{:.4f} RS:{:.4f}".format(np.mean(loss_arr), 
                                                                                     np.mean(loss_mlm_arr), 
                                                                                     np.mean(loss_rs_arr)))
            
            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar("loss", (tr_loss - logging_loss) / args.logging_steps, global_step)
                    logging_loss = tr_loss

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    
                    if args.evaluate_during_training and args.n_gpu == 1:
                        results = evaluate(args, model, dev_loader, tokenizer)
                        for key, value in results.items():
                            tb_writer.add_scalar("eval_{}".format(key), value, global_step)
                    else:
                        results = {}
                        results["loss"] = best_loss - 0.1 # always saving
                        
                    if results["loss"] < best_loss:
                        patience = 0
                        best_loss = results["loss"]
                        
                        checkpoint_prefix = "checkpoint"
                        # Save model checkpoint
                        output_dir = os.path.join(args.output_dir, "{}-{}".format(checkpoint_prefix, global_step))
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                            model_to_save = (
                            model.module if hasattr(model, "module") else model
                        )  # Take care of distributed/parallel training
                        model_to_save.save_pretrained(output_dir)
                        tokenizer.save_pretrained(output_dir)

                        torch.save(args, os.path.join(output_dir, "training_args.bin"))
                        logger.info("Saving model checkpoint to %s", output_dir)

                        _rotate_checkpoints(args, checkpoint_prefix)

                        torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                        torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                        logger.info("Saving optimizer and scheduler states to %s", output_dir)
                    else:
                        patience += 1
                        logger.info("Current patience: patience {}".format(patience))

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
            
            if patience > args.patience:
                logger.info("Ran out of patience...")
                break
            
        if (args.max_steps > 0 and global_step > args.max_steps) or patience > args.patience:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step


def evaluate(args, model, dev_loader, tokenizer, prefix=""):
    eval_output_dir = args.output_dir

    if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir)

    eval_dataloader = dev_loader
    
    # multi-gpu evaluate
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataloader))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        
        inputs = batch["context"].clone()
        
        #inputs, labels = mask_tokens(inputs, tokenizer, args) if args.mlm else (inputs, inputs)
        if args.mlm:
            inputs, labels = mask_tokens(inputs, tokenizer, args)
        else:
            labels = inputs.clone()
            masked_indices = (labels == 0)
            labels[masked_indices] = -100 

        inputs = inputs.to(args.device)
        labels = labels.to(args.device)

        with torch.no_grad():
            outputs = model(inputs, 
                            masked_lm_labels=labels,
                            attention_mask=inputs>0) if args.mlm else model(inputs, labels=labels)
            
            lm_loss = outputs[0]
            eval_loss += lm_loss.mean().item()
        nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps
    perplexity = torch.exp(torch.tensor(eval_loss))

    result = {"perplexity": perplexity, "loss":eval_loss}

    output_eval_file = os.path.join(eval_output_dir, prefix, "eval_results.txt")
    with open(output_eval_file, "w") as writer:
        logger.info("***** Eval results {} *****".format(prefix))
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))

    return result


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--model_type", default="bert", type=str, help="The model architecture to be fine-tuned.")
    parser.add_argument(
        "--model_name_or_path",
        default="bert-base-uncased",
        type=str,
        help="The model checkpoint for weights initialization.",
    )
    parser.add_argument(
        "--mlm", action="store_true", help="Train with masked-language modeling loss instead of language modeling."
    )
    parser.add_argument(
        "--mlm_probability", type=float, default=0.15, help="Ratio of tokens to mask for masked language modeling loss"
    )
    parser.add_argument(
        "--config_name",
        default="",
        type=str,
        help="Optional pretrained config name or path if not the same as model_name_or_path",
    )
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path",
    )
    parser.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help="Optional directory to store the pre-trained models downloaded from s3 (instead of the default one)",
    )
    parser.add_argument(
        "--block_size",
        default=-1,
        type=int,
        help="Optional input sequence length after tokenization."
        "The training dataset will be truncated in block of this size for training."
        "Default to the model max input length for single sentence inputs (take into account special tokens).",
    )
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
    parser.add_argument(
        "--evaluate_during_training", action="store_true", help="Run evaluation during training at each logging step."
    )
    parser.add_argument(
        "--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model."
    )

    parser.add_argument("--per_gpu_train_batch_size", default=4, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument(
        "--per_gpu_eval_batch_size", default=4, type=int, help="Batch size per GPU/CPU for evaluation."
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--num_train_epochs", default=300, type=int, help="Total number of training epochs to perform."
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")

    parser.add_argument("--logging_steps", type=int, default=100, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=100, help="Save checkpoint every X updates steps.")
    parser.add_argument(
        "--save_total_limit",
        type=int,
        default=1,
        help="Limit the total amount of checkpoints, delete the older checkpoints in the output_dir, does not delete by default",
    )
    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help="Evaluate all checkpoints starting with the same prefix as model_name_or_path ending and ending with step number",
    )
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument(
        "--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory"
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
        "See details at https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--server_ip", type=str, default="", help="For distant debugging.")
    parser.add_argument("--server_port", type=str, default="", help="For distant debugging.")
    
    ## My add
    parser.add_argument(
        '-dpath','--data_path', 
        help='path to dataset folder', 
        required=False, 
        default='/export/home/dialog_datasets', 
        type=str)
    parser.add_argument(
        '-ds','--dataset', 
        help='which dataset to be used for training.',
        required=False, 
        default='["multiwoz", "camrest676", "woz", "smd", "frames", "msre2e", "taskmaster", "metalwoz", "schema"]', 
        type=str)
    parser.add_argument(
        '-hds','--holdout_dataset', 
        help='which dataset to be held out as dev and test set.',
        required=False, 
        default='["multiwoz"]', 
        type=str)
    parser.add_argument(
        '-task','--task', 
        help='task in ["nlu", "dst", "dm", "nlg", "e2e"] to decide which dataloader to use', 
        default="usdl",
        required=False)
    parser.add_argument(
        '--usr_token', 
        help='use to identify user utterance', 
        required=False, 
        default="[USR]", 
        type=str)
    parser.add_argument(
        '--sys_token', 
        help='use to identify system response', 
        required=False, 
        default="[SYS]", 
        type=str)
    parser.add_argument(
        "--add_rs_loss",
        action="store_true",
        help="whether to add RCL loss during training") 
    parser.add_argument(
        "--nb_addition_negresponse_per_sample",
        default=0,
        type=int,
        help="number of negative responses per sample added to the in-batch negative samples")
    parser.add_argument(
        "--negative_sampling_by_kmeans",
        action="store_true",
        help="whether use kmeans to select negative samples or select randomly",) 
    parser.add_argument(
        "--nb_kmeans",
        default=500,
        type=int,
        help="number of kmeans clusters")
    parser.add_argument(
        "--patience", 
        type=int, 
        default=15, 
        help="waiting to earlystop")
    
    ## data reading related setting (can be ignored here)
    parser.add_argument(
        '--max_line', help='maximum line for reading data (for quick testing)', required=False, default=None, type=int)
    parser.add_argument(
        '--example_type', help='type in ["turn", "dial"] for data reading',  required=False,  default="turn")
    parser.add_argument(
        "--train_data_ratio", default=1.0, type=float, help="")
    parser.add_argument(
        "--ratio_by_random", action="store_true", help="read data by random with a defined ratio")  
    parser.add_argument(
        "--domain_act", action="store_true", help="use domain_act for mwoz")  
    parser.add_argument(
        '-task_name', '--task_name', help='', required=False, default="") 
    parser.add_argument(
        "--only_last_turn", action="store_true", help="")  
    parser.add_argument(
        "--oracle_domain", action="store_true", help="") 
    parser.add_argument(
        "--ontology_version", default="", type=str, help="['', '1.0']")
    parser.add_argument(
        "--dstlm", action="store_true", help="") 
    parser.add_argument(
        "--max_seq_length", default=512, type=int, help="")
    parser.add_argument(
        "--nb_shots", default=-1, type=int, help="")

    args = parser.parse_args()
    args_dict = vars(args)

    if args.model_type in ["bert", "roberta", "distilbert", "camembert"] and not args.mlm:
        raise ValueError(
            "BERT and RoBERTa do not have LM heads but masked LM heads. They must be run using the --mlm "
            "flag (masked language modeling)."
        )
    #if args.eval_data_file is None and args.do_eval:
    #    raise ValueError(
    #        "Cannot do evaluation without an evaluation data file. Either supply a file to --eval_data_file "
    #        "or remove the --do_eval argument."
    #    )

    if (
        os.path.exists(args.output_dir)
        and os.listdir(args.output_dir)
        and args.do_train
        and not args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd

        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )

    # Set seed
    set_seed(args)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training download model & vocab

    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    #config.output_hidden_states = True
    
    tokenizer = tokenizer_class.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        do_lower_case=args.do_lower_case,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    
    if args.block_size <= 0:
        args.block_size = (
            tokenizer.max_len_single_sentence
        )  # Our input block size will be the max possible for the model
    args.block_size = min(args.block_size, tokenizer.max_len_single_sentence)
    model = model_class.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    model.to(args.device)
    
    # Add new tokens to the vocabulary and embeddings of our model
    tokenizer.add_tokens([args.sys_token, args.usr_token])
    model.resize_token_embeddings(len(tokenizer))

    if args.local_rank == 0:
        torch.distributed.barrier()  # End of barrier to make sure only the first process in distributed training download model & vocab

    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        
        # Barrier to make sure only the first process in distributed training process the dataset, 
        # and the others will use the cache
        if args.local_rank not in [-1, 0]:
            torch.distributed.barrier()  

        ## Read datasets and create global set of candidate responses
        datasets = {}
        cand_uttr_sys = set()
        for ds_name in ast.literal_eval(args.dataset):
            data_trn, data_dev, data_tst, data_meta = globals()["prepare_data_{}".format(ds_name)](args_dict)
            # held-out mwoz for now
            if ds_name in ast.literal_eval(args.holdout_dataset):
                datasets[ds_name] = {"train": data_trn, "dev":data_dev, "test": data_tst, "meta":data_meta}
            else:
                datasets[ds_name] = {"train": data_trn + data_dev + data_tst, "dev":[], "test": [], "meta":data_meta}

            for d in datasets[ds_name]["train"]:
                cand_uttr_sys.add(d["turn_sys"])
                cand_uttr_sys.update(set([sent for si, sent in enumerate(d["dialog_history"]) if si%2==0]))
        unified_meta = get_unified_meta(datasets)  

        ## process candidate responses
        if args.nb_addition_negresponse_per_sample > 0:
            cand_uttr_sys = list(cand_uttr_sys)
            cand_uttr_sys = [s.lower() for s in cand_uttr_sys if len(s.split(" ")) <= 100] # remove too long responses
            cand_uttr_sys_tokens = []
            for cand in tqdm(cand_uttr_sys):
                cand_ids = tokenizer.tokenize("[CLS] [SYS]") + tokenizer.tokenize(cand)
                cand_ids = torch.tensor(tokenizer.convert_tokens_to_ids(cand_ids))
                cand_uttr_sys_tokens.append(cand_ids)
            cand_uttr_sys_dict = {a:b for a, b in zip(cand_uttr_sys, cand_uttr_sys_tokens)}
        else:
            cand_uttr_sys_dict = {}
        
        ## batch size
        args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        args_dict["batch_size"] = args.train_batch_size
        args_dict["eval_batch_size"] = args.eval_batch_size

        ## Create Dataloader
        trn_loader = get_loader(args_dict, "train", tokenizer, datasets, unified_meta, "train")
        dev_loader = get_loader(args_dict, "dev"  , tokenizer, datasets, unified_meta, "dev")
        
        ## additional information for negative sampling
        others = {}
        if args.negative_sampling_by_kmeans:
            others["datasets"] = datasets
            others["unified_meta"] = unified_meta
        
        if args.local_rank == 0:
            torch.distributed.barrier()

        global_step, tr_loss = train(args, trn_loader, dev_loader, model, tokenizer, cand_uttr_sys_dict, others)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Evaluation
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        checkpoints = list(
            os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True))
        )
        logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            prefix = checkpoint.split("/")[-1] if checkpoint.find("checkpoint") != -1 else ""

            model = model_class.from_pretrained(checkpoint)
            model.to(args.device)
            result = evaluate(args, model, dev_loader, tokenizer, prefix=prefix)
            result = dict((k + "_{}".format(global_step), v) for k, v in result.items())
            results.update(result)
    
    print(results)


if __name__ == "__main__":
    main()
