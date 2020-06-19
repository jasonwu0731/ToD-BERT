import os.path
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from torch.nn import CrossEntropyLoss
from torch.nn import CosineEmbeddingLoss
from sklearn.metrics import f1_score #, average_precision_score
import numpy as np


from transformers import *


class multi_label_classifier(nn.Module):
    def __init__(self, args): #, num_labels, device):
        super(multi_label_classifier, self).__init__()
        self.args = args
        self.hidden_dim = args["hdd_size"]
        self.rnn_num_layers = args["num_rnn_layers"]
        self.num_labels = args["num_labels"]
        self.bce = nn.BCELoss()
        self.sigmoid = nn.Sigmoid()
        self.n_gpu = args["n_gpu"]

        ### Utterance Encoder
        self.utterance_encoder = args["model_class"].from_pretrained(self.args["model_name_or_path"])

        self.bert_output_dim = args["config"].hidden_size
        #self.hidden_dropout_prob = self.utterance_encoder.config.hidden_dropout_prob
        
        if self.args["fix_encoder"]:
            print("[Info] fix_encoder")
            for p in self.utterance_encoder.parameters():
                p.requires_grad = False
        
        if self.args["more_linear_mapping"]:
            self.one_more_layer = nn.Linear(self.bert_output_dim, self.bert_output_dim)
        
        self.classifier = nn.Linear(self.bert_output_dim, self.num_labels)
        print("self.classifier", self.bert_output_dim, self.num_labels)

        ## Prepare Optimizer
        def get_optimizer_grouped_parameters(model):
            param_optimizer = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01,
                 'lr': args["learning_rate"]},
                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
                 'lr': args["learning_rate"]},
            ]
            return optimizer_grouped_parameters

        if self.n_gpu == 1:
            optimizer_grouped_parameters = get_optimizer_grouped_parameters(self)
        else:
            optimizer_grouped_parameters = get_optimizer_grouped_parameters(self.module)

        
        self.optimizer = AdamW(optimizer_grouped_parameters,
                                 lr=args["learning_rate"],)

    def optimize(self):
        self.loss_grad.backward()
        clip_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), self.args["grad_clip"])
        self.optimizer.step()
    
    def forward(self, data):
                #input_ids, input_len, labels=None, n_gpu=1, target_slot=None):
       
        self.optimizer.zero_grad()
        
        inputs = {"input_ids": data[self.args["input_name"]], "attention_mask":(data[self.args["input_name"]] > 0).long()}
        
        if "gpt2" in self.args["model_type"]:
            hidden = self.utterance_encoder(**inputs)[0]
            hidden_head = hidden.mean(1)
        elif self.args["model_type"] == "dialogpt":
            transformer_outputs = self.utterance_encoder.transformer(
                inputs["input_ids"],
                attention_mask=(inputs["input_ids"] > 0).long())[0]
            hidden_head = transformer_outputs.mean(1)
        else:
            hidden = self.utterance_encoder(**inputs)[0]
            hidden_head = hidden[:, 0, :]
        
        # loss
        if self.args["more_linear_mapping"]:
            hidden_head = self.one_more_layer(hidden_head)
        
        logits = self.classifier(hidden_head)
        prob = self.sigmoid(logits)
        loss = self.bce(prob, data[self.args["task_name"]])

        if self.training: 
            self.loss_grad = loss
            self.optimize()
        
        predictions = (prob > 0.5)
        
        outputs = {"loss":loss.item(), 
                   "pred":predictions.detach().cpu().numpy(), 
                   "label":data[self.args["task_name"]].detach().cpu().numpy()} 

        return outputs
    
    def evaluation(self, preds, labels):
        preds = np.array(preds)
        labels = np.array(labels)
        results = {}
        for avg_name in ['micro', 'macro', 'weighted', 'samples']:
            my_f1_score = f1_score(y_true=labels, y_pred=preds, average=avg_name)
            results["f1_{}".format(avg_name)] = my_f1_score

        return results

