# TOD-BERT: Pre-trained Natural Language Understanding for Task-Oriented Dialogues 

Authors: [Chien-Sheng Wu](https://jasonwu0731.github.io/), [Steven Hoi](http://mysmu.edu.sg/faculty/chhoi/), [Richard Socher](https://www.socher.org/) and [Caiming Xiong](http://cmxiong.com/). 

EMNLP 2020. Paper: https://arxiv.org/abs/2004.06871


## Introduction
The underlying difference of linguistic patterns between general text and task-oriented dialogue makes existing pre-trained language models less useful in practice. In this work, we unify nine human-human and multi-turn task-oriented dialogue datasets for language modeling. To better model dialogue behavior during pre-training, we incorporate user and system tokens into the masked language modeling. We propose a contrastive objective function to simulate the response selection task. Our pre-trained task-oriented dialogue BERT (TOD-BERT) outperforms strong baselines like BERT on four downstream task-oriented dialogue applications, including intention recognition, dialogue state tracking, dialogue act prediction, and response selection. We also show that TOD-BERT has a stronger few-shot ability that can mitigate the data scarcity problem for task-oriented dialogue.


## Citation
If you use any source codes, pretrained models or datasets included in this repo in your work, please cite the following paper. The bibtex is listed below:
<pre>
@inproceedings{wu-etal-2020-tod,
    title = "{TOD}-{BERT}: Pre-trained Natural Language Understanding for Task-Oriented Dialogue",
    author = "Wu, Chien-Sheng  and
      Hoi, Steven C.H.  and
      Socher, Richard  and
      Xiong, Caiming",
    booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)",
    month = nov,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.emnlp-main.66",
    doi = "10.18653/v1/2020.emnlp-main.66",
    pages = "917--929"
}
</pre>


## Update
* (2020.10.01) More training and inference information added. Release TOD-DistilBERT.
* (2020.07.10) Loading model from [Huggingface](https://huggingface.co/) is now supported.
* (2020.04.26) Pre-trained models are available.


## Pretrained Models
You can easily load the pre-trained model using huggingface [Transformers](https://github.com/huggingface/transformers) library using the AutoModel function. Several pre-trained versions are supported:
* TODBERT/TOD-BERT-MLM-V1: TOD-BERT pre-trained only using the MLM objective
* TODBERT/TOD-BERT-JNT-V1: TOD-BERT pre-trained using both the MLM and RCL objectives
* TODBERT/TOD-DistilBERT-JNT-V1: TOD-DistilBERT pre-trained using both the MLM and RCL objectives
```
import torch
from transformers import *
tokenizer = AutoTokenizer.from_pretrained("TODBERT/TOD-BERT-JNT-V1")
tod_bert = AutoModel.from_pretrained("TODBERT/TOD-BERT-JNT-V1")
```

You can also downloaded the pre-trained models from the following links:
* [ToD-BERT-mlm V1](https://drive.google.com/file/d/1vxqTda4MIYb1VDIA4NOokq7uCM4MW_1J/view?usp=sharing)
* [ToD-BERT-jnt V1](https://drive.google.com/file/d/17F-wS4PwR6iz-Ubj0TaNsxNyMscgO3VV/view?usp=sharing)
```
model_name_or_path = <path_to_the_downloaded_tod-bert>
model_class, tokenizer_class, config_class = BertModel, BertTokenizer, BertConfig
tokenizer = tokenizer_class.from_pretrained(model_name_or_path)
tod_bert = model_class.from_pretrained(model_name_or_path)
```

## Direct Usage
Please refer to the following guide how to use our pre-trained ToD-BERT models. Our model is built on top of the [PyTorch](https://pytorch.org/) library and huggingface [Transformers](https://github.com/huggingface/transformers) library. Let's do a very quick overview of the model architecture and code. Detailed examples for model architecturecan be found in the paper.

```
# Encode text 
input_text = "[CLS] [SYS] Hello, what can I help with you today? [USR] Find me a cheap restaurant nearby the north town."
input_tokens = tokenizer.tokenize(input_text)
story = torch.Tensor(tokenizer.convert_tokens_to_ids(input_tokens)).long()

if len(story.size()) == 1: 
    story = story.unsqueeze(0) # batch size dimension

if torch.cuda.is_available(): 
    tod_bert = tod_bert.cuda()
    story = story.cuda()

with torch.no_grad():
    input_context = {"input_ids": story, "attention_mask": (story > 0).long()}
    hiddens = tod_bert(**input_context)[0] 
```

## Training and Testing
If you would like to train the model yourself, you can download those datasets yourself from each of their original papers or sources. You can also direct download a zip file [here](https://drive.google.com/file/d/1EnGX0UF4KW6rVBKMF3fL-9Q2ZyFKNOIy/view?usp=sharing).

The repository is currently in this structure:
```
.
└── image
    └── ...
└── models
    └── multi_class_classifier.py
    └── multi_label_classifier.py
    └── BERT_DST_Picklist.py
    └── dual_encoder_ranking.py
└── utils.py
    └── multiwoz
        └── ...
    └── metrics
        └── ...
    └── loss_function
        └── ...
    └── dataloader_nlu.py
    └── dataloader_dst.py
    └── dataloader_dm.py
    └── dataloader_nlg.py
    └── dataloader_usdl.py
    └── ...
└── README.md
└── evaluation_pipeline.sh
└── evaluation_ratio_pipeline.sh
└── run_tod_lm_pretraining.sh
└── main.py
└── my_tod_pretraining.py
```

* Run Pretraining
```console
❱❱❱ ./run_tod_lm_pretraining.sh 0 bert bert-base-uncased save/pretrain/ToD-BERT-MLM --only_last_turn
❱❱❱ ./run_tod_lm_pretraining.sh 0 bert bert-base-uncased save/pretrain/ToD-BERT-JNT --only_last_turn --add_rs_loss
```

* Run Fine-tuning
```console
❱❱❱ ./evaluation_pipeline.sh 0 bert bert-base-uncased save/BERT
```

* Run Fine-tuning (Few-Shot)
```console
❱❱❱ ./evaluation_ratio_pipeline.sh 0 bert bert-base-uncased save/BERT --nb_runs=3 
```

## Report
Feel free to create an issue or send email to the first author at cswu@salesforce.com

