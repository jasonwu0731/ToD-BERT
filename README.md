# ToD-BERT: Pre-trained Natural Language Understanding for Task-Oriented Dialogues 

Authors: [Chien-Sheng Wu](https://jasonwu0731.github.io/), [Steven Hoi](http://mysmu.edu.sg/faculty/chhoi/), [Richard Socher](https://www.socher.org/) and [Caiming Xiong](http://cmxiong.com/).

Paper: https://arxiv.org/abs/2004.06871

## Introduction
The use of pre-trained language models has emerged as a promising direction for improving dialogue systems. However, the underlying difference of linguistic patterns between conversational data and general text makes the existing pre-trained language models not as effective as they have been shown to be. Recently, there are some pre-training approaches based on open-domain dialogues, leveraging large-scale social media data such as Twitter or Reddit. Pre-training for task-oriented dialogues, on the other hand, is rarely discussed because of the long-standing and crucial data scarcity problem. In this work, we combine nine English-based, human-human, multi-turn and publicly available task-oriented dialogue datasets to conduct language model and response selection pre-training. The experimental results show that our pre-trained task-oriented dialogue BERT (ToD-BERT) surpasses BERT and other strong baselines in four downstream task-oriented dialogue applications, including intention detection, dialogue state tracking, dialogue act prediction, and response selection. Moreover, in the simulated limited data experiments, we show that ToD-BERT has stronger few-shot capacity that can mitigate the data scarcity problem in task-oriented dialogues.

## Citation
If you use any source codes, pretrained models or datasets included in this repo in your work, please cite the following paper. The bibtex is listed below:
<pre>
@article{wu2020tod,
  title={ToD-BERT: Pre-trained Natural Language Understanding for Task-Oriented Dialogues},
  author={Wu, Chien-Sheng and Hoi, Steven and Socher, Richard and Xiong, Caiming},
  journal={arXiv preprint arXiv:2004.06871},
  year={2020}
}
</pre>

## Pretrained Models
Please downloaded the pre-trained models from the following links.
* [ToD-BERT-mlm](https://drive.google.com/file/d/1vxqTda4MIYb1VDIA4NOokq7uCM4MW_1J/view?usp=sharing)
* [ToD-BERT-jnt](https://drive.google.com/file/d/17F-wS4PwR6iz-Ubj0TaNsxNyMscgO3VV/view?usp=sharing)

## Usage
Please refer to the following guide how to use our pre-trained ToD-BERT models. Full training and evaluation code will be released soon.Our model is built on top of the [PyTorch](https://pytorch.org/) library and huggingface [Transformer](https://github.com/huggingface/transformers) library.
```console
❱❱❱ pip install transformers
```

Let's do a very quick overview of the model architecture and code. Detailed examples for model architecturecan be found in the paper.
```
import torch
from transformers import *

MODELS = { "bert": (BertModel,       BertTokenizer,       BertConfig)}
model_name_or_path = <path_to_the_downloaded_pretrained_models>

model_class, tokenizer_class, config_class = MODELS["bert"]
tokenizer = tokenizer_class.from_pretrained(model_name_or_path)
tod_bert = model_class.from_pretrained(model_name_or_path)

# Encode text (Add special tokens takes care of adding [CLS], [SEP], <s>... tokens in the right way for each model.)
input_text = "[CLS] [SYS] Hello, what can I help with you today? [USR] Find me a cheap restaurant nearby the north town."
input_tokens = self.tokenizer.tokenize(input_text)
story = torch.Tensor(self.tokenizer.convert_tokens_to_ids(input_tokens))

if torch.cuda.is_available(): 
    tod_bert = tod_bert.cuda()
    story = story.cuda()

with torch.no_grad():
    input_context = {"input_ids": story, "attention_mask": (story > 0).long()}
    hiddens = self.utterance_encoder(**input_context)[0] 
```


