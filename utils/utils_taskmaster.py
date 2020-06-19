import json
import ast
import collections
import os

from .utils_function import get_input_example


def read_langs_turn(args, dials, ds_name, max_line):
    print(("Reading from {} for read_langs_turn".format(ds_name)))
    
    data = []
    turn_sys = ""
    turn_usr = ""
    
    cnt_lin = 1
    for dial in dials:
        dialog_history = []
        for ti, turn in enumerate(dial["utterances"]):
            if turn["speaker"] == "USER":
                turn_usr = turn["text"].lower().strip()
                
                data_detail = get_input_example("turn")
                data_detail["ID"] = "{}-{}".format(ds_name, cnt_lin)
                data_detail["turn_id"] = ti % 2
                data_detail["turn_usr"] = turn_usr
                data_detail["turn_sys"] = turn_sys
                data_detail["dialog_history"] = list(dialog_history)
                
                if (not args["only_last_turn"]):
                    data.append(data_detail)
                
                dialog_history.append(turn_sys)
                dialog_history.append(turn_usr)
            elif turn["speaker"] == "ASSISTANT":
                turn_sys = turn["text"].lower().strip()
            else:
                turn_usr += " {}".format(turn["text"])
        
        if args["only_last_turn"]:
            data.append(data_detail)
        
        cnt_lin += 1
        if(max_line and cnt_lin >= max_line):
            break

    return data


def read_langs_dial(file_name, ontology, dialog_act, max_line = None, domain_act_flag=False):
    print(("Reading from {} for read_langs_dial".format(file_name)))
    
    raise NotImplementedError



def prepare_data_taskmaster(args):
    ds_name = "TaskMaster"
    
    example_type = args["example_type"]
    max_line = args["max_line"]
    
    fr_trn_id = open(os.path.join(args["data_path"], 'Taskmaster/TM-1-2019/train-dev-test/train.csv'), 'r')
    fr_dev_id = open(os.path.join(args["data_path"], 'Taskmaster/TM-1-2019/train-dev-test/dev.csv'), 'r')
    fr_trn_id = fr_trn_id.readlines()
    fr_dev_id = fr_dev_id.readlines()
    fr_trn_id = [_id.replace("\n", "").replace(",", "") for _id in fr_trn_id]
    fr_dev_id = [_id.replace("\n", "").replace(",", "") for _id in fr_dev_id]

    fr_data_woz = open(os.path.join(args["data_path"], 'Taskmaster/TM-1-2019/woz-dialogs.json'), 'r')
    fr_data_self = open(os.path.join(args["data_path"], 'Taskmaster/TM-1-2019/self-dialogs.json'), 'r')
    dials_all = json.load(fr_data_woz) + json.load(fr_data_self)

    _example_type = "dial" if "dial" in example_type else example_type
    pair_trn = globals()["read_langs_{}".format(_example_type)](args, dials_all, ds_name, max_line)
    pair_dev = []
    pair_tst = []
    
    print("Read {} pairs train from {}".format(len(pair_trn), ds_name))
    print("Read {} pairs valid from {}".format(len(pair_dev), ds_name))
    print("Read {} pairs test  from {}".format(len(pair_tst), ds_name))  
    
    meta_data = {"num_labels":0}

    return pair_trn, pair_dev, pair_tst, meta_data

