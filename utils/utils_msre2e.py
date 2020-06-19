import json
import ast
import collections
import os

from .utils_function import get_input_example


def read_langs_turn(args, file_name, max_line = None, ds_name=""):
    print(("Reading from {} for read_langs_turn".format(file_name)))
    
    data = []
    
    with open(file_name) as f:
        dials = f.readlines()
        
        cnt_lin = 1
        dialog_history = []
        turn_usr = ""
        turn_sys = ""
        turn_idx = 0
        
        for dial in dials[1:]:
            dial_split = dial.split("\t")
            session_ID, Message_ID, Message_from, Message = dial_split[0], dial_split[1], dial_split[3], dial_split[4]
            
            if Message_ID == "1" and turn_sys != "":
                
                if args["only_last_turn"]:
                    data.append(data_detail)
                
                turn_usr = ""
                turn_sys = ""
                dialog_history = []
                cnt_lin += 1
                turn_idx = 0

            if Message_from == "user":
                turn_usr = Message.lower().strip()
                data_detail = get_input_example("turn")
                data_detail["ID"] = "{}-{}".format(ds_name, cnt_lin)
                data_detail["turn_id"] = turn_idx
                data_detail["turn_usr"] = turn_usr
                data_detail["turn_sys"] = turn_sys
                data_detail["dialog_history"] = list(dialog_history)
                
                if not args["only_last_turn"]:
                    data.append(data_detail)
                
                dialog_history.append(turn_sys)
                dialog_history.append(turn_usr)
                turn_idx += 1
            elif Message_from == "agent":
                turn_sys = Message.lower().strip()
            
            if(max_line and cnt_lin >= max_line):
                break

    return data


def read_langs_dial(file_name, ontology, dialog_act, max_line = None, domain_act_flag=False):
    print(("Reading from {} for read_langs_dial".format(file_name)))
    
    raise NotImplementedError



def prepare_data_msre2e(args):
    ds_name = "MSR-E2E"
    
    example_type = args["example_type"]
    max_line = args["max_line"]
    
    file_mov = os.path.join(args["data_path"], 'e2e_dialog_challenge/data/movie_all.tsv')
    file_rst = os.path.join(args["data_path"], 'e2e_dialog_challenge/data/restaurant_all.tsv')
    file_tax = os.path.join(args["data_path"], 'e2e_dialog_challenge/data/taxi_all.tsv')

    _example_type = "dial" if "dial" in example_type else example_type
    pair_mov = globals()["read_langs_{}".format(_example_type)](args, file_mov, max_line, ds_name+"-mov")
    pair_rst = globals()["read_langs_{}".format(_example_type)](args, file_rst, max_line, ds_name+"-rst")
    pair_tax = globals()["read_langs_{}".format(_example_type)](args, file_tax, max_line, ds_name+"-tax")
    
    pair_trn = pair_mov + pair_rst + pair_tax
    pair_dev = []
    pair_tst = []
    
    print("Read {} pairs train from {}".format(len(pair_trn), ds_name))
    print("Read {} pairs valid from {}".format(len(pair_dev), ds_name))
    print("Read {} pairs test  from {}".format(len(pair_tst), ds_name))  
    
    meta_data = {"num_labels":0}

    return pair_trn, pair_dev, pair_tst, meta_data

