import json
import ast
import collections
import os

from .utils_function import get_input_example


def read_langs_turn(args, file_name, max_line = None):
    print(("Reading from {} for read_langs_turn".format(file_name)))
    
    data = []
    
    with open(file_name) as f:
        dials = json.load(f)
        
        cnt_lin = 1
        for dial_dict in dials:
            dialog_history = [""]

            # Reading data
            for ti, turn in enumerate(dial_dict["dial"]):
                assert ti == turn["turn"]
                turn_usr = turn["usr"]["transcript"].lower().strip()
                turn_sys = turn["sys"]["sent"].lower().strip()

                data_detail = get_input_example("turn")
                data_detail["ID"] = "camrest676-"+str(cnt_lin)
                data_detail["turn_id"] = turn["turn"]
                data_detail["turn_usr"] = turn_usr
                data_detail["turn_sys"] = turn_sys
                data_detail["dialog_history"] = list(dialog_history)
                
                if not args["only_last_turn"]:
                    data.append(data_detail)

                dialog_history.append(turn_usr)
                dialog_history.append(turn_sys)
            
            if args["only_last_turn"]:
                data.append(data_detail)
            
            cnt_lin += 1
            if(max_line and cnt_lin >= max_line):
                break

    return data


def read_langs_dial(file_name, ontology, dialog_act, max_line = None, domain_act_flag=False):
    print(("Reading from {} for read_langs_dial".format(file_name)))
    
    raise NotImplementedError



def prepare_data_camrest676(args):
    example_type = args["example_type"]
    max_line = args["max_line"]
    
    file_trn = os.path.join(args["data_path"], 'CamRest676/CamRest676.json')

    _example_type = "dial" if "dial" in example_type else example_type
    pair_trn = globals()["read_langs_{}".format(_example_type)](args, file_trn, max_line)
    pair_dev = [] 
    pair_tst = [] 

    print("Read %s pairs train from CamRest676" % len(pair_trn))

    meta_data = {"num_labels":0}

    return pair_trn, pair_dev, pair_tst, meta_data

