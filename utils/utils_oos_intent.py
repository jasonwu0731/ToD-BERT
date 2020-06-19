import json
import ast
import os
import random
from .utils_function import get_input_example


def read_langs(args, dtype, _data, _oos_data):
    print(("Reading [OOS Intent] for read_langs {}".format(dtype)))
    
    data = []
    intent_counter = {}
    
    for cur_data in [_data, _oos_data]:
        for d in cur_data:
            sentence, label = d[0], d[1]

            data_detail = get_input_example("turn")
            data_detail["ID"] = "OOS-INTENT-{}-{}".format(dtype, len(data))
            data_detail["turn_usr"] = sentence
            data_detail["intent"] = label
            data.append(data_detail)

            # count number of each label
            if label not in intent_counter.keys():
                intent_counter[label] = 0
            intent_counter[label] += 1

    #print("len of OOS Intent counter: ", len(intent_counter))
    
    return data, intent_counter


def prepare_data_oos_intent(args):
    example_type = args["example_type"]
    max_line = args["max_line"]
    
    file_input = os.path.join(args["data_path"], 'oos-intent/data/data_full.json')
    data = json.load(open(file_input, "r"))

    pair_trn, intent_counter_trn = read_langs(args, "trn", data["train"], data["oos_train"])
    pair_dev, intent_counter_dev = read_langs(args, "dev", data["val"], data["oos_val"])
    pair_tst, intent_counter_tst = read_langs(args, "tst", data["test"], data["oos_test"])

    print("Read %s pairs train from OOS Intent" % len(pair_trn))
    print("Read %s pairs valid from OOS Intent" % len(pair_dev))
    print("Read %s pairs test  from OOS Intent" % len(pair_tst)) 
    
    intent_class = list(intent_counter_trn.keys())
    
    meta_data = {"intent":intent_class, "num_labels":len(intent_class)}
    print("len(intent_class)", len(intent_class))

    return pair_trn, pair_dev, pair_tst, meta_data

