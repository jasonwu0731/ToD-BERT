from tqdm import tqdm
import torch.nn as nn
import logging
import ast
import glob
import numpy as np
import copy

# utils 
from utils.config import *
from utils.utils_general import *
from utils.utils_multiwoz import *
from utils.utils_oos_intent import *
from utils.utils_universal_act import *

# models
from models.multi_label_classifier import *
from models.multi_class_classifier import *
from models.BERT_DST_Picklist import *
from models.dual_encoder_ranking import *

# hugging face models
from transformers import *

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter
    
## model selection
MODELS = {"bert": (BertModel,       BertTokenizer,       BertConfig),
          "todbert": (BertModel,       BertTokenizer,       BertConfig),
          "gpt2": (GPT2Model,       GPT2Tokenizer,       GPT2Config),
          "todgpt2": (GPT2Model,       GPT2Tokenizer,       GPT2Config),
          "dialogpt": (AutoModelWithLMHead, AutoTokenizer, GPT2Config),
          "albert": (AlbertModel, AlbertTokenizer, AlbertConfig),
          "roberta": (RobertaModel, RobertaTokenizer, RobertaConfig),
          "distilbert": (DistilBertModel, DistilBertTokenizer, DistilBertConfig),
          "electra": (ElectraModel, ElectraTokenizer, ElectraConfig)}

## Fix torch random seed
if args["fix_rand_seed"]: 
    torch.manual_seed(args["rand_seed"])

    
## Reading data and create data loaders
datasets = {}
for ds_name in ast.literal_eval(args["dataset"]):
    data_trn, data_dev, data_tst, data_meta = globals()["prepare_data_{}".format(ds_name)](args)
    datasets[ds_name] = {"train": data_trn, "dev":data_dev, "test": data_tst, "meta":data_meta}
unified_meta = get_unified_meta(datasets)  
if "resp_cand_trn" not in unified_meta.keys(): unified_meta["resp_cand_trn"] = {}
args["unified_meta"] = unified_meta


## Create vocab and model class
args["model_type"] = args["model_type"].lower()
model_class, tokenizer_class, config_class = MODELS[args["model_type"]]
tokenizer = tokenizer_class.from_pretrained(args["model_name_or_path"], cache_dir=args["cache_dir"])
args["model_class"] = model_class
args["tokenizer"] = tokenizer
if args["model_name_or_path"]:
    config = config_class.from_pretrained(args["model_name_or_path"], cache_dir=args["cache_dir"]) 
else:
    config = config_class()
args["config"] = config
args["num_labels"] = unified_meta["num_labels"]

    
## Training and Testing Loop
if args["do_train"]:
    result_runs = []
    output_dir_origin = str(args["output_dir"])
    
    ## Setup logger
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                        datefmt='%m-%d %H:%M',
                        filename=os.path.join(args["output_dir"], "train.log"),
                        filemode='w')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

    ## training loop
    for run in range(args["nb_runs"]):
         
        ## Setup random seed and output dir
        rand_seed = SEEDS[run]
        if args["fix_rand_seed"]: 
            torch.manual_seed(rand_seed)
            args["rand_seed"] = rand_seed
        args["output_dir"] = os.path.join(output_dir_origin, "run{}".format(run)) 
        os.makedirs(args["output_dir"], exist_ok=False)
        logging.info("Running Random Seed: {}".format(rand_seed))
        
        ## Loading model
        model = globals()[args['my_model']](args)
        if torch.cuda.is_available(): model = model.cuda()
        
        ## Create Dataloader
        trn_loader = get_loader(args, "train", tokenizer, datasets, unified_meta)
        dev_loader = get_loader(args, "dev"  , tokenizer, datasets, unified_meta, shuffle=args["task_name"]=="rs")
        tst_loader = get_loader(args, "test" , tokenizer, datasets, unified_meta, shuffle=args["task_name"]=="rs")
        
        ## Create TF Writer
        tb_writer = SummaryWriter(comment=args["output_dir"].replace("/", "-"))

        # Start training process with early stopping
        loss_best, acc_best, cnt, train_step = 1e10, -1, 0, 0
        
        try:
            for epoch in range(args["epoch"]):
                logging.info("Epoch:{}".format(epoch+1)) 
                train_loss = 0
                pbar = tqdm(trn_loader)
                for i, d in enumerate(pbar):
                    model.train()
                    outputs = model(d)
                    train_loss += outputs["loss"]
                    train_step += 1
                    pbar.set_description("Training Loss: {:.4f}".format(train_loss/(i+1)))

                    ## Dev Evaluation
                    if (train_step % args["eval_by_step"] == 0 and args["eval_by_step"] != -1) or \
                                                  (i == len(pbar)-1 and args["eval_by_step"] == -1):
                        model.eval()
                        dev_loss = 0
                        preds, labels = [], []
                        ppbar = tqdm(dev_loader)
                        for d in ppbar:
                            with torch.no_grad():
                                outputs = model(d)
                            #print(outputs)
                            dev_loss += outputs["loss"]
                            preds += [item for item in outputs["pred"]]
                            labels += [item for item in outputs["label"]] 

                        dev_loss = dev_loss / len(dev_loader)
                        results = model.evaluation(preds, labels)
                        dev_acc = results[args["earlystop"]] if args["earlystop"] != "loss" else dev_loss

                        ## write to tensorboard
                        tb_writer.add_scalar("train_loss", train_loss/(i+1), train_step)
                        tb_writer.add_scalar("eval_loss", dev_loss, train_step)
                        tb_writer.add_scalar("eval_{}".format(args["earlystop"]), dev_acc, train_step)

                        if (dev_loss < loss_best and args["earlystop"] == "loss") or \
                            (dev_acc > acc_best and args["earlystop"] != "loss"):
                            loss_best = dev_loss
                            acc_best = dev_acc
                            cnt = 0 # reset
                            
                            if args["not_save_model"]:
                                model_clone = globals()[args['my_model']](args)
                                model_clone.load_state_dict(copy.deepcopy(model.state_dict()))
                            else:
                                output_model_file = os.path.join(args["output_dir"], "pytorch_model.bin")
                                if args["n_gpu"] == 1:
                                    torch.save(model.state_dict(), output_model_file)
                                else:
                                    torch.save(model.module.state_dict(), output_model_file)
                                logging.info("[Info] Model saved at epoch {} step {}".format(epoch, train_step))
                        else:
                            cnt += 1
                            logging.info("[Info] Early stop count: {}/{}...".format(cnt, args["patience"]))

                        if cnt > args["patience"]: 
                            logging.info("Ran out of patient, early stop...")  
                            break

                        logging.info("Trn loss {:.4f}, Dev loss {:.4f}, Dev {} {:.4f}".format(train_loss/(i+1), 
                                                                                              dev_loss,
                                                                                              args["earlystop"],
                                                                                              dev_acc))

                if cnt > args["patience"]: 
                    tb_writer.close()
                    break 
                    
        except KeyboardInterrupt:
            logging.info("[Warning] Earlystop by KeyboardInterrupt")
        
        ## Load the best model
        if args["not_save_model"]:
            model.load_state_dict(copy.deepcopy(model_clone.state_dict()))
        else:
            # Start evaluating on the test set
            if torch.cuda.is_available(): 
                model.load_state_dict(torch.load(output_model_file))
            else:
                model.load_state_dict(torch.load(output_model_file, lambda storage, loc: storage))
        
        ## Run test set evaluation
        pbar = tqdm(tst_loader)
        for nb_eval in range(args["nb_evals"]):
            test_loss = 0
            preds, labels = [], []
            for d in pbar:
                with torch.no_grad():
                    outputs = model(d)
                test_loss += outputs["loss"]
                preds += [item for item in outputs["pred"]]
                labels += [item for item in outputs["label"]] 

            test_loss = test_loss / len(tst_loader)
            results = model.evaluation(preds, labels)
            result_runs.append(results)
            logging.info("[{}] Test Results: ".format(nb_eval) + str(results))
    
    ## Average results over runs
    if args["nb_runs"] > 1:
        f_out = open(os.path.join(output_dir_origin, "eval_results_multi-runs.txt"), "w")
        f_out.write("Average over {} runs and {} evals \n".format(args["nb_runs"], args["nb_evals"]))
        for key in results.keys():
            mean = np.mean([r[key] for r in result_runs])
            std  = np.std([r[key] for r in result_runs])
            f_out.write("{}: mean {} std {} \n".format(key, mean, std))
        f_out.close()

else:
    
    ## Load Model
    print("[Info] Loading model from {}".format(args['my_model']))
    model = globals()[args['my_model']](args)    
    if args["load_path"]:
        print("MODEL {} LOADED".format(args["load_path"]))
        if torch.cuda.is_available(): 
            model.load_state_dict(torch.load(args["load_path"]))
        else:
            model.load_state_dict(torch.load(args["load_path"], lambda storage, loc: storage))
    else:
        print("[WARNING] No trained model is loaded...")
    
    if torch.cuda.is_available():
        model = model.cuda()
    
    print("[Info] Start Evaluation on dev and test set...")
    dev_loader = get_loader(args, "dev"  , tokenizer, datasets, unified_meta)
    tst_loader = get_loader(args, "test" , tokenizer, datasets, unified_meta, shuffle=args["task_name"]=="rs")
    model.eval()
    
    for d_eval in ["tst"]: #["dev", "tst"]:
        f_w = open(os.path.join(args["output_dir"], "{}_results.txt".format(d_eval)), "w")

        ## Start evaluating on the test set
        test_loss = 0
        preds, labels = [], []
        pbar = tqdm(locals()["{}_loader".format(d_eval)])
        for d in pbar:
            with torch.no_grad():
                outputs = model(d)
            test_loss += outputs["loss"]
            preds += [item for item in outputs["pred"]]
            labels += [item for item in outputs["label"]] 

        test_loss = test_loss / len(tst_loader)
        results = model.evaluation(preds, labels)
        print("{} Results: {}".format(d_eval, str(results)))
        f_w.write(str(results))
        f_w.close()
