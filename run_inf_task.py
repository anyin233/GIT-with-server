# Code Reference: (https://github.com/dolphin-zs/Doc2EDAG)

import argparse
import os
import torch.distributed as dist
import pprint
import json
import logging

from dee.utils import set_basic_log_config, strtobool, BERTChineseCharacterTokenizer
from dee.dee_task import DEETask, DEETaskSetting
from dee.dee_helper import aggregate_task_eval_info, print_total_eval_info, print_single_vs_multi_performance
from dee.event_type import *

set_basic_log_config()

BERT_MODEL = 'bert-base-chinese'
MODEL_SETTINGS = {
    "task_name": "try",
    "data_dir": "./Data/dev",
    "exp_dir": "./Exps",
    "save_cpt_flag": True,
    "skip_train": True,
    "eval_model_names": "GIT",
    "re_eval_flag": False,
    "eval_epoch": 26,
    "cpt_file_name": "GIT",
    "eval_batch_size": 1
}

############# initial model ###############
task_dir = os.path.join(MODEL_SETTINGS['exp_dir'], MODEL_SETTINGS['task_name'])
if not os.path.exists(task_dir):
    logging.info("Cannot find task dir, please check your settings")
    
MODEL_SETTINGS["model_dir"] = os.path.join(task_dir, "Model")
MODEL_SETTINGS["output_dir"] = os.path.join(task_dir, "Output")

dee_setting = DEETaskSetting(
    **MODEL_SETTINGS
)
dee_setting.summary_dir_name = os.path.join(task_dir, "Summary")

# build task
dee_task = DEETask(dee_setting, load_train=False, load_dev=False, parallel_decorate=False, only_inference=True, inf_epoch=MODEL_SETTINGS['eval_epoch'])


def decode_info_process(decode_data):
    '''
    decode event information from model output
    model output like this
    [
        (
            id: int,
            event_type_onehot: [int or None : 5],
            event_type_fields: [
                [int] or None : 5
            ], // use bert to convert id to chinese token, then fill them into event fields
            DocSpaninfo,
            [int]
        )
    ]
    '''
    
    # initial tokenizer
    tokenizer = BERTChineseCharacterTokenizer.from_pretrained(BERT_MODEL)
    extracted_events = []
    for data in decode_data:
        doc_id = data[0]
        event_type_onehot = data[1]
        event_type_fields = data[2]
        event_path = data[4]
        event_list = []
        for index, (has_event, events) in enumerate(zip(event_type_onehot, event_type_fields)):
            if has_event == 0:
                event_list.append([])
                continue
            event_detail = event_type_fields_list[index]
            event_pertype_list = []
            # pair all event with field, and create event
            for event in events:
                event_dict = dict(zip(event_detail[1], convert_bert_id_to_token(event, tokenizer=tokenizer)))
                event_obj = event_type2event_class[event_detail[0]]()
                event_obj.update_by_dict(event_dict)
                if event_obj.is_good_candidate():
                    event_dict['event_type'] = event_detail[0]
                    event_pertype_list.append(event_dict)
            event_list.append(event_pertype_list)
                
        extracted_events.append({
            "id": doc_id,
            "event_type_list": event_type_onehot,
            "events": event_list,
            "event_path": event_path
        })
        
    return extracted_events
                


def convert_bert_id_to_token(id_list, tokenizer = None):
    if tokenizer is None:
        tokenizer = BERTChineseCharacterTokenizer.from_pretrained(BERT_MODEL)
    tokens = []
    for ids in id_list:
        if not ids is None:
            tokens.append("".join(tokenizer.convert_ids_to_tokens(ids)))
            
    return tokens 


def run_inf_task():
    dee_task.reload_data()
    
    res = dee_task.inf_only(MODEL_SETTINGS["eval_epoch"])
    res = decode_info_process(res)
    
    # ensure every processes exit at the same time
    if dist.is_initialized():
        print("initialized")
        dist.barrier()
    
    # return decoded and reformat resout
    return res
    
if __name__ == '__main__':
    run_inf_task() # debug enterence



