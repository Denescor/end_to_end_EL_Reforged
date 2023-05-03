#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 09:23:43 2022

@author: carpentier
"""
import os

import model.config as config
import model.evaluate as evaluate
import model.fun_eval as fun_eval
import model.train as train

from model.usemodel import yield_documents#, dump_documents


if __name__ == "__main__":
    args, train_args = evaluate._process_args(evaluate._parse_args())
    print(args)
    if not os.path.exists(args.output_folder): os.makedirs(args.output_folder)
    train_args.checkpoint_model_num = args.checkpoint_model_num
    train_args.entity_extension = args.entity_extension
    if train_args.context_bert_lstm is None:  train_args.context_bert_lstm = False
    train.args = train_args
    #args.context_bert_lstm = False
    args.batch_size = train_args.batch_size
    args.output_folder = train_args.output_folder
    args.eval_cnt = None
    el_datasets, el_names, model = fun_eval.retrieve_model(train.args, args, mode="train")
    with model.sess as sess:
        iterators, handles = fun_eval.ed_el_dataset_handles(sess, el_datasets)
        print("YIELD DOCUMENT")
        yield_documents(args, iterators, handles, el_names, model)