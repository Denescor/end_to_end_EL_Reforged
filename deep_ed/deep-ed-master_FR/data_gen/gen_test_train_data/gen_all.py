#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  8 16:30:19 2021

@author: carpentier
"""

# -- Generates all training and test data for entity disambiguation.

import argparse
import data_gen.gen_test_train_data.gen_aida_test as gen_aida_test
import data_gen.gen_test_train_data.gen_aida_train as gen_aida_train
import data_gen.gen_test_train_data.gen_ace_msnbc_aquaint_csv as gen_ace_msnbc_aquaint_csv

parser = argparse.ArgumentParser()
parser.add_argument("--root_data_dir",defaut="./",help="Root Path of the data : $DATA_PATH")
#parser.add_argument("--redirect_file",defaut="wiki_redirects.txt")

args = parser.parse_args()

gen_aida_test.dofile(args)
gen_aida_train.dofile(args)
gen_ace_msnbc_aquaint_csv.dofile(args)
