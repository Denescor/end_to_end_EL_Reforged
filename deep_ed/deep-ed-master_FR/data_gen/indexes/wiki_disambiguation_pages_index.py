#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  8 16:10:59 2021

@author: carpentier
"""

# -- Loads the link disambiguation index from Wikipedia

import torch
import argparse
import utils.utils as utils

def dofile():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_data_dir",defaut="./",help="Root Path of the data : $DATA_PATH")
    #parser.add_argument("--redirect_file",defaut="wiki_redirects.txt")
    
    args = parser.parse_args()
    args.wiki_disambiguation_pages = "wiki_disambiguation_pages.txt"
    
    print('==> Loading disambiguation index')
    wiki_disambiguation_index = dict()
    
    with open(args.root_data_dir+"/basic_data/"+args.wiki_disambiguation_pages,"r") as f:
        for line in f:
            parts = line.split("\t")
            try: wiki_disambiguation_index[int(parts[0])] = 1
            except : print("not a number : {}".format(parts[0]))
            
        assert 579 in wiki_disambiguation_index
        assert 41535072 in wiki_disambiguation_index
        
    print('Done loading disambiguation index')