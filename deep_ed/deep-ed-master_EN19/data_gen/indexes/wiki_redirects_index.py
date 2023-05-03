#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 14:49:16 2021

@author: carpentier
"""

import torch
import argparse
import utils.utils as utils

# Loads the link redirect index from Wikipedia

def get_redirected_ent_title(ent_name):
    if wiki_redirects_index[ent_name] :
        return wiki_redirects_index[ent_name]
    else:
        return ent_name

def dofile():    
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_data_dir",defaut="./",help="Root Path of the data : $DATA_PATH")
    #parser.add_argument("--redirect_file",defaut="wiki_redirects.txt")
    
    args = parser.parse_args()
    args.redirect_file = "wiki_redirects.txt"
    
    print('==> Loading redirects index')
    
    wiki_redirects_index = dict()
    
    
    with open(args.root_data_dir+"'basic_data/"+args.redirect_file,"r") as f:
        for line in f:
            parts = line.split("\t")
            wiki_redirects_index[parts[0]] = parts[1]
            
    assert wiki_redirects_index['Coercive'] == 'Coercion'
    assert wiki_redirects_index['Hosford, FL'] == 'Hosford, Florida'
    
    print('Done loading redirects index')
    return wiki_redirects_index