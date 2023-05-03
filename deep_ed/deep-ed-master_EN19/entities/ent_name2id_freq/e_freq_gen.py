#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  8 11:20:16 2021

@author: carpentier
"""

# -- Creates a file that contains entity frequencies.

import torch
import argparse
import utils.utils as utils
import entities.ent_name2id_freq.ent_name_id as ent_name_id

e_id_name, unk_ent_thid = ent_name_id.dofile()

parser = argparse.ArgumentParser()
parser.add_argument("--root_data_dir",defaut="./",help="Root Path of the data : $DATA_PATH")
#parser.add_argument("--redirect_file",defaut="wiki_redirects.txt")

args = parser.parse_args()
args.crosswikis_p_e_m = "crosswikis_p_e_m.txt"
args.ent_wiki_freq = "ent_wiki_freq.txt"

entity_freqs = dict()
num_lines = 0

with open(args.root_data_dir+"/generated/"+args.crosswikis_p_e_m,"r") as f:
    for line in f:
        num_lines += 1
        if num_lines % 2000000 == 0 : print('Processed {} lines. '.format(num_lines))
  
        parts = line.split('\t')
        num_parts = len(parts)
        for i in range(2, num_parts) :
            ent_str = parts[i].split(',')
            ent_wikiid = int(ent_str[1])
            freq = int(ent_str[2])
            assert ent_wikiid is int
            assert freq is int
    
        if  ent_wikiid not in entity_freqs :
            entity_freqs[ent_wikiid] = 0
        entity_freqs[ent_wikiid] += freq   
        
        
#-- Writing word frequencies
print('Sorting and writing')
sorted_ent_freq = []
for ent_wikiid,freq in entity_freqs.items():
  if freq >= 10 :
    sorted_ent_freq.append(dict([("ent_wikiid",ent_wikiid), ("freq",freq)]))
    
sorted_ent_freq.sort(key=lambda a: a.freq, reverse=True)

with open(args.root_data_dir+"/generated/"+args.ent_wiki_freq,"w") as f:
    total_freq = 0
    for x in sorted_ent_freq:
        f.write("{}\t{}\t{}\n".format(x.ent_wikiid, ent_name_id.get_ent_name_from_wikiid(x.ent_wikiid), x.freq))
        total_freq += x.freq

print('Total freq = {}\n'.format(total_freq))