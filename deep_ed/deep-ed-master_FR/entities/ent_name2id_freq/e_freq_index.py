#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  8 14:07:31 2021

@author: carpentier
"""

# Loads an index containing entity -> frequency pairs. 
# TODO: rewrite this file in a simpler way (is complicated because of some past experiments).

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--root_data_dir",defaut="./",help="Root Path of the data : $DATA_PATH")
#parser.add_argument("--redirect_file",defaut="wiki_redirects.txt")

args = parser.parse_args()

print('==> Loading entity freq map') 

def dofile(args):
    min_freq = 1
    e_freq = dict()
    e_freq["ent_f_start"] = dict()
    e_freq["ent_f_end"] = dict()
    e_freq["total_freq"] = 0
    e_freq["sorted"] = dict()
    
    cur_start = 1
    cnt = 0
    
    args.ent_freq_file = "ent_wiki_freq.txt"
    with open(args.root_data_dir+"/generated/"+args.ent_freq_file,"r") as f:
        for line in f:
            parts = line.split('\t')
            ent_wikiid = int(parts[0])
            ent_f = int(parts[1])
            assert ent_wikiid is int
            assert ent_f is int
            if (not rewtr) or rewtr.reltd_ents_wikiid_to_rltdid[ent_wikiid] : #TODO retrouver rewtr
                e_freq["ent_f_start"][ent_wikiid] = cur_start
                e_freq["ent_f_end"][ent_wikiid] = cur_start + ent_f - 1
                e_freq["sorted"][cnt] = ent_wikiid
                cur_start += ent_f
                cnt += 1
                
    e_freq["total_freq"] = cur_start - 1
    print('    Done loading entity freq index. Size = {}'.format(cnt))
    return e_freq


def get_ent_freq(ent_wikiid) :
    if e_freq["ent_f_start"][ent_wikiid] :
        return e_freq["ent_f_end"][ent_wikiid] - e_freq["ent_f_start"][ent_wikiid] + 1
    return 0