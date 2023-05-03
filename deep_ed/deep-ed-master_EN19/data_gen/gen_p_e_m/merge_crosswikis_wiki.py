#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  8 09:41:08 2021

@author: carpentier
"""
# -- Merge Wikipedia and Crosswikis p(e|m) indexes
# -- Run: th data_gen/gen_p_e_m/merge_crosswikis_wiki.lua -root_data_dir $DATA_PATH

import torch
import argparse
import utils.utils as utils
import entities.ent_name2id_freq.ent_name_id as ent_name_id

merged_e_m_counts = dict()

parser = argparse.ArgumentParser()
parser.add_argument("--root_data_dir",defaut="./",help="Root Path of the data : $DATA_PATH")
#parser.add_argument("--redirect_file",defaut="wiki_redirects.txt")

args = parser.parse_args()
args.wiki_p_e_m_file = "wikipedia_p_e_m.txt"
args.crosswikis_p_e_m = "crosswikis_p_e_m.txt"
args.crosswikis_wikipedia_p_e_m = "crosswikis_wikipedia_p_e_m.txt"

def merge_file(file,merged_e_m_counts):
    with open(args.root_data_dir+"/"+file,"r") as f:
        for line in f:
            parts = line.split("\t")
            mention  = parts[0]
    
            if (not utils.findLUA_bool(mention,'Wikipedia')) and (not utils.findLUA_bool(mention,'wikipedia')) :
                if not merged_e_m_counts[mention] :
                    merged_e_m_counts[mention] = dict()
                    
                total_freq = int(parts[1])
                assert total_freq is int
                num_ents = len(parts)
                for i in range(2, num_ents):
                    ent_str = parts[i].split(",")
                    ent_wikiid = int(ent_str[1])
                    assert ent_wikiid is int
                    freq = int(ent_str[2])
                    assert freq is int
    
            if not merged_e_m_counts[mention][ent_wikiid] :
                merged_e_m_counts[mention][ent_wikiid] = 0
            merged_e_m_counts[mention][ent_wikiid] += freq       
    return merged_e_m_counts
       
print('Process Crosswikis')
merged_e_m_counts = merge_file(args.wiki_p_e_m_file, merged_e_m_counts)

print('Process Wikipedia')
merged_e_m_counts = merge_file(args.crosswikis_p_e_m, merged_e_m_counts)

print('Now sorting and writing ..')
with open(args.root_data_dir+"/"+args.crosswikis_wikipedia_p_e_m,"w") as f:
    for mention, liste in merged_e_m_counts.items() :
        if len(mention) >= 1 :
            tbl = []
            for ent_wikiid, freq in liste.items() :
                tbl.append(dict([("ent_wikiid",ent_wikiid), ("freq",freq)]))
            tbl.sort(key=lambda a: a.freq, reverse=True)
    
            string = ""
            total_freq = 0
            num_ents = 0
            for el in tbl :
                if ent_name_id.is_valid_ent(el["ent_wikiid"]) :
                    string = "{}{},{},{}\t".format(string,el["ent_wikiid"],el["freq"], ent_name_id.get_ent_name_from_wikiid(el["ent_wikiid"].replace(' ', '_')))
                    num_ents += num_ents
                    total_freq += el["freq"]
    
                    if num_ents >= 100 : #-- At most 100 candidates
                        break
    
            f.write("{}\t{}\t{}\n".format(mention,total_freq,string))    
    
    
print('Done sorting and writing.')