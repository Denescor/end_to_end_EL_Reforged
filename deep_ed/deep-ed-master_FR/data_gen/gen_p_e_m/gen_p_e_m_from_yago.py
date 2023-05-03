#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  8 10:35:56 2021

@author: carpentier
"""

# -- Generate p(e|m) index from Wikipedia
# -- Run: th data_gen/gen_p_e_m/gen_p_e_m_from_yago.lua -root_data_dir $DATA_PATH

import torch
import argparse
import utils.utils as utils
import data_gen.gen_p_e_m.unicode_map as unicode_map
import data_gen.indexes.wiki_redirects_index as wiki_redirects_index
import entities.ent_name2id_freq.ent_name_id as ent_name_id

parser = argparse.ArgumentParser()
parser.add_argument("--root_data_dir",defaut="./",help="Root Path of the data : $DATA_PATH")
#parser.add_argument("--redirect_file",defaut="wiki_redirects.txt")

args = parser.parse_args()
args.aida_means = "aida_means.tsv"

get_redirected_ent_title = wiki_redirects_index.dofile()
e_id_name, unk_ent_thid = ent_name_id.dofile(args)
unk_ent_wikiid = -1

print('Computing YAGO p_e_m')

num_lines = 0
wiki_e_m_counts = dict()

with open(args.root_data_dir+"/basic_data/p_e_m_data/"+args.aida_means,"r") as f:
    for line in f:
        num_lines = num_lines + 1
        if num_lines % 5000000 == 0 : print('Processed {} lines.'.format(num_lines))
        parts = line.split('\t')
        assert len(parts) == 2
        assert parts[0][0] == '"'
        assert parts[0][-1] == '"'
  
        mention = parts[0][1:-1] #la mention sans les '"'
        ent_name = parts[1]
        ent_name = ent_name.replace('&amp;', '&')
        ent_name = ent_name.replace('&quot;', '"')
        x = 0
        while utils.findLUA_bool(ent_name,'\\u',start=x) :
            x,_ = utils.findLUA(ent_name,'\\u',start=x)
            code = ent_name[x:x + 5]
            assert code in unicode_map.unicode2ascii
            replace = unicode_map.unicode2ascii[code]
            if(replace == "%") :
                replace = "%%"
            ent_name = ent_name.replace(code, replace)

        ent_name = ent_name_id.preprocess_ent_name(ent_name)
        ent_wikiid = ent_name_id.get_ent_wikiid_from_name(ent_name, True)
        if ent_wikiid != unk_ent_wikiid :
            if not wiki_e_m_counts[mention] :
                wiki_e_m_counts[mention] = dict()
            wiki_e_m_counts[mention][ent_wikiid] = 1

print('Now sorting and writing ..')

with open(args.root_data_dir+"/generated/yago_p_e_m.txt") as f:
    for mention, liste in wiki_e_m_counts.items():
        string = ""
        total_freq = 0
        for ent_wikiid in liste.keys() :
            string = "{}{},{}\t".format(string, ent_wikiid, ent_name_id.get_ent_name_from_wikiid(ent_wikiid.replace(' ', '_')))
            total_freq += 1
        f.write("{}\t{}\t{}\n".format(mention, total_freq, string))
  
print('Done sorting and writing.')