#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  8 14:50:29 2021

@author: carpentier
"""

import torch
import argparse
import utils.utils as utils
import data_gen.parse_wiki_dump.parse_wiki_dump_tools as parse_wiki_dump_tools
import entities.ent_name2id_freq.e_freq_index as e_freq_index
import entities.ent_name2id_freq.ent_name_id as ent_name_id

print('\nExtracting text only from Wiki dump. Output is wiki_canonical_words.txt containing on each line an Wiki entity with the list of all words in its canonical Wiki page.')

parser = argparse.ArgumentParser()
parser.add_argument("--root_data_dir",defaut="./",help="Root Path of the data : $DATA_PATH")
#parser.add_argument("--redirect_file",defaut="wiki_redirects.txt")

args = parser.parse_args()
args.anchors_file = "textWithAnchorsFromAllWikipedia2014Feb.txt"
args.wiki_canonical_words = "wiki_canonical_words.txt"

e_id_name, unk_ent_thid = ent_name_id.dofile()
e_freq = e_freq_index.dofile(args)

num_lines = 0
num_valid_ents = 0
num_error_ents = 0 #-- Probably list or disambiguation pages.
unk_ent_wikiid = -1

empty_valid_ents = ent_name_id.get_map_all_valid_ents()

cur_words = ""
cur_ent_wikiid = -1

with open(args.root_data_dir+"/basic_data/"+args.anchors_file,"r") as f:
    with open(args.root_data_dir+"/generated/"+args.wiki_canonical_words,"w") as ouf:
        for line in f:
            num_lines += 1
            if num_lines % 5000000 == 0 : print('Processed {} lines. Num valid ents = {}. Num errs = {}'.format(num_lines,num_valid_ents,num_error_ents))

            if (not utils.findLUA_bool(line,'<doc id="')) and (not utils.findLUA_bool(line,'</doc>')) :
                _, text, _, _ , _ , _ = parse_wiki_dump_tools.extract_text_and_hyp(line, False)
                words = utils.split_in_words(text)
                cur_words = "{}{} ".format(cur_words," ".join(words))
    
            elif utils.findLUA_bool(line,'<doc id="') :
                if (cur_ent_wikiid > 0 and cur_words != '') :
                    if cur_ent_wikiid != unk_ent_wikiid and ent_name_id.is_valid_ent(cur_ent_wikiid) :
                        ouf.write("{}\t{}\t{}\n".format(cur_ent_wikiid,ent_name_id.get_ent_name_from_wikiid(cur_ent_wikiid),cur_words ))
                        empty_valid_ents[cur_ent_wikiid] = None
                        num_valid_ents += 1
                    else:
                        num_error_ents += 1
    
                cur_ent_wikiid = parse_wiki_dump_tools.extract_page_entity_title(line)
                cur_words = ''           