#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 09:27:51 2023

@author: carpentier
"""

import argparse
from time import time
import model.config as config
import preprocessing.util as util

parser = argparse.ArgumentParser()
parser.add_argument("--wikiid2nnid_file", default="wikiid2nnid.txt")
parser.add_argument("--wiki_id_map_file", default="wiki_name_id_map.txt")
args = parser.parse_args()
top = time()

wikiid2nnid = util.load_wikiid2nnid(None, txt_file=args.wikiid2nnid_file)
_, wiki_id_name_map = util.load_wiki_name_id_map(filepath=args.wiki_id_map_file)

nb_tot_wikid = len(wikiid2nnid)
nb_tot_wikimap = len(wiki_id_name_map)
nb_comp_nnid = 0
nb_comp_map = 0

for idmap in wiki_id_name_map:
    if idmap in wikiid2nnid: nb_comp_map += 1
for nnid in wikiid2nnid:
    if nnid in wiki_id_name_map: nb_comp_nnid += 1
    
print("compatible nnid :\t{}/{} ({:.2f}%)".format(nb_comp_nnid, nb_tot_wikid, 100*(nb_comp_nnid/nb_tot_wikid)))
print("compatible map id :\t{}/{} ({:.2f}%)".format(nb_comp_map, nb_tot_wikimap, 100*(nb_comp_map/nb_tot_wikimap)))
print("DONE IN {}s".format(int(time()-top)))