#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 16:00:17 2023

@author: carpentier
"""

import os
import time
import argparse
from wiki2vec_txt_from_npy import unify_entity_name 

def make_list_TREntities(out_filepath, data_folders):
    mention_list = []
    for data_folder in data_folders:
        list_doc = [x for x in os.listdir(data_folder) if os.path.isfile(data_folder+"/"+x) and os.path.splitext(x)[1]==".mentions"]
        for doc in list_doc:
            mention_file = "{}/{}".format(data_folder,doc)
            mention_list_temp = extract_TR_entities(mention_file)
            mention_list.extend(mention_list_temp)
    if len(mention_list) > 0:
        with open(out_filepath, args.write_mode) as fout:
            for entity in mention_list:
                fout.write("{}\n".format(entity))
    return len(mention_list)
            
def extract_TR_entities(mention_filepath):
    with open(mention_filepath) as fin:
        mention_list = []
        for line in fin:
            bg, nd, ent_en, ent_fr, hard = line.split("\t")
            if args.entity_language == "fr": mention_list.append(unify_entity_name(ent_fr)) #marche car il n'y peut pas y avoir 2 mentions dans 1 documents qui débutent au même endroit
            else: mention_list.append(unify_entity_name(ent_en)) #marche car il n'y peut pas y avoir 2 mentions dans 1 documents qui débutent au même endroit
    return mention_list

def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_folder", help="folder with the wikipedia2vec data")
    parser.add_argument("--data_folder", help="folder with the dataset")
    parser.add_argument("--entity_language", default="fr", help="'fr' or 'en'")
    parser.add_argument("--append_mode", dest="write_mode", action="store_const", const='a', help="add new lines to reference files")
    parser.add_argument("--replace_mode", dest="write_mode", action="store_const", const='w', help="default mode : replace existing file")
    parser.set_defaults(write_mode='w')
    args = parser.parse_args()
    return args 

if __name__ == "__main__":
    args = _parse_args()
    walltime = time.time()
    data_type = os.path.basename(os.path.normpath(args.data_folder))
    entity_file = args.in_folder+"{}_list_entities.txt".format(args.entity_language)
    print("MAKE {}_list_entities.txt FROM {} ...".format(args.entity_language, data_type), end="", flush=True)
    
    nb_TR = make_list_TREntities(entity_file, [args.data_folder+x for x in ["train", "test", "dev"]])
    print(" {} ENTITIES IN {}s".format(nb_TR,int(time.time()-walltime)))