#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 15:43:20 2022

@author: carpentier
"""

import time
import tqdm
import argparse
import os
import preprocessing.util as util
from wiki2vec_txt_from_npy import unify_entity_name 

def process_TR_file(txt_filepath, mention_filepath, fout, entityNameIdMap):
    unknown_gt_ids = 0   # counter of ground truth entity ids that are not in the wiki_name_id.txt
    nb_mentions = 0
    with open(mention_filepath) as fin:
        mention_dict = dict()
        for line in fin:
            try:
                bg, nd, ent_en, ent_fr, hard = line.split("\t")
                if args.entity_language == "fr": 
                    if args.unify: final_ent = unify_entity_name(ent_fr) #marche car il n'y peut pas y avoir 2 mentions dans 1 documents qui débutent au même endroit
                    else: final_ent = ent_fr
                else: 
                    if args.unify: final_ent = unify_entity_name(ent_en)
                    else: final_ent = ent_en
                mention_dict[int(bg)] = (int(nd), final_ent)
                nb_mentions += 1
            except ValueError: print(">> ValueError at line '{}' of file '{}'".format(line,mention_filepath))
    with open(txt_filepath) as fin :
        text = ""
        current_end = -1
        mention = ""
        is_mention = False
        for line in fin:
            text += line
        for i, car in enumerate(text): #on va placer les entités au fur et à mesure en comptant par caractère    
            if i in mention_dict:
                current_mention = mention_dict[i]
                true_mention = current_mention[1]
                ent_id = entityNameIdMap.compatible_ent_id(name=true_mention)
                if ent_id is not None:
                    mention += car
                    is_mention = True
                    current_end = current_mention[0]
                else:
                    unknown_gt_ids += 1
                    #print("unknow gt ids : {} -> {} ({} - {})".format(current_mention[1],true_mention,i,current_mention[0]))
            elif i == current_end-1:
                if car == '\n': continue
                else: 
                    mention += car
                    is_mention = False
                    if args.unify: fout.add(unify_entity_name(mention))
                    else: fout.add(mention)
                    mention = ""
            elif is_mention :
                mention += car
            else: continue #do not copy '\n'
    return unknown_gt_ids, nb_mentions, fout

def process_TR(folder):
    # _, wiki_id_name_map = util.load_wiki_name_id_map(lowercase=False)
    #_, wiki_id_name_map = util.entity_name_id_map_from_dump()
    entityNameIdMap = util.EntityNameIdMap()
    entityNameIdMap.init_compatible_ent_id(wiki_map_file=args.wiki_path)
    print(list(entityNameIdMap.wiki_name_id_map.keys())[:15])
    unknown_gt_ids = 0   # counter of ground truth entity ids that are not in the wiki_name_id.txt
    nb_mention = 0
    os.chdir(folder)
    list_mention = set()
    list_doc = [os.path.splitext(x)[0] for x in os.listdir() if os.path.isfile(x) and os.path.splitext(x)[1]==".mentions"]
    for doc in tqdm.tqdm(list_doc, total=len(list_doc)):
        mention_file = "{}.mentions".format(doc)
        txt_file = "{}.txt".format(doc)
        unknown_gt_ids_temp, nb_mention_temp, list_mention = process_TR_file(txt_file,mention_file, list_mention, entityNameIdMap)
        unknown_gt_ids += unknown_gt_ids_temp
        nb_mention += nb_mention_temp
    if len(list_doc) > 0: print("process_TR\tunknown_gt_ids: {}/{} ({:.2f}%)".format(unknown_gt_ids,nb_mention,100*(unknown_gt_ids/nb_mention)))
    return list_mention
    
    
def process_p_e_m(filepath):
    cross_wikis_wikipedia = []
    mention_list = set()
    nb_vide = 0
    with open(filepath, "r") as pemf:
        for line in pemf:
            pem = line.strip().split("\t")
            try: 
                if int(pem[1]) == 0: nb_vide += 1
            except ValueError : pass
            cross_wikis_wikipedia.append(pem)
    for i in range(len(cross_wikis_wikipedia)):
        if len(cross_wikis_wikipedia[i]) > 1:
            mention = cross_wikis_wikipedia[i][0]
            if args.unify: mention_list.add(unify_entity_name(mention))
            else: mention_list.add(mention)
    return mention_list
        

def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--TR_folder", default="../data/basic_data/test_datasets/AIDA/")
    parser.add_argument("--entity_language", default="fr", help="'fr' or 'en'")
    parser.add_argument("--wiki_path", default="wiki_name_id_map.txt")
    parser.add_argument("--p_e_m_file", default="")
    parser.add_argument("--unify_entity_name", dest="unify", action='store_true')
    parser.set_defaults(unify=False)
    return parser.parse_args()

if __name__ == "__main__":
    args = _parse_args()
    current_dir = os.getcwd()
    print("START TR")
    mention_TR = process_TR(args.TR_folder+"train")
    os.chdir(current_dir)
    if os.path.exists(args.TR_folder+"test"): 
        mention_TR = mention_TR.union(process_TR(args.TR_folder+"test"))
        os.chdir(current_dir)
    print("exemples mention_TR : {}".format(list(mention_TR)[:10]))
    print("START PEM")
    mention_PEM = process_p_e_m(args.p_e_m_file)
    print("exemples mention_PEM : {}".format(list(mention_PEM)[:10]))
    print("COMPARAISON")
    top = time.time()
    print("- nombre de mentions TR : {}".format(len(mention_TR)))
    print("- nombre de mentions PEM : {}".format(len(mention_PEM)))
    nb_inTR = 0
    nb_inPEM = 0
    for mention in mention_PEM:
        if mention in mention_TR: nb_inTR += 1
    for mention in mention_TR:
        if mention in mention_PEM: nb_inPEM += 1
    print("- mention of TR in PEM file : {}/{} ({:.2f}%)".format(nb_inPEM, len(mention_PEM), 100*(nb_inPEM/len(mention_PEM)) ))
    print("- mention of PEM in TR file : {}/{} ({:.2f}%)".format(nb_inTR, len(mention_TR), 100*(nb_inTR/len(mention_TR)) ))
    print("comparaison done in {}s".format(int(time.time()-top)))