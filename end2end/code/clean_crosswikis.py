#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 13 16:27:30 2022

@author: carpentier
"""

import os
import time
import argparse
from wiki2vec_txt_from_npy import unify_entity_name 
from preprocessing.util import load_wiki_name_id_map

ent_dict_path = "../data/basic_data/TR_fr_en_entmap.txt"

def make_dict_TREntities(out_filepath, TR_folders):
    mention_list = dict()
    for TR_folder in TR_folders:
        list_doc = [x for x in os.listdir(TR_folder) if os.path.isfile(TR_folder+"/"+x) and os.path.splitext(x)[1]==".mentions"]
        for doc in list_doc:
            mention_file = "{}/{}".format(TR_folder,doc)
            mention_list_temp = make_TR_dict(mention_file)
            mention_list.update(mention_list_temp)
    if len(mention_list) > 0:
        with open(out_filepath, 'w') as fout:
            for ent_fr, ent_en in mention_list.items():
                fout.write("{}\t{}\n".format(ent_fr, ent_en))
    return len(mention_list)

def make_TR_dict(mention_filepath):
    with open(mention_filepath) as fin:
        mention_list = dict()
        for line in fin:
            bg, nd, ent_en, ent_fr, hard = line.split("\t")
            mention_list[unify_entity_name(ent_fr)] = unify_entity_name(ent_en) #marche car il n'y peut pas y avoir 2 mentions dans 1 documents qui débutent au même endroit
    return mention_list

def load_dict_TREntities(file, reverse=False):
    """
        default : fr --> en
        reverse : en --> fr
    """
    dico = dict()
    lenght = 0
    with open(file, "r") as f:
        for line in f:
            ent_fr, ent_en = line.strip().split("\t")
            if reverse : dico[ent_en] = ent_fr
            else: dico[ent_fr] = ent_en
            lenght += 1
    print("number of line : {}\nnumber of final entities : {}".format(lenght, len(dico)))
    print("exemple fr ({}) : {}".format(len(list(dico.keys())), list(dico.keys())[:20]))
    print("exemple en ({}) : {}".format(len(list(dico.values())), list(dico.values())[:20]))
    return dico

def clean_crosswikis(file, wiki_dic=None, wiki_lower=None, ent_dic=None):
    cross_wikis_wikipedia = []
    unmodify = 0
    unmatched = 0
    tomodify = 0
    nb_vide = 0
    in_lower = 0
    set_cand = set()
    assert (wiki_dic is not None and wiki_lower is not None) or (wiki_dic is None and wiki_lower is None), "état entre wiki_id et wiki_lower incompatible"
    with open(file, "r") as cross_read:
        for line in cross_read:
            pem = line.strip().split("\t")
            try: 
                if int(pem[1]) == 0: nb_vide += 1
            except ValueError : pass
            cross_wikis_wikipedia.append(pem)
    for i in range(len(cross_wikis_wikipedia)):
        if len(cross_wikis_wikipedia[i]) > 1:
            mention = cross_wikis_wikipedia[i][0]
            freq = cross_wikis_wikipedia[i][1]
            entities = cross_wikis_wikipedia[i][2:]
            for j in range(len(entities)):
                entity = entities[j]
                entity_split = entity.split(",")
                entity_split[-1] = unify_entity_name(entity_split[-1])
                set_cand.add(entity_split[-1])
                if wiki_dic is not None:
                    try: 
                        ent_name = entity_split[-1]
                        if ent_dic is not None:
                            try: ent_name = ent_dic[ent_name] #find the english name of the entity
                            except KeyError: unmatched += 1
                            if ent_name.islower(): 
                                try: ent_name = wiki_lower[ent_name] #from lowercase to casual case
                                except KeyError: in_lower += 1
                            entity_split[0] = wiki_dic[ent_name] #find the id in an english wiki_name_id_map
                            entity_split[-1] = ent_name
                        else: entity_split[0] = wiki_dic[ent_name] #directly find the id in a french wiki_name_id_map
                    except KeyError: unmodify += 1
                    finally: tomodify += 1
                entity = ",".join(entity_split)
                entities[j] = entity
            pem = [mention, freq]
            pem.extend(entities)
            cross_wikis_wikipedia[i] = pem
    with open(file,"w") as cross_write:
        for pem in cross_wikis_wikipedia:
            cross_write.write("{}\n".format("\t".join(pem)))
    if wiki_dic is not None: print("exemple ref wiki ({}) : {}".format(len(list(wiki_dic.keys())), list(wiki_dic.keys())[:20]))
    if ent_dic is not None: print("exemple cand ({}) : {}".format(len(list(set_cand)), list(set_cand)[:20]))
    if ent_dic is not None and wiki_dic is not None:
        nb_in_wiki = len([x for x in ent_dic.values() if x in wiki_dic])
        nb_in_cand = len([x for x in ent_dic.keys() if x in set_cand])
        nb_in_TR = len([x for x in set_cand if x in ent_dic.keys()])
        print("nombre de TR dans wiki : {}/{} ({:.2f}%)".format(nb_in_wiki, len(ent_dic), 100*(nb_in_wiki/len(ent_dic))))
        print("nombre de TR dans cand : {}/{} ({:.2f}%)".format(nb_in_cand, len(ent_dic), 100*(nb_in_cand/len(ent_dic))))
        print("nombre de cand dans TR : {}/{} ({:.2f}%)".format(nb_in_TR, len(set_cand), 100*(nb_in_TR/len(set_cand))))
        print("nombre de lowercase dans TR EN : {}/{} ({:.2f}%)".format(in_lower, len(set_cand), 100*(in_lower/len(set_cand))))
    print("taille pem : {}".format(len(cross_wikis_wikipedia)))
    print("dont vide : {} ({:.2f}%)".format(nb_vide, 100*(nb_vide/len(cross_wikis_wikipedia))))
    if ent_dic is not None: print("dont échec modification id : {} --> {}/{} ({:.2f}%)".format(unmatched, unmodify, tomodify, 100*(unmodify/tomodify)))
    elif wiki_dic is not None: print("dont échec modification id : {}/{} ({:.2f}%)".format(unmodify, tomodify, 100*(unmodify/tomodify)))

if __name__ == "__main__":    
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", default="../data/basic_data/crosswikis_wikipedia_p_e_m.txt", help="path from the folder 'code'")
    parser.add_argument("--wiki_name_id_map", help="file to the 'wiki map' with the new ids")
    parser.add_argument("--convert_fr_to_en", dest="convert", action='store_true', help="add argument to use a dict that convert french name into english name before searching into the wiki_name_id_map")
    parser.set_defaults(convert=False)
    args = parser.parse_args()
    top = time.time()
    dico_id = None
    dico_lower = None
    dico_ent = None
    if args.convert: #defaut path only here
        # regarder si le fichier existe déjà
        #if not os.path.exists(ent_dict_path): 
        # sinon le refaire
        lenght = make_dict_TREntities(ent_dict_path, ["../../en_entities/TR/"+x for x in ["train", "test", "dev"]])
        print("{} entities wrote in '{}'".format(lenght, ent_dict_path))
        # Une fois refait (ou dans tous les cas), on le load
        dico_ent = load_dict_TREntities(ent_dict_path)
    if args.wiki_name_id_map is not None: 
        dico_id = dict()
        dico_lower = dict()
        dico_temp, _ = load_wiki_name_id_map(filepath=args.wiki_name_id_map, verbose=False)
        for name, i in dico_temp.items(): dico_id[unify_entity_name(name)] = i
        for name in dico_id.keys(): 
            try:
                lowername = name.lower()
                assert lowername.islower(), "erreur dans la mise en lowercase"
                dico_lower[lowername] = name
            except AssertionError : continue
        print("taille wiki\ncasual : {}\nlowercase : {}".format(len(dico_id), len(dico_lower)))
        assert len(dico_lower)>0, "erreur dans la mise en lowercase"
    clean_crosswikis(args.file, wiki_dic=dico_id, wiki_lower=dico_lower, ent_dic=dico_ent)
    print("DONE IN {}s".format(int(time.time()-top)))
