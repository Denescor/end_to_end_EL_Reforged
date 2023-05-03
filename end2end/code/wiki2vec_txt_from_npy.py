#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 13 14:40:28 2022

@author: carpentier
"""

#process wiki2vec
#- extraire les entités de frwiki_20180420_300d.txt
#    - format initial
#        - 1er ligne : nb d'entité \t dimension
#        - mot " " embeddings (séparé par " ")
#    - mettre sous format "ent_vecs.txt" + "wiki_name_id_map.txt"
#        - "ent_vecs.txt" : embeddings (séparé par " ")
#        - "wiki_name_id_map.txt" : nom entité \t id
#            - id = osef (un int quelconque)
#- créer un "wiki2nnid.txt"
#    - format : id wiki \t id npy
#    - id wiki = id de "wiki_name_id_map.txt"
#    - id npy = position dans "ent_vecs.txt"
#- utiliser le script "ent_vecs_from_txt_to_npy.py"
#    ==> créer "ent_vecs.npy"

import re
import os
import time
import os.path
import argparse
from shutil import copy, copy2, move
import model.config as config
import preprocessing.bridge_code_lua.ent_vecs_from_txt_to_npy as txt_to_npy

clear_parenthesis = re.compile(r"\([^()]*\)")
clear_parenthesis2 = re.compile(r"\[[^\[\]]*\]")
clear_parenthesis3 = re.compile(r"{[^{}}]*}")

# dossier de lancement du script : "config.base_folder/code"

def unify_entity_name(entity):
    """
    Doit :
        - remplacer les "_", "-", ":" par des espaces
        - retire les espaces superflus qui resteraient
        - retire tous les espaces entre les mots
    """
    try:
        entity = entity.strip()
        entity = entity.replace("_", " ")
        entity = entity.replace("-", " ")
        entity = entity.replace(":", " ")
        entity = "".join(entity.split(" "))
        #entity = clear_parenthesis.sub("", entity)
        #assert "(" not in entity1, "old : {} | new : {}".format(entity, entity1)
        #entity = " ".join([x for x in entity1.split(" ") if x != ''])
        #entity = entity.lower()
    except Exception as e: print(e)
    return entity

def load_wiki_name_id_map(folder, lowercase=False, filepath=None, verbose=True):
    wall_start = time.time()
    wiki_name_id_map = dict()
    wiki_id_name_map = dict()
    wiki_name_id_map_errors = 0
    duplicate_names = 0    # different lines in the doc with the same title
    duplicate_ids = 0      # with the same id
    if filepath is None:
        filepath = "wiki_name_id_map.txt"
    with open(folder+filepath) as fin:
        for line in fin:
            line = line.rstrip()
            try:
                wiki_title, wiki_id = line.split("\t")
                wiki_title = unify_entity_name(wiki_title)
                if lowercase:
                    wiki_title = wiki_title.lower()

                if wiki_title in wiki_name_id_map:
                    duplicate_names += 1
                if wiki_id in wiki_id_name_map:
                    duplicate_ids += 1

                wiki_name_id_map[wiki_title] = wiki_id
                wiki_id_name_map[wiki_id] = wiki_title
            except ValueError:
                wiki_name_id_map_errors += 1
    if verbose:
        print("load wiki_name_id_map. wall time:", (time.time() - wall_start)/60, " minutes")
        print("wiki_name_id_map_errors: ", wiki_name_id_map_errors)
        print("duplicate names: ", duplicate_names)
        print("duplicate ids: ", duplicate_ids)
    return wiki_name_id_map, wiki_id_name_map

def split_wikifile_to_ent_vecs(folder, folder_wiki_base, wikifile, entfile, generate_index=True):
    """
        Prend un fichier de vecteur de wikipedia2vec et une list d'entité de TR
            - format txt
        Renvoie :
            - ent_vecs.txt : les vecteurs sur chaque ligne dans un fichier pouvant être transformé par np.loadtxt()
            - wiki_name_id_map.txt : une map entre nom d'entités et id numérique
            - wikiid2nnid.txt : une map entre id numérique de l'entité et position dans la matrice de vecteur
            - nnid2wikiid.txt : la map de wikiid2nid inversée
    """
    top = time.time()
    vector_list = dict()
    wikiid2nnid = dict()
    entity_list = set()
    nb_entity_total = 0
    if not generate_index:
        wiki_base, _ = load_wiki_name_id_map(folder_wiki_base)
        error_wiki_base = 0
    with open(entfile, "r") as tr:
        nb_entities_ref = 0
        for line in tr:
            entity_temp = line.strip().split("\t")
            if len(entity_temp) == 1: entity = unify_entity_name(entity_temp[0])
            elif len(entity_temp) == 2: entity = unify_entity_name(entity_temp[0]) #TODO sans doute obsolète
            else: print(entity_temp)
            entity_list.add(entity)
        nb_entities_ref = len(entity_list)
    print("Nombre d'entité de référence : {}".format(len(entity_list)))
    with open(folder+wikifile,"r") as wiki:
        for i, line in enumerate(wiki):
            if i == 0: continue #on exclue la première ligne qui n'est pas un vecteur
            if "\t" in line: #format "ent\tvalue1 value2 value3 ... valueDim"
                linesplit = line.split("\t")
                word = linesplit[0]
                vector = linesplit[1].split(" ")
                assert len(vector) > 1, "bad separator of vector"
                assert "\t" not in word, "bad separation of word"
            else: #format "ent value1 value2 value3 ... valueDim"
                linesplit = line.split(" ")
                word, vector = linesplit[0], linesplit[1:]
            if("ENTITY/" in word):
                nb_entity_total += 1
                entity_clean = unify_entity_name(word.split("/")[1])
                if generate_index: ent_id = i
                else: 
                    try: ent_id = wiki_base[entity_clean]
                    except KeyError: 
                        error_wiki_base += 1
                        continue #on zap l'entité non trouvé
                entity = "{}\t{}".format(ent_id,entity_clean)
                if entity_clean in entity_list: vector_list[entity] = vector
    print("Nombre d'entité in {}s : ".format(int(time.time()-top)))
    print("\t- par rapport à Wikipedia2Vec : {}/{} ({:.2f}%)".format(len(vector_list), nb_entity_total, 100*(len(vector_list)/nb_entity_total)))
    print("\t- par rapport à la référence : {}/{} ({:.2f}%)".format(len(vector_list) ,nb_entities_ref, 100*(len(vector_list)/nb_entities_ref)))
    if not generate_index: print("\t- unknow entities from original dump : {}/{} ({:.2f}%)".format(error_wiki_base, nb_entity_total, 100*(error_wiki_base/nb_entity_total)))
    print("exemples : {}".format(list(vector_list.keys())[:10]))
    with open(folder+"wiki_name_id_map.txt", "w") as wiki_name_id_map:
        with open(folder+"ent_vecs.txt", "w") as ent_vecs:
            with open(folder+"wiki_name_id_map_unify.txt", "w") as wiki_name_id_map_unify:
                idnpy = 1 #simulate index like torch (lua)
                for entity, vector in vector_list.items():
                    idwiki, ent = entity.split("\t") #idwiki = position de l'entité dans la matrice initiale d'embedding de mot (wikifile)
                    wiki_name_id_map.write("{}\t{}\n".format(ent,idwiki))
                    wiki_name_id_map_unify.write("{}\t{}\n".format(unify_entity_name(ent),idwiki))
                    ent_vecs.write("{}\n".format(" ".join(vector)))
                    wikiid2nnid[idwiki] = idnpy
                    idnpy += 1
    print("wikiid2nnid : {} in {}s".format(len(wikiid2nnid),int(time.time()-top)))
    with open(folder+"wikiid2nnid.txt","w") as wikiid2nnidfile:
        with open(folder+"nnid2wikiid.txt","w") as nnid2wikiidfile:
            for wikiid, nnid in wikiid2nnid.items():
                wikiid2nnidfile.write("{}\t{}\n".format(wikiid,nnid))
                nnid2wikiidfile.write("{}\t{}\n".format(nnid,wikiid))
    print("SPLIT WIKIPEDIA2VEC DONE IN {}s".format(int(time.time()-top)))
    return len(wikiid2nnid)

def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--entity_extension", default=False)
    parser.add_argument("--in_folder", help="folder with the wikipedia2vec data")
    parser.add_argument("--folder_wiki_base", default="/people/carpentier/Modeles/en_entities/deep-ed-master_true_map/basic_data/")
    parser.add_argument("--wikifile", default="frwiki_20180420_300d.txt", help="file with the embeddings in txt")
    parser.add_argument("--entfile", default="ent_fr_wiki2vec.txt", help="file with the list of entities")
    parser.add_argument("--not-generate-index", dest="generate_index", action="store_false")
    parser.set_defaults(generate_index=True)
    parser.add_argument("--entity_vectors", default="ent_vecs_wiki2vec.txt")
    parser.add_argument("--wikiid2nnid", default="wikiid2nnid_wiki2vec.txt")
    parser.add_argument("--entity_language", default="fr", help="'fr' or 'en'")
    args = parser.parse_args()
    return args    
    
def from_wiki2vec_to_entnpy(args):
    walltime = time.time()
    entity_file = args.in_folder+"{}_list_entities.txt".format(args.entity_language)
    print("START {}".format(args.entity_language))
    print("Entities Reference : {}".format(entity_file))
    test = split_wikifile_to_ent_vecs(args.in_folder, args.folder_wiki_base, args.wikifile, entity_file, generate_index=args.generate_index)
    try: assert test > 0
    except: print("ABANDON")
    else:
        top = time.time()
        filename, _ = os.path.splitext(args.wikiid2nnid)
        key_name = filename.split("_")[-1] #defaut = "wiki2vec"
        #for file in ["wiki_name_id_map", "ent_vecs", "wikiid2nnid"]
        if os.path.exists(config.base_folder+"data/basic_data/wiki_name_id_map_"+key_name+".txt"): os.remove(config.base_folder+"data/basic_data/wiki_name_id_map_"+key_name+".txt")
        if os.path.exists(config.base_folder+"data/basic_data/wiki_name_id_map_"+key_name+".txt"): os.remove(config.base_folder+"data/basic_data/wiki_name_id_map_unify_"+key_name+".txt")
        if not args.generate_index: copy2(args.folder_wiki_base+"wiki_name_id_map.txt", args.in_folder) #erase existing and generated wiki_name_id_map. We only work with the original one
        os.rename(args.in_folder+"wiki_name_id_map.txt", args.in_folder+"wiki_name_id_map_"+key_name+".txt")
        move(args.in_folder+"wiki_name_id_map_"+key_name+".txt", config.base_folder+"data/basic_data/")
        os.rename(args.in_folder+"wiki_name_id_map_unify.txt", args.in_folder+"wiki_name_id_map_unify_"+key_name+".txt")
        move(args.in_folder+"wiki_name_id_map_unify_"+key_name+".txt", config.base_folder+"data/basic_data/")
        if os.path.exists(config.base_folder+"data/entities/ent_vecs/ent_vecs_"+key_name+".txt"): os.remove(config.base_folder+"data/entities/ent_vecs/ent_vecs_"+key_name+".txt")
        os.rename(args.in_folder+"ent_vecs.txt", args.in_folder+"ent_vecs_"+key_name+".txt")
        move(args.in_folder+"ent_vecs_"+key_name+".txt", config.base_folder+"data/entities/ent_vecs/")
        if os.path.exists(config.base_folder+"data/entities/wikiid2nnid/wikiid2nnid_"+key_name+".txt"): os.remove(config.base_folder+"data/entities/wikiid2nnid/wikiid2nnid_"+key_name+".txt")
        os.rename(args.in_folder+"wikiid2nnid.txt", args.in_folder+"wikiid2nnid_"+key_name+".txt")
        move(args.in_folder+"wikiid2nnid_"+key_name+".txt", config.base_folder+"data/entities/wikiid2nnid/")
        if os.path.exists(config.base_folder+"data/entities/wikiid2nnid/nnid2wikiid_"+key_name+".txt"): os.remove(config.base_folder+"data/entities/wikiid2nnid/nnid2wikiid_"+key_name+".txt")
        os.rename(args.in_folder+"nnid2wikiid.txt", args.in_folder+"nnid2wikiid_"+key_name+".txt")
        move(args.in_folder+"nnid2wikiid_"+key_name+".txt", config.base_folder+"data/entities/wikiid2nnid/")
        args.entity_vectors = "ent_vecs_"+key_name+".txt"
        args.wikiid2nnid = "wikiid2nnid_"+key_name+".txt"
        print("COPY DONE IN {}s".format(int(time.time()-top)))
        top = time.time()
        txt_to_npy.main(args) #les fichiers nécessaire sont écris et copiés dans les bons dossiers. On peut utiliser le script de base pour passer au numpy
        os.rename(config.base_folder+"data/entities/ent_vecs/ent_vecs.npy", config.base_folder+"data/entities/ent_vecs/ent_vecs_"+key_name+".npy")
        print("CONVERSION TO NUMPY DONE IN {}s".format(int(time.time()-top)))
    print("ALL DONE IN {}s".format(int(time.time()-walltime)))
    
if __name__ == "__main__":
    args = _parse_args()
    from_wiki2vec_to_entnpy(args)
