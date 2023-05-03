#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 15:51:30 2021

@author: carpentier
"""

import argparse

def make_wikiid2nnid(fileEnt, fileWikiName):
    wikiid = load_dict(fileWikiName) # format txt -> idwiki
    nnid = load_dict(fileEnt) # format idnn -> txt
    wikiid2nnid = dict() # format idwiki -> idnn
    nnid2wikiid = dict() # format idnn -> idwiki
    nb_error = 0
    print("\nsize wiki_name_id_map : {}\nsize entities_universe : {}".format(len(wikiid), len(nnid)))
    for i, (idnn, txt) in enumerate(nnid.items()):
        try:
            idwiki = wikiid[txt]
            wikiid2nnid[idwiki] = i+1
            nnid2wikiid[i] = idnn
        except KeyError:
            nb_error += 1
            continue
    print("size wikiid2nnid : {}\nnb_error : {}\n".format(len(wikiid2nnid),nb_error))
    return wikiid2nnid, nnid2wikiid

def make_wikiid2nnid_bis(fileEnt, fileWikiName):
    wikiid = load_dict(fileWikiName) # format txt -> idwiki
    #nnid = load_dict(fileEnt) # format idnn -> txt
    wikiid2nnid = dict() # format idwiki -> idnn
    nnid2wikiid = dict() # format idnn -> idwiki
    print("\nsize wiki_name_id_map : {}".format(len(wikiid)))
    for i, (txt, idwiki) in enumerate(wikiid.items()):
        wikiid2nnid[idwiki] = i+1
        nnid2wikiid[i] = idwiki
    print("size wikiid2nnid : {}\n".format(len(wikiid2nnid)))
    return wikiid2nnid, nnid2wikiid
    
def load_dict(filepath):
    current_dict = dict()
    with open(filepath, "r") as lines:
        for line in lines:
            key, value = line.strip().split("\t")
            key = " ".join(key.split("_"))
            value = " ".join(value.split("_"))
            current_dict[key] = value
    print("current dict : {}".format(list(current_dict.items())[:10]))
    return current_dict

def write_dict(dico, filepath):
    with open(filepath, "w") as output:
        for key, value in dico.items():
            output.write("{}\t{}\n".format(key,value))
    print("{} done".format(filepath))


parser = argparse.ArgumentParser()
parser.add_argument("--nnpath", default="/people/carpentier/Modeles/end2end_neural_el-master/data/entities/entities_universe.txt")
parser.add_argument("--wikipath", default="/people/carpentier/Modeles/end2end_neural_el-master/data/basic_data/wiki_name_id_map.txt")   
parser.add_argument("--outputpath", default="/people/carpentier/Modeles/end2end_neural_el-master/data/entities/wikiid2nnid/")   
args = parser.parse_args()

#print("test file")
#cnt_1, cnt_2, cnt_3, cnt_4, cnt_5, cnt_6, cnt_7, cnt_8 = 0, 0, 0, 0, 0, 0, 0, 0
#with open("/people/carpentier/Modeles/en_entities/deep-ed-master_true_map/generated/wiki_canonical_words.txt","r") as lines:
#    for line in lines:
#        cnt_1 += 1
#with open("/people/carpentier/Modeles/en_entities/deep-ed-master_true_map/generated/wiki_canonical_words_RLTD.txt","r") as lines:
#    for line in lines:
#        cnt_2 += 1
#with open("/people/carpentier/Modeles/en_entities/deep-ed-master_true_map/generated/empty_page_ents.txt","r") as lines:
#    for line in lines:
#        cnt_3 += 1
#with open("/people/carpentier/Modeles/en_entities/deep-ed-master_true_map/generated/ent_wiki_freq.txt","r") as lines:
#    for line in lines:
#        cnt_4 += 1
#with open("/people/carpentier/Modeles/en_entities/deep-ed-master_true_map/generated/word_wiki_freq.txt","r") as lines:
#    for line in lines:
#        cnt_5 += 1
#with open("/people/carpentier/Modeles/en_entities/deep-ed-master_true_map/generated/wikipedia_p_e_m.txt","r") as lines:
#    for line in lines:
#        cnt_6 += 1
#with open("/people/carpentier/Modeles/en_entities/deep-ed-master_true_map/generated/yago_p_e_m.txt","r") as lines:
#    for line in lines:
#        cnt_7 += 1
#with open("/people/carpentier/Modeles/en_entities/deep-ed-master_true_map/generated/crosswikis_wikipedia_p_e_m.txt","r") as lines:
#    for line in lines:
#        cnt_8 += 1
#import gensim
#model = gensim.models.KeyedVectors.load_word2vec_format("/people/carpentier/Modeles/en_entities/deep-ed-master_true_map/basic_data/wordEmbeddings/Word2Vec/GoogleNews-vectors-negative300.bin", binary=True)
#cnt_9 = len(model.vocab)
#print("len word all : {}\nlen word RLTD : {}".format(cnt_1, cnt_2))
#print("len empty pages : {}\nlen ents freq : {}\nlen word freq : {}".format(cnt_3, cnt_4, cnt_5))
#print("len wiki pem : {}\nlen yago pem : {}\nlen crosswikis pem : {}".format(cnt_6, cnt_7, cnt_8))
#print("len W2V : {}".format(cnt_9))
#print("DONE")

print("START")
wikiid2nnid, nnid2wikiid = make_wikiid2nnid_bis(args.nnpath, args.wikipath)
write_dict(wikiid2nnid, args.outputpath+"wikiid2nnid.txt")
write_dict(nnid2wikiid, args.outputpath+"nnid2wikiid.txt")
print("DONE")