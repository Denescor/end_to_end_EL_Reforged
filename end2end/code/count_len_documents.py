#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 15:40:44 2022

@author: carpentier
"""

import os
import time
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--folder", default="new_datasets", help="name of the folder where pick documents")
args = parser.parse_args()
top = time.time()
len_doc = dict()
nb_word = dict()

folder = "../data/{}".format(args.folder)
list_doc = [folder+"/"+x for x in os.listdir(folder) if os.path.isfile(folder+"/"+x)]
for doc in list_doc:
    len_doc[doc] = {"char": [], "word": []}
    with open(doc) as f:
        dataset = f.readlines()
        for line in dataset:
            if line.startswith("DOCSTART_"): #begin doc
                len_doc[doc]["char"].append(0)
                len_doc[doc]["word"].append(0)
            elif line.startswith("MMSTART_") or line.startswith("MMEND") or line.startswith("DOCEND") or line.startswith("*NL*") : #end doc
                continue #nothing to do
            else: #in a doc
                len_doc[doc]["char"][-1] += len(line)
                len_doc[doc]["word"][-1] += 1
print("DONE IN {}s".format(int(time.time()-top)))
for doc, allen in len_doc.items():
    charlen = allen["char"]
    wordlen = allen["word"]
    print("{} :".format(doc))
    print("\t- nb doc : {}\n\t{}\n\t- mean len : {:.1f}\n\t- median len : {:.1f}\n\t- min len : {}\n\t- max len : {}".format(len(charlen), 20*"-", np.mean(charlen), np.median(charlen), min(charlen), max(charlen) ))
    print("\t{}".format(20*"-"))
    print("\t- mean word : {:.1f}\n\t- median word : {:.1f}\n\t- min word : {}\n\t- max word : {}".format(np.mean(wordlen), np.median(wordlen), min(wordlen), max(wordlen)))
    print(20*"#")
