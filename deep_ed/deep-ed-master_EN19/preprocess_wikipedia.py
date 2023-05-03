#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 25 13:13:21 2021

@author: carpentier
"""

import re
import os
import argparse

doc_balise = re.compile(r'<doc id="([0-9]+)" url=".+" title="(.+)">')

def make_wiki_name_id_map(folder,file):
    list_ent = []
    with open(folder+file) as fin:
        for line in fin:
            if doc_balise.match(line):
                extract_doc = doc_balise.search(line)
                ent_id = extract_doc.group(1)
                ent_title = extract_doc.group(2)
                list_ent.append((ent_id,ent_title))
    with open(folder+"wiki_name_id_map.txt",'w') as fout:
        for ent_id, ent_title in list_ent:
            fout.write("{}\t{}\n".format(ent_title,ent_id))
    print("DONE :\t{} entities".format(len(list_ent)))
       
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", default="./")
    parser.add_argument("--file", default="textWithAnchorsFromAllWikipedia2014Feb.txt")
    args = parser.parse_args()
    
    os.chdir(args.folder)
    
    print("infile :\t{}/{}\noutfile :\t{}/wiki_name_id_map.txt".format(os.getcwd(),args.file,os.getcwd()))
    make_wiki_name_id_map(args.folder, args.file)
