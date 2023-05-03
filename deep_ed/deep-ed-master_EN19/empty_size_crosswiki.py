#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  9 15:36:34 2022

@author: carpentier
"""

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--file", default="/people/carpentier/Modeles/en_entities/deep-ed-master_true_map/generated/wikipedia_p_e_m.txt", help="path from the folder 'code'")
args = parser.parse_args()

nb_vide = 0
nblines = 0
nb_error = 0
with open(args.file, "r") as cross_read:
    for line in cross_read:
        pem = line.strip().split("\t")
        try: 
            if int(pem[1]) == 0: nb_vide += 1
        except ValueError : nb_error += 1
        finally: nblines += 1
print("Taille Crosswiki : {}\n\tdont vide : {} ({:.2f}%)\n\tdont erreur : {} ({:.2f}%)".format( nblines, nb_vide, 100*(nb_vide/nblines), nb_error, 100*(nb_error/nblines) ))