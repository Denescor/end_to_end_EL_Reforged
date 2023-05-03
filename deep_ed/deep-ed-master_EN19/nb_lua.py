#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  8 14:16:50 2021

@author: carpentier
"""

import os

folder = "/people/carpentier/Modeles/en_entities/deep-ed-master"
print(folder)


def count_lua(path):
    liste = [x for x in os.listdir(path) if os.path.isfile(path+"/"+x) and os.path.splitext(x)[1]==".lua"]
    if len(liste) > 0:
        print("{} : {}\t{}".format("/".join(path.split("/")[5:]),liste,len(liste)))
    return len(liste)

def count_py(path):
    liste = [x for x in os.listdir(path) if os.path.isfile(path+"/"+x) and os.path.splitext(x)[1]==".py"]
    if len(liste) > 0:
        print("{} : {}".format("/".join(path.split("/")[5:]),liste))
    return len(liste)    

def parcours_folder(path,python=False):
    liste = [x for x in os.listdir(path) if os.path.isdir(path+"/"+x)]
    if len(liste)==0:
        if python : return count_py(path)
        else: return count_lua(path)
    else:
        cnt = 0
        for x in liste:
            cnt += parcours_folder(path+"/"+x,python=python)
        return cnt
        
total = parcours_folder(folder)
print("-----------------------")
total_py = parcours_folder(folder,python=True)
print("total : {} fichiers lua".format(total))
print("total : {} fichiers py déjà écrits".format(total_py))