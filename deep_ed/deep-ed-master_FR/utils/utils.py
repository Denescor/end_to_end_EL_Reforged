#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 11:32:26 2021

@author: carpentier
"""

import torch
import re

def findLUA_bool(string,sub,start=0,end=0):
    bg = string.find(sub,start,end)
    if bg != -1:
        return True
    else:
        return False

def topk(one_dim_tensor, k):
    bestk, indices = torch.topk(one_dim_tensor, k, smallest=True)
    sorted, newindices = torch.sort(bestk, True)
    oldindices = torch.LongTensor(k)
    for i in range(1,k):
        oldindices[i] = indices[newindices[i]]
    return sorted, oldindices

def list_with_scores_to_str(liste, scores):
    string = ""
    for i,v in liste.items() :
        string = "{}{}[{:.2f}];".format(string,list[i],scores[i])
    return string


def table_len(t):
    count = 0
    for _ in t : count += 1
    return count


def split(inputstr, sep=None):
    if sep is None :
        sep = "%s" #TODO convertir le pattern en python
    t=[]
    pat = re.compile("([^{}]+)".format(sep)) #TODO vérifier le pattern
    for string in pat.finditer(inputstr):
        t.append(string)
    return t

# Unit test:
assert 6 == split('aa_bb cc__dd   ee  _  _   __ff' , '_ ')



def correct_type(data) : #TODO convertir les "data:" en python
    if args.type == 'float' : return None #data:float()
    elif args.type == 'double' : return None #data:double()
    elif findLUA_bool(str(args.type), 'cuda') : return None #data:cuda()
    else: print('Unsuported type')

# color fonts:
def red(s) : return '\27[31m{}\27[39m'.format(s)

def green(s): return '\27[32m{}\27[39m'.format(s)

def yellow(s): return '\27[33m{}\27[39m'.format(s)

def blue(s): return '\27[34m{}\27[39m'.format(s)

def violet(s): return '\27[35m{}\27[39m'.format(s)

def skyblue(s): return '\27[36m{}\27[39m'.format(s)


def split_in_words(inputstr):
    words = []
    pat = re.compile("%w+") #TODO convertir pattern en regex
    for word in pat.finditer(inputstr): words.append(word)
    return words

def first_letter_to_uppercase(s): return "{}{}".format(s[1:1].upper(),s[2:])

def modify_uppercase_phrase(s):
    if (s == s.upper()) :
        words = split_in_words(s.lower())
        res = []
        for w in words :
            res.append(first_letter_to_uppercase(w))
        return " ".join(res) 
    else:
        return s

def blue_num_str(n): blue("{:.3f}".format(n))

def string_starts(s, m): return s[1:len(m)] == m

# trim:
def trim1(s):
    pat = re.compile("^%s*(.-)%s*$") #TODO convertir pattern en regex
    return re.subn(pat,"%1",s)


def nice_print_red_green(a,b):
    s = "{:.3f}:{:.3f}[".format(a,b)
    if a > b :
        return "{}{}]".format(s,red("{:.3f}".format(a-b)))
    elif a < b :
        return "{}{}]".format(s,green("{:.3f}".format(b-a)))
    else:
        return "{}0]".format(s)