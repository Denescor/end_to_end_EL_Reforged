#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 15:08:49 2021

@author: carpentier
"""

# ------------------ Load entity name-id mappings ------------------
# -- Each entity has:
# --   a) a Wikipedia URL referred as 'name' here
# --   b) a Wikipedia ID referred as 'ent_wikiid' or 'wikiid' here
# --   c) an ID that will be used in the entity embeddings lookup table. Referred as 'ent_thid' or 'thid' here.

import os
import torch
import utils.utils as utils
import data_gen.indexes.wiki_redirect_index as wiki_redirect_index

def dofile(args):
    rltd_only = False    
    if args and args.entities and args.entities != 'ALL' :
        assert rewtr.reltd_ents_wikiid_to_rltdid, 'Import relatedness.lua before ent_name_id.lua' #TODO trouver ola lib rewtr
        rltd_only = True
        
    unk_ent_wikiid = 1
    
    entity_wiki_txtfilename = "{}basic_data/wiki_name_id_map.txt".format(args.root_data_dir)
    entity_wiki_t7filename = "{}generated/ent_name_id_map.t7".format(args.root_data_dir)
    if rltd_only :
        entity_wiki_t7filename = "{}generated/ent_name_id_map_RLTD.t7".format(args.root_data_dir)
    else:
        entity_wiki_t7filename = "{}generated/ent_name_id_map.t7".format(args.root_data_dir)
        
    print('==> Loading entity wikiid - name map') 
    
    e_id_name = None
    
    if os.isfile(entity_wiki_t7filename) :
        print("  ---> from t7 file: {}".format(entity_wiki_t7filename))
        e_id_name = torch.load(entity_wiki_t7filename)
    else:
        print('---> t7 file NOT found. Loading from disk (slower). Out f = {}'.format(entity_wiki_t7filename))
        wiki_disambiguation_index = wiki_redirect_index.dofile()
        print('Still loading entity wikiid - name map ...') 
        
        e_id_name = dict()
        # map for entity name to entity wiki id
        e_id_name["ent_wikiid2name"] = dict()
        e_id_name["ent_name2wikiid"] = dict()
    
        # map for entity wiki id to tensor id. Size = 4.4M
        if not rltd_only :
            e_id_name["ent_wikiid2thid"] = dict()
            e_id_name["ent_thid2wikiid"] = dict()
            
        cnt = 0
        cnt_freq = 0
        with open(entity_wiki_txtfilename, "r") as lines:
            for line in lines :
                parts = line.split('\t')
                ent_name = parts[0]
                ent_wikiid = int(parts[1])      
            
                if (not wiki_disambiguation_index[ent_wikiid]) : 
                    if (not rltd_only) or rewtr.reltd_ents_wikiid_to_rltdid[ent_wikiid] : #TODO trouver la lib rewtr
                        e_id_name["ent_wikiid2name"][ent_wikiid] = ent_name
                        e_id_name["ent_name2wikiid"][ent_name] = ent_wikiid
                    if not rltd_only :
                        cnt = cnt + 1
                        e_id_name["ent_wikiid2thid"][ent_wikiid] = cnt
                        e_id_name["ent_thid2wikiid"][cnt] = ent_wikiid
                        
        if not rltd_only :
            cnt += 1
            e_id_name["ent_wikiid2thid"][unk_ent_wikiid] = cnt
            e_id_name["ent_thid2wikiid"][cnt] = unk_ent_wikiid
            
        e_id_name["ent_wikiid2name"][unk_ent_wikiid] = 'UNK_ENT'
        e_id_name["ent_name2wikiid"]['UNK_ENT'] = unk_ent_wikiid
    
        torch.save(entity_wiki_t7filename, e_id_name)
        
    if not rltd_only :
        unk_ent_thid = e_id_name["ent_wikiid2thid"][unk_ent_wikiid]
    else:
        unk_ent_thid = rewtr.reltd_ents_wikiid_to_rltdid[unk_ent_wikiid] #TODO trouver la lib rewtr
    return e_id_name, unk_ent_thid


# ------------------------ Functions for wikiids and names-----------------
def get_map_all_valid_ents():
    m = dict()
    for ent_wikiid in e_id_name["ent_wikiid2name"].keys() :
        m[ent_wikiid] = 1
    return m  

def is_valid_ent(ent_wikiid):
    if e_id_name.ent_wikiid2name[ent_wikiid] :
        return True
    return False


def get_ent_name_from_wikiid(ent_wikiid):
  ent_name = e_id_name.ent_wikiid2name[ent_wikiid]
  if (not ent_wikiid) or (not ent_name) :
    return 'NIL'
  return ent_name

def preprocess_ent_name(ent_name):
    ent_name = utils.trim1(ent_name)
    ent_name = ent_name.replace('&amp;', '&')
    ent_name = ent_name.replace('&quot;', '"')
    ent_name = ent_name.replace('_', ' ')
    ent_name = utils.first_letter_to_uppercase(ent_name)
    try : ent_name = wiki_redirect_index.get_redirected_ent_title(ent_name)
    except : pass
    return ent_name

def get_ent_wikiid_from_name(ent_name, not_verbose):
    verbose = (not not_verbose)
    ent_name = preprocess_ent_name(ent_name)
    ent_wikiid = e_id_name.ent_name2wikiid[ent_name]
    if (not ent_wikiid) or (not ent_name) :
        if verbose : print(utils.red('Entity {} not found. Redirects file needs to be loaded for better performance.'.format(ent_name)))
        return unk_ent_wikiid
    return ent_wikiid

# ------------------------ Functions for thids and wikiids -----------------
# -- ent wiki id -> thid
def get_thid(ent_wikiid):
    if rltd_only :
        ent_thid = rewtr.reltd_ents_wikiid_to_rltdid[ent_wikiid] #TODO to convert
    else:
        ent_thid = e_id_name["ent_wikiid2thid"][ent_wikiid]
    if (not ent_wikiid) or (not ent_thid) :
        return unk_ent_thid
    return ent_thid

def contains_thid(ent_wikiid):
    if rltd_only :
        ent_thid = rewtr.reltd_ents_wikiid_to_rltdid[ent_wikiid] #TODO to convert
    else:
        ent_thid = e_id_name["ent_wikiid2thid"][ent_wikiid]
    if ent_wikiid == None or ent_thid == None :
        return False
    return True

def get_total_num_ents():
    if rltd_only :
        assert table_len(rewtr.reltd_ents_wikiid_to_rltdid) == rewtr.num_rltd_ents #TODO to convert
        return table_len(rewtr.reltd_ents_wikiid_to_rltdid) #TODO to convert
    else:
        return e_id_name["ent_thid2wikiid"]
  
def get_wikiid_from_thid(ent_thid):
    if rltd_only :
        ent_wikiid = rewtr.reltd_ents_rltdid_to_wikiid[ent_thid] #TODO to convert
    else:
        ent_wikiid = e_id_name["ent_thid2wikiid"][ent_thid]
    if ent_wikiid == None or ent_thid == None :
        return unk_ent_wikiid
    return ent_wikiid

#-- tensor of ent wiki ids --> tensor of thids
def get_ent_thids(ent_wikiids_tensor):
    ent_thid_tensor = torch.clone(ent_wikiids_tensor)
    if ent_wikiids_tensor.dim() == 2 :
        for i in range(0,ent_thid_tensor.size(0)) :
            for j in range(0,ent_thid_tensor.size(1)) :
                ent_thid_tensor[i][j] = get_thid(ent_wikiids_tensor[i][j])
    elif ent_wikiids_tensor.dim() == 1 :
        for i in range(0,ent_thid_tensor.size(1)):
            ent_thid_tensor[i] = get_thid(ent_wikiids_tensor[i])
    else:
        print('Tensor with > 2 dimentions not supported')
        os.exit()
    return ent_thid_tensor

print('Done loading entity name - wikiid. Size thid index = {}'.format(get_total_num_ents()))