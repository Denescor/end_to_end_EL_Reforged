#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 14:22:49 2023

@author: carpentier
"""

import os
import argparse
import jsonlines
from tqdm import tqdm
from time import time
import preprocessing.util as util
from wiki2vec_txt_from_npy import unify_entity_name 

def convert_to_aida(data_name, kilt_file, aida_file, entityNameIdMap):
    unknown_gt_ids = 0
    nb_mentions = 0
    with jsonlines.open(kilt_file) as kilt, open(aida_file, "w") as aida:
        for item in tqdm(kilt, desc="process {}".format(data_name)):
            context = item["meta"]["context"]
            mentions = item["meta"]["mentions"]
            entities = item["meta"]["entities"]
            id_item = item["id"]
            nb_mentions += len(mentions)
            assert len(context) == len(mentions) + 1
            assert len(mentions) == len(entities)
            text = "DOCSTART_{} ".format(id_item)
            for i in range(len(mentions)):
                left = context[i]
                mention = mentions[i]
                if args.unify: entity = unify_entity_name(entities[i])
                else: entity = entities[i]
                ent_id = entityNameIdMap.compatible_ent_id(name=entity, ent_id=entities[i])
                if ent_id is not None:
                    text += "{} MMSTART_{} {} MMEND ".format(left, ent_id, mention)
                else:
                    text += "{} {} ".format(left, mention)
                    unknown_gt_ids += 1
            text += "{} DOCEND".format(context[-1])
            text = [x.strip() for x in text.split(" ")]
            for word in text:
                aida.write("{}\n".format(word))
    return unknown_gt_ids, nb_mentions


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", help="path for kilt files")
    parser.add_argument("--output_path", help="path to save txt files")
    parser.add_argument("--input_filenames", help="files separate by 'separator'")
    parser.add_argument("--separator", default="|", help="filennames separator")
    parser.add_argument("--wiki_path", default="wiki_name_id_map.txt")
    parser.add_argument("--unify_entity_name", dest="unify", action='store_true')
    parser.set_defaults(unify=False)    
    return parser.parse_args()

if __name__ == "__main__":
    args = _parse_args()
    entityNameIdMap = util.EntityNameIdMap()
    entityNameIdMap.init_compatible_ent_id(wiki_map_file=args.wiki_path)
    datasets = [(os.path.splitext(x)[0], os.path.join(args.input_path, x)) for x in args.input_filenames.split(args.separator)]
    top = time()
    for name, dataset in datasets:
        aida_file = os.path.join(args.output_path, name)
        unknown_gt_ids, nb_mentions = convert_to_aida(name, dataset, "{}.txt".format(aida_file), entityNameIdMap)
        print("skip mentions (unknown) : {}/{} ({:.2f}%)".format(unknown_gt_ids, nb_mentions, 100*(unknown_gt_ids/nb_mentions)))
    print("DONE {} DATASETS IN {}s".format(len(datasets), int(time()-top)))