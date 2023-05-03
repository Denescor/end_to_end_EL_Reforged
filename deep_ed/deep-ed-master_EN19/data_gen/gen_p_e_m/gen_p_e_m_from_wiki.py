# -*- coding: utf-8 -*-
"""
Éditeur de Spyder

Ceci est un script temporaire.
"""

# Generate p(e|m) index from Wikipedia
# Run: th data_gen/gen_p_e_m/gen_p_e_m_from_wiki.lua -root_data_dir $DATA_PATH

import torch
import argparse
import utils.utils as utils
import data_gen.parse_wiki_dump.parse_wiki_dump_tools as parse_wiki_dump_tools

if __name__ =="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_data_dir",defaut="./",help="Root Path of the data : $DATA_PATH")
    parser.add_argument("--anchors_file",defaut="textWithAnchorsFromAllWikipedia2014Feb.txt")
    
    args = parser.parse_args()
    
    print('Computing Wikipedia p_e_m')
    
    wiki_e_m_counts = dict()
    
    num_lines = 0
    parsing_errors = 0
    list_ent_errors = 0
    diez_ent_errors = 0
    disambiguation_ent_errors = 0
    num_valid_hyperlinks = 0
    
    with open(args.root_data_dir+"'basic_data/"+args.anchors_file,"r") as f:
        for line in f:
            num_lines += 1
            if num_lines % 5000000 == 0 :
                print("Processed {} lines. Parsing errs = {} List ent errs = {} diez errs = {} disambig errs = {} Num valid hyperlinks = {}".format(num_lines,parsing_errors,list_ent_errors,diez_ent_errors,disambiguation_ent_errors,num_valid_hyperlinks))
            if '<doc id="' not in line:
                list_hyp, text, le_errs, p_errs, dis_errs, diez_errs = parse_wiki_dump_tools.extract_text_and_hyp(line, False)
                parsing_errors = parsing_errors + p_errs
                list_ent_errors = list_ent_errors + le_errs
                disambiguation_ent_errors = disambiguation_ent_errors + dis_errs
                diez_ent_errors = diez_ent_errors + diez_errs
                for el in list_hyp :
                    mention = el["mention"]
                    ent_wikiid = el["ent_wikiid"]
                    # A valid (entity,mention) pair
                    num_valid_hyperlinks += 1
                    if not wiki_e_m_counts["mention"]: wiki_e_m_counts["mention"] = dict()
                    if not wiki_e_m_counts["mention"]["ent_wikiid"]: wiki_e_m_counts[mention][ent_wikiid] = 0
                    wiki_e_m_counts["mention"]["ent_wikiid"] += 1
                

    