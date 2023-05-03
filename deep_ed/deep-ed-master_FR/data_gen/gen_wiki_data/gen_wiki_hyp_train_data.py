#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  8 15:13:48 2021

@author: carpentier
"""

# -- Generate training data from Wikipedia hyperlinks by keeping the context and
# -- entity candidates for each hyperlink

# -- Format: 
# -- ent_wikiid \t ent_name \t mention \t left_ctxt \t right_ctxt \t CANDIDATES \t [ent_wikiid,p_e_m,ent_name]+ \t GT: \t pos,ent_wikiid,p_e_m,ent_name

import torch
import argparse
import utils.utils as utils
import data_gen.parse_wiki_dump.parse_wiki_dump_tools as parse_wiki_dump_tools
import data_gen.indexes.yago_crosswikis_wiki as yago_crosswikis_wiki
import entities.ent_name2id_freq.ent_name_id as ent_name_id

print('\nGenerating training data from Wiki dump')

parser = argparse.ArgumentParser()
parser.add_argument("--root_data_dir",defaut="./",help="Root Path of the data : $DATA_PATH")
#parser.add_argument("--redirect_file",defaut="wiki_redirects.txt")

args = parser.parse_args()
args.anchors_file = "textWithAnchorsFromAllWikipedia2014Feb.txt"
args.wiki_hyperlink_contexts = "wiki_hyperlink_contexts.txt"
unk_ent_wikiid = -1

num_lines = 0
num_valid_hyp = 0

cur_ent_wikiid = -1

with open(args.root_data_dir+"/basic_data/"+args.anchors_file,"r") as f:
    with open(args.root_data_dir+"/generated/"+args.wiki_hyperlink_contexts,"w") as ouf:
        for line in f:
            cur_words_num = 0
            cur_words = []
            cur_mentions = []
            cur_mentions_num = 0
            num_lines = num_lines + 1
            if num_lines % 1000000 == 0 : print('Processed {} lines. Num valid hyp = {}'.format(num_lines,num_valid_hyp))

            #-- If it's a line from the Wiki page, add its text words and its hyperlinks
            if (not utils.findLUA_bool(line,'<doc id="')) and (not utils.findLUA_bool(line,'</doc>')) :
                list_hyp, text, _, _ , _ , _ = parse_wiki_dump_tools.extract_text_and_hyp(line, True)
       
                words_on_this_line = utils.split_in_words(text)
                num_added_hyp = 0
                line_mentions = dict()
                for w in words_on_this_line :
                    wstart = utils.string_starts(w, 'MMSTART')
                    wend = utils.string_starts(w, 'MMEND')
                    if (not wstart) and (not wend) :
                        cur_words.append(w)
                        cur_words_num = cur_words_num + 1
                    elif wstart :
                        mention_idx = int(w[len('MMSTART'):])
                        #assert mention_idx, w #TODO Que fait ce assert ?
                        line_mentions[mention_idx] = dict([("start_off",cur_words_num + 1), ("end_off",-1)])
                    elif wend :
                        num_added_hyp = num_added_hyp + 1
                        mention_idx = int(w[len('MMEND'):])
                        #assert mention_idx, w #TODO Que fait ce assert ?
                        assert line_mentions[mention_idx]
                        line_mentions[mention_idx]["end_off"] = cur_words_num
    
                assert len(list_hyp) == num_added_hyp, "{} :: {} :: {} {}".format(line, text, num_added_hyp, len(list_hyp))
                for hyp in list_hyp :
                    assert hyp["cnt"] in line_mentions
                    cur_mentions_num += 1
                    cur_mentions[cur_mentions_num] = dict()
                    cur_mentions[cur_mentions_num]["mention"] = hyp["mention"]
                    cur_mentions[cur_mentions_num]["ent_wikiid"] = hyp["ent_wikiid"]
                    cur_mentions[cur_mentions_num]["start_off"] = line_mentions[hyp["cnt"]]["start_off"]
                    cur_mentions[cur_mentions_num]["end_off"] = line_mentions[hyp["cnt"]]["end_off"]
                
            elif utils.findLUA_bool(line,'<doc id="') :
    
                #-- Write results:
                if cur_ent_wikiid != unk_ent_wikiid and ent_name_id.is_valid_ent(cur_ent_wikiid) :
                    header = "{}\t{}\t".format(cur_ent_wikiid, ent_name_id.get_ent_name_from_wikiid(cur_ent_wikiid))
                for hyp in cur_mentions :
                    if ent_p_e_m_index[hyp["mention"]] and len(ent_p_e_m_index[hyp["mention"]]) > 0 : #TODO trouver lib de "ent_p_e_m_index"
                        assert len(hyp["mention"]) > 0, line
                        string = "{}{}\t".format(header, hyp["mention"])
    
                        left_ctxt = []
                        for i in range(max(0, hyp["start_off"] - 100), hyp["start_off"] - 1) :
                            left_ctxt.append(cur_words[i])
                        if len(left_ctxt) == 0 :
                            left_ctxt.append('EMPTYCTXT')
                        string = "{}{}\t".format(string, " ".join(left_ctxt))
    
                        right_ctxt = []
                        for i in range(hyp["end_off"] + 1, min(cur_words_num, hyp["end_off"] + 100)) :
                            right_ctxt.append(cur_words[i])
                        if len(right_ctxt) == 0 :
                            right_ctxt.append('EMPTYCTXT')
                        string = "{}{}\tCANDIDATES\t".format(string, " ".join(right_ctxt))
              
                        #-- Entity candidates from p(e|m) dictionary
                        unsorted_cand = []
                        for ent_wikiid,p in ent_p_e_m_index[hyp["mention"]].items() : #TODO trouver lib de "ent_p_e_m_index"
                            unsorted_cand.append(dict([("ent_wikiid",ent_wikiid), ("p",p)]))
                        unsorted_cand.sort(unsorted_cand, lambda a: a["p"], reverse=True)
              
                        candidates = []
                        gt_pos = -1
                        for pos,e in enumerate(unsorted_cand) :
                            if pos <= 32 :
                                candidates.append("{},{:.3f},{}".format(e["ent_wikiid"],["p"],ent_name_id.get_ent_name_from_wikiid(e["ent_wikiid"])))
                                if e["ent_wikiid"] == hyp["ent_wikiid"] : gt_pos = pos
                            else: break
                        string = "{}{}\tGT:\t".format(string, "\t".join(candidates))
              
                        if gt_pos > 0 :
                            num_valid_hyp += 1
                            ouf.write("{}{},{}\n".format(string, gt_pos, candidates[gt_pos]))
    
            cur_ent_wikiid = parse_wiki_dump_tools.extract_page_entity_title(line)  
            
print('Done generating training data from Wiki dump. Num valid hyp = {}'.format(num_valid_hyp))