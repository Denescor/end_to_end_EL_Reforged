#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 10:05:06 2021

@author: carpentier
"""

import utils.utils as utils
import data_gen.indexes.wiki_redirects_index as wiki_redirects_index
import entities.ent_name2id_freq.ent_name_id as ent_name_id

wiki_redirects_index = wiki_redirects_index.dofile()
e_id_name, unk_ent_thid = ent_name_id.dofile(args)
unk_ent_wikiid = 1

def findLUA(string,sub,start=0,end=0):
    bg = string.find(sub,start,end)
    if bg != -1:
        return bg, bg+len(sub)
    else:
        return False, False

def extract_text_and_hyp(line, mark_mentions):
    list_hyp = [] # (mention, entity) pairs
    text = ""
    list_ent_errors = 0
    parsing_errors = 0
    disambiguation_ent_errors = 0
    diez_ent_errors = 0
  
    end_end_hyp = 0
    begin_end_hyp = 0
    begin_start_hyp, end_start_hyp = findLUA(line,'<a href="')
  
    num_mentions = 0
    
    while begin_start_hyp is not False:
        text = text+line[end_end_hyp+1:begin_start_hyp]
        try :
            next_quotes,end_quotes = findLUA(line,'">', start=end_start_hyp + 1)
            ent_name = line[end_start_hyp + 1, next_quotes - 1]
            begin_end_hyp, end_end_hyp = findLUA(line,'</a>', start=end_quotes + 1)
            if begin_end_hyp :
                mention = line[end_quotes + 1, begin_end_hyp - 1]
                mention_marker = False
                
                good_mention = True
                good_mention = good_mention and ("Wikipedia" not in mention)
                good_mention = good_mention and ("wikipedia" not in mention)
                good_mention = good_mention and (len(mention) >= 1)
        
                if good_mention :
                    i,_ = findLUA(ent_name,'wikt:')
                    if i  is not False : # i not False
                        ent_name = ent_name[6:]
                    ent_name = ent_name_id.preprocess_ent_name(ent_name)
    
                    i,_ = findLUA(ent_name,'List of ')
                    if i is False : #i False
                        if findLUA(ent_name,'#')[0] is not False : diez_ent_errors += 1
                        else :
                            ent_wikiid = ent_name_id.get_ent_wikiid_from_name(ent_name, True)
                            if ent_wikiid == unk_ent_wikiid : disambiguation_ent_errors += 1
                            else :
                                # A valid (entity,mention) pair
                                num_mentions += 1
                                list_hyp.append(dict([("mention",mention),("ent_wikiid",ent_wikiid),("cnt",num_mentions)]))
                                if mark_mentions : mention_marker = True
                    else: list_ent_errors += 1
            
                if (not mention_marker) : text = "{} {}".format(text, mention)
                else: text = "{} MMSTART{} {} MMEND{}".format(text, num_mentions, mention, num_mentions)
            else :
                parsing_errors += 1
                begin_start_hyp = False
        except ValueError:
            parsing_errors += 1
            begin_start_hyp = False
        if begin_start_hyp is not False: begin_start_hyp, end_start_hyp = findLUA(line,'<a href="', start=end_start_hyp + 1) 
    if end_end_hyp is not False:
        text = text + line[(end_end_hyp + 1):]
    else:
        if (not mark_mentions) :
            text = line # Parsing did not succed, but we don't throw this line away.
        else:
            text = ""
            list_hyp = dict()
    return list_hyp, text, list_ent_errors, parsing_errors, disambiguation_ent_errors, diez_ent_errors     

def extract_page_entity_title(line):
    startoff, endoff = findLUA(line,'<doc id="')
    assert startoff is not False, line
    startquotes, _ = findLUA(line,'"', start=endoff + 1)
    ent_wikiid = int(line[endoff + 1, startquotes - 1])
    assert ent_wikiid is int, '{} ==> {}'.format(line, line[startoff + 1, startquotes - 1])
    starttitlestartoff, starttitleendoff = findLUA(line,' title="')
    endtitleoff, _ = findLUA(line,'">')
    ent_name = line[starttitleendoff + 1, endtitleoff - 1]
    if (ent_wikiid != ent_name_id.get_ent_wikiid_from_name(ent_name, True)):
        # Most probably this is a disambiguation or list page
        new_ent_wikiid = ent_name_id.get_ent_wikiid_from_name(ent_name, True)
        #print(red('Error in Wiki dump: ' .. line .. ' ' .. ent_wikiid .. ' ' .. new_ent_wikiid))
        return new_ent_wikiid
    return ent_wikiid

if __name__== "__main__":
    #  ----------------------------- Unit tests -------------
    print('\n Unit tests:')
    test_line_1 = '<a href="Anarchism">Anarchism</a> is a <a href="political philosophy">political philosophy</a> that advocates<a href="stateless society">stateless societies</a>often defined as <a href="self-governance">self-governed</a> voluntary institutions, but that several authors have defined as more specific institutions based on non-<a href="Hierarchy">hierarchical</a> <a href="Free association (communism and anarchism)">free associations</a>..<a href="Anarchism">Anarchism</a>'
    
    test_line_2 = 'CSF pressure, as measured by <a href="lumbar puncture">lumbar puncture</a> (LP), is 10-18 <a href="Pressure#H2O">'
    test_line_3 = 'Anarchism'
    
    list_hype, text = extract_text_and_hyp(test_line_1, False)
    print(list_hype)
    print(text)
    print("")
    
    list_hype, text = extract_text_and_hyp(test_line_1, True)
    print(list_hype)
    print(text)
    print("")
    
    list_hype, text = extract_text_and_hyp(test_line_2, True)
    print(list_hype)
    print(text)
    print()
    
    list_hype, text = extract_text_and_hyp(test_line_3, False)
    print(list_hype)
    print(text)
    print()
    print('    Done unit tests.')
    # ---------------------------------------------------------
    
    test_line_4 = '<doc id="12" url="http://en.wikipedia.org/wiki?curid=12" title="Anarchism">'
    
    print(extract_page_entity_title(test_line_4))