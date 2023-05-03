import os
import sys
import copy
import time
import argparse
import numpy as np
import operator
import pickle as pk
from functools import reduce
from collections import Counter
from statistics import median

import tensorflow as tf2
import tensorflow.compat.v1 as tf
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics.pairwise import cosine_similarity

import model.util as util
import model.train as train
import model.fun_eval as fun_eval
from evaluation.metrics import _filtered_spans_and_gm_gt_list
import plotting.tsne as tsne

"""
#############################################################################################################################################################################################
# FONCTIONS POUR EXTRAIRE LES ENTITÉS ET LES EMBEDDINGS DE MOTS ET FAIRE LES GRAPHES T-SNE DE VOISINAGE SÉMANTIQUES DES ENTITÉS                                                             #
#                                                                                                                                                                                           #
# ENTITIES EXTRACTION                                                                                                                                                                       #
# - entities_discrimination_tensorboard : MAIN FONCTION LANÇANT LES EXTRACTIONS. PEUT ÉGALEMENT CONSTRUIRE UNE REPRÉSENTATION POUR TENSORBOARD.                                             #
# - extract_entities_embeddings : FONCTION LANÇANT LES MODÈLES ET EXTRAYANT LES ENTITÉS SELON SI ELLES SONT BIEN PRÉDITES OU PAS                                                            #
# - create_mask : FONCTION QUI SELON LE FILTRE "BIEN PRÉDIT"/"MAL PRÉDIT" SÉLECTIONNE LES ENTITÉS À EXTRAIRE                                                                                #
#                                                                                                                                                                                           #
# EMBEDDINGS EXTRACTION                                                                                                                                                                     #
# - entities_discrimination_viewWord : MAIN FONCTION LANÇANT LES EXTRACTIONS NÉCESSAIRES OU CHARGEANT CELLES PRÉEXISTANTES. CONSTRUITS ÉGALEMENT LES ENSEMBLES D'ENTITÉS PERTINENTS         #
# - extract_word_embeddings : FONCTION LANÇANT LES MODÈLES ET EXTRAYANT LES EMBEDDINGS DEMANDÉS (CONTEXTE, MOT, MENTION)                                                                    #
#                                                                                                                                                                                           #
# GRAPHS GENERATION                                                                                                                                                                         #
# - generating_graphs : FONCTION QUI A PARTIR DES ENSEMBLES D'ENTITÉS CONSTRUITS LES GROUPES DE POINTS À REPRÉSENTER. PRÉPARE ÉGALEMENT LE MODÈLE T-SNE ET PRÉPARE LES LÉGENDES             #
# - generate_graph_word : FONCTION QUI PRÉPARE LES POINTS À AFFICHER ET LANCE LE T-SNE AVANT DE DONNER SES RÉSULTATS À LA FONCTION S'OCCUPANT D'EFFECTUER CONCRÈTEMENT LE PLOT              #
# - compute_cosine_similarity : CALCULE LE COSINUS SIMILARITÉ ENTRE UN VECTEUR SPÉCIFIQUE ET TOUS LES VECTEURS D'UN ENSEMBLE DONNÉ. RENVOIE LES "topn" RÉSULTATS LES PLUS PROCHES           #
#                                                                                                                                                                                           #
# OCCURENCES COMPUTATION                                                                                                                                                                    #
# - occ_entities_alldataset : FONCTION QUI À PARTIR DES EXTRACTIONS ENREGISTRÉES CALCULENT LES OCCURENCES DES ENTITÉS EN FONCTION DES DIFFÉRENTS DATASET ET LES RENVOIES DANS LA CONSOLE.   #
#############################################################################################################################################################################################
"""

def compute_cosine_similarity(entity,words,metadata,topn=10):
    """
    INPUT :
        - entity : vecteur de comparaison
        - words : list of vecteurs de même dimension
        - metadata : list of string de même dimension
    OUTPUTS :
        - list of tuple (vecteur, string). Ce sont les "topn" avec la similarité cosinus la plus élevée
    """
    #print("entity : {}\nwords : {}".format(np.shape([entity]), np.shape(words)))
    sim = cosine_similarity(words,[entity])
    sim = [(sim[x],np.array(words[x]),x) for x in range(len(sim))]
    sim.sort(reverse=True, key=lambda sim: sim[0])
    words_list = np.array(words)
    #print("sim :\n\ttype : {}\n\tlen : {}".format(type(sim),len(sim)))
    #print("words:\n\ttype : {}\n\tlen : {}".format(type(words),len(words)))
    #print("words_list:\n\ttype : {}\n\tlen : {}".format(type(words_list),len(words_list)))
    #print("meta:\n\ttype : {}\n\tlen : {}".format(type(metadata),len(metadata)))
    meta_sim = [metadata[x[2]] for x in sim]
    return [(sim[x][1], meta_sim[x]) for x in range(topn)] #sim[:topn], meta_sim[:topn]

def create_mask(model, result_l, args_for_load, opt_thr, test_handle, search_tp = False, verbose=True):
    """
    INPUT : 
        - model : le modèle chargé
        - result_l : liste des données du modèles (cf. "retrieve_l" in "extract_entities_embeddings")
        - args_for_load : arguments permettant le chargement des dictionnaires
        - opt_th : seuil minimal de validité du score. En dessous, tout résultat est considéré comme faux
        - test_handle : handle du dataset
        - search_tp : True si le critère est de sélectionner les entités bien prédites. False sinon
    OUTPUTS :
        - mask : list of tuple (doc_num, 
                                indices de l'embeddings dans les structures,
                                indices de début du span dans le document,
                                score de l'entité prédite,
                                entité de référence -- celle qui doit être prédite)
    """
    idtocand, id2word, id2char = fun_eval.load_data(args_for_load, verbose=verbose)
    test_index = 0
    # Format de result_l
    #    - final_scores, cand_entities_len, cand_entities
    #    - begin_span, end_span, spans_len
    #    - begin_gm, end_gm, ground_truth, ground_truth_len,
    #    - words_len, words, chars, chars_len, chunk_id
    cand_entities = result_l[2]
    mask = []
    for doc_num in range(result_l[0].shape[0]):
        ## Etape 1 : récupération des couples (mention, meilleur mention) et (gold mention, ground truth) pour chaque document
        filtered_spans, gm_gt_list = _filtered_spans_and_gm_gt_list(doc_num, result_l[0], result_l[1], result_l[2], result_l[3], result_l[4], 
                                                                            result_l[5], result_l[6], result_l[7], result_l[8], result_l[9], result_l[10],
                                                                            indice = True)
        
        # Format des structures :
        #     - filtered_spans : [(best_cand_score, begin_idx, end_idx, best_cand_id), ...]
        #     - gm_gt_list : [(begin_gm, end_gm, ground_truth), ...]
                                                                    
        ## Etape 2 : conversion des ids en mots
        _, chunk_words, _ = fun_eval.reconstruct_chunk_word(result_l, doc_num, id2word, id2char)
        # Récupération des True Positifs, False Positifs & False Entities
        for span in filtered_spans:
            (best_cand_score, begin_idx, _, _), (indiceI, indiceJ) = span
            span_list, best_cand, _ = fun_eval.reconstruct_span(span[0], chunk_words, idtocand)
            for (bgm, egm, gt) in gm_gt_list:
                is_tp, _, _, find, _, _ = fun_eval.reconstruct_true_positif(span[0], (bgm, egm, gt), chunk_words, "", idtocand, best_cand, opt_thr, False)
                if (find) and (search_tp == is_tp) : mask.append((doc_num,indiceI,indiceJ,begin_idx,best_cand_score,gt)) # mask.append((doc_num,indiceI,indiceJ,best_cand_score,gt))
    return mask
    
    
def extract_entities_embeddings(experiment_name, training_name, dataset="test", emb_t=1, tp=False, reshape=False, wikiid2nnid_name="wikiid2nnid.txt"):
    """
    INPUT :
        - experiment_name : expérience étudiée
        - training_name : modèle de l'expérience à charger
        - dataset : dataset que l'on souhaite extraire parmi "train", "test" et "dev" (DEFAUT : "test")
        - emb_t : type d'embeddings à extraire (DEFAUT : 1)
                  0 : embeddings de mots
                  1 : embeddings d'entité
                  2 : embeddings de contexte
                  3 : embeddings de mention
        - tp : True si le critère est de sélectionner les entités bien prédites. False sinon
    OUTPUT :
        - mask_entities : list of tuple de listes de taille identique :
            - entities_final : les embeddings extrait
            - metadata_final : le label des embeddings ("1er mot de la mention | entité prédite | entité de référence")
            - best_score_final : score de similarité associé à l'entité prédite
        - eval_args : arguments permettant le chargement du modèle pour évaluation (générés par la fonction)
    """
    ## INIT DATA
    entities_final = []
    metadata_final = []
    best_score_final = []
    ## RETRIEVE THE MODEL
    (datasets, names, model), eval_args, _ = fun_eval.retrieve_model_args(experiment_name, training_name, wikiid2nnid_name=wikiid2nnid_name)
    retrieve_l = [model.final_scores, model.cand_entities_len, model.cand_entities,
                  model.begin_span, model.end_span, model.spans_len,
                  model.begin_gm, model.end_gm,
                  model.ground_truth, model.ground_truth_len,
                  model.words_len, model.words, model.chars, model.chars_len, model.chunk_id,
                  model.tf_param_summaries, model.projector_embeddings]
    ## RUN THE MODEL
    model.restore_session("el")
    train.args.context_bert_lstm = model.args.context_bert_lstm
    test_iterator, test_handle = fun_eval.ed_el_testset_handles(model.sess, names, datasets, data_to_pick=dataset)
    val_iterators, val_handles = fun_eval.ed_el_dataset_handles(model.sess, datasets)
    opt_thr, _ = train.optimal_thr_calc(model, val_handles, val_iterators, True)
    #raise KeyboardInterrupt
    model.sess.run(test_iterator.initializer)
    size_it = 0
    keyerror = 0
    keytotal = 0
    eval_args = fun_eval.eval_parsing()
    eval_args.experiment_name = experiment_name
    eval_args.training_name = training_name
    eval_args.wikiid2nnid_name = wikiid2nnid_name
    idtocand, id2word, id2char = fun_eval.load_data(eval_args, verbose=False)
    while True:
        try :
            result_l = model.sess.run(retrieve_l, feed_dict={model.input_handle_ph: test_handle, model.dropout: 1})
            projectors = result_l[-1] #les différents embeddings à extraire
            if size_it == 0:
                print("##### PROJECTOR : {}({}) of {} #####".format(type(projectors),len(projectors),type(projectors[0])))
                print("shape word : {}".format(np.shape(projectors[0])))
                print("shape entities : {}".format(np.shape(projectors[1])))
                print("shape context : {}".format(np.shape(projectors[2])))
                print("shape mention : {}".format(np.shape(projectors[3])))
                print("shape cand_entities : {}x{} of {} {} of {}".format(len(result_l[2]), len(result_l[2][0]), type(result_l[2][0]), np.shape(result_l[2][0]), type(result_l[2][0][0][0])))
                print("shape ground_truth : {}x{} of {} {} of {}".format(len(result_l[8]), len(result_l[8][0]), type(result_l[8][0]), np.shape(result_l[8][0]), type(result_l[8][0][0])))
                print("shape words : {}x{} of {} {} of {}".format(len(result_l[-6]), len(result_l[-6][0]), type(result_l[-6][0]), np.shape(result_l[-6][0]), type(result_l[-6][0][0])))
                print("shape chars : {}x{} of {} {} of {}".format(len(result_l[-5]), len(result_l[-5][0]), type(result_l[-5][0]), np.shape(result_l[-5][0]), type(result_l[-5][0][0])))
                print("shape chars_len : {}x{} of {} {} of {}".format(len(result_l[-4]), len(result_l[-4][0]), type(result_l[-4][0]), np.shape(result_l[-4][0]), type(result_l[-4][0][0])))
                print("##### ##### ##### ##### #####")
            modelChars = result_l[-5]
            modelCharsLen = result_l[-4]
            words = result_l[-6]
            cand_entities = result_l[2]
            ## MASK CREATION
            mask = create_mask(model, result_l[0:-2], eval_args, opt_thr, test_handle, search_tp=tp, verbose=False)
            ## APPLICATION DU MASK
            entities = projectors[emb_t] #depends of will
            if size_it == 0: print("type projectors : {}".format(type(entities)))
            metadata = [] 
            entities_temp = []
            metadata_temp = []
            best_score_temp = []
            #for x in range(len(cand_entities)):
            #    metadata.append(np.reshape(fun_eval.generate_cand_metadata(cand_entities[x],eval_args,details=False),np.shape(cand_entities[x])))
            for (x,i,j,bi,b,gt) in mask:
                if type(entities[x][bi]) == np.ndarray: entities_temp.append(entities[x][bi])#[j])
                else: entities_temp.append((entities[x][bi]).numpy)
                word = ""
                if words[x][bi] != 0: word = id2word[words[x][bi]]
                else : 
                    word_char = []
                    for y in range((modelCharsLen[x])[bi]):
                        word_char.append(id2char[(modelChars[x])[bi][y]])
                    word = ''.join(word_char)
                keytotal += 1
                try: 
                    if cand_entities[x][i][j] != 0 : cand = idtocand[cand_entities[x][i][j]]
                    else : cand = "<wunk>"
                except KeyError: 
                    cand = "<wunk>"
                    keyerror += 1
                try: metadata_temp.append("{} | {} | {}".format(word,cand,idtocand[gt])) #(word,idtocand[gt]))
                except KeyError: metadata_temp.append("{} | {} | <wunk>".format(word,cand)) #(word,"<wunk>"))
                best_score_temp.append(b)
            entities_final.extend(entities_temp)
            metadata_final.extend(metadata_temp)
            best_score_final.extend(best_score_temp)
            size_it += 1
        except tf.errors.OutOfRangeError as e: 
            print("End of Sequence (size : {})".format(size_it))
            break
        except Exception as e:
            print("unexpected error : '{}'".format(e))
            size_it += 1
            continue
    print("nb KeyError : {}/{} ({:.2f}%)".format(keyerror, keytotal, 100*(keyerror/keytotal)))
    print("Size entities : {} ({} {})".format( np.shape(entities_final), type(entities_final), len(np.shape(entities_final)) ))
    print("wunk final : {:.2f}%".format(100*(len([x for x in metadata_final if x.split("|")[-1] == "<wunk>"])/sum([len(x) for x in metadata_final]))))
    size_template = np.shape(entities_final[0])
    consistance = (sum([np.shape(x)==size_template for x in entities_final]) == len(entities_final))
    print("Is entities_final consistant : {}".format(consistance))
    if not consistance:
        dico_shape = dict()
        for entity in entities_final:
            shape = np.shape(entity)
            try: dico_shape[shape] += 1
            except KeyError: dico_shape[shape] = 1
        for shape, tot in dico_shape.items():
            print("\tshape '{}' : {}".format(shape, tot))
    ## FINAL RETURN
    if reshape and (len(np.shape(entities_final)) >= 3 or not consistance): 
        print("reshaping\nbefore : {}".format(np.shape(entities_final)))
        entities_final = [x[0] for x in entities_final]
        print("after : {}".format(np.shape(entities_final)))
    mask_entities = (entities_final,metadata_final,best_score_final)
    
    return mask_entities, eval_args
 
def extract_word_embeddings(experiment_name, training_name, dataset="test", emb_t="words", wikiid2nnid_name="wikiid2nnid.txt"):
    """
    INPUT :
        - experiment_name : expérience étudiée
        - training_name : modèle de l'expérience à charger
        - dataset : dataset que l'on souhaite extraire parmi "train", "test" et "dev" (DEFAUT : "test")
        - emb_t : type d'embeddings à extraire parmi "words", "context" et "mention" (DEFAUT : "words")
    OUTPUT :
        - tuple de listes de taille identique
            - words_embs_final : embeddings de mots extraits
            - metadata_final : label associé aux mots (le mot)
    """
    if emb_t=="context": emb_t_int = 2 #embedding de contexte
    elif emb_t=="words": emb_t_int = 0 #embedding de mot
    elif emb_t=="mention": emb_t_int = 3 #embedding de mention
    else: emb_t_int = 1 #embedding d'entités
    ## INIT DATA
    words_embs_final = []
    words_final = []
    metadata_final = []
    nb_keyerror = 0
    nb_tot = 0   
    if emb_t_int == 1: 
        (words_embs_final, metadata_final, _), _ = extract_entities_embeddings(experiment_name, training_name, dataset=dataset, emb_t=emb_t_int, tp=True,reshape=False,wikiid2nnid_name=wikiid2nnid_name)
        print("before:\nentities shape : {}\nmetadata shape : {}\n---- ---- ---- ----".format(np.shape(words_embs_final), np.shape(metadata_final)))
        words_embs_final = [x[0] for x in words_embs_final] #On prend le premier == le plus probable
        metadata_final = [x.split("|")[-1] for x in metadata_final] #On ne récupère que l'entité du label
        print("after reshape:\nentities shape : {}\nmetadata shape : {}".format(np.shape(words_embs_final), np.shape(metadata_final)))
    else:
        ## RETRIEVE THE MODEL
        (datasets, names, model), eval_args, _ = fun_eval.retrieve_model_args(experiment_name, training_name)
        retrieve_l = [model.words, model.words_len,
                      model.chars, model.chars_len, 
                      model.chunk_id,
                      model.tf_param_summaries, model.projector_embeddings]
        eval_args.wikiid2nnid_name = wikiid2nnid_name
        ## RUN THE MODEL
        model.restore_session("el")
        train.args.context_bert_lstm = model.args.context_bert_lstm
        test_iterator, test_handle = fun_eval.ed_el_testset_handles(model.sess, names, datasets, data_to_pick=dataset)
        val_iterators, val_handles = fun_eval.ed_el_dataset_handles(model.sess, datasets)
        opt_thr, _ = train.optimal_thr_calc(model, val_handles, val_iterators, True)
        model.sess.run(test_iterator.initializer)
        _, id2word, id2char = fun_eval.load_data(eval_args, verbose=False)
        size_it = 0
        while True:
            try :
                result_l = model.sess.run(retrieve_l, feed_dict={model.input_handle_ph: test_handle, model.dropout: 1}) 
                projectors = result_l[-1]
                words = result_l[0]
                modelCharsLen = result_l[3]
                modelChars = result_l[2]
                if size_it == 0:
                    print("##### PROJECTOR : {}({}) of {} #####".format(type(projectors),len(projectors),type(projectors[0])))
                    print("shape word : {}".format(np.shape(projectors[0])))
                    print("shape entities : {}".format(np.shape(projectors[1])))
                    print("shape context : {}".format(np.shape(projectors[2])))
                    print("shape mention : {}".format(np.shape(projectors[3])))
                    print("##### ##### ##### ##### #####")
                ## CREATION DES METADATA
                words_vectors = projectors[emb_t_int] #Context Embeddings ==> 1 par mot, de même dimension que les entités
                metadata = []
                for x in range(len(words)):
                    for y in range(len(words[x])):
                        try: 
                            if words[x][y] != 0: word_meta = id2word[words[x][y]]
                            else:
                                word_char = []
                                for j in range((modelCharsLen[x])[y]):
                                    word_char.append(id2char[(modelChars[x])[y][j]])
                                word_meta = ''.join(word_char)
                        except KeyError: 
                            word_meta = "wunk [{}]".format(words[x][y])
                            nb_keyerror += 1
                        finally : 
                            metadata.append(word_meta)
                            nb_tot += 1
                metadata = np.reshape(metadata,np.shape(words))
                size_it += 1
                for i in range(len(words)):
                    words_embs_final.extend(words_vectors[i])
                    words_final.extend(words[i])
                    metadata_final.extend(metadata[i])
            except tf.errors.OutOfRangeError as e:
                print("End of Sequence (size : {})".format(size_it))
                print("Key Error : {}/{} ({:.2f}%)".format(nb_keyerror,nb_tot,100*(nb_keyerror/nb_tot)))
                break 
    return (words_embs_final, metadata_final)
    
def entities_discrimination_tensorboard(melting_args, summary_folder, use_old_data=False, dataset="test", emb_t="entity", tp=False, only_extract = True, verbose=True,wikiid2nnid_name="wikiid2nnid.txt"):
    """
    INPUT :
        - meltings_args : liste of tuple of string (nom de l'expérience, nom du modèle)
        - summary_folder : dossier de référence où sauver ou charger les embeddings extraits
        - use_old_data : True si on veut charger une extraction pré-enregistrée. False si on souhaite refaire l'extraction des embeddings
        - dataset : dataset que l'on souhaite extraire parmi "train", "test" et "dev" (DEFAUT : "test")
        - emb_t : type d'embeddings à extraire parmi "entity", "words", "context" et "mention" (DEFAUT : "entity")
        - tp : True si le critère est de sélectionner les entités bien prédites. False sinon
        - only_extract : True si la fonction ne doit effectuer que l'extraction des embeddings et les sauver sans effectuer la projection. False sinon
        NB : use_old_data et only_extract ne peuvent pas être à True simulatanément. La fonction s'annule si c'est le cas.
    OUTPUT :
        - NONE : génère des fichiers pickle si demandés. Génère une projection pour tensorboard si demandé.
    """
    if emb_t=="context": emb_t_int = 2 #embedding de contexte
    elif emb_t=="words": emb_t_int = 0 #embedding de mot
    elif emb_t=="mention": emb_t_int = 3 #embedding de mention
    else: emb_t_int = 1 #embedding d'entités
    if use_old_data and only_extract:
        print("les options choisies rendent la fonction inutile => ANNULATION")
        return None
    final_emb_summary = SummaryWriter(summary_folder + 'global_embedding_projector/masked_entities/')
    if tp: print("POSITIF ENTITIES")
    else : print("FALSE ENTITIES")
    if use_old_data : 
        print("LOADING DATA... ",end="")
        if tp: tp_text = "tp"
        else : tp_text = "fp"
        names = pk.load(open(summary_folder+"{}_embeddings_extraction/".format(emb_t)+"masked_names_names_{}_{}.pk".format(tp_text,dataset),"rb"))
        embs = pk.load(open(summary_folder+"{}_embeddings_extraction/".format(emb_t)+"masked_embs_embs_{}_{}.pk".format(tp_text,dataset),"rb"))
        print("DONE")
    else : 
        names = []
        embs = []
        print("GENERATING AND SAVING DATA...")
        for experiment_name, training_name in melting_args: 
            entities_masked, _ = extract_entities_embeddings(experiment_name,training_name,dataset=dataset,emb_t=emb_t_int,tp=tp,reshape=True,wikiid2nnid_name=wikiid2nnid_name)
            tf.reset_default_graph()
            names.append("{};{}".format(experiment_name,training_name))
            embs.append(entities_masked)
        if tp: tp_text = "tp"
        else : tp_text = "fp"
        if not os.path.exists(summary_folder+"{}_embeddings_extraction/".format(emb_t)): os.makedirs(summary_folder+"{}_embeddings_extraction/".format(emb_t))
        pk.dump(names,open(summary_folder+"{}_embeddings_extraction/".format(emb_t)+"masked_names_names_{}_{}.pk".format(tp_text,dataset),"wb"))
        pk.dump(embs,open(summary_folder+"{}_embeddings_extraction/".format(emb_t)+"masked_names_embs_{}_{}.pk".format(tp_text,dataset),"wb"))
        print("DONE")
    ## PRINT DATA STATS
    print("data about embeddings :")
    print("names:\n\tlen : {}\n\ttype : {}".format(len(names),type(names)))
    print("embs:\n\tlen : {}x{}\n\ttype : {}x{}".format(len(embs),len(embs[0]),type(embs),type(embs[0])))
    print("tuple 1:\n\tlen : {}\n\ttype : {}".format(np.shape(embs[0][0]),type(embs[0][0][0])))
    print("tuple 2:\n\tlen : {}\n\ttype : {}".format(np.shape(embs[0][1]),type(embs[0][1][0])))
    if only_extract: return None
    ## GENERATE TENSORBOARD PROJECTIONS
    labels = "entities_embeddings"
    for j in range(len(embs[0])) : # on a un batch de taille 4
        step_zero = 10*j
        for x in range(len(embs)):
            if verbose : 
                print("Modèle {} ({}) - batch {}".format(x,names[x],j))
                print("entities summary :\n\tentities embs : {}\n\tentities labels : {}".format(len(embs[x][0]),len(embs[x][1])))
            step = step_zero + x
            label = names[x]+"_"+labels+"_{}".format(j)
            try :final_emb_summary.add_embedding(np.array(embs[x][0]), tag=label, 
                                            metadata=embs[x][1],
                                            global_step=step)
            except Exception as e: 
                print(e)
                continue
            else:
                if verbose : print("done")

def entities_discrimination_viewWord(melting_args, summary_folder, tp=False, dataset="test", emb_t="entity", use_old_data=False, verbose=True, wikiid2nnid_name="wikiid2nnid.txt"):
    """
    INPUT :
        - meltings_args : liste of tuple of string (nom de l'expérience, nom du modèle)
        - summary_folder : dossier de référence où sauver ou charger les embeddings extraits ainsi que les plots
        - tp : True si le critère est de sélectionner les entités bien prédites. False sinon
        - dataset : dataset que l'on souhaite extraire parmi "train", "test" et "dev" (DEFAUT : "test")
        - emb_t : type d'embeddings à extraire parmi "entity", "words", "context" et "mention" (DEFAUT : "entity")
        - use_old_data : True si on veut charger une extraction pré-enregistrée. False si on souhaite refaire l'extraction des embeddings
    OUTPUT :
        - entities : list of dictionnaire (entité --> embedding correspondant)
        - scores : list of tuple (score de similarité, label de l'entité fractionner -- 1er mot du span, entité prédite, entité de référence) trié par ordre décroissant selon le score
        - scores_reverse : list "scores" triée dans le sens inverse
        - identic_entities : ensemble des entités prédites présentes dans tous les modèles
        - diff_entities : ensemble des entités prédites qui sont absentes d'au moins 1 modèle, regroupé par modèle
    """
    emb_type = "{}_embeddings".format(emb_t) # "words_embeddings" # "mention_embeddings"
    if tp: 
        tp_folder = "true"
        tp_text = "tp"
    else : 
        tp_folder = "wrong"
        tp_text = "fp"
    final_emb_summary = SummaryWriter(summary_folder + "global_embedding_projector/masked_entities/")
    if use_old_data : 
        print("LOADING DATA... ",end="")
        names = pk.load(open(summary_folder+"{}_extraction/".format(emb_type)+"masked_names_names_{}_{}.pk".format(tp_text,dataset),"rb"))#[:2]
        embs = pk.load(open(summary_folder+"{}_extraction/".format(emb_type)+"masked_names_embs_{}_{}.pk".format(tp_text,dataset),"rb"))#[:2]
        (words_vectors,metadata_words) = pk.load(open(summary_folder+"{}_extraction/".format(emb_type)+"words_vectors_{}.pk".format(dataset),"rb"))
        #words_vectors = words_vectors[:2]
        #metadata_words = metadata_words[:2]
        print("DONE")
    else : 
        names = []
        embs = []
        words_vectors = []
        metadata_words = []
        print("GENERATING AND SAVING DATA...")
        for experiment_name, training_name in melting_args: 
            words_embs_final, metadata_final = extract_word_embeddings(experiment_name, training_name, emb_t=emb_t, wikiid2nnid_name=wikiid2nnid_name)
            tf.reset_default_graph()
            words_vectors.append(words_embs_final)
            metadata_words.append(metadata_final)
        if not os.path.exists(summary_folder+"{}_extraction/".format(emb_t)): os.makedirs(summary_folder+"{}_extraction/".format(emb_t))
        names = pk.load(open(summary_folder+"{}_extraction/".format(emb_type)+"masked_names_names_{}_{}.pk".format(tp_text,dataset),"rb"))
        embs = pk.load(open(summary_folder+"{}_extraction/".format(emb_type)+"masked_names_embs_{}_{}.pk".format(tp_text,dataset),"rb"))
        pk.dump((words_vectors,metadata_words),open(summary_folder+"{}_extraction/".format(emb_type)+"words_vectors_{}.pk".format(dataset),"wb"))
        print("DONE")
    ## PRINT DATA STATS
    print("data about embeddings :")
    print("names:\n\tlen : {}\n\ttype : {} of {}".format(len(names),type(names),type(names[0])))
    print("embs:\n\tlen : {}x{}\n\ttype : {}x{}".format(len(embs),len(embs[0]),type(embs),type(embs[0])))
    print("entities : {}\nmeta : {}x{}\nscore : {}".format(len(embs[0][0]),len(embs[0][1]),len(embs[0][1][0].split("|")),len(embs[0][2])))
    print("words_vectors:\n\tlen : {}x{}\n\ttype : {}".format(np.shape(words_vectors),np.shape(words_vectors[0][0]),type(words_vectors)))
    print("metadata_words:\n\tlen : {}\n\ttype : {}".format(np.shape(metadata_words),type(metadata_words)))
    #return None #### NONE STOP ####
    ### ### ### ### ### ### ### ### ### ### #
    #   x : model                           #
    #   entité (0) / meta (1) / score (2)   #
    #   i : sample                          #
    ### ### ### ### ### ### ### ### ### ### #
    ## REORGANISE ENTITIES DATA
    # dictionnaire : nom de l'entité --> embedding de l'entité
    lenx = len(embs)
    iWord = 0
    iCand = 1
    iEnt = 2
    entities = [dict([(embs[x][1][i].split(" | ")[iEnt],embs[x][0][i]) for i in range(len(embs[x][0]))]) for x in range(lenx)]
    # liste trié par score : (score de l'entité, embedding de l'entité)
    scores_set = set()
    scores = [(embs[x][2][i],embs[x][1][i].split(" | ")) for x in range(lenx) for i in range(len(embs[x][1])) if (embs[x][1][i].split(" | ")[iEnt] not in scores_set) and (scores_set.add(embs[x][1][i].split(" | ")[-1]) or True)]
    scores_occ = [Counter([embs[x][1][i].split(" | ")[iEnt] for i in range(len(embs[x][1]))]) for x in range(lenx)]    
    scores.sort(reverse=True, key = lambda x : x[0])
    scores_reverse = copy.deepcopy(scores)
    scores_reverse.reverse()
    assert scores != scores_reverse, "bad copy of score"
    identic_ent = len([x for x in scores if x[1][iCand] == x[1][iEnt]])
    identic_ent_all = [[x for i in range(len(embs[x][1])) if (embs[x][1][i].split(" | ")[iCand] == embs[x][1][i].split(" | ")[iEnt])] for x in range(lenx)]
    print("scores organisation:\n\tentities : {}x{}\n\tscores_set : {}\n\tscores : :{}".format(len(entities),[len(entities[x]) for x in range(lenx)],len(scores_set),len(scores)))
    print("\tidentic prediction / entities all : {}/{} ({:.2f}%)".format(identic_ent,len(scores),100*(identic_ent/len(scores))))
    print("\tidentic prediction / entities by model :")
    for id_ent_all in identic_ent_all:
        x = id_ent_all[0]
        print("\t\t{} : {}/{} ({:.2f}%)".format(names[x],len(id_ent_all),len(embs[x][0][1]),100*(len(id_ent_all)/len(embs[x][0][1]))))
    ## GENERATE ENTITIES SETS
    identic_entities = set([embs[0][1][i].split(" | ")[iEnt] for i in range(len(embs[0][1]))])
    diff_entities = [set() for x in range(len(embs))]
    for x in range(lenx):
        identic_entities = identic_entities.intersection(set([embs[x][1][i].split(" | ")[iEnt] for i in range(len(embs[x][1]))]))
    for x in range(lenx):
        diff_entities[x] = set([embs[x][1][i].split(" | ")[iEnt] for i in range(len(embs[x][1]))]).difference(identic_entities)
    ## PRINT SETS STATS
    print("stats set:\n\tlen identic_entities : {}/{} ({:.2f}%)\n\tonly entities : {}".format(len(identic_entities),len(scores),100*(len(identic_entities)/len(scores)),[len(x) for x in diff_entities]))
    for x in range(lenx):
        for y in range(x+1,lenx):
            print("\t{} <-> {} : {}".format(names[x],names[y],len(diff_entities[x].intersection(diff_entities[y]))))
    for x in range(lenx):
        occ = list(scores_occ[x].values())
        som = sum(occ)
        med = median(occ)
        moy = som/len(entities[x])
        print("\tocc {} : somme = {} || médiane = {} || moyenne = {:.2f}".format(names[x],som,med,moy))
    #return None #### NONE STOP -- PREVENT GRAPH GENERATION ####
    ## WRITE SETS DATA
    with open(summary_folder+"occ_entities.txt","w") as dump:
        dump.write("Entities miss predict\n")
        dump.write(10*"########## "+"\n")
        i = 0
        for elt in scores_occ:
            dump.write("\t{}\n".format(elt))
            dump.write(10*"########## "+"\n")
    with open(summary_folder+"difference_entities.txt", "w") as dump:
        dump.write("Difference of miss prediction\n")
        dump.write(10*"########## "+"\n")
        dump.write("IDENTIC MISS PREDICTION ({}):\n".format(len(identic_entities)))
        for x in identic_entities:
            dump.write("\t- {}\n".format(x))
        for x in range(len(diff_entities)):
            dump.write(10*"########## "+"\n")
            dump.write("DEVIANCE OF {} ({})\n".format(names[x],len(diff_entities[x])))
            for i in diff_entities[x]:
                dump.write("\t- {}\n".format(i))
        dump.write(10*"########## "+"\nFIN")
    print("WRITTING SET FINISHED")
    ## CREATE GRAPHS
    return names, words_vectors, metadata_words, entities, scores, scores_reverse, identic_entities, diff_entities

def generating_graphs(names, words_vectors, metadata_words, entities, scores, scores_reverse, identic_entities, diff_entities, summary_folder, emb_t="entity", maxent=10, maxmot=10, tp=True):    
    """
    INPUT :
        - entities : list of dictionnaire (entité --> embedding correspondant)
        - scores : list of tuple (score de similarité, label de l'entité fractionner -- 1er mot du span, entité prédite, entité de référence) trié par ordre décroissant selon le score
        - scores_reverse : list "scores" triée dans le sens inverse
        - identic_entities : ensemble des entités prédites présentes dans tous les modèles
        - diff_entities : ensemble des entités prédites qui sont absentes d'au moins 1 modèle, regroupé par modèle
        - summary_folder : dossier de référence où sauver les plots
        - emb_t : type d'embeddings à extraire parmi "entity", "words", "context" et "mention" (DEFAUT : "entity")
        - maxent : nombre maximum d'entité à afficher sur le même graphe (DEFAUT : 10)
        - maxmot : nombre maximum de mots maximum à afficher par entité sur le même graphe (DEFAUT : 10)
        - tp : True si le critère est de sélectionner les entités bien prédites. False sinon
    OUTPUT :
        - NONE : génère des plots
    """
    iWord = 0
    iCand = 1
    iEnt = 2
    print("GENERATING GRAPHS")
    #print("shape vectors : {}\nshape entities : {}\nshape metadata : {}".format(np.shape(words_vectors), np.shape(entities), np.shape(metadata_words)))
    # Paramètres
    highscore_label = "entities with highest similarity scores"
    lowscore_label = "entities with lowest similarity scores"
    emb_type = "{}_embeddings".format(emb_t) # "words_embeddings" # "mention_embeddings"
    if tp:
        tp_folder = "true" 
        tp_label = "true entities"
    else :
        tp_folder = "wrong" 
        tp_label = "wrong entities"
    save_folder = summary_folder+"tsne_graph/{}/{}/".format(emb_type,tp_folder)
    if not os.path.exists(save_folder): os.makedirs(save_folder)
    tsne_model = tsne._TSNE(iteration=5000, component=2, learning_rate=10, perplexity=30)
    # entités communes
    set_label = "shared by all the models"
    keys = [x[1] for x in scores if x[1][iEnt] in identic_entities][:maxent] # les maxent entités communes les mieux notés
    label = " - {} - {} set\n{} {}\n{}".format(emb_type,dataset,tp_label,set_label,highscore_label)
    if(len(keys) > 0): generate_graph_word(keys, zip([i for i in range(len(names))], words_vectors, metadata_words, names), entities, tsne_model, label, save_folder+"identic_highscore", maxmot = maxmot)
    #return None #### NONE STOP
    keys = [x[1] for x in scores_reverse if x[1][iEnt] in identic_entities][:maxent] # les maxent entités communes les moins bien notés
    label = " - {} - {} set\n{} {}\n{}".format(emb_type,dataset,tp_label,set_label,lowscore_label)
    if(len(keys) > 0): generate_graph_word(keys, zip([i for i in range(len(names))], words_vectors, metadata_words, names), entities, tsne_model, label, save_folder+"identic_lowscore", maxmot = maxmot)
    # entités duo
    set_label = "shared by 2 models"
    duo = [(i,j) for i in range(len(diff_entities)) for j in range(i+1,len(diff_entities))]
    print("duo : {}".format(duo))
    for i,j in duo:
        set1, set2 = diff_entities[i],diff_entities[j]
        keys = [x[1] for x in scores if (x[1][iEnt] in set1.intersection(set2))][:maxent] # les maxent entités du duo les mieux notés
        print("set duo {},{} : {} ({})".format(i,j,len(set1.intersection(set2)),len(keys)))
        label = " - {} - {} set\n{} {}\n{}".format(emb_type,dataset,tp_label,set_label,highscore_label)
        if(len(keys) > 0): generate_graph_word(keys, [(x, words_vectors[x], metadata_words[x], names[x]) for x in (i,j)], entities, tsne_model, label, save_folder+"duo{}{}_highscore".format(i,j), maxmot = maxmot)
        keys = [x[1] for x in scores_reverse if (x[1][iEnt] in set1.intersection(set2))][:maxent] # les maxent entités du duo les moins bien notés
        label = " - {} - {} set\n{} {}\n{}".format(emb_type,dataset,tp_label,set_label,lowscore_label)
        if(len(keys) > 0): generate_graph_word(keys, [(x, words_vectors[x], metadata_words[x], names[x]) for x in (i,j)], entities, tsne_model, label, save_folder+"duo{}{}_lowscore".format(i,j), maxmot = maxmot)
    # entités seules
    set_label = "only present in one model"
    label = "{} {}".format(tp_label,set_label)
    for i in range(len(diff_entities)):
        notset = [diff_entities[x] for x in range(i-1,i-len(diff_entities),-1)]
        currentset = diff_entities[i]
        for sx in notset:
            currentset = currentset.difference(diff_entities[i].intersection(sx))
        #currentset = currentset.intersection(diff_entities[i])
        print("set only : {}".format(len(currentset)))
        keys = [x[1] for x in scores if (x[1][iEnt] in currentset)][:maxent] # les maxent entités du solo les mieux notés
        for key in keys:
            print("\t{} : {}".format(key,[x[0] for x in scores if x[1]==key][0]))
        label = " - {} - {} set\n{} {}\n{}".format(emb_type,dataset,tp_label,set_label,highscore_label)
        if(len(keys) > 0): generate_graph_word(keys, [(i, words_vectors[i], metadata_words[i], names[i])], entities, tsne_model, label, save_folder+"only_highscore", maxmot = maxmot)
        keys = [x[1] for x in scores_reverse if (x[1][iEnt] in currentset)][:maxent] # les maxent entités du solo les moins bien notés
        label = " - {} - {} set\n{} {}\n{}".format(emb_type,dataset,tp_label,set_label,lowscore_label)
        if(len(keys) > 0): generate_graph_word(keys, [(i, words_vectors[i], metadata_words[i], names[i])], entities, tsne_model, label, save_folder+"only_lowscore", maxmot = maxmot)
    print("GENERATING FINISHED")
    
def generate_graph_word(keys, words_data, entities_data, tsne_model_en_2d, label, save_file, maxmot=10):
    """
    INPUTS : 
        - keys : liste des entités à afficher
        - words_data : list of tuple :
            - l'indice du modèle
            - les embeddings de mots du modèle
            - les label associés aux embeddings de mots
            - le nom du modèle (utilisé pour la légende)
        - entities_data : list of dictionnaire (entité --> embedding correspondant)
        - tsne_model_en_2d : modèle tsne prêt à l'emploi
        - label : légende (partiel) pour le plot
        - save_file : nom (partiel) du fichier devant être sauvé
        - maxmot : nombre maximum de mots maximum à afficher par entité sur le même graphe (DEFAUT : 10)
    OUTPUT :
        - NONE : génère des plots
    """
    print("BEGIN {}... ".format(label),end="")
    sys.stdout.flush() 
    top = time.time()
    entities = entities_data
    emb_type = label.split("-")[1]
    for x, words_vectors, metadata_words, name in words_data:
        #print("words : {}\tentities : {}".format(np.shape(words_vectors[0])[0],np.shape(entities[keys[0]])[0]))
        #if np.shape(words_vectors[0]) != np.shape(entities[keys[0]]):
        #    output_size = np.shape(entities[keys[0]])[0]
        #    new_words_vectors = []
        #    for words_emb in words_vectors:
        #        output_weights = tf.get_variable("output_weights", [np.shape(words_emb)[1], output_size])#, initializer=output_weights_initializer)
        #        output_bias = tf.get_variable("output_bias", [output_size])
        #        outputs = tf.matmul(tf.constant(words_emb), output_weights) + output_bias
        #        new_words_vectors.append(outputs.numpy())
        #    words_vectors = new_words_vectors
        #try:
        embedding_clusters = []
        word_clusters = []
        for key in keys:
            key = key[-1] #entité
            embeddings = []
            words = []
            for similar_vector, similar_word in compute_cosine_similarity(entities[x][key],words_vectors,metadata_words,topn=maxmot):
                words.append(similar_word)
                embeddings.append(similar_vector)
            embedding_clusters.append(embeddings)
            word_clusters.append(words)
        embedding_clusters = np.array(embedding_clusters)
        n, m, k = embedding_clusters.shape
        embeddings_en_2d = np.array(tsne._TSNE_fit(embedding_clusters.reshape(n * m, k),tsne_model_en_2d)).reshape(n, m, 2)
        legend = "Modèle {}{}".format(name,label)
        tsne.tsne_plot_similar_words(keys, embeddings_en_2d, word_clusters, "{}_{}_tsne".format(save_file,name.replace("/","-")), legend)
        #except Exception as e:
        #    print("{} : {}".format(name,e))
    print("DONE IN {:.0f}s".format(time.time()-top))
    sys.stdout.flush() 
    
def occ_entities_alldataset(melting_args, summary_folder, emb_t="entity", verbose=True):
    """
    INPUT : 
        - meltings_args : liste of tuple of string (nom de l'expérience, nom du modèle)
        - summary_folder : dossier de référence où sauver les fichiers
        - emb_t : type d'embeddings à extraire parmi "entity", "words", "context" et "mention" (DEFAUT : "entity")
    OUTPUT : 
        - None : génère les stats dans la console et génère des fichiers texte avec le détails des occurences des entités
    """
    emb_type = "{}_embeddings".format(emb_t)
    dataset = dict()
    dataset["true"] = dict()
    dataset["wrong"] = dict()
    iWord = 0
    iCand = 1
    iEnt = 2
    ## Loads data
    for tp_text,tp_dict in [("tp","true"),("fp","wrong")]:
        for data in ["train","dev","test"]:
            print("LOADING DATA... ",end="")
            names = pk.load(open(summary_folder+"{}_extraction/".format(emb_type)+"masked_names_names_{}_{}.pk".format(tp_text,data),"rb"))#[:2]
            embs = pk.load(open(summary_folder+"{}_extraction/".format(emb_type)+"masked_names_embs_{}_{}.pk".format(tp_text,data),"rb"))#[:2]
            (words_vectors,metadata_words) = pk.load(open(summary_folder+"{}_extraction/".format(emb_type)+"words_vectors_{}.pk".format(data),"rb"))
            print("DONE")
            dataset[tp_dict][data] = [names,embs,words_vectors,metadata_words]
    ## Compact all entities
    lenx = len(dataset["true"]["train"][1])
    print("size embs : {}x".format(len(dataset["true"]["train"])),end="")
    print("{}x".format(lenx),end="")
    print("{}x".format(len(dataset["true"]["train"][1][0])),end="")
    print("{}".format(len(dataset["true"]["train"][1][0][0])),end="")
    print(" of {}; {}; {}".format(type(dataset["true"]["train"][1][0][0][0]),type(dataset["true"]["train"][1][0][0][1]),type(dataset["true"]["train"][1][0][0][2])))
    train_ent = [dataset["true"]["train"][1][x][1][i].split(" | ")[iEnt] 
                 for x in range(lenx)
                 for i in range(len(dataset["true"]["train"][1][x][1]))]+\
                 [dataset["wrong"]["train"][1][x][1][i].split(" | ")[iEnt] 
                 for x in range(lenx) 
                 for i in range(len(dataset["wrong"]["train"][1][x][1]))]
    dev_ent_true = [dataset["true"]["dev"][1][x][1][i].split(" | ")[iEnt] for x in range(lenx) for i in range(len(dataset["true"]["dev"][1][x][1]))]
    dev_ent_false = [dataset["wrong"]["dev"][1][x][1][i].split(" | ")[iEnt] for x in range(lenx) for i in range(len(dataset["wrong"]["dev"][1][x][1]))]
    test_ent_true = [dataset["true"]["test"][1][x][1][i].split(" | ")[iEnt] for x in range(lenx) for i in range(len(dataset["true"]["test"][1][x][1]))]
    test_ent_false = [dataset["wrong"]["test"][1][x][1][i].split(" | ")[iEnt] for x in range(lenx) for i in range(len(dataset["wrong"]["test"][1][x][1]))] 
    ## Create Entities Dictionnaries
    occ_train = dict(Counter(train_ent))
    occ_all_test = dict(Counter(test_ent_true+test_ent_false))
    occ_all_dev = dict(Counter(dev_ent_true+dev_ent_false))
    sum_test = len(test_ent_true)+len(test_ent_false)
    sum_dev = len(dev_ent_true)+len(dev_ent_false)
    sum_occ = len({**occ_train,**occ_all_test,**occ_all_dev})
    ## FALSE ENTITIES STATS
    for true,dev_ent,test_ent in [(True,dev_ent_true,test_ent_true),(False,dev_ent_false,test_ent_false)]:
        if true: true_text="true" 
        else: true_text="false"
        occ_dev = dict(Counter(dev_ent))
        occ_test = dict(Counter(test_ent))
        occ_test_v = list(occ_test.values())
        occ_test_k = list(occ_test.keys())
        occ_dev_v = list(occ_dev.values())
        occ_dev_k = list(occ_dev.keys())
        occ_test_train = [(occ_test_k[i],occ_test_v[i],occ_train[occ_test_k[i]]) for i in range(len(occ_test)) if occ_test_k[i] in occ_train]
        occ_test_train.sort(reverse=True, key = lambda x : x[2])
        occ_test_only = [(occ_test_k[i],occ_test_v[i]) for i in range(len(occ_test)) if occ_test_k[i] not in occ_train]
        occ_test_only.sort(reverse=True, key = lambda x : x[1])
        occ_dev_train = [(occ_dev_k[i],occ_dev_v[i],occ_train[occ_dev_k[i]]) for i in range(len(occ_dev)) if occ_dev_k[i] in occ_train]
        occ_dev_train.sort(reverse=True, key = lambda x : x[2])
        occ_dev_only = [(occ_dev_k[i],occ_dev_v[i]) for i in range(len(occ_dev)) if occ_dev_k[i] not in occ_train]
        occ_dev_only.sort(reverse=True, key = lambda x : x[1])
        all_test_train = [x for x in test_ent if x in train_ent]
        all_test_only = [x for x in test_ent if x not in train_ent]
        all_dev_train = [x for x in dev_ent if x in train_ent]
        all_dev_only = [x for x in dev_ent if x not in train_ent]
        if true: print("True stats : ") 
        else: print("False stats :")
        print("\ttrain_ent : {}\n\ttest_ent : {} ({:.2f}%)\n\tdev_ent : {} ({:.2f}%)".format(len(train_ent),len(test_ent),100*(len(test_ent)/sum_test),len(dev_ent),100*(len(dev_ent)/sum_dev)))
        print("\t"+20*"-")
        print("\tocc_dev : {} ({:.2f}%)\n\tocc_test : {} ({:.2f}%)\n\tocc_train : {} ({:.2f}%)".format(len(occ_dev),100*(len(occ_dev)/sum_occ),len(occ_test),100*(len(occ_test)/sum_occ),len(occ_train),100*(len(occ_train)/sum_occ)))
        print("\t"+20*"-")
        print("\tocc_test_train : {} ({:.2f}%)\n\tocc_test_only : {} ({:.2f}%)".format(len(occ_test_train),100*(len(occ_test_train)/len(occ_test)),len(occ_test_only),100*(len(occ_test_only)/len(occ_test))))
        print("\tocc_dev_train : {} ({:.2f}%)\n\tocc_dev_only : {} ({:.2f}%)".format(len(occ_dev_train),100*(len(occ_dev_train)/len(occ_dev)),len(occ_dev_only),100*(len(occ_dev_only)/len(occ_dev))))
        print("\t"+20*"-")
        print("\tall_test_train : {} ({:.2f}%)\n\tall_test_only : {} ({:.2f}%)".format(len(all_test_train),100*(len(all_test_train)/len(test_ent)),len(all_test_only),100*(len(all_test_only)/len(test_ent))))
        print("\tall_dev_train : {} ({:.2f}%)\n\tall_dev_only : {} ({:.2f}%)".format(len(all_dev_train),100*(len(all_dev_train)/len(dev_ent)),len(all_dev_only),100*(len(all_dev_only)/len(dev_ent))))
        
        with open(summary_folder+"occ_test_train_{}.txt".format(true_text),"w") as dump:
            dump.write("Occurence of False test in train\n")
            for elt in occ_test_train:
                dump.write("\t{} : {} | {} ({:.2f}%)\n".format(elt[0],elt[1],elt[2],100*(elt[1]/len(all_test_train))))
        with open(summary_folder+"occ_test_only_{}.txt".format(true_text),"w") as dump:
            dump.write("Occurence of False test not in train\n")
            for elt in occ_test_only:
                dump.write("\t{} : {} ({:.2f}%)\n".format(elt[0],elt[1],100*(elt[1]/len(all_test_only))))
        with open(summary_folder+"occ_dev_train_{}.txt".format(true_text),"w") as dump:
            dump.write("Occurence of False dev in train\n")
            for elt in occ_dev_train:
                dump.write("\t{} : {} | {} ({:.2f}%)\n".format(elt[0],elt[1],elt[2],100*(elt[1]/len(all_dev_train))))
        with open(summary_folder+"occ_dev_only_{}.txt".format(true_text),"w") as dump:
            dump.write("Occurence of False dev not in train\n")
            for elt in occ_dev_only:
                dump.write("\t{} : {} ({:.2f}%)\n".format(elt[0],elt[1],100*(elt[1]/len(all_dev_only))))
    
def melt_to_melting_args(melt_args):
    experiment_names = melt_args.experiment_name.split("_z_")
    training_names = melt_args.training_name.split("_z_") 
    melting_args = list(zip(experiment_names,training_names))
    return melting_args
        
def _parse_args():
    #arg par défaut : --experiment_name="base_mode_reajusted_z_context_bert_lstm_z_word_bert" --training_name="base_model_model_1_z_BERT&LSTM_model_1_z_Word_BERT_model_1" --load_existing_data
    #arg bis : --experiment_name="base_mode_reajusted_z_context_bert_lstm_z_context_bert_lstm" --training_name="base_model_model_1_z_BERT&LSTM_model_1_z_bert_lstm_ffnn2_300_embr6"
    #arg ters : --experiment_name="base_mode_reajusted_z_base_mode_reajusted_z_context_bert_lstm" --training_name="model_transformer_0_z_base_model_model_1_z_BERT&LSTM_model_1"
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", help="under folder data/tfrecords/\nList of folder separate by '_z_'")
    parser.add_argument("--training_name", help="under folder data/tfrecords/\nList of folder separate by '_z_'")
    parser.add_argument("--load_existing_data",dest="load_data",action="store_true")
    parser.add_argument("--compact_batch",dest="compact_batch",action="store_true")
    parser.add_argument("--only_occ",dest="only_occ",action="store_true")
    parser.add_argument("--wikiid2nnid_name", default="wikiid2nnid.txt")
    parser.set_defaults(load_data=False)
    parser.set_defaults(compact_batch=False)
    parser.set_defaults(only_occ=False)
    args = parser.parse_args()
    return args
        
if __name__=="__main__":
    args = _parse_args() #parse list of model to load and use
    summary_folder = "../data/tfrecords/"
    melting_args = melt_to_melting_args(args)
    it = 0
    topGlob = time.time()
    for emb_type in ["entities","words","context"]: #,"mention"]:
        for dataset in ["test","train","dev"]: #["dev"]: #
            if args.only_occ: break
            top = time.time()
            print(100*"/"+"\n"+40*"/"+"{} {}".format(emb_type,dataset)+40*"/"+"\n"+100*"/")
            ##ENTITIES TENSORBOARD
            #entities_discrimination_tensorboard(melting_args, summary_folder, tp=False, only_extract=False, use_old_data=args.load_data)
            #entities_discrimination_tensorboard(melting_args, summary_folder, tp=True, only_extract=False, use_old_data=args.load_data)
            ##TRUE ENTITIES
            print(10*"/"+" TRUE ENTITIES "+10*"/")
            entities_discrimination_tensorboard(melting_args, summary_folder, tp=True, dataset=dataset, emb_t=emb_type, use_old_data=args.load_data, wikiid2nnid_name=args.wikiid2nnid_name)
            names, words_vec, meta_words, ent, scores, s_r, i_e, d_e = entities_discrimination_viewWord(melting_args, summary_folder, tp=True, dataset=dataset, emb_t=emb_type, use_old_data=args.load_data, wikiid2nnid_name=args.wikiid2nnid_name)
            generating_graphs(names, words_vec, meta_words, ent, scores, s_r, i_e, d_e, summary_folder, emb_t=emb_type, maxent=10, maxmot=10, tp=True)
            print(100*"#")
            ##FALSE ENTITIES
            print(10*"/"+" FALSE ENTITIES "+10*"/")
            entities_discrimination_tensorboard(melting_args, summary_folder, tp=False, dataset=dataset, emb_t=emb_type, use_old_data=args.load_data, wikiid2nnid_name=args.wikiid2nnid_name)
            names, words_vec, meta_words, ent, scores, s_r, i_e, d_e = entities_discrimination_viewWord(melting_args, summary_folder, tp=False, dataset=dataset, emb_t=emb_type, use_old_data=True, wikiid2nnid_name=args.wikiid2nnid_name)
            generating_graphs(names, words_vec, meta_words, ent, scores, s_r, i_e, d_e, summary_folder, emb_t=emb_type, maxent=10, maxmot=10, tp=False)
            print(100*"#")
            print(100*"/"+"\n"+40*"/"+"{} {} IN {:.2f}s".format(emb_type,dataset,time.time()-top),40*"/"+"\n"+100*"/"+"\n")
            it += 1
        occ_entities_alldataset(melting_args, summary_folder, emb_t=emb_type)
    print(20*"#"+"END IN {:.2f}s".format(time.time()-topGlob)+20*"#")
