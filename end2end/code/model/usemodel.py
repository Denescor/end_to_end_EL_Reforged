# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 09:44:08 2020

@author: carpentier
"""
import numpy as np
import time
try: import tensorflow.compat.v1 as tf
except ImportError: import tensorflow as tf

import model.config as config
import model.evaluate as evaluate
import model.fun_eval as fun_eval
import model.train as train
from model.model import Model
from evaluation.metrics import _filtered_spans_and_gm_gt_list #Evaluator, metrics_calculation, threshold_calculation
from evaluation.print_predictions import PrintPredictions

"""
################################################################################################################################################################################################
#FONCTIONS RÉCUPÉRANT LES SORTIES DU MODÈLES ET LES METTANT EN FORME DANS UN FICHIER                                                                                                           #
#                                                                                                                                                                                              #
#DOCUMENTS UNIQUEMENT                                                                                                                                                                          #
# - yield_documents : RESTORE LA SESSION DU MODÈLE DÉJÀ CHARGÉ ET LANCE LA FONCTION "dump_documents"                                                                                           #
# - dump_documents : CRÉER UN FICHIER DANS LEQUEL TOUS LES DOCUMENTS DES DATASETS SONT SAUVÉS                                                                                                  #
#                                                                                                                                                                                              #
#PRÉDICTION UNIQUEMENT                                                                                                                                                                         #
# - yield_outputs : RESTORE LA SESSION DU MODÈLE DÉJÀ CHARGÉ ET LANCE LA FONCTION "dump_outputs"                                                                                               #
# - dump_output : CRÉER UN FICHIER DANS LEQUEL LES PRÉDICTIONS DU MODÈLES SONT SAUVÉS ET CLASSÉES PAR EXACTITUDE. DES STATS SUR TOUTES LES PRÉDICTIONS SONT ENSUITE DONNÉES DANS LA CONSOLE.   #
#                                                                                                                                                                                              #
#SCORES DU MODÈLE                                                                                                                                                                              #
# - evaluate_score : RESTORE LA SESSION DU MODÈLE DÉJÀ CHARGÉ ET CALCULE LES SCORES DU MODÈLES                                                                                                 #
################################################################################################################################################################################################
"""


def dump_documents(args, model, handles, names, iterators):
    print("SAVING DOCUMENTS...")
    with open(args.output_folder+"documents.txt", "w") as dump:
        # Reconstruction des dicos de conversion id <--> mot
        _, id2word, id2char = fun_eval.load_data(args, verbose=True)
        for test_handle, test_name, test_it in zip(handles, names, iterators):
            top = time.time()
            nbchunkword = 0 # Compteur du nombre de mots dans le document
            nbwuntchunk = 0 # Compteur du nombre de mots du document non trouvés dans le dico
            print("begining of {}".format(test_name))
            model.sess.run(test_it.initializer) # Initialisation de l'itérateur
            list_sentences = []
            while True:                 
                try:
                    result_l = fun_eval.run_model_return_args(model, test_handle)
                    for doc_num in range(result_l[0].shape[0]):
                        docid = str(result_l[-1][doc_num].split(b"&*", 1)[0]) # ID utilisé uniquement lors de l'écriture du fichier outputs
                        sentence, chunk_words, (nbchunktemp,nbwunttemp) = fun_eval.reconstruct_chunk_word(result_l, doc_num, id2word, id2char)
                        nbchunkword += nbchunktemp
                        nbwuntchunk += nbwunttemp
                        list_sentences.append((docid,sentence))
                except tf.errors.OutOfRangeError :
                    break
            ## Print de la forme 
            ## '### [Dataset] ###'
            ## '> [Docnum]'
            ## '[document]'
            ## '> [Docnum2]'
            ## '[document2]'
            ## etc...
            dump.write("### {} ###\n".format(test_name))
            for docid,sentence in list_sentences:
                dump.write("> {}\n".format(docid))
                dump.write("{}\n".format(sentence))
    print("... DOCUMENTS SAVED IN {:.2f}s TO : {}".format(time.time()-top,args.output_folder+"documents.txt")) 

def dump_output(args, model, handles, names, iterators, opt_thr, val_f1 = None):
    print("SAVING OUTPUTS...")
    with open(args.output_folder+"outputs.txt", "w") as dump:
        idtocand, id2word, id2char = fun_eval.load_data(args, verbose=True)
        # On parcours les datasets (dev, test et train)
        for test_handle, test_name, test_it in zip(handles, names, iterators):
            top = time.time()
            print("begining of {}".format(test_name))
            model.sess.run(test_it.initializer) # Initialisation de l'itérateur
            tp_count = dict()
            fp_count = dict()
            fn_count = dict()
            dict_id = 0
            nbchunkword = 0 # Compteur du nombre de mots dans le document
            nbwuntchunk = 0 # Compteur du nombre de mots du document non trouvés dans le dico
            nbword = 0 # Compteur de mots global vues par le script (pour le ratio de mots introuvés)
            nbwunt = 0 # Compteur de Mots non trouvés dans le dico
            nbcand = 0 # Compteur du nombre d'entités prédites
            nbcandword = 0 # Compteur du nombre d'entités total vues par le script (pour le ratio d'entités introuvées)
            nb_fp_cand = 0 # Compteur du nombre de Fausses Entités prédites
            nbwuntcand = 0 # Compteur du nombre d'entités non trouvées dans le dico
            nb_gm = 0 # Compteur global de gold mention pour le ratio de Faux Négatifs
            nb_fn = 0 # Compteur de Faux Négatifs
            nb_nogm = 0 # Compteur de Faux Positifs (cad nombre de span qui n'ont été liée à aucune gold mention)
            nb_expgm = 0 # Compteur global de gold mention pour le ratio de Faux Positifs (==> nombre de span total)
            fp_scores = [] # Stockage des scores des Faux Positifs (pour filtrer ceux inférieurs au seuil de score limite)
            nb_gmall = [] # Structure pour compter les spans matchant avec plusieurs gold mention (cad de doublons)            
            nb_tp = 0 # Compteur du nombre de True Positifs (mention + entité correctement prédites)
            gt_all_list = set() # Liste des ground truth
            cand_false = set() # Ensemble des Faux Positifs - Entités (unique)
            all_spans = set() # Ensemble des mentions (unique)
            not_fp_gt = 0 # Compteur du nombre de Faux Positifs - Mentions dont l'Entité matchent tout de même
            nb_ffp = 0 # Compteur du nombre de Faux Positifs - Mentions en dessous dont le score est en dessous du seuil critique
            nb_ftp = 0 # Compteur du nombre de True Positifs en dessous dont le score est en dessous du seuil critique
            nb_fcf = 0 # Compteur du nombre de Faux Positifs - Entités en dessous dont le score est en dessous du seuil critique 
            ## On parours chaque batch du dataset en cours
            ## On y extrait les spans et gold
            ## On en déduit les True Positifs, False Positifs et False Negatifs tels que :
            ##  True Positifs : span (ou mention) prédit correctement associé à une gold mention et dont l'entité prédite correspond également à la gold entité (ou ground truth)
            ##     False Positifs : span prédit ne correspondant à aucune gold mention
            ##     False Négatifs : gold mention n'ayant été prédit par aucun span
            ##     False Entities : span prédit correctement associé à une gold mention mais dont l'entité prédite ne correspond pas à la gold entité (ou ground truth)
            ## On compte également chaque sous-ensemble de manière globale de manière à effectuer ensuite une stat globale à chaque dataset
            while True:                 
                try:
                    result_l = fun_eval.run_model_return_args(model, test_handle)   
                    # Format de result_l
                    #    - final_scores, cand_entities_len, cand_entities
                    #    - begin_span, end_span, spans_len
                    #    - begin_gm, end_gm, ground_truth, ground_truth_len,
                    #    - words_len, words, chars, chars_len, chunk_id
                    for doc_num in range(result_l[0].shape[0]):
                        ## Etape 1 : récupération des couples (mention, meilleur mention) et (gold mention, ground truth) pour chaque document
                        current_dict_id = "{}:{}".format(dict_id,doc_num)
                        tp_count[current_dict_id] = 0
                        fp_count[current_dict_id] = 0
                        fn_count[current_dict_id] = 0
                        docid = str(result_l[-1][doc_num].split(b"&*", 1)[0]) # ID utilisé uniquement lors de l'écriture du fichier outputs
                        filtered_spans, gm_gt_list = _filtered_spans_and_gm_gt_list(doc_num, result_l[0], result_l[1], result_l[2], result_l[3], result_l[4], 
                                                                                    result_l[5], result_l[6], result_l[7], result_l[8], result_l[9], result_l[10])
                        # Format des structures :
                        #     - filtered_spans : [(best_cand_score, begin_idx, end_idx, best_cand_id), ...]
                        #     - gm_gt_list : [(begin_gm, end_gm, ground_truth), ...]                                          
                        ## Etape 2 : conversion des ids en mots
                        sentence, chunk_words, (nbchunktemp,nbwunttemp) = fun_eval.reconstruct_chunk_word(result_l, doc_num, id2word, id2char)
                        nbchunkword += nbchunktemp
                        nbwuntchunk += nbwunttemp
                        # Traitement des spans et entités
                        fn = [] # Liste des Faux Négatifs
                        tp = [] # Liste des True Positifs
                        fp = [] # Liste des Faux Positifs - Mentions
                        fp_entities = [] # Listes des Faux Positifs - Entités
                        #TEST cand_all_list = set([x[3] for x in filtered_spans]) # Liste des entités prédites
                        gt_all_list = gt_all_list.union(set([x[2] for x in gm_gt_list])) # Listes des ground truth
                        # Récupération des True Positifs, False Positifs & False Entities
                        for span in filtered_spans:
                            nb_gmdoublon = 0 # Compteur du nombre de gold mention associées en double pour un même span
                            best_cand_score, begin_idx, end_idx, best_cand_id = span
                            # On compte l'entité et le span actuelle à la fin de la boucle, on sait à l'avance qu'il suffit d'ajouter la longueur de la liste des spans
                            span_list, best_cand, (nbwuntcandtemp, nbwunttemp, nbwordtemp) = fun_eval.reconstruct_span(span, chunk_words, idtocand)
                            all_spans.add(span_list)
                            nbwuntcand += nbwuntcandtemp
                            nbwunt += nbwunttemp
                            nbword += nbwordtemp
                            # Récupération des True Positifs & False Entities
                            #     On trouve pour chaque span les gold mention matchant. 
                            #     A la première trouvée, on récupère la gold entité associée et on la compare avec l'entité prédite pour différencier les True Positifs des False Entities
                            #     Les autres gold mentions ne sont pas traités mais comptabilisées comme doublons
                            gm_find = False # Passe à "True" à la première gold mention qui match avec notre span
                            gt_find = False #                                    
                            for (bgm, egm, gt) in gm_gt_list:
                                ## Reconstruction de la Gold Mention par chevauchement #### #### #### #### #### #### #### #### ####
                                _, gm_list, (fp_count, tp_count, tp, fp_entities, cand_false), gm_find, (nbwunttemp, nbwordtemp, nb_gmdoublontemp, nbwuntcandtemp, nbcandwordtemp), (nb_ftptemp, nb_fcftemp, nb_fp_candtemp, nb_tptemp) = fun_eval.reconstruct_true_positif(span, 
                                                                                                                        (bgm, egm, gt), 
                                                                                                                        chunk_words, 
                                                                                                                        span_list,
                                                                                                                        idtocand,
                                                                                                                        best_cand, 
                                                                                                                        opt_thr, 
                                                                                                                        gm_find, 
                                                                                                                        current_dict_id, 
                                                                                                                        (fp_count, tp_count, tp, fp_entities, cand_false))
                                #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####
                                nb_ftp += nb_ftptemp
                                nb_fcf += nb_fcftemp
                                nb_fp_cand += nb_fp_candtemp
                                nb_tp += nb_tptemp
                                nbwunt += nbwunttemp
                                nbword += nbwordtemp
                                nb_gmdoublon += nb_gmdoublontemp
                                nbwuntcand += nbwuntcandtemp
                                nbcandword += nbcandwordtemp
                            # Cas où aucun candidat n'aurait matché avec le span (donc pas de prédiction récupérée)
                            if((not gm_find) and (len(gm_gt_list) > 0)):
                                nb_nogm += 1
                                gm_list, gt_word = ["[NONE]","[NONE]"], "[NONE]" # On oublie pas de définir "gm_list" pour éviter que la version précédente de l'itération soit prise
                                if(best_cand_score >= opt_thr): 
                                    fp.append((begin_idx,span_list))
                                    fp_count[current_dict_id] += 1
                                else: nb_ffp += 1 # Si le score est inférieur au seuil, le modèle ne le décompte pas. On récupère donc juste le compteur pour évaluer l'impact
                                if(gt_find): not_fp_gt += 1
                            elif(len(gm_gt_list) == 0): 
                                gm_list, gt_word = ["[NONE]","[NONE]"], "[NONE]"
                            nb_gmall.append((nb_gmdoublon,(nb_gmdoublon + 1 if gm_find else 0))) # Couple (nombre de gold mention en double, nombre de gold mention trouvées)
                        # Récupération des Faux Négatifs
                        #   On parcours toutes les gold mention et pour chacune on regarde si au moins 1 span le chevauche
                        for (bgm, egm, gt) in gm_gt_list:
                            is_fn = True
                            nb_gm += 1                            
                            for span in filtered_spans:
                                span_list, best_cand, _ = fun_eval.reconstruct_span(span, chunk_words, idtocand)
                                is_tp, _, _, _, _, _ = fun_eval.reconstruct_true_positif(span, (bgm, egm, gt), chunk_words, span_list, idtocand, best_cand, opt_thr, False)
                                is_fn = not is_tp
                                if(is_tp): break
                            if (is_fn): # C'est un Faux Négatif ==> On récupère la glod mention + gold entité sous forme de texte et on l'ajoute à la liste "fn"
                                nb_fn += 1
                                # Reconstruction de la gold mention complète
                                gm_list, _ = fun_eval.reconstruct_span_words(bgm, egm+1, chunk_words)
                                try: gt_word = idtocand[gt]
                                except KeyError:
                                    gt_word = "[wunt {}]".format(gt)
                                fn.append((bgm,gm_list,gt_word))
                                fn_count[current_dict_id] += 1
                        nb_expgm += len(filtered_spans) # On pourrait incrémenter le compteur dans la boucle sur la liste des spans mais on connait le résultat à l'avance.
                        nbcandword += len(filtered_spans) # On pourrait incrémenter le compteur dans la boucle sur la liste des spans mais on connait le résultat à l'avance.
                        ## Etapé 3 : écriture des résultats
                        dump.write("######### document : '{}' ##########\n".format(docid))
                        # Print stats
                        if(len(gm_gt_list) == 0 or len(filtered_spans) == 0):
                            print("error list {} doc {} ({}) : gm list = {} ; span list = {} ; tp list = {}".format(test_name,docid,doc_num,len(gm_gt_list),len(filtered_spans),len(tp)))
                            dump.write("Faux Negatif:\t\t'[calcul impossible]'\n")
                            dump.write("Faux Positif:\t\t'[calcul impossible]'\n")
                        else:
                            dump.write("Faux Negatif:\t\t{:.2f}%\n".format(100*(len(fn)/len(gm_gt_list))))
                            dump.write("Faux Positif:\t\t{:.2f}%\n".format(100*(len(fp)/len(filtered_spans))))
                        if(len(tp) > 0):
                            dump.write("True Positif:\t\t{:.2f}%\n".format(100*(len(tp)/len(filtered_spans))))
                            dump.write("\tdont entités correctes: {}%\n".format(100-100*(len(fp_entities)/len(tp))))
                        else:
                            #print("error list {} doc {} ({}) : gm list = {} ; span list = {} ; tp list = {}".format(test_name,docid,doc_num,len(gm_gt_list),len(filtered_spans),len(tp)))
                            dump.write("True Positif:\t\t0%\n")
                            dump.write("\tdont entités correctes: 0%\n")
                        # Print document
                        dump.write("'{}'\n".format(sentence))
                        #TEST for elt in result_words:
                        #TEST    output = "--mention (position {}) : '{}'\n\t--entité : '{}'\n\t--gold_mention (position {}) : '{}'\n\t--gold_entité : '{}'\n".format(elt[0],elt[1],elt[2],elt[3],elt[4],elt[5])
                        #TEST    dump.write(output)
                        # Print values
                        dump.write("True Positif\n")
                        for idx,span,cand in tp:
                            dump.write("\t--mention (position {}) : '{}'\n".format(idx,span))
                            dump.write("\t\t--entité : '{}'\n".format(cand))
                        dump.write("False Entity\n")
                        for idx,span,cand,gt in fp_entities:
                            dump.write("\t--mention (position {}) : '{}'\n".format(idx,span))
                            dump.write("\t\t--entité : '{}'\n".format(cand))
                            dump.write("\t\t--gold entité : '{}'\n".format(gt))
                        dump.write("False Negatif\n")
                        for idx,gm,gt in fn:
                            dump.write("\t--mention (position {}) : '{}'\n".format(idx,gm))
                            dump.write("\t\t--entité : '{}'\n".format(gt))
                        dump.write("False Positif\n")
                        for idx,span in fp:
                            dump.write("\t--mention (position {}) : '{}'\n".format(idx,span))
                        dump.write("########## end : '{}' ##########\n\n".format(docid))
                except tf.errors.OutOfRangeError :#TEST as e:
                    # FIN DE LA BOUCLE ==> donc du traitement du dataset
                    doublon_mean = np.mean([x[0] for x in nb_gmall if x[1] > 0])
                    gm_mean = np.mean([x[1] for x in nb_gmall if x[1] > 0])
                    # Calcul des Précisions / Rappel et F1 pour le dataset entier
                    micro_pr, micro_re, micro_f1, macro_pr, macro_re, macro_f1 = fun_eval.compute_score(tp_count,fp_count,fn_count)
                    print("////////////////// compare counter //////////////////")
                    print("nombre de spans : {}".format(nb_expgm))
                    print("nombre de spans (unique) : {}".format(len(all_spans)))
                    print("nombre de gold mention : {}".format(nb_gm))
                    print("nombre de ground truth (unique) : {}".format(len(gt_all_list)))
                    print("nombre de True Positif : {}".format(nb_tp))
                    print("nombre de False Positif : {}".format(nb_nogm))
                    print("nombre de False Negatif : {}".format(nb_fn))
                    print("nombre d'entités prédites : {}".format(nbcand))
                    print("nombre de mauvaies entités prédites : {}".format(nb_fp_cand))
                    print("nombre de mauvaies entités prédites (unique) : {}".format(len(cand_false)))
                    print("////////////////// stats reconstruction du document //////////////////")
                    print("chunk non trouvés : \t{}/{} ({:.2f}%)".format(nbwuntchunk,nbchunkword,100*(nbwuntchunk/nbchunkword)))
                    print("Mots non trouvés : \t{}/{} ({:.2f}%)".format(nbwunt,nbword,100*(nbwunt/nbword)))
                    print("entités non trouvées : \t{}/{} ({:.2f}%)".format(nbwuntcand,nbcandword,100*(nbwuntcand/nbcandword)))
                    print("////////////////// scores P / R / F1 //////////////////")
                    print("micro", "P: %.1f" % micro_pr, "\tR: %.1f" % micro_re, "\tF1: %.1f" % micro_f1)
                    print("macro", "P: %.1f" % macro_pr, "\tR: %.1f" % macro_re, "\tF1: %.1f" % macro_f1)      
                    print("////////////////// ratio TP / FP / FN //////////////////")
                    print("Faux Négatifs Globaux : \t{}/{} ({:.2f}%)".format(nb_fn,nb_gm,100*(nb_fn/nb_gm)))
                    print("Faux Positifs Globaux : \t{}/{} ({:.2f}%)".format(nb_nogm,nb_expgm,100*(nb_nogm/nb_expgm)))
                    print("\tdont score inférieur au seuil : {}/{} ({:.2f}%)".format(nb_ffp,nb_nogm,100*(nb_ffp/nb_nogm)))
                    print("\tdont l'entité est bien trouvée : {}/{} ({:.2f}%)".format(not_fp_gt,nb_nogm,100*(not_fp_gt/nb_nogm)))
                    print("True Positifs Globaux : \t{}/{} ({:.2f}%)".format(nb_tp,nb_expgm,100*(nb_tp/nb_expgm)))
                    print("\tdont score inférieur au seuil : {}/{} ({:.2f}%)".format(nb_ftp,nb_tp,100*(nb_ftp/nb_tp)))
                    print("Entités incorrectes globaux : \t{}/{} ({:.2f}%)".format(nb_fp_cand,nb_expgm,100*(nb_fp_cand/nb_expgm))) #TEST replace "nb_expgm" by "nb_cand"
                    print("\tdont score inférieur au seuil : {}/{} ({:.2f}%)".format(nb_fcf,nb_fp_cand,100*(nb_fcf/nb_fp_cand)))
                    print("Entités incorrectes (pdv entités) : \t{}/{} ({:.2f}%)".format(len(cand_false),len(gt_all_list),100*(len(cand_false)/len(gt_all_list))))#TEST replace "len(cand_false)" by "nb_fp_cand"      
                    print("doublon moyen dans les gold mentions : {}/{} ({:.2f}%)".format(doublon_mean,gm_mean,100*(doublon_mean/gm_mean)))
                    print("////////////////////////////////////")
                    print("End of {} in {}s".format(test_name, time.time()-top))
                    break
                finally:
                    dict_id += 1
    print("... OUTPUTS SAVED TO : {}".format(args.output_folder+"outputs.txt")) 

def evaluate_score(iterators, handles, model):
    """
    print the Precision, Recall and F1 Score of the current model to the terminal
    """
    print(">>> YIELD SCORES <<<")
    top = time.time()
    el_scores = train.compute_ed_el_scores(model, handles, el_names, iterators, el_mode=True)
    print("el_scores {}, taille {}".format(type(el_scores),len(el_scores)))
    print(">>> DONE IN {:.2f}s <<<".format(time.time()-top))

def yield_outputs(args, iterators, handles, el_names, model):
    """
    Return the prediction of the model to a file named "outputs.txt"
    """
    print(">>> YIELD OUTPUTS <<<")
    top = time.time()
    opt_thr, val_f1 = train.optimal_thr_calc(model, handles, iterators, True)
    dump_output(args, model, handles, el_names, iterators, opt_thr, val_f1)
    print(">>> DONE IN {:.2f}s <<<".format(time.time()-top))

def yield_documents(args, iterators, handles, el_names, model):
    """
    Return the prediction of the model to a file named "documents.txt"
    """
    print(">>> DUMP DOCUMENTS <<<")
    top = time.time()
    #opt_thr, val_f1 = train.optimal_thr_calc(model, handles, iterators, True)
    dump_documents(args, model, handles, el_names, iterators)
    print(">>> DONE IN {:.2f}s <<<".format(time.time()-top))

if __name__ == "__main__":
    args, train_args = evaluate._process_args(evaluate._parse_args())
    print(args)
    train_args.checkpoint_model_num = args.checkpoint_model_num
    train_args.entity_extension = args.entity_extension
    if train_args.context_bert_lstm is None:  train_args.context_bert_lstm = False
    train.args = train_args
    #args.context_bert_lstm = False
    args.batch_size = train_args.batch_size
    args.output_folder = train_args.output_folder
    args.eval_cnt = None
    el_datasets, el_names, model = fun_eval.retrieve_model(train.args, args)
    print("START EVALUATE")
    try:
        printPredictions = PrintPredictions(config.base_folder+"data/tfrecords/"+
                         args.experiment_name+"/", args.predictions_folder, args.entity_extension,
                                            args.gm_bucketing_pempos, args.print_global_voters,
                                            args.print_global_pairwise_scores,
                                            wikiid2nnid_name=args.wikiid2nnid_name)
    except : print("printPrediction doesn't work")
    with model.sess as sess:
        iterators, handles = fun_eval.ed_el_dataset_handles(sess, el_datasets)
        evaluate_score(iterators, handles, model)
        print("START WRITTING")
        yield_outputs(args, iterators, handles, el_names, model)
        yield_documents(args, iterators, handles, el_names, model)
