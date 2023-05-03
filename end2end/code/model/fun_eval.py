## Functions for evaluation based on mentions

import time
import pickle
import numpy as np
try: import tensorflow.compat.v1 as tf
except ImportError: import tensorflow as tf

from preprocessing.util import load_wiki_name_id_map, load_wikiid2nnid #reverse_dict, 
from model.model import Model
import model.config as config
import model.evaluate as evaluate
import model.train as train

"""
#############################################################################################################################################
#Fonctions disponibles :                                                                                                                    #
#	- eval_parsing()                                                --> eval_args                                                           #
#	- load_data(args, verbose=True)                                 --> idtocand, id2word, id2char                                          #
#	- generate_cand_metadata(index,args, details=True)              --> metadata                                                            #
#	- ed_el_dataset_handles(sess, datasets)                         --> test_iterators, test_handles                                        #
#	- retrieve_model(train_args,eval_args)                          --> el_datasets, el_names, model                                        #
#	- retrieve_model_args(experiment_name, training_name)           --> model, eval_args, train_args                                        #
#	- run_model_return_args(model,test_handle)                      --> result_l                                                            #
#	- run_model_ent_and_proj(model, test_handle, verbose=True)      --> cand_entities, ground_truth, projectors	                            #
#	- reconstruct_chunk_word(result_l, docnum, id2word, id2char)    --> sentence, chunk_words, (nbchunkword, nbwuntchunk)                   #
#	- reconstruct_span(span, chunk_words)                           --> span_list, best_cand, (nbwuntcand, nbwunt, nbword)                  #
#	- reconstruct_span_words(bgi, endi, chunk_words)                --> span_list, (nbwunt, nbword)                                         #
#	- reconstruct_true_positif(span, gm, chunk_words, idtocand,                                                                             #
#	                           opt_thr, gm_find,                                                                                            #
#	                           current_dict_id=-1, dicos=None)      --> (nb_tp > 0),                                                        #
#	                                                                    gm_list,                                                            #
#	                                                                    (fp_count, tp_count, tp, fp_entities, cand_false),                  #
#	                                                                    gm_find,                                                            #
#	                                                                    (nbwunt, nbword, nb_gmdoublon, nbwuntcand),                         #
#	                                                                    (nb_ftp, nb_fcf, nb_fp_cand, nb_tp)                                 #
#	- compute_score(tp_dict,fp_dict,fn_dict)                        --> micro_pr, micro_re, micro_f1, macro_pr, macro_re, macro_f1          #
#############################################################################################################################################
"""

def eval_parsing(wikiid2nnid_name="wikiid2nnid.txt"):
    """
    INPUT : NONE
    OUTPUT :
        - eval_args : arguments permettant de charger le modèle en évaluation
    """
    eval_args = evaluate._parse_args() #defaut parsing. To complete with "experiment_name" and "training_name"
    eval_args.all_spans_training = True
    eval_args.ed_datasets = ""
    eval_args.wikiid2nnid_name = wikiid2nnid_name
    return eval_args

def load_data(args, verbose=False):
    """
    INPUT :
        - args : argument de base contenant le nom de l'expérience et la valeur de l'option "entity_extension"
    OUTPUT :
        - idtocand : dictionnaire (int --> entité)
        - id2word : dictionnaire (int --> mot)
        - id2char : dictionnaire (int --> caractère)
    """
    # Reconstruction des dicos de conversion id <--> mot
    #TO DELETE with open(config.base_folder+"data/vocabulary/vocab_freq.pickle", 'rb') as vocab_freq:
    #TO DELETE     word_freq, _ = pickle.load(vocab_freq)
    with open(config.base_folder+"data/tfrecords/"+args.experiment_name+"/word_char_maps.pickle", 'rb') as wordmap:
        _, id2word, _, id2char, _, _ = pickle.load(wordmap)
    #TO DELETE id2word = reverse_dict(word_freq)
    wikiid2nnid = load_wikiid2nnid(args.entity_extension, txt_file=args.wikiid2nnid_name, verbose=False) #reverse_dict(load_wikiid2nnid(args.entity_extension)) 
    wiki_name_id_file = "wiki_name_id_map_{}".format(args.wikiid2nnid_name.split("_")[-1])
    if(verbose): print("wikiid2nid : {}\nwiki_name_id_map : {}".format(args.wikiid2nnid_name, wiki_name_id_file))
    _, wiki_id_name_map = load_wiki_name_id_map(filepath=wiki_name_id_file, verbose=False) 
    idtocand = dict()
    notInDict = 0
    nbElts = 0
    for key,elt in wikiid2nnid.items():
        if key in wiki_id_name_map:
            idtocand[elt] = wiki_id_name_map[key]
        else:
            notInDict += 1
        nbElts += 1
    if(verbose):
        print("id2word : \t\t{}".format([(k,id2word[k]) for k in list(id2word.keys())[:4]]))   
        print("wikiid2nnid : \t{}".format([(k,wikiid2nnid[k]) for k in list(wikiid2nnid.keys())[:4]]))
        print("wiki_id_name_map : {}".format([(k,wiki_id_name_map[k]) for k in list(wiki_id_name_map.keys())[:4]]))
        print("idtocand : \t{}".format([(k,idtocand[k]) for k in list(idtocand.keys())[:4]]))
        print("match idtocand impossible : \t{}/{} ({}%)".format(notInDict,nbElts,100*(notInDict/nbElts)))
    return idtocand, id2word, id2char

def generate_cand_metadata(index,args, details=True):
    """
    INPUT :
        - index : matrice d'int. dimension : taille du batch x nb de candidats
        - args : argument de base contenant le nom de l'expérience et la valeur de l'option "entity_extension"
        - details : True si l'on souhaite avec les 50 premières erreurs de récupération générées
    OUTPUT :
        list of string : la liste des entités
    """
    idtocand, _, _ = load_data(args, verbose=False)
    metadata = []
    error = 0
    total = 0
    error_details = []
    (x,y) = np.shape(index)
    for x_i in range(x):
        for y_i in range(y):
            cand_index = index[x_i][y_i]
            try: metadata.append(idtocand[cand_index])
            except KeyError:
                metadata.append("<wunk>")
                error +=1
                error_details.append(cand_index)
            total += 1
    
    if details : 
        print("{}/{} ({:.2f}%) metadata remplacé par '<wunk>'".format(error,total,100*(error/total)))
        print("bad keys : \n\t{}...".format(error_details[:50]))
    return metadata

def ed_el_dataset_handles(sess, datasets):
    """
    INPUT :
        - sess : session active du modèle
        - datasets : liste des datasets
    OUTPUT :
        - list of iterators
        - list of handles
    """
    test_iterators = []
    test_handles = []
    for dataset in datasets:
        test_iterator = dataset.make_initializable_iterator()
        test_iterators.append(test_iterator)
        test_handles.append(sess.run(test_iterator.string_handle()))
    return test_iterators, test_handles
    
def ed_el_testset_handles(sess, names, datasets, data_to_pick="test"):
    """
        
    INPUT :
        - sess : session active du modèle
        - datasets : liste des datasets
        - data_to_pick : dataset spécifique à extraire parmi "train", "dev" or "test" (DEFAUT : "test")
        Varie de la version précédente en ne générant un itérateur et un handle que pour 1 seul des datasets
    OUTPUT :
        - iterator
        - handle
    """
    for i in range(len(names)):
    	if(data_to_pick in names[i].split("_")):  
    		print("dataset : {}".format(names[i]))
    		dataset = datasets[i] #uniquement le dataset de test
    		break
    test_iterator = dataset.make_initializable_iterator()
    test_handle = sess.run(test_iterator.string_handle())
    return test_iterator, test_handle
    
def retrieve_model(train_args, eval_args, mode="evaluation"):
    """
    INPUT :
        - train_args : arguments permettant de charger le modèle en apprentissage
        - eval_args : arguments permettant de charger le modèle en évaluation
    OUTPUT :
        - el_datasets : list des datasets
        - el_names : list of string (les noms des datasets)
        - model : le modèle chargé
    """
    print(">>> retrieve datasets")    
    ed_datasets, ed_names = train.create_el_ed_pipelines(gmonly_flag=True, filenames=eval_args.ed_datasets, args=train_args)
    el_datasets, el_names = train.create_el_ed_pipelines(gmonly_flag=False, filenames=eval_args.el_datasets, args=train_args)

    input_handle_ph = tf.placeholder(tf.string, shape=[], name="input_handle_ph")
    sample_dataset = ed_datasets[0] if ed_datasets != [] else el_datasets[0]
    iterator = tf.data.Iterator.from_string_handle(
        input_handle_ph, sample_dataset.output_types, sample_dataset.output_shapes)
    next_element = iterator.get_next()
    train_args.running_mode = mode #Mode évaluation
    
    print(">>> retrieve model")

    model = Model(train_args, next_element)
    model.build()
    model.input_handle_ph = input_handle_ph    # just for convenience so i can access it from everywhere
    
    print("model variables : {}".format(tf.trainable_variables()))
    print("global variables : {}".format(tf.global_variables()))
    if eval_args.p_e_m_algorithm:
        model.final_scores = model.cand_entities_scores
    model.tf_writers = None

    #print_variables_values(model)
    
    print(">>> run")
    return el_datasets, el_names, model

def retrieve_model_args(experiment_name, training_name, wikiid2nnid_name="wikiid2nnid.txt"):
    """
    INPUT :
        - experiment_name : expérience étudiée
        - training_name : modèle de l'expérience à charger
    OUTPUT :
        - model : tuple
            - el_datasets : list des datasets
            - el_names : list of string (les noms des datasets)
            - model : le modèle chargé
        - eval_args : arguments permettant de charger le modèle en évaluation
        - train_args : arguments permettant de charger le modèle en apprentissage
    """
    #Process args
    print("/"*50)
    eval_args = eval_parsing()
    eval_args.training_name = training_name
    eval_args.experiment_name = experiment_name
    eval_args.wikiid2nnid_name = wikiid2nnid_name
    #if eval_args.transformer is None: 
    #eval_args.transformer = False
    eval_process_args, train_args = evaluate._process_args(eval_args)
    #Ajust train_args and eval_process_args
    train_args.checkpoint_model_num = eval_process_args.checkpoint_model_num
    train_args.entity_extension = eval_process_args.entity_extension
    train.args = eval_process_args
    eval_process_args.batch_size = train_args.batch_size
    eval_process_args.eval_cnt = None
    #Retrieve and save Model
    model = retrieve_model(train_args,eval_process_args) #model1 : el_datasets, el_names, model  
    print("/"*50) 
    return model, eval_args, train_args

def run_model_return_args(model, test_handle):
    """
    INPUT :
        - model : le modèle chargé
        - test_handle : l'handle du dataset à utiliser
    OUTPUT :
        - result_l : la liste des données contenues dans le modèle (cf. variable "retrive_l")
    """
    retrieve_l = [model.final_scores, model.cand_entities_len, model.cand_entities,
                  model.begin_span, model.end_span, model.spans_len,
                  model.begin_gm, model.end_gm,
                  model.ground_truth, model.ground_truth_len,
                  model.words_len, model.words, model.chars, model.chars_len, model.chunk_id]
    result_l = model.sess.run(retrieve_l, feed_dict={model.input_handle_ph: test_handle, model.dropout: 1})  
    return result_l
    
def run_model_ent_and_proj(model, test_handle, verbose=True):
    """
    INPUT :
        - model : le modèle chargé
        - test_handle : l'handle du dataset à utiliser
    OUTPUT :
        - cand_entities : la matrice des entités candidates (matrice of int)
        - ground_truth : la matrice des entités de références (matrice of int)
        - projectors : list (len : 4) of matrices d'embeddings 
    """
    cand_entities, ground_truth, _, projectors = model.sess.run(
                                                   [model.cand_entities,
                                                    model.ground_truth,
                                                    model.tf_param_summaries, 
                                                    model.projector_embeddings], 
                                                   feed_dict={model.input_handle_ph: test_handle,
                                                              model.dropout: 1})#, model.lr: model.args.lr})
    ## STAT EMBEDDINGS
    if(verbose):
        print("##### PROJECTOR : {}({}) of {} #####".format(type(projectors),len(projectors),type(projectors[0])))
        print("shape word : {}".format(np.shape(projectors[0])))
        print("shape entities : {}".format(np.shape(projectors[1])))
        print("shape context : {}".format(np.shape(projectors[2])))
        print("shape mention : {}".format(np.shape(projectors[3])))
        print("shape cand_entities : {}x{} of {} {} of {}".format(len(cand_entities),
                                                                  len(cand_entities[0]),
                                                                  type(cand_entities[0]),
                                                                  np.shape(cand_entities[0]),
                                                                  type(cand_entities[0][0][0])))
        print("shape ground_truth : {}x{} of {} {} of {}".format(len(ground_truth),
                                                                 len(ground_truth[0]),
                                                                 type(ground_truth[0]),
                                                                 np.shape(ground_truth[0]),
                                                                 type(ground_truth[0][0])))
        print("##### ##### ##### ##### #####")
    return cand_entities, ground_truth, projectors

def reconstruct_chunk_word(result_l, docnum, id2word, id2char):
    """
    INPUT : 
        - result_l : la liste des données contenues dans le modèle (cf. variable "retrive_l" in "run_model_return_args")
        - docnum : id du document
        - id2word : dictionnaire (int --> mot)
        - id2char : dictionnaire (int --> caractère)
    OUTPUT : 
        - sentence : string (document reconstruit depuis les chunk)
        - chunk_words : list of string (les différents mots du document)
        - (nbchunkword, nbwuntchunk) : stats (nombre de chunk total, nombre de chunk inconnus)
    """
    nbwuntchunk = 0
    chunk_words = []
    # Récupération des structures des mots et caractères
    modelWordsLen = result_l[-5]
    modelWords = result_l[-4]
    modelChars = result_l[-3]#.tolist()
    modelCharsLen = result_l[-2]
    # Reconstruction de la 'phrase' / 'document'
    nbchunkword = len(modelWords[docnum])
    for i,wordid in enumerate(modelWords[docnum]):#TEST in range(modelWordsLen[docid]):
        try:
            if wordid != 0: chunk_words.append(id2word[wordid])
            else:
                word_char = []
                for j in range((modelCharsLen[docnum])[i]):
                    word_char.append(id2char[(modelChars[docnum])[i][j]])
                    chunk_words.append(''.join(word_char))
        except KeyError:
            chunk_words.append("[wunt {}]".format(wordid))
            nbwuntchunk += 1
    sentence = " ".join(chunk_words) # Document final reconstruit
    return sentence, chunk_words, (nbchunkword, nbwuntchunk)

def reconstruct_span(span, chunk_words, idtocand):
    """
    INPUT :
        - span : tuple
            - best_cand_score : score de similarité l'entité prédite
            - begin_idx : indice du début de la mention dans le document
            - end_idx :indice de fin de la mention dans le document
            - best_cand_id : entité prédite (int)
        - chunk_words : list of string (les différents mots du document)
        - idtocand : dictionnaire (int --> entité)
    OUTPUT :
        - span_list : string (la mention complète)
        - best_cand : id de l'entité prédite (int)
        - (nbwuntcand, nbwunt, nbword) : stat (nb d'entités inconnues, nombre de mots inconnues, nombre de mots total)
    """
    # Récupération de l'Entité Prédites
    best_cand_score, begin_idx, end_idx, best_cand_id = span
    nbwuntcand, nbwunt, nbword = 0, 0, 0
    try: 
        best_cand = idtocand[best_cand_id]
    except KeyError:
        best_cand = "[wunt {}]".format(best_cand_id)
        nbwuntcand += 1
    # Reconstruction de la mention
    span_list, (nbwunt, nbword) = reconstruct_span_words(begin_idx, end_idx, chunk_words)
    return span_list, best_cand, (nbwuntcand, nbwunt, nbword)
    
def reconstruct_span_words(bgi, endi, chunk_words):
    """
    INPUT: 
        - bgi : indice du début de la mention dans le document
        - endi : indice de fin de la mention dans le document
        - chunk_words : list of string (les différents mots du document)
    OUTPUT :
        - span_list : string (la mention complète)
        - (nbwunt, nbword) : stat (nombre de mots inconnues, nombre de mots total)
    """
    span_list = []
    nbwunt = 0
    for index in range(bgi, endi):
        try : span_list.append(chunk_words[index])# + "[{}]".format(id2)) #gm_list.append(id2word[id]) #
        except KeyError :
            span_list.append("[wunt {}]".format(index))
            nbwunt += 1
        except Exception as e : print("error in reconstruct_span : {} (bgm : {} ; egm : {}/{})".format(e,bgi,endi,len(chunk_words)))
    nbword = endi - bgi # Nombre de mots vues
    span_list = " ".join(span_list) #reconstruction de la gold mention complète
    return span_list, (nbwunt, nbword)
                          
def reconstruct_true_positif(span, gm, chunk_words, sentence, idtocand, best_cand, opt_thr, gm_find, current_dict_id=-1, dicos=None):
    """
    INPUTS : 
        - span : tuple
            - best_cand_score : score de similarité l'entité prédite
            - begin_idx : indice du début de la mention dans le document
            - end_idx :indice de fin de la mention dans le document
            - best_cand_id : entité prédite (int)
        - gm : tuple
            - bgm : indice du début de l'entité de référence dans le document
            - egm : indice de la fin de l'entité de référence dans le document
            - gt : l'entité de référence (int)
        - chunk_words : list of string (les différents mots du document)
        - sentence : string (document reconstruit depuis les chunk)
        - idtocand : dictionnaire (int --> entité)
        - best_cand : l'entité prédite (string)
        - opt_thr : seuil minimal de validité du score. En dessous, tout résultat est considéré comme faux
        - gm_find : True si la mention a déjà été matché à une autre entité auparavant. False sinon
        - current_dict_id : indice actuel à mettre à jour dans les dictionnaires de compatage (DEFAUT : -1 -- ne pas prendre en compte les dictionnaires de comptage)
        - dicos : dictionnaire de comptage (DEFAUT : None -- ne pas prendre en compte les dictionnaires de comptage)
            - nombre de faux positifs -- (int --> int),
            - nombre de vrais positifs -- (int --> int),
            - Vrais positifs - list of tuple 
                - indice de début de la mention (int)
                - phrase (string)
                - entité prédite (string)
            - Faux Positifs - list of tuple
                - indice de début de la mention (int)
                - phrase (string)
                - entité prédite (string)
                - entité de référence (string)
            - ensemble des entités de référence mal prédites (int)
    OUTPUTS : 
        - "booleen" : True si l'entité prédite est correcte. False sinon
        - gm_list
        - (fp_count, tp_count, tp, fp_entities, cand_false) : dictionnaires de comptage mis à jour
            - nombre de faux positifs -- (int --> int),
            - nombre de vrais positifs -- (int --> int),
            - Vrais positifs - list of tuple 
                - indice de début de la mention (int)
                - phrase (string)
                - entité prédite (string)
            - Faux Positifs - list of tuple
                - indice de début de la mention (int)
                - phrase (string)
                - entité prédite (string)
                - entité de référence (string)
            - ensemble des entités de référence mal prédites (int)
        - gm_find : booléen réactualisé si un match a été effectué entre la mention et l'entité
        - (nbwunt, nbword, nb_gmdoublon, nbwuntcand, nbcandword) : stats (nombre de mots inconnus, 
                                                                          nombre de mots total, 
                                                                          nombre de doublon de prédiction pour la mention,
                                                                          nombre d'entités inconnues,
                                                                          nombre d'entités total)
        - (nb_ftp, nb_fcf, nb_fp_cand, nb_tp) : stats (nombre de faux positifs pour cause de scores trop bas,
                                                       nombre de vrais faux négatifs dont le scores est trop bas,
                                                       nombre de vrais faux négatifs,
                                                       nombre vrais positifs)
    """
    # Reconstruction de la Gold Mention par chevauchement
    # Définition des compteurs
    if dicos is None: return_counter = False
    else: return_counter = True
    if return_counter: fp_count, tp_count, tp, fp_entities, cand_false = dicos
    else : fp_count, tp_count, tp, fp_entities, cand_false = dict([(-1,0)]), dict([(-1,0)]), [], [], set() # structures factices par défaut
    nb_ftp, nb_fcf, nb_fp_cand, nb_tp = 0, 0, 0, 0 
    nb_gmdoublon, nbwuntcand, nbcandword, nbcand, nbwunt, nbword = 0, 0, 0, 0, 0, 0
    # Récupérations des spans et candidats
    best_cand_score, begin_idx, end_idx, best_cand_id = span
    bgm, egm, gt = gm
    #Début du traitement
    gm_list = [] #cas par défaut si le match ne s'effectue pas
    if (begin_idx<=bgm and end_idx<=egm and bgm<end_idx) or (begin_idx>=bgm and end_idx>=egm and begin_idx<egm) or (begin_idx<=bgm and end_idx>=egm) or (begin_idx>=bgm and end_idx<=egm):
        ###### MATCH ACQUIS SI ######
        #   - la mention est légèrement trop sur la gauche à cheval
        #   - la mention est légèrement trop sur la droite à cheval
        #   - la mention déborde légèrement des deux côtés
        #   - la mention est légèrement trop ressérée
        ### ### ### ### ### ### ### ##
        ### CAS OÙ ÇA NE MATCH PAS ###
        #   - la mention finit avant le démarrage du véritable span
        #   - la mention commence après la fin du véritable span
        ### ### ### ### ### ### ### ##
        # On récupère la mention identifiée et on compte les doublons
        if gm_find : nb_gmdoublon += 1
        else:
            gm_list_temp, (nbwunt, nbword) = reconstruct_span_words(bgm, egm, chunk_words)
            gm_list = (bgm,gm_list_temp)
            try: gt_word = idtocand[gt]
            except KeyError:
                gt_word = "[wunt {}]".format(gt)
                nbwuntcand += 1
            finally : nbcandword += 1            
            # Détermination des Entités Erronées
            if(gt == best_cand_id): # La gold mention match & l'entité est la bonne ==> c'est un Vrai Positif
                tp.append((begin_idx,sentence,best_cand))
                if (best_cand_score >= opt_thr) : tp_count[current_dict_id] += 1
                else : nb_ftp += 1 # Si le score est inférieur au seuil, le modèle ne le décompte pas. On récupère donc juste le compteur pour évaluer l'impact
                nb_tp += 1
            else:
                fp_entities.append((begin_idx,sentence,best_cand,gt_word))
                cand_false.add(gt)
                if (best_cand_score >= opt_thr) : fp_count[current_dict_id] += 1
                else : nb_fcf += 1 # Si le score est inférieur au seuil, le modèle ne le décompte pas. On récupère donc juste le compteur pour évaluer l'impact
                nb_fp_cand += 1
            nbcand += 1
        gm_find = True
    if(gt == best_cand_id):
        gt_find = True
        # Récupération des Faux Positifs
        #     Si gm_find == False à ce stade-ci, c'est qu'aucune gold_mention ne match avec notre mention (cad qu'on a un Faux Positif)
    return (tp_count[current_dict_id] > 0), gm_list, (fp_count, tp_count, tp, fp_entities, cand_false), gm_find, (nbwunt, nbword, nb_gmdoublon, nbwuntcand, nbcandword), (nb_ftp, nb_fcf, nb_fp_cand, nb_tp)

def compute_score(tp_dict,fp_dict,fn_dict):
    """
    INPUT :
        - tp_dict : dictionnaire (int --> int)
        - fp_dict : dictionnaire (int --> int)
        - fn_dict : dictionnaire (int --> int)
    OUTPUT :
        - micro_pr : float
        - micro_re : float
        - micro_f1 : float
        - macro_pr : float
        - macro_re : float
        - macro_f1 ; float
    """
    docs = tp_dict.keys()
    assert(docs == fp_dict.keys() and docs==fn_dict.keys())
    micro_tp, micro_fp, micro_fn = 0, 0, 0
    macro_pr, macro_re = 0, 0

    for docid in docs:
        tp, fp, fn = tp_dict[docid], fp_dict[docid], fn_dict[docid]
        micro_tp += tp
        micro_fp += fp
        micro_fn += fn

        doc_precision = tp / (tp + fp + 1e-6)
        macro_pr += doc_precision

        doc_recall = tp / (tp + fn + 1e-6)
        macro_re += doc_recall

    #assert(self.gm_num == micro_tp + micro_fn)

    micro_pr = 100 * micro_tp / (micro_tp + micro_fp + 1e-6)
    micro_re = 100 * micro_tp / (micro_tp + micro_fn + 1e-6)
    micro_f1 = 2*micro_pr*micro_re / (micro_pr + micro_re + 1e-6)

    macro_pr = 100 * macro_pr / len(docs)
    macro_re = 100 * macro_re / len(docs)
    macro_f1 = 2*macro_pr*macro_re / (macro_pr + macro_re + 1e-6)

    return micro_pr, micro_re, micro_f1, macro_pr, macro_re, macro_f1
