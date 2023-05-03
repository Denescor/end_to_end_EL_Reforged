import argparse
import os
import sys

import torch
import tensorflow.compat.v1 as tf
import tensorflow as tf2
from tfdeterminism import patch

import time
import pickle as pk
import numpy as np

import operator
import random as rn
from torch.utils.tensorboard import SummaryWriter
from functools import reduce

import model.reader as reader
import model.config as config
import model.train as train
import model.evaluate as evaluate
import model.fun_eval as fun_eval
#import plotting.tsne as tsne_plot
from model.model import Model
from model.util import load_train_args, load_ent_vecs
from preprocessing.util import reverse_dict, load_wiki_name_id_map, load_wikiid2nnid
from evaluation.metrics import Evaluator, metrics_calculation, threshold_calculation, _filtered_spans_and_gm_gt_list

"""
#############################################################################################################################################################################################
# FONCTIONS                                                                                                                                                                                 #
# - tsne_fusion : GÉNÈRE UNE REPRÉSENTATION FUSIONNANT SUR UN MÊME GRAPHES LES T-SNE DES EMBEDDINGS DE MÊME TYPE POUR DIFFÉRENTS MODÈLES. S'OCCUPE DE RÉCUPÉRER LES EMBEDDINGS NÉCESSAIRES. #
# - melting_embeddings : CONCATÈNE LES EMBEDDINGS DE DIFFÉRENTS TYPES POUR EN FAIRE UN AFFICHAGE TENSORBOARD. /!\ INNOPÉRANT, NE FONCTIONNE PAS NI NE PEUT ÊTRE DÉBUGGUÉ /!\                #
# - extract_embeddings : EXTRAIT LES EMBEDDINGS (METADATA INCLUS POUR LES ENTITÉS) DEMANDÉS POUR TOUS LES MODÈLES DEMANDÉS                                                                  #
#############################################################################################################################################################################################
"""

tf.disable_eager_execution() #Compatibilité V1 to V2
tf.disable_v2_behavior() #Compatibilité V1 to V2
#tf.debugging.set_log_device_placement(True)

def extract_embeddings(melting_args, to_extract=set([0,1,2,3]), verbose=True):
    final_emb_model = dict()
    for experiment_name, training_name in melting_args: # OLDVERSION for el_datasets, el_names, model in list_model:
        ## RETRIEVE THE MODEL
        (el_datasets, el_names, model), eval_args, train_args = fun_eval.retrieve_model_args(experiment_name, training_name)
        model_name = model.args.experiment_name
        final_emb_model[model_name] = [[],[],[],[],[]]
        # extract the embeddings from the model
        for datasets, names in zip([el_datasets], [el_names]):
            ## RUN THE MODEL
            model.restore_session("el")
            test_iterator, test_handle = fun_eval.ed_el_testset_handles(model.sess, names, datasets)
            model.sess.run(test_iterator.initializer)
            cand_entities, ground_truth, projectors, _ = fun_eval.run_model_ent_and_proj(model, test_handle, verbose=verbose)
            ## TRAITEMENT PROJECTORS
            for index_tensor in range(len(projectors)):
                if index_tensor > 3: break #le cas spécifique du "bert_embeddings" ne nous intéresse pas
                if index_tensor not in to_extract: continue #only requested
                for index_batch in range(len(projectors[index_tensor])):
                    tensor = projectors[index_tensor][index_batch]
                    ts = np.shape(tensor)
                    nts = (reduce(operator.mul,ts[0:-1],1),ts[-1])
                    tensor = np.reshape(tensor,nts)
                    if index_tensor == 1: # catch metadata for entities
                        metadata = fun_eval.generate_cand_metadata(cand_entities[index_batch], eval_args, details=True)
                        final_emb_model[model_name][-1].append(metadata)
                    final_emb_model[model_name][index_tensor].append(tensor)
        tf.reset_default_graph()
    return final_emb_model

def tsne_fusion(melting_args, summary_folder, use_old_data=False, verbose=True):
    final_emb_summary = SummaryWriter(summary_folder + 'global_embedding_projector/')
    if use_old_data :
        print("LOADING DATA... ",end="")
        names = pk.load(open(summary_folder+"names_embs.pk","rb"))
        embs = pk.load(open(summary_folder+"embs_embs.pk","rb"))
        print("DONE")
    else:
        print("GENERATING AND SAVING DATA")
        top = time.time()
        extraction = set([0,2,3]) # 0 : Word EMbedding - 1 : Entities Embeddings - 2 : Context Embeddings - 3 : Mention Embeddings
        final_emb_model = extract_embeddings(melting_args,to_extract = extraction)
        names = list(final_emb_model.keys()) # list of string. Dimension = nb de model chargés
        embs = list(final_emb_model.values()) # list of list of list of tensor. Dimension = nb model x 4 x 4
        pk.dump(names,open(summary_folder+"names_embs.pk","wb"))
        pk.dump(embs,open(summary_folder+"embs_embs.pk","wb"))
        print("DONE IN {}s".format(int(time.time()-top)))
    # Print data stats
    labels = ["word_embeddings_global", "entities_embeddings_global", "context_bi_lstm_global", "mention_embeddings_global"]
    print("data about embeddings :")
    print("names:\n\tlen : {}\n\ttype : {}".format(len(names),type(names),))
    print("embs:\n\tlen : {}x{}x{}\n\ttype : {}x{}x{}".format(len(embs),len(embs[0]),len(embs[0][0]),type(embs),type(embs[0]),type(embs[0][0])))
    # Create TSNE
    if verbose : print("Create TSNE model ...",end="")
    top = time.time()
    tsne_model = tsne_plot._TSNE(iteration=5000,component=3,learning_rate=10,perplexity=30)
    if verbose : print("... done in {}s\n##### ##### #####".format(int(time.time()-top)))
    toptop = time.time()
    if melting_args.compact_batch :
        for i in range(4):
            if len(embs[0][i]) == 0: continue
            for j in range(1,4):
                for x in range(len(embs)):
                    print("shape : {} <- {}".format(np.shape(embs[x][i][0]),np.shape(embs[x][i][j]))) 
                    embs[x][i][0] = np.append(embs[x][i][0],embs[x][i][j],axis=0)
            embs[x][i] = [embs[x][i][0]]
        print("de-batch embs:\n\tlen : {}x{}x{}\n\ttype : {}x{}x{}".format(len(embs),len(embs[0]),len(embs[0][0]),type(embs),type(embs[0]),type(embs[0][0])))    
        lenJ = 1
    else : lenJ = 4
    for i in range(4): # on a 4 embeddings
        if len(embs[0][i]) == 0 : continue
        for j in range(lenJ) : # on a un batch de taille lenJ
            name = labels[i]+"_{}".format("all") if melting_args.compact_batch else labels[i]+"_{}".format(j)
            file_name = summary_folder+"images/{}".format(name)
            metadata = []
            legend = []
            for x in range(len(names)): 
                metadata.extend([x for plouf in range(len(embs[x][i][j]))])
                legend.append(names[x])
            if verbose : 
                print("Generating TSNE embeddings ...",end="")
                sys.stdout.flush() 
            top = time.time()
            TSNE_embs = [tsne_plot._TSNE_fit(np.array(embs[x][i][j]),tsne_model) for x in range(len(embs))]
            if verbose : 
                print("... done in {}s".format(int(time.time()-top)))
                sys.stdout.flush() 
            X = TSNE_embs[0] 
            for tsne_emb in TSNE_embs[1:]: X = np.append(X,tsne_emb,axis=0) #np.concatenate(TSNE_embs,axis=0)
            if verbose: print("TSNE shape : {}\nConcat Shape : {}\nplotting TSNE ...".format(" ".join([str(np.shape(t)) for t in TSNE_embs]),np.shape(X)),end="")
            top = time.time()
            tsne_plot.PlotterMultiTriDimension(1,[X], #La concaténation des représentations d'embeddings à afficher
                                                 [legend], #La légende : le nom des modèles (x)
                                                 [name], #Le titre du graphe : le type d'embeddings représenté
                                                 [metadata], #les int associés à chaque embeddings permettant de les regrouper par modèle
                                                 file_name+".png")
            tsne_plot.PlotterMultiTriDimension(len(embs),
                                               TSNE_embs, #list des représentations d'embeddings à afficher (3)
                                               [[names[x]] for x in range(len(names))], #La Légende : le nom du modèle pour chacune des représentation (1 par groupe - 3 groupes)
                                               [name+"_{}".format(x) for x in names], #Le titre des graphes : le type d'embeddigs représenté + nom du modèle
                                               [[x for plouf in range(len(embs[x][i][j]))] for x in range(len(names))], #les int associé à chaque embeddings (1 couleur par groupe - 3 groupe) 
                                               file_name+"_separated.png")
            if verbose : 
                print("... done in {}s\nImage to '{}'".format(int(time.time()-top),file_name))
                sys.stdout.flush() 
    if verbose : print("##### ##### #####\nALL DONE IN {}s".format(int(time.time()-toptop)))
    
def melting_embeddings(melting_args, summary_folder, use_old_data=False, verbose=True):
    final_emb_summary = SummaryWriter(summary_folder + 'global_embedding_projector/fusion_projectors/')
    if use_old_data :
        print("LOADING DATA")
        names = pk.load(open(summary_folder+"names_embs.pk","rb"))
        embs = pk.load(embs,open(summary_folder+"embs_embs.pk","rb"))
    else:
        print("GENERATING AND SAVING DATA")
        final_emb_model = extract_embeddings(melting_args, to_extract=set([1])) #to_extract = [1] : only entities
        names = list(final_emb_model.keys()) # list of string. Dimension = nb de model chargés
        embs = list(final_emb_model.values()) # list of list of list of tensor. Dimension = nb model x 4 x 4
        pk.dump(names,open(summary_folder+"names_embs.pk","wb"))
        pk.dump(embs,open(summary_folder+"embs_embs.pk","wb"))
    # Print data stats
    labels = ["word_embeddings_global", "entities_embeddings_global", "context_bi_lstm_global", "mention_embeddings_global"]
    print("data about embeddings :")
    print("names:\n\tlen : {}\n\ttype : {}".format(len(names),type(names),))
    print("embs:\n\tlen : {}x{}x{}\n\ttype : {}x{}x{}".format(len(embs),len(embs[0]),len(embs[0][0]),type(embs),type(embs[0]),type(embs[0][0])))
    for i in range(4): # on a 4 embeddings
        if len(embs[0][i]) == 0 : continue
        for j in range(4) : # on a un batch de taille 4
            step_zero = 100*i+10*j
            label = labels[i]+"_{}".format(j)
            global_emb = np.concatenate([np.array(embs[x][i][j]) for x in range(len(embs))],axis=-1) #IMPOSSIBLE A CONCATENER
            metadata = []
            for x in range(len(names)): metadata.extend([names[x] for plouf in range(len(embs[x][i][j]))])
            if(verbose):
                print("test range:\n\tarrayxlen(embs) = {}\n\tlen(names)xrange(embs) = {}".format((len(embs[0][i][j])*len(embs)),
                                                                                                  (len(names)*len(embs[0][i][j]))))
                print("\tconcatenation = {}".format(sum([len(plouf) for plouf in [np.array(embs[x][i][j]) for x in range(len(embs))]])))
                print("data global_emb:\n\tmetadata : {}\n\tnb embs : {}\n\tlabel : {}".format(len(metadata),len(global_emb),label))
            #final_emb_summary.add_embedding(global_emb, tag=label, 
            #                                metadata=metadata,
            #                                global_step=step_zero)
            if i==1 : 
                for x in range(len(embs)): 
                    if verbose : print("entities summary :\n\tentities embs : {}\n\tentities labels : {}".format(len(embs[x][i][j]),len(embs[x][-1][j])))
                    step = step_zero + x
                    final_emb_summary.add_embedding(embs[x][i][j], tag=label, 
                                                    metadata=embs[x][-1][j],
                                                    global_step=step)

def generating_metadata(args):
    save_file = "../data/basic_data/idtoent.txt"
    args.entity_extension = None
    args.wikiid2nnid_name = "wikiid2nnid_FR.txt"
    args.entity_vecs_name = "ent_vecs_fr_true_15.npy"
    entities_embeddings = load_ent_vecs(args)
    idtocand, _, _ = fun_eval.load_data(args, verbose=True)
    idonly = list(idtocand.keys())
    print("idtocand : {}".format(list(idtocand.items())[:10]))
    print("id min : {}\nid max : {}".format(min(idonly), max(idonly)))
    print("ent size = {}\nidtocand size = {}".format(len(entities_embeddings),len(idtocand)))
    with open(save_file,"w") as txt:
        for i in range(len(entities_embeddings)):
            if i in idtocand: txt.write("{}\n".format(idtocand[i]))
            else: txt.write("<wunk>\n")
    print("metadata done at '{}'".format(save_file))
    

     
def melt_to_melting_args(melt_args):
    experiment_names = melt_args.experiment_name.split("_z_")
    training_names = melt_args.training_name.split("_z_") 
    melting_args = list(zip(experiment_names,training_names))
    return melting_args
        
def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", help="under folder data/tfrecords/\nList of folder separate by '_z_'")
    parser.add_argument("--training_name", help="under folder data/tfrecords/\nList of folder separate by '_z_'")
    parser.add_argument("--load_existing_data",dest="load_data",action="store_true")
    parser.add_argument("--compact_batch",dest="compact_batch",action="store_true")
    parser.set_defaults(load_data=False)
    parser.set_defaults(compact_batch=False)
    args = parser.parse_args()
    return args
    
if __name__=="__main__":
    args = _parse_args() #parse list of model to load and use
    summary_folder = "../data/tfrecords/"
    generating_metadata(args)
    #melting_args = melt_to_melting_args(args)
    #print("nb model : {}\ntype model : \n\tel_dataset : {}\n\tel_names : {}\n\tmodel : {}".format(len(list_model),type(list_model[0][0]),type(list_model[0][1]),type(list_model[0][2])))
    #melting_embeddings(melting_args, summary_folder, use_old_data=args.load_data))
    #tsne_fusion(melting_args, summary_folder, use_old_data=args.load_data)
    
