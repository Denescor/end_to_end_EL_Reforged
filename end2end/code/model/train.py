import argparse
import model.reader as reader
import model.config as config
import os
import sys
try: 
    import tensorflow.compat.v1 as tf
    import tensorflow as tf2
except ImportError: 
    import tensorflow as tf
    import tensorflow as tf2
else:
    tf.disable_eager_execution() #Compatibilité V1 to V2
    tf.disable_v2_behavior() #Compatibilité V1 to V2
    #tf.debugging.set_log_device_placement(True)
#import tensorflow.contrib as contrib
#from fwd9m.tensorflow import patch
from evaluation.metrics import Evaluator, metrics_calculation, threshold_calculation, _filtered_spans_and_gm_gt_list
import time
import pickle
import numpy as np
from preprocessing.util import reverse_dict, load_wiki_name_id_map, load_wikiid2nnid
from model.util import load_train_args
#from model.usemodel import dump_output
import random as rn
#import torch
try: from torch.utils.tensorboard import SummaryWriter
except ImportError: print("Unable to import SummaryWriter")
from functools import reduce
import operator

def create_training_pipelines(args):
    folder = "../data/tfrecords/" + args.experiment_name + \
             ("/allspans/" if args.all_spans_training else "/gmonly/")
    training_dataset = reader.train_input_pipeline([folder + file for file in args.train_datasets], args)
    return training_dataset


def create_el_ed_pipelines(gmonly_flag, filenames, args):
    if filenames is None:
        return [], []

    folder = config.base_folder+"data/tfrecords/" + args.experiment_name + ("/gmonly/" if gmonly_flag else "/allspans/")
    test_datasets = []
    for file in filenames:
        test_datasets.append(reader.test_input_pipeline([folder+file], args))

    return test_datasets, filenames


def tensorboard_writers(graph):
    tf_writers = dict()
    tf_writers["train"] = tf.summary.FileWriter(args.summaries_folder + 'train/', graph)
    tf_writers["ed_pr"] = tf.summary.FileWriter(args.summaries_folder + 'ed_pr/')
    tf_writers["ed_re"] = tf.summary.FileWriter(args.summaries_folder + 'ed_re/')
    tf_writers["ed_f1"] = tf.summary.FileWriter(args.summaries_folder + 'ed_f1/')
    #tf_writers["ed_pr@1"] = tf.summary.FileWriter(args.summaries_folder + 'ed_pr@1/')

    tf_writers["el_pr"] = tf.summary.FileWriter(args.summaries_folder + 'el_pr/')
    tf_writers["el_re"] = tf.summary.FileWriter(args.summaries_folder + 'el_re/')
    tf_writers["el_f1"] = tf.summary.FileWriter(args.summaries_folder + 'el_f1/')
    
    tf_writers["projector"] = SummaryWriter(args.summaries_folder + 'projector/')
    tf.summary.FileWriter(args.summaries_folder + 'projector/')
    return tf_writers


def validation_loss_calculation(model, iterator, dataset_handle, opt_thr, el_mode, name=""):
    # name is the name of the dataset e.g. aida_test.txt, aquaint.txt
    # Run one pass over the validation dataset.
    model.sess.run(iterator.initializer)
    evaluator = Evaluator(opt_thr, name=name)
    while True:
        try:
            if(args.context_bert_lstm): 
                retrieve_l = [model.final_scores, model.cand_entities_len, model.cand_entities,
                              model.begin_span, model.end_span, model.spans_len,
                              model.begin_gm, model.end_gm,
                              model.ground_truth, model.ground_truth_len,
                              model.words_len, model.chunk_id, model.context_bert]
            else :
                retrieve_l = [model.final_scores, model.cand_entities_len, model.cand_entities,
                              model.begin_span, model.end_span, model.spans_len,
                              model.begin_gm, model.end_gm,
                              model.ground_truth, model.ground_truth_len,
                              model.words_len, model.chunk_id]
            result_l = model.sess.run(
                    retrieve_l, 
                    feed_dict={
                        model.input_handle_ph: dataset_handle, 
                        model.dropout: 1})
            if(args.context_bert_lstm): metrics_calculation(evaluator, *(result_l[:-1]), el_mode)
            else: metrics_calculation(evaluator, *result_l, el_mode)
            
        except tf.errors.OutOfRangeError:
            print(name)
            micro_f1, macro_f1 = evaluator.print_log_results(model.tf_writers, args.eval_cnt, el_mode)
            global_result = list(evaluator._score_computation(el_mode))
            break
    return [name]+global_result



def optimal_thr_calc(model, handles, iterators, el_mode):
    val_datasets = args.el_val_datasets if el_mode else args.ed_val_datasets
    tp_fp_scores_labels = []
    fn_scores = []
    for val_dataset in val_datasets:  # 1, 4
        dataset_handle = handles[val_dataset]
        iterator = iterators[val_dataset]
        model.sess.run(iterator.initializer)
        #print("len handles : {}".format(len(handles)))
        #print("handles : \n{}".format(handles[:3]))
        #print("len iterators : {}".format(len(iterators)))
        #print("iterators : \n{}".format(iterators[:3]))
        while True:
            try:
                if(args.context_bert_lstm): 
                    retrieve_l = [model.final_scores, model.cand_entities_len, model.cand_entities,
                                  model.begin_span, model.end_span, model.spans_len,
                                  model.begin_gm, model.end_gm,
                                  model.ground_truth, model.ground_truth_len,
                                  model.words_len, model.chunk_id, model.context_bert]
                else :
                    retrieve_l = [model.final_scores, model.cand_entities_len, model.cand_entities,
                                  model.begin_span, model.end_span, model.spans_len,
                                  model.begin_gm, model.end_gm,
                                  model.ground_truth, model.ground_truth_len,
                                  model.words_len, model.chunk_id]
                #print("len retrieve_l : {}".format(len(retrieve_l)))
                #print("dataset_handle type : {}".format(type(dataset_handle)))
                #print("len dataset_handle : {}".format(len(dataset_handle)))
                #print("dropout type : {}".format(type(model.dropout)))
                #print("len dropout : {}".format(np.shape(model.dropout)))
                #print("dataset_handle :\n {}".format(dataset_handle))
                #print("len dropout : {}")
                result_l = model.sess.run(
                    retrieve_l, 
                    feed_dict={
                        model.input_handle_ph: dataset_handle, 
                        model.dropout: 1})
                if(args.context_bert_lstm) : tp_fp_batch, fn_batch = threshold_calculation(*(result_l[:-1]), el_mode)
                else: tp_fp_batch, fn_batch = threshold_calculation(*result_l, el_mode)
                tp_fp_scores_labels.extend(tp_fp_batch)
                fn_scores.extend(fn_batch)
            except tf.errors.OutOfRangeError:
                break
    return optimal_thr_calc_aux(tp_fp_scores_labels, fn_scores)


def optimal_thr_calc_aux(tp_fp_scores_labels, fn_scores):
    # based on tp_fp_scores and fn_scores calculate optimal threshold
    tp_fp_scores_labels = sorted(tp_fp_scores_labels)   # low --> high
    fn_scores = sorted(fn_scores)
    tp, fp = 0, 0
    fn_idx = len(fn_scores)    # from [0, fn_idx-1] is fn. [fn_idx, len(fn_scores)) isn't.
    # initially i start with a very high threshold which means I reject everything, hence tp, fp =0
    # and all the gold mentions are fn. so fn = len(fn_scores)
    best_thr = tp_fp_scores_labels[-1][0]+1  # the highest (rightmost) possible threshold + 1 (so everything is rejected)
    best_f1 = -1
    # whatever is on the right or at the position we point is included in the tp, fp
    # whatever is on the left remains to be processed-examined
    tp_fp_idx = len(tp_fp_scores_labels)  # similar to fn_idx
    while tp_fp_idx > 0:  # if we point to 0 then nothing on the left to examine (smaller thresholds)
        # from right to left loop
        tp_fp_idx -= 1
        new_thr, label = tp_fp_scores_labels[tp_fp_idx]
        tp += label
        fp += (1 - label)
        while tp_fp_idx > 0 and tp_fp_scores_labels[tp_fp_idx-1][0] == new_thr:
            tp_fp_idx -= 1
            new_thr, label = tp_fp_scores_labels[tp_fp_idx]
            tp += label
            fp += (1 - label)

        while fn_idx > 0 and fn_scores[fn_idx-1] >= new_thr:  # move left one position
            fn_idx -= 1
        assert( 0 <= tp <= len(tp_fp_scores_labels) and
                0 <= fp <= len(tp_fp_scores_labels) and
                0 <= fn_idx <= len(fn_scores))
        precision = 100 * tp / (tp + fp + 1e-6)
        recall = 100 * tp / (tp + fn_idx + 1e-6)
        f1 = 2 * precision * recall / (precision + recall + 1e-6)
        assert(0 <= precision <= 100 and 0 <= recall <= 100 and 0 <= f1 <= 100)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = new_thr

    print('Best validation threshold = %.3f with F1=%.1f ' % (best_thr, best_f1))
    return best_thr, best_f1


def compute_ed_el_scores(model, handles, names, iterators, el_mode):
    # first compute the optimal threshold based on validation datasets.
    if args.hardcoded_thr:
        opt_thr = args.hardcoded_thr
    else:
        opt_thr, val_f1 = optimal_thr_calc(model, handles, iterators, el_mode)
    micro_results = []
    macro_results = []
    global_result = []
    for test_handle, test_name, test_it in zip(handles, names, iterators):
        results = validation_loss_calculation(model, test_it, test_handle, opt_thr,
                                               el_mode=el_mode, name=test_name)
        micro_results.append(results[3])
        macro_results.append(results[6])
        #for i in range(len(results)):
        global_result.append(results)
    val_datasets = args.el_val_datasets if el_mode else args.ed_val_datasets
    if not args.hardcoded_thr and len(val_datasets) == 1 and abs(micro_results[val_datasets[0]] - val_f1) > 0.1:
        print("ASSERTION ERROR: optimal threshold f1 calculalation differs from normal"
              "f1 calculation!!!!", val_f1, "  and ", micro_results[val_datasets[0]])
    return micro_results, global_result

    
def print_score_log(log_file,scores):
    f1_scores = []
    for i in range(len(scores)):
        name = scores[i][0]
        micro_pr, micro_re, micro_f1, macro_pr, macro_re, macro_f1 = scores[i][1:]
        log_file.write("Scores Finales {}:\n".format(name))
        log_file.write( "micro\tP: {:.1f}\tR: {:.1f}\tF1: {:.1f}\n".format(micro_pr,micro_re,micro_f1) )
        log_file.write( "macro\tP: {:.1f}\tR: {:.1f}\tF1: {:.1f}\n".format(macro_pr,macro_re,macro_f1) )
        f1_scores.append( (name, "{:.1f} / {:.1f}".format(micro_f1, macro_f1)) )
    log_file.write("LaTeX\n")  
    log_file.write( "[...] & {} & {} & {} \\\ \n".format(f1_scores[0][0],f1_scores[1][0],f1_scores[2][0]) )
    log_file.write( "[...] & {} & {} & {} \\\ \n".format(f1_scores[0][1],f1_scores[1][1],f1_scores[2][1]) )

    
def train():
    training_dataset = create_training_pipelines(args)

    ed_datasets, ed_names = create_el_ed_pipelines(gmonly_flag=True, filenames=args.ed_datasets, args=args)
    el_datasets, el_names = create_el_ed_pipelines(gmonly_flag=False, filenames=args.el_datasets, args=args)

    input_handle_ph = tf.placeholder(tf.string, shape=[], name="input_handle_ph")
    iterator = tf.data.Iterator.from_string_handle( #tf.contrib.data.Iterator.from_string_handle(
        input_handle_ph, training_dataset.output_types, training_dataset.output_shapes)
    next_element = iterator.get_next()
    #print(next_element)
    print("#####\nwords shape : {}\noutput shape : {}\nhandle shape : {}\n#####".format(np.shape(next_element[1]),training_dataset.output_shapes,np.shape(input_handle_ph)))
    
    if args.ablations:
        from model.model_ablations import Model
    else:
        from model.model import Model
    model = Model(args, next_element)
    model.build()
    model.input_handle_ph = input_handle_ph    # just for convenience so i can access it from everywhere
    #print(tf.global_variables())

    tf_writers = tensorboard_writers(model.sess.graph)
    model.tf_writers = tf_writers   # for accessing convenience

    a = tf.random_uniform([1])
    
    # The `Iterator.string_handle()` method returns a tensor that can be evaluated
    # and used to feed the `handle` placeholder.
    name_log_scores = "./log_exp/{}_{}".format(args.experiment_name, (((args.training_name.split("/"))[-1]).split("_"))[0]  )
    if args.transfert_learning: name_log_scores += "_transfering.txt" #pour différencier le transfert de la baseline
    else: name_log_scores += ".txt"
    try: log_score = open(name_log_scores,"x")
    except: None
    else: log_score.close()
    with model.sess as sess:
        print("random rn flag : {}".format(rn.random()))
        print("random np flag : {}".format(np.random.random()))
        print("random tf flag : {}".format(sess.run(a)))
        
        def ed_el_dataset_handles(datasets):
            test_iterators = []
            test_handles = []
            for dataset in datasets:
                test_iterator = dataset.make_initializable_iterator()
                test_iterators.append(test_iterator)
                test_handles.append(sess.run(test_iterator.string_handle()))
            return test_iterators, test_handles

        training_iterator = training_dataset.make_one_shot_iterator()
        training_handle = sess.run(training_iterator.string_handle())

        ed_iterators, ed_handles = ed_el_dataset_handles(ed_datasets)
        el_iterators, el_handles = ed_el_dataset_handles(el_datasets)
        
        opt_thr, _ = optimal_thr_calc(model, el_handles, el_iterators, el_mode=True)

        # Loop forever, alternating between training and validation.
        best_ed_score = -1
        best_el_score = -1
        termination_ed_score = 0
        termination_el_score = 0
        nepoch_no_imprv = 0  # for early stopping
        train_step = 0
        if args.eval_cnt >= args.nepoch_max: print("!!!! epoch limit already done !!!!")
        while args.eval_cnt < args.nepoch_max:
        # Mettre "True" pour le early stopping
        # Mettre "args.eval_cnt < [n]" pour effectuer [n] epoch
            total_train_loss = 0
            # for _ in range(args.steps_before_evaluation):          # for training based on training steps
            wall_start = time.time()
            while ( (time.time() - wall_start) / 60 ) <= args.evaluation_minutes:
                train_step += 1
                if args.ffnn_l2maxnorm:
                    sess.run(model.ffnn_l2normalization_op_list)
                if(args.eval_cnt>=0):
                    _, loss, w_param, projectors = sess.run([model.train_op, model.loss, model.tf_param_summaries, model.projector_embeddings], 
                                                          feed_dict={input_handle_ph: training_handle,
                                                              model.dropout: args.dropout,
                                                              model.lr: model.args.lr})
                    
                # else:
                    # _, loss = sess.run([model.train_op, model.loss],  
                                                  # feed_dict={input_handle_ph: training_handle
                                                  # model.dropout: args.dropout,
                                                  # model.lr: model.args.lr})
                total_train_loss += loss

            args.eval_cnt += 1

            tf_writers["train"].add_summary(w_param, args.eval_cnt)
            summary = tf.Summary(value=[tf.Summary.Value(tag="total_train_loss", simple_value=total_train_loss)])
            tf_writers["train"].add_summary(summary, args.eval_cnt)
            
            if args.eval_cnt == args.nepoch_max-1 or args.eval_cnt == 1:
                # On sauve les embeddings au début et à la fin (moins de données à stocker).
                print("shape word : {}".format(np.shape(projectors[0])))
                print("shape entities : {}".format(np.shape(projectors[1])))
                print("shape context : {}".format(np.shape(projectors[2])))
                print("shape mention : {}".format(np.shape(projectors[3])))
                label_tensor = ["word_embeddings", "entities_embeddings", "context_bi_lstm", "mention_embeddings"]
                if(args.context_bert_lstm): label_tensor.append("bert_embeddings")
                assert len(label_tensor) == len(projectors), "label : {}\nprojectors : {}".format(len(label_tensor),len(projectors))
                #for index_tensor in range(len(projectors)):
                #    index_batch = 0
                #    for i, tensor in enumerate(projectors):
                #        ts = np.shape(tensor)
                #        nts = (reduce(operator.mul,ts[0:-1],1),ts[-1])
                #        index_set = 10*args.eval_cnt+index_batch
                        #print("ts : {}\nnew ts : {}".format(ts,nts))
                #        tensor = np.reshape(tensor,nts)
                #        tf_writers["projector"].add_embedding(tensor, tag=label_tensor[index_tensor], 
                #                                            metadata=list(range(len(tensor))),
                #                                            global_step=index_set)
                #        index_batch+=1

            print("args.eval_cnt = ", args.eval_cnt)
            #summary = sess.run(model.merged_summary_op)
            #tf_writers["train"].add_summary(summary, args.eval_cnt)

            wall_start = time.time()
            comparison_ed_score = comparison_el_score = -0.1
            if ed_names:
                print("Evaluating ED datasets")
                ed_scores, ed_global_scores = compute_ed_el_scores(model, ed_handles, ed_names, ed_iterators, el_mode=False)
                comparison_ed_score = np.mean(np.array(ed_scores)[args.ed_val_datasets])
            if el_names:
                print("Evaluating EL datasets")
                el_scores, el_global_scores = compute_ed_el_scores(model, el_handles, el_names, el_iterators, el_mode=True)
                comparison_el_score = np.mean(np.array(el_scores)[args.el_val_datasets])
            print("Evaluation duration in minutes: ", (time.time() - wall_start) / 60)

            #comparison_ed_score = (ed_scores[1] + ed_scores[4]) / 2   # aida_dev + acquaint
            #comparison_score = ed_scores[1]  # aida_dev
            if model.args.lr_decay > 0:
                model.args.lr *= model.args.lr_decay  # decay learning rate
            text = ""
            best_ed_flag = False
            best_el_flag = False
            # otherwise not significant improvement 75.2 to 75.3 micro_f1 of aida_dev
            if comparison_ed_score >= best_ed_score + 0.1: # args.improvement_threshold:
                text = "- new best ED score!" + " prev_best= " + str(best_ed_score) +\
                       " new_best= " + str(comparison_ed_score)
                best_ed_flag = True
                best_ed_score = comparison_ed_score
            if comparison_el_score >= best_el_score + 0.1: #args.improvement_threshold:
                text += "- new best EL score!" + " prev_best= " + str(best_el_score) +\
                       " new_best= " + str(comparison_el_score)
                best_el_flag = True
                best_el_score = comparison_el_score
            if best_ed_flag or best_el_flag: # keep checkpoint
                print(text)
                if args.nocheckpoints is False: 
                    model.save_session(args.eval_cnt, best_ed_flag, best_el_flag)
                    model.save_weight_model(args, model, sess, input_handle_ph, el_handles[args.el_val_datasets[0]], args.checkpoints_folder)
            # check for termination now.
            if comparison_ed_score >= termination_ed_score + args.improvement_threshold\
                    or comparison_el_score >= termination_el_score + args.improvement_threshold:
                print("significant improvement. reset termination counter")
                termination_ed_score = comparison_ed_score
                termination_el_score = comparison_el_score
                nepoch_no_imprv = 0
            else:
                nepoch_no_imprv += 1
                if nepoch_no_imprv >= args.nepoch_no_imprv:
                    print("- early stopping {} epochs without "
                                     "improvement".format(nepoch_no_imprv))
                    # Dump Output
                    #dump_output(args, model, el_handles, el_names, el_iterators, opt_thr)
                    # Write Final Scores
                    pass
                    #break
            log_args(args, args.output_folder+"train_args.txt")
            sys.stdout.flush()
            os.system("chgrp -R endtoendEL {}".format(args.training_name))
        log_score = open(name_log_scores,"a")
        log_score.write(20*"#"+"BEGIN"+20*"#"+"\n")
        log_score.write("\t -- epoch : {}/{} --".format(args.eval_cnt, args.nepoch_max))
        try: print_score_log(log_score, el_global_scores)
        except: pass
        log_score.write(20*"#"+"END"+20*"#"+"\n")
        log_score.close()
        # Close model and Tee
        terminate()
        model.close_session()
            #dump_output(model, el_handles, el_names, el_iterators) #Commande de test #TOREMOVE
            #terminate()
            #break

def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", default="alldatasets_perparagr", #"standard",
                        help="under folder data/tfrecords/")
    parser.add_argument("--training_name", default=None,
                        help="under folder data/tfrecords/")
    parser.add_argument("--shuffle_capacity", type=int, default=500)
    parser.add_argument("--debug", type=bool, default=False)

    parser.add_argument("--nepoch_max", type=int, default=50)
    parser.add_argument("--nepoch_no_imprv", type=int, default=5)
    parser.add_argument("--improvement_threshold", type=float, default=0.3, help="if improvement less than this then"
                            "it is considered not significant and we have early stopping.")
    parser.add_argument("--clip", type=int, default=-1, help="if negative then no clipping")
    parser.add_argument("--lr_decay", type=float, default=-1.0, help="if negative then no decay")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--lr_method", default="adam")
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--train_ent_vecs", dest='train_ent_vecs', action='store_true')
    parser.add_argument("--no_train_ent_vecs", dest='train_ent_vecs', action='store_false')
    parser.set_defaults(train_ent_vecs=False)

    parser.add_argument("--dim_word_emb", type=int, default=300)
    parser.add_argument("--dim_ent_emb", type=int, default=300)
    
    parser.add_argument("--steps_before_evaluation", type=int, default=10000)
    parser.add_argument("--evaluation_minutes", type=int, default=15, help="every this number of minutes pause"
                                                                           " training and run an evaluation epoch")
    parser.add_argument("--dim_char", type=int, default=100)
    parser.add_argument("--hidden_size_char", type=int, default=100, help="lstm on chars")
    parser.add_argument("--hidden_size_lstm", type=int, default=100, help="lstm on word embeddings")
    parser.add_argument("--transformer", dest="transformer", action="store_true", help="replace the bi-LSTM of context by a Transformer")
    parser.set_defaults(transformer=False)
    parser.add_argument("--transformer_neurons", type=int, default=300, help="size of transformer's FeedFoward")
    parser.add_argument("--transformer_size", type=int, default=1, help="number of encoder/decoder of the transformer")

    parser.add_argument("--use_chars", dest="use_chars", action='store_true', help="use character embeddings or not")
    parser.add_argument("--no_use_chars", dest="use_chars", action='store_false')
    parser.set_defaults(use_chars=True)
    
    parser.add_argument("--use_word_BERT", dest="word_bert", action="store_true", help="use BERT word embeddings instead of Word2Vec")
    parser.add_argument("--no_use_word_BERT", dest="word_bert", action="store_false", help="default option")
    parser.set_defaults(word_bert=False)
    parser.add_argument("--use_context_BERT", dest="context_bert", action="store_true", help="use BERT contextual embeddings instead of Word2Vec + bi-LSTM")
    parser.add_argument("--no_use_context_BERT", dest="context_bert", action="store_false", help="default option")
    parser.set_defaults(context_bert=False)
    parser.add_argument("--use_addition_BERT", dest="context_bert_lstm", action="store_true", help="use BERT contextual embeddings in addition the bi-LSTM")
    parser.add_argument("--no_use_addition_BERT", dest="context_bert_lstm", action="store_false", help="default option")
    parser.set_defaults(context_bert_lstm=False)    

    parser.add_argument("--model_heads_from_bilstm", type=bool, default=False,
                        help="use the bilstm vectors for the head instead of the word embeddings")
    parser.add_argument("--span_boundaries_from_wordemb", type=bool, default=False, help="instead of using the "
                                "output of contextual bilstm for start and end of span we use word+char emb")
    parser.add_argument("--span_emb", default="boundaries_head", help="boundaries for start and end, and head")


    parser.add_argument("--max_mention_width", type=int, default=10)
    parser.add_argument("--use_features", type=bool, default=False, help="like mention width")
    parser.add_argument("--feature_size", type=int, default=20)   # each width is represented by a vector of that size


    parser.add_argument("--ent_vecs_regularization", default="l2dropout", help="'no', "
                                "'dropout', 'l2', 'l2dropout'")
    parser.add_argument("--entity_vecs_name", default="ent_vecs.npy", help="choose the entity embedding to use : 'ent_vecs.npy', 'ent_vecs_reinforced.npy', 'ent_vecs_glove.npy'")
    parser.add_argument("--wikiid2nnid_name", default="wikiid2nnid.txt", help="choose the correct wiki id dict with correct number of entry corresponding to the entity vectors")

    parser.add_argument("--span_emb_ffnn", default="0_0", help="int_int  the first int"
                        "indicates the number of hidden layers and the second the hidden size"
                        "so 2_100 means 2 hidden layers of width 100 and then projection to output size"
                        ". 0_0 means just projecting without hidden layers")
    parser.add_argument("--final_score_ffnn", default="1_100", help="int_int  look span_emb_ffnn")


    parser.add_argument("--gamma_thr", type=float, default=0.2)

    parser.add_argument("--nocheckpoints", type=bool, default=False)
    parser.add_argument("--checkpoints_num", type=int, default=1, help="maximum number of checkpoints to keep")

    parser.add_argument("--ed_datasets", default="")
    parser.add_argument("--ed_val_datasets", default="1", help="based on these datasets pick the optimal"
                                                               "gamma thr and also consider early stopping")
                                                #--ed_val_datasets=1_4  # aida_dev, aquaint
    parser.add_argument("--el_datasets", default="")
    parser.add_argument("--el_val_datasets", default="1") #--el_val_datasets=1_4   # aida_dev, aquaint

    parser.add_argument("--train_datasets", default="aida_train.txt")
                        #--train_datasets=aida_train.txt_z_wikidumpRLTD.txt

    parser.add_argument("--continue_training", type=bool, default=False,
                        help="if true then just restore the previous command line"
                             "arguments and continue the training in exactly the"
                             "same way. so only the experiment_name and "
                             "training_name are used from here. Retrieve values from"
                             "latest checkpoint.")
    parser.add_argument("--transfert_learning", type=bool, default=False, help="restore the previous session but with a different datasets and/or entities as precised")
    parser.add_argument("--pretrained_model", default="data/final_model.pkl", help="folder (from base_folder) where find the pickle file with the pretrained layers")
    parser.add_argument("--onleohnard", type=bool, default=False)

    parser.add_argument("--comment", default="", help="put any comment here that describes your experiment"
                                                      ", for logging purposes only.")

    parser.add_argument("--all_spans_training", type=bool, default=False)
    parser.add_argument("--fast_evaluation", type=bool, default=False, help="if all_spans training then evaluate only"
                                            "on el tests, corresponding if gm training evaluate only on ed tests.")


    parser.add_argument("--entity_extension", default=None, help="extension_entities or extension_entities_all etc")

    parser.add_argument("--nn_components", default="pem_lstm", help="each option is one scalar, then these are fed to"
                            "the final ffnn and we have the final score. choose any combination you want: e.g"
                            "pem_lstm_attention_global, pem_attention, lstm_attention, pem_lstm_global, etc")
    parser.add_argument("--attention_K", type=int, default=100, help="K from left and K from right, in total 2K")
    parser.add_argument("--attention_R", type=int, default=30, help="hard attention")
    parser.add_argument("--attention_use_AB", type=bool, default=False)
    parser.add_argument("--attention_on_lstm", type=bool, default=False, help="instead of using attention on"
                    "original pretrained word embedding. use it on vectors or lstm, "
                    "needs also projection now the context vector x_c to 100 dimensions")
    parser.add_argument("--attention_ent_vecs_no_regularization", type=bool, default=False)
    parser.add_argument("--attention_retricted_num_of_entities", type=int, default=None,
                        help="instead of using 30 entities for creating the context vector we use only"
                             "the top x number of entities for reducing noise.")
    parser.add_argument("--global_thr", type=float, default=0.1)   # 0.0, 0.05, -0.05, 0.2
    parser.add_argument("--global_mask_scale_each_mention_voters_to_one", type=bool, default=False)
    parser.add_argument("--global_topk", type=int, default=None)
    parser.add_argument("--global_gmask_based_on_localscore", type=bool, default=False)   # new
    parser.add_argument("--global_topkthr", type=float, default=None)   # 0.0, 0.05, -0.05, 0.2
    parser.add_argument("--global_score_ffnn", default="1_100", help="int_int  look span_emb_ffnn")
    parser.add_argument("--global_one_loss", type=bool, default=False)
    parser.add_argument("--global_norm_or_mean", default="norm")
    parser.add_argument("--global_topkfromallspans", type=int, default=None)
    parser.add_argument("--global_topkfromallspans_onlypositive", type=bool, default=False)
    parser.add_argument("--global_gmask_unambigious", type=bool, default=False)

    parser.add_argument("--hardcoded_thr", type=float, default=None, help="if this is specified then we don't calculate"
                           "optimal threshold based on the dev dataset but use this one.")
    parser.add_argument("--ffnn_dropout", dest="ffnn_dropout", action='store_true')
    parser.add_argument("--no_ffnn_dropout", dest="ffnn_dropout", action='store_false')
    parser.set_defaults(ffnn_dropout=True)
    parser.add_argument("--ffnn_l2maxnorm", type=float, default=None, help="if positive"
                        " then bound the Frobenius norm <= value for the weight tensor of the "
                        "hidden layers and the output layer of the FFNNs")
    parser.add_argument("--ffnn_l2maxnorm_onlyhiddenlayers", type=bool, default=False)

    parser.add_argument("--cand_ent_num_restriction", type=int, default=None, help="for reducing memory usage and"
                                "avoiding OOM errors in big NN I can reduce the number of candidate ent for each span")
    # --ed_datasets=  --el_datasets="aida_train.txt_z_aida_dev.txt"     which means i can leave something empty
    # and i can also put "" in the cla

    parser.add_argument("--no_p_e_m_usage", type=bool, default=False, help="use similarity score instead of "
                                                                           "final score for prediction")
    parser.add_argument("--pem_without_log", type=bool, default=False)
    parser.add_argument("--pem_buckets_boundaries", default=None,
                        help="example: 0.03_0.1_0.2_0.3_0.4_0.5_0.6_0.7_0.8_0.9_0.99")
    # the following two command line arguments
    parser.add_argument("--gpem_without_log", type=bool, default=False)
    parser.add_argument("--gpem_buckets_boundaries", default=None,
                        help="example: 0.03_0.1_0.2_0.3_0.4_0.5_0.6_0.7_0.8_0.9_0.99")
    parser.add_argument("--stage2_nn_components", default="local_global", help="each option is one scalar, then these are fed to"
                                                                    "the final ffnn and we have the final score. choose any combination you want: e.g"
                                                                    "pem_local_global, pem_global, local_global, global, etc")
    parser.add_argument("--ablations", type=bool, default=False)
    args = parser.parse_args()

    return preprocess_args(args)
    
def preprocess_args(args):
    if(args.word_bert):
        print("args forced : span_boundaries_from_wordemb = False ; model_heads_from_bilstm = False")# ; use_chars = False")
        args.span_boundaries_from_wordemb = False
        args.model_heads_from_bilstm = False
        #args.use_chars=False

    if(args.context_bert):
        print("args forced : span_boundaries_from_wordemb = True ; model_heads_from_bilstm = False ; attention_on_lstm = False ; use_chars = False")
        args.span_boundaries_from_wordemb = True
        args.model_heads_from_bilstm = False
        args.attention_on_lstm = False
        args.use_chars=False
        
    
    if(args.context_bert_lstm):
        print("args forced : span_boundaries_from_wordemb = False ; context_bert = False ; word_bert : False")
        args.span_boundaries_from_wordemb = False
        args.context_bert = False
        args.word_bert = False
    
    if args.training_name is None:
        from datetime import datetime
        args.training_name = "{:%d_%m_%Y____%H_%M}".format(datetime.now())

    temp = "all_spans_" if args.all_spans_training else ""
    args.output_folder = config.base_folder+"data/tfrecords/" + \
                         args.experiment_name+"/{}training_folder/".format(temp)+\
                         args.training_name+"/"

    if args.continue_training:
        print("continue training...")
        train_args = load_train_args(args.output_folder, "train_continue")
        train_args.running_mode = "train_continue"
        return train_args
    elif args.transfert_learning: 
        print("Transfert Learning ongoing...")
        args.running_mode = "transfert_learning"
    else:
        args.running_mode = "train"                 # "evaluate"  "ensemble_eval"  "gerbil"

    if os.path.exists(args.output_folder) and not (args.continue_training):
        print("!!!!!!!!!!!!!!\n"
              "experiment: ", args.output_folder, "already exists and args.continue_training=False."
                            "folder will be deleted in 20 seconds. Press CTRL+C to prevent it.")
        time.sleep(20)
        import shutil
        shutil.rmtree(args.output_folder)

    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    args.checkpoints_folder = args.output_folder + "checkpoints/"
    if args.onleohnard:
        args.checkpoints_folder = "/cluster/home/nkolitsa/checkpoints_folder/"+\
            args.experiment_name + "/" + args.training_name + "/"

    args.summaries_folder = args.output_folder + "summaries/"
    if not os.path.exists(args.summaries_folder):
        os.makedirs(args.summaries_folder)

    args.ed_datasets = args.ed_datasets.split('_z_') if args.ed_datasets != "" else None
    args.el_datasets = args.el_datasets.split('_z_') if args.el_datasets != "" else None
    args.train_datasets = args.train_datasets.split('_z_') if args.train_datasets != "" else None

    args.ed_val_datasets = [int(x) for x in args.ed_val_datasets.split('_')]
    args.el_val_datasets = [int(x) for x in args.el_val_datasets.split('_')]

    args.span_emb_ffnn = [int(x) for x in args.span_emb_ffnn.split('_')]
    args.final_score_ffnn = [int(x) for x in args.final_score_ffnn.split('_')]
    args.global_score_ffnn = [int(x) for x in args.global_score_ffnn.split('_')]

    args.eval_cnt = 0
    args.zero = 1e-6

    if args.pem_buckets_boundaries:
        args.pem_buckets_boundaries = [float(x) for x in args.pem_buckets_boundaries.split('_')]
    if args.gpem_buckets_boundaries:
        args.gpem_buckets_boundaries = [float(x) for x in args.gpem_buckets_boundaries.split('_')]

    if args.fast_evaluation:
        if args.all_spans_training:  # destined for el so omit the evaluation on ed
            args.ed_datasets = None
        else:
            args.el_datasets = None
    return args


def log_args(args, filepath):
    with open(filepath, "w") as fout:
        attrs = vars(args)
        # {'kids': 0, 'name': 'Dog', 'color': 'Spotted', 'age': 10, 'legs': 2, 'smell': 'Alot'}
        fout.write('\n'.join("%s: %s" % item for item in attrs.items()))

    with open(args.output_folder+"train_args.pickle", 'wb') as handle:
        pickle.dump(args, handle)

def terminate():
    tee.close()
    with open(args.output_folder+"train_args.pickle", 'wb') as handle:
        pickle.dump(args, handle)


if __name__ == "__main__":
    from tfdeterminism import patch
    args = _parse_args()
    print(20*"#"+"BEGIN"+20*"#")
    print(args)
    SEED = 1234
    os.environ['PYTHONHASHSEED']=str(SEED)
    tf.set_random_seed(SEED)
    tf2.random.set_seed(SEED)
    rn.seed(SEED)
    np.random.seed(SEED)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)
    patch()
    log_args(args, args.output_folder+"train_args.txt")
    from model.util import Tee
    tee = Tee(args.output_folder+'log.txt', 'a')
    try:
        train()
    except KeyboardInterrupt:
        terminate()
    print(20*"#"+"END"+20*"#")

