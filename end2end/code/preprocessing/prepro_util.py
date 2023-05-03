import pickle
from collections import defaultdict, namedtuple
import numpy as np
import argparse

import os
import model.config as config
import preprocessing.util as util
from termcolor import colored
try: import tensorflow.compat.v1 as tf
except ImportError: import tensorflow as tf
#import torch

word2vec = ""

class VocabularyCounter(object):
    """counts the frequency of each word and each character in the corpus. With each
    file that it processes it increases the counters. So one frequency vocab for all the files
    that it processes."""
    def __init__(self, vocab_name="vocab_freq.pickle", new_datasets="new_datasets", lowercase_emb=False):
        import gensim
        self.model = gensim.models.KeyedVectors.load_word2vec_format(
        config.base_folder+"data/basic_data/wordEmbeddings/Word2Vec/"+word2vec, binary=True)
        """lowercase_emb=False if True then we lowercase the word for counting of
        frequencies and hence for finding the pretrained embedding."""
        self.word_freq = defaultdict(int)
        self.char_freq = defaultdict(int)    # how many times each character is encountered
        self.lowercase_emb = lowercase_emb
        self.not_in_word2vec_cnt = 0
        self.all_words_cnt = 0
        self.vocab_file = vocab_name
        self.new_datasets = new_datasets

    def add(self, filepath):
        """the file must be in the new dataset format."""
        with open(filepath) as fin:
            for line in fin:
                if line.startswith("DOCSTART_") or line.startswith("DOCEND") or\
                        line.startswith("MMSTART_") or line.startswith("MMEND") or \
                        line.startswith("*NL*"):
                    continue
                line = line.rstrip()       # omit the '\n' character
                word = line.lower() if self.lowercase_emb else line
                self.all_words_cnt += 1
                if word not in self.model:
                    self.not_in_word2vec_cnt += 1
                else:
                    self.word_freq[word] += 1
                for c in line:
                    self.char_freq[c] += 1

    def print_statistics(self, word_edges=None,
                         char_edges=None):
        """Print some statistics about word and char frequency."""
        if word_edges is None:
            word_edges = [1, 2, 3, 6, 11, 21, 31, 51, 76, 101, 201, np.inf]
        if char_edges is None:
            char_edges = [1, 6, 11, 21, 51, 101, 201, 501, 1001, 2001, np.inf]
        print("not_in_word2vec_cnt = ", self.not_in_word2vec_cnt)
        print("all_words_cnt = ", self.all_words_cnt)
        print("some frequency statistics. The bins are [...) ")
        for d, name, edges in zip([self.word_freq, self.char_freq], ["word", "character"], [word_edges, char_edges]):
            hist_values, _ = np.histogram(list(d.values()), edges)
            cum_sum = np.cumsum(hist_values[::-1])
            print(name, " frequency histogram, edges: ", edges)
            print("absolute values:        ", hist_values)
            print("absolute cumulative (right to left):    ", cum_sum[::-1])
            print("probabilites cumulative (right to left):", (cum_sum / np.sum(hist_values))[::-1])

    def serialize(self, folder=None, name="vocab_freq.pickle"):
        if folder is None:
            folder = config.base_folder+"data/vocabulary/"
        if not os.path.exists(folder):
            os.makedirs(folder)
        with open(folder+name, 'wb') as handle:
            pickle.dump((self.word_freq, self.char_freq), handle)

    def count_datasets_vocabulary(self):
        new_dataset_folder = "{}data/{}/".format(config.base_folder, self.new_datasets)
        """
        datasets = ['aida_train.txt', 'aida_dev.txt', 'aida_test.txt', 'ace2004.txt',
                    'aquaint.txt', 'clueweb.txt', 'msnbc.txt', 'wikipedia.txt']
        """
        for dataset in util.get_immediate_files(new_dataset_folder):
            dataset = os.path.basename(os.path.normpath(dataset))
            if(not (dataset[-4:] == ".txt")): continue
            print("Processing dataset: ", dataset)
            self.add(new_dataset_folder+dataset)
        self.print_statistics()
        self.serialize(folder=config.base_folder+"data/vocabulary/",
                       name=self.vocab_file)

def build_word_char_maps_bert():
    """
    Construit le mapping de manière naïve :
        - chaque mot de chaque document est un mot unique du dictionnaire
        - pas besoin de compter l'occurence des mots, il en existe par définition au moins 1 (et 1 seul)
        - pas besoin de compter l'occurence des caractères, on peut le faire en même temps que la création du dictionnaire de mapping
        - les dictionnaires finaux seront volumineux :
            - 
            - embedding de dimension : 
    """
    output_folder = config.base_folder+"data/tfrecords/"+args.experiment_name+"/"
    output_name = "embedding_bert.npy" if args.context_bert else "embeddings_array.npy"
    word_char_name = "word_char_maps_bert.pickle" if args.context_bert else "word_char_maps.pickle"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    # récupération des datasets
    # train, test, dev
    # dictionnaire doc -> [[(mot,emb),...],...]
    dev_name = "{}data/bert_emb/aida_dev_bert.pickle".format(config.base_folder)
    test_name = "{}data/bert_emb/aida_test_bert.pickle".format(config.base_folder)
    train_name = "{}data/bert_emb/aida_train_bert.pickle".format(config.base_folder)
    dev_bert = pickle.load(open(dev_name,'rb'))
    test_bert = pickle.load(open(test_name,'rb'))
    train_bert = pickle.load(open(train_name,'rb'))
    dev = dict(dev_bert)
    test = dict(test_bert)
    train = dict(train_bert)
    # other dataset
    ace = dict(pickle.load(open("{}data/bert_emb/ace2004_bert.pickle".format(config.base_folder),'rb')))
    clueweb = dict(pickle.load(open("{}data/bert_emb/clueweb_bert.pickle".format(config.base_folder),'rb')))
    aquaint = dict(pickle.load(open("{}data/bert_emb/aquaint_bert.pickle".format(config.base_folder),'rb')))
    msnbc = dict(pickle.load(open("{}data/bert_emb/msnbc_bert.pickle".format(config.base_folder),'rb')))
    wikipedia = dict(pickle.load(open("{}data/bert_emb/wikipedia_bert.pickle".format(config.base_folder),'rb')))
    print("key exemple : {}".format(list(train.keys())[:5]))
    print("len :\n\ttrain : {}\n\ttest : {}\n\tdev : {}".format(len(train),len(test),len(dev)))
    print("\tace : {}\n\tclueweb : {}\n\taquaint : {}\n\tmsnbc : {}\n\twikipedia : {}".format(len(ace),len(clueweb),len(aquaint),len(msnbc),len(wikipedia)))
    #print("len :\n\ttrain : {} - keys : {}\n\ttest : {} - keys : {}\n\tdev : {} - keys - {}".format(len(train),train.keys(),len(test),test.keys(),len(dev),dev.keys()))
    #assert False
    word2id = []
    id2word = dict()
    char2id = dict()
    id2char = dict()
    model = dict()
    chunk_word2id = dict()

    wcnt = 0   # unknown word
    word2id.append(("<wunk>",wcnt))
    id2word[wcnt] = "<wunk>"
    wcnt += 1
    ccnt = 0   # unknown character
    char2id["<u>"] = ccnt
    id2char[ccnt] = "<u>"
    ccnt += 1
    
    for dataset in [train, test, dev, ace, aquaint, clueweb, msnbc, wikipedia]:
        for doc_id,doc in dataset.items():
            chunk_word2id[doc_id] = []
            for sentence in doc:
                for word,emb in sentence:
                    chunk_word2id[doc_id].append(wcnt)
                    word2id.append((word,wcnt))
                    id2word[wcnt] = word
                    model[wcnt] = emb
                    wcnt += 1
                    
                    for c in word:
                        if c not in char2id:
                            char2id[c] = ccnt
                            id2char[ccnt] = c
                            ccnt += 1

    print("words in vocabulary: {}".format(wcnt))
    print("characters in vocabulary: {}".format(ccnt))
    embedding_dim = len(list(train.values())[0][0][0][1])
    print("embeddings dimension: {}".format(embedding_dim))
    embeddings_array = np.empty((wcnt, embedding_dim))   # id2emb
    embeddings_array[0] = np.zeros(embedding_dim)
    for i in range(1, wcnt):
        embeddings_array[i] = model[i]    
    
    np.save(output_folder+output_name, embeddings_array)
    with open(output_folder+word_char_name, 'wb') as handle:
        pickle.dump((word2id, id2word, char2id, id2char, args.word_freq_thr,
                     args.char_freq_thr), handle)
    return chunk_word2id, char2id
                     
def build_word_char_maps(vocab_file):
    output_folder = config.base_folder+"data/tfrecords/"+args.experiment_name+"/"
    word2vec_model = config.base_folder+"data/basic_data/wordEmbeddings/Word2Vec/"+word2vec
    word_char_map_name = output_folder+"word_char_maps.pickle"
    embedding_array_name = output_folder+'embeddings_array.npy'
    if  os.path.isfile(word_char_map_name) and os.path.isfile(embedding_array_name): 
        print("use existing word_char_maps")
        return build_word_char_maps_restore() #return word2id, char2id from existing word_char_maps.pickle
    else:
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        with open(config.base_folder+"data/vocabulary/{}".format(vocab_file), 'rb') as handle:
            word_freq, char_freq = pickle.load(handle)
        word2id = dict()
        id2word = dict()
        char2id = dict()
        id2char = dict()
    
        wcnt = 0   # unknown word
        word2id["<wunk>"] = wcnt
        id2word[wcnt] = "<wunk>"
        wcnt += 1
        ccnt = 0   # unknown character
        char2id["<u>"] = ccnt
        id2char[ccnt] = "<u>"
        ccnt += 1
    
        # for every word in the corpus (we have already filtered out the words that are not in word2vec)
        for word in word_freq:
            if word_freq[word] >= args.word_freq_thr:
                word2id[word] = wcnt
                id2word[wcnt] = word
                wcnt += 1
    
        for c in char_freq:
            if char_freq[c] >= args.char_freq_thr:
                char2id[c] = ccnt
                id2char[ccnt] = c
                ccnt += 1
        assert(len(word2id) == wcnt)
        assert(len(char2id) == ccnt)
        print("words in vocabulary: ", wcnt)
        print("characters in vocabulary: ", ccnt)
        with open(word_char_map_name, 'wb') as handle:
            pickle.dump((word2id, id2word, char2id, id2char, args.word_freq_thr,
                         args.char_freq_thr), handle)
        import gensim
        model = gensim.models.KeyedVectors.load_word2vec_format(word2vec_model, binary=True)
        embedding_dim = len(model['queen'])
        print("embedding dimension : {}".format(embedding_dim))
        embeddings_array = np.empty((wcnt, embedding_dim))   # id2emb
        embeddings_array[0] = np.zeros(embedding_dim)
        for i in range(1, wcnt): embeddings_array[i] = model[id2word[i]]
        np.save(embedding_array_name, embeddings_array)
        return word2id, char2id


def build_word_char_maps_restore():
    output_folder = config.base_folder+"data/tfrecords/"+args.experiment_name+"/"
    with open(output_folder+"word_char_maps.pickle", 'rb') as handle:
        word2id, _, char2id, _, _, _ = pickle.load(handle)
    return word2id, char2id


class Chunker(object):
    def __init__(self):
        self.separator = args.chunking
        self.chunk_ending = {'DOCEND'}
        if self.separator == "per_paragraph":
            self.chunk_ending.add('*NL*')
        if self.separator == "per_sentence":
            self.chunk_ending.add('.')
            self.chunk_ending.add('*NL*')
        self.parsing_errors = 0

    def new_chunk(self):
        self.chunk_words = []
        self.begin_gm = []          # the starting positions of gold mentions
        self.end_gm = []            # the end positions of gold mentions
        self.ground_truth = []      # list with the correct entity ids

    def compute_result(self, docid):
        chunk_id = docid
        if self.separator == "per_paragraph":
            chunk_id = chunk_id + "&*" + str(self.par_cnt)
        if self.separator == "per_sentence":
            chunk_id = chunk_id + "&*" + str(self.par_cnt) + "&*" + str(self.sent_cnt)
        result = (chunk_id, self.chunk_words, self.begin_gm, self.end_gm, self.ground_truth)

        # correctness checks. not necessary
        no_errors_flag = True
        if len(self.begin_gm) != len(self.end_gm) or \
            len(self.begin_gm) != len(self.ground_truth):
            no_errors_flag = False
        for b, e in zip(self.begin_gm, self.end_gm):
            if e <= b or b >= len(self.chunk_words) or e > len(self.chunk_words):
                no_errors_flag = False

        self.new_chunk()
        if no_errors_flag == False:
            self.parsing_errors += 1
            #print("chunker parse error: ", result)
            return None
        else:
            return result

    def process(self, filepath):
        with open(filepath) as fin:
            self.new_chunk()
            docid = ""
            # paragraph and sentence counter are not actually useful. only for debugging purposes.
            self.par_cnt = 0      # paragraph counter (useful if we work per paragraph)
            self.sent_cnt = 0      # sentence counter (useful if we work per sentence)
            len_file = 0
            for line in fin:
                len_file += 1
                line = line.rstrip()     # omit the '\n' character
                if line in self.chunk_ending:
                    if len(self.chunk_words) > 0:  # if we have continues *NL* *NL* do not return empty chunks
                        temp = self.compute_result(docid)
                        if temp is not None:
                            yield temp
                    # do not add the chunk separator, no use
                    if line == '*NL*':
                        self.par_cnt += 1
                        self.sent_cnt = 0
                    if line == '.':
                        self.sent_cnt += 1
                elif line == '*NL*':
                    self.par_cnt += 1
                    self.sent_cnt = 0
                    # do not add this in our words list
                elif line == '.':
                    self.sent_cnt += 1
                    self.chunk_words.append(line)
                elif line.startswith('MMSTART_'):
                    ent_id = line[8:]   # assert that ent_id in wiki_name_id_map
                    self.ground_truth.append(ent_id)
                    self.begin_gm.append(len(self.chunk_words))
                elif line == 'MMEND':
                    self.end_gm.append(len(self.chunk_words))
                elif line.startswith('DOCSTART_'):
                    docid = line[9:]
                    self.par_cnt = 0
                    self.sent_cnt = 0
                else:
                    self.chunk_words.append(line)

        print(filepath, " chunker parsing errors: {}/{} ({:.2f}%)".format(self.parsing_errors,len_file, 100*(self.parsing_errors/len_file)))
        self.parsing_errors = 0



GmonlySample = namedtuple("GmonlySample",
                          ["chunk_id", "chunk_words", 'begin_gm', "end_gm",
                          "ground_truth", "cand_entities", "cand_entities_scores"])
AllspansSample = namedtuple("AllspansSample",
                            ["chunk_id", "chunk_words", "begin_spans", "end_spans",
                         "ground_truth", "cand_entities", "cand_entities_scores",
                         "begin_gm", "end_gm"])


class SamplesGenerator(object):
    def __init__(self, mode="allspans"):
        self.mode = mode
        self._generator = Chunker()
        self.fetchFilteredCoreferencedCandEntities = util.FetchFilteredCoreferencedCandEntities(args)
        self.all_gm_misses = 0
        self.all_gt_misses = 0
        self.all_gm = 0   # all the gm encountered in all the datasets

    def set_gmonly_mode(self):
        self.mode = "gmonly"

    def set_allspans_mode(self):
        self.mode = "allspans"

    def is_gmonly_mode(self):
        return True if self.mode == "gmonly" else False

    def is_allspans_mode(self):
        return True if self.mode == "allspans" else False

    def process(self, filepath):
        if self.is_allspans_mode():
            return self._process_allspans(filepath)
        else:
            return self._process_gmonly(filepath)

    def _process_allspans(self, filepath):
        gt_misses = 0
        gm_misses = 0
        gm_this_file = 0  # how many gold mentions are in this document - dataset. so we can find percentage for misses

        max_mention_width_violations = 0
        if(args.chunking == "bert_token") :
            filename = os.path.normpath(filepath)[:-4]  # omit the '.txt'
            chunkspath = "{}_chunk.pickle".format(filename)
            with open(chunkspath,"rb") as chf:
                chunks = pickle.load(chf)
            print("Taille chunks : {}".format(len(chunks)))
            #old_chunks = list(self._generator.process(filepath))
            #print("old : {} {} : new|-|".format(old_chunks[0][1],chunks[0][1]),end="")
        else: 
            chunks = self._generator.process(filepath)
        for chunk in chunks:
            self.fetchFilteredCoreferencedCandEntities.init_coref(el_mode=True)
            begin_spans = []
            end_spans = []
            cand_entities = []   # list of lists     candidate entities
            cand_entities_scores = []
            chunk_id, chunk_words, begin_gm, end_gm, ground_truth = chunk
            gm_this_file += len(begin_gm)
            for left, right in self.all_spans(chunk_words):
                cand_ent, scores = self.fetchFilteredCoreferencedCandEntities.process(left, right, chunk_words)
                if cand_ent is not None:
                    begin_spans.append(left)
                    end_spans.append(right)
                    cand_entities.append(cand_ent)
                    cand_entities_scores.append(scores)

            if args.calculate_stats:
                # check if gold mentions are inside the candidate spans and if yes check if ground truth is in cand ent.
                gm_spans = list(zip(begin_gm, end_gm))   # [(3, 5), (10, 11), (15, 18)]
                all_spans = list(zip(begin_spans, end_spans))
                for i, gm_span in enumerate(gm_spans):
                    flag_continue = False
                    if gm_span not in all_spans:
                        gm_misses += 1
                        flag_continue = True
                        #print("gm not in spans\t\t\t", colored(' '.join(chunk_words[gm_span[0]:gm_span[1]]), 'red'))
                    elif ground_truth[i] not in cand_entities[all_spans.index(gm_span)]:
                        gt_misses += 1
                        flag_continue = True
                        #print("gt not in cand ent", colored(' '.join(chunk_words[gm_span[0]:gm_span[1]]), 'green'))
                        #print("gt: ", ground_truth[i], "cand_ent: ", cand_entities[all_spans.index(gm_span)])
                    if flag_continue: continue #TODO remove

                for b, e in zip(begin_gm, end_gm):
                    if e - b > args.max_mention_width:
                        max_mention_width_violations += 1

            if begin_spans:  # there are candidate spans in the processed text
                yield AllspansSample(chunk_id, chunk_words, begin_spans, end_spans,
                                     ground_truth, cand_entities, cand_entities_scores,
                                     begin_gm, end_gm)
        if args.calculate_stats:
            print("max_mention_width_violations :", max_mention_width_violations)
            print("gt_misses", gt_misses)
            print("gm_misses", gm_misses)
            print("gm_this_file: ", gm_this_file)
            print("recall %     : ", (1 - (gm_misses+gt_misses)/gm_this_file)*100, " %")
            self.all_gt_misses += gt_misses
            self.all_gm_misses += gm_misses
            self.all_gm += gm_this_file

    @staticmethod
    def all_spans(chunk_words):
        # this function produces all possible text spans that do not include spans separators (fullstops).
        # divide the list of words to lists of lists based on spans_separator.
        # e.g. if chunk_words is for the whole document divide it to sentences (a list of
        # sentences) since no span extend above a fullstop.
        separation_indexes = []
        spans_separator = set(config.spans_separators)
        for idx, word in enumerate(chunk_words):
            if word in spans_separator:
                separation_indexes.append(idx)

        separation_indexes.append(len(chunk_words))

        def all_spans_aux(begin_idx, end_idx):
            for left_idx in range(begin_idx, end_idx):
                for length in range(1, args.max_mention_width + 1):
                    if left_idx + length > end_idx:
                        break
                    yield left_idx, left_idx + length

        begin_idx = 0
        for end_idx in separation_indexes:
            for left, right in all_spans_aux(begin_idx, end_idx):
                # print(left, right, chunk_words[left:right])
                # print(left, right, ' '.join(chunk_words[left:right])
                yield left, right
            begin_idx = end_idx + 1

    def _process_gmonly(self, filepath):
        gt_misses = 0
        gm_misses = 0
        gm_this_file = 0
        max_mention_width_violations = 0
        if(args.chunking == "bert_token") :
            filename = os.path.normpath(filepath)[:-4]  # omit the '.txt'
            chunkspath = "{}_chunk.pickle".format(filename)
            with open(chunkspath,"rb") as chf:
                chunks = pickle.load(chf)
            print("Taille chunks : {}".format(len(chunks)))
        else: 
            chunks = self._generator.process(filepath)
        for chunk in chunks:
            self.fetchFilteredCoreferencedCandEntities.init_coref(el_mode=False)
            cand_entities = []   # list of lists     candidate entities
            cand_entities_scores = []
            chunk_id, chunk_words, begin_gm, end_gm, ground_truth = chunk
            gm_this_file += len(begin_gm)
            for left, right, gt in zip(begin_gm, end_gm, ground_truth):
                cand_ent, scores = self.fetchFilteredCoreferencedCandEntities.process(left, right, chunk_words)
                if cand_ent is None:
                    gm_misses += 1
                    cand_ent, scores = [], []
                    #print("gm not in p_e_m\t\t\t", colored(' '.join(chunk_words[left:right]), 'red'))
                elif args.calculate_stats and gt not in cand_ent:
                    gt_misses += 1
                    #print("gt not in cand ent", colored(' '.join(chunk_words[left:right]), 'green'))
                    #print("gt: ", gt, "cand_ent: ", cand_ent)

                if right - left > args.max_mention_width:
                    max_mention_width_violations += 1

                #print(' '.join(chunk_words[left:right])
                #print(cand_ent, scores)
                cand_entities.append(cand_ent)
                cand_entities_scores.append(scores)

            if begin_gm:  #not emtpy
                yield GmonlySample(chunk_id, chunk_words, begin_gm, end_gm, ground_truth,
                                   cand_entities, cand_entities_scores)

        if args.calculate_stats:
            print("max_mention_width_violations :", max_mention_width_violations)
            print("gt_misses", gt_misses)
            print("gm_misses", gm_misses)
            print("gm_this_file", gm_this_file)
            print("recall %     : ", (1 - (gm_misses+gt_misses)/gm_this_file)*100, " %")
            self.all_gt_misses += gt_misses
            self.all_gm_misses += gm_misses
            self.all_gm += gm_this_file


SampleEncoded = namedtuple("SampleEncoded",
                                ["chunk_id",
                                "words", 'words_len',   # list,  scalar
                                'chars', 'chars_len',   # list of lists,  list
                                'begin_spans', "end_spans",  'spans_len',   # the first 2 are lists, last is scalar
                                "cand_entities", "cand_entities_scores", 'cand_entities_labels',  # lists of lists
                                'cand_entities_len',  # list
                                "ground_truth", "ground_truth_len",
                                'begin_gm', 'end_gm'])  # list
                                
SampleEncoded2 = namedtuple("SampleEncoded",
                                ["chunk_id",
                                "words", 'words_len',   # list,  scalar
                                "context", # list
                                'chars', 'chars_len',   # list of lists,  list
                                'begin_spans', "end_spans",  'spans_len',   # the first 2 are lists, last is scalar
                                "cand_entities", "cand_entities_scores", 'cand_entities_labels',  # lists of lists
                                'cand_entities_len',  # list
                                "ground_truth", "ground_truth_len",
                                'begin_gm', 'end_gm'])  # list


class EncoderGenerator(object):
    """receives samples Train or Test samples and encodes everything to numbers ready to
    be transformed to tfrecords. Also filters out candidate entities that are not in the
    entity universe."""
    def __init__(self):
        self._generator = SamplesGenerator()
        if(args.word_bert): self._word2id, self._char2id = build_word_char_maps_bert()
        else : self._word2id, self._char2id = build_word_char_maps(args.vocab_file)
        if(args.context_bert): self._contextid, _ = build_word_char_maps_bert()
        #self._word2id, self._char2id = build_word_char_maps_restore()  # alternative
        self._wikiid2nnid = util.load_wikiid2nnid(args.entity_extension, txt_file=args.wikiid2nnid_file)
        #if "<u>" not in self._wikiid2nnid: self._wikiid2nnid["<u>"] = 0 #len(self._wikiid2nnid)+1

    def set_gmonly_mode(self):
        self._generator.set_gmonly_mode()

    def set_allspans_mode(self):
        self._generator.set_allspans_mode()

    def is_gmonly_mode(self):
        return self._generator.is_gmonly_mode()

    def is_allspans_mode(self):
        return self._generator.is_allspans_mode()

    def process(self, filepath):
        ground_truth_errors_cnt = 0
        total_gt = 0
        cand_entities_not_in_universe_cnt = 0
        total_cand = 0
        samples_with_errors_1 = 0
        samples_with_errors_2 = 0
        total_sample = 0
        for sample in self._generator.process(filepath):
            words = []
            chars = []
            if(args.word_bert): 
                if sample.chunk_id in self._word2id: words=self._word2id[sample.chunk_id]
                else: 
                    print("sample ignored : {}".format(sample.chunk_id))
                    continue
            elif(args.context_bert):
                if sample.chunk_id in self._contextid: context=self._contextid[sample.chunk_id]
                else: 
                    print("sample ignored : {}".format(sample.chunk_id))
                    continue
                #print("{} {}|-|".format(len(sample.chunk_words),len(context)),end="")
				#assert len(words) == len(sample.chunk_words), "words : {} | chunk = {}".format(len(words),len(sample.chunk_words))
            for word in sample.chunk_words:
                if(not args.word_bert): words.append(self._word2id[word] if word in self._word2id else self._word2id["<wunk>"])
                chars.append([self._char2id[c] if c in self._char2id else self._char2id["<u>"]
                              for c in word])
            chars_len = [len(word) for word in chars]
            
            # Groung Truth Filtering
            ground_truth_enc = []
            begin_gm = []
            end_gm = []
            
            if total_sample == 0:
                print("dim gt : {} ({})".format(np.shape(sample.ground_truth), type(sample.ground_truth)))
                print("dim begin_gm : {} ({})".format(np.shape(sample.begin_gm), type(sample.begin_gm)))
                print("dim end_gm : {} ({})".format(np.shape(sample.end_gm), type(sample.end_gm)))
                print("exemple begin_gm : '{}'".format(sample.begin_gm[:5]))
                print("exemple end_gm : '{}'".format(sample.end_gm[:5]))
            
            for i, gt in enumerate(sample.ground_truth):
                total_gt += 1
                if gt in self._wikiid2nnid:
                    ground_truth_enc.append(self._wikiid2nnid[gt])
                    begin_gm.append(sample.begin_gm[i])
                    end_gm.append(sample.end_gm[i])
                else:
                    try: ground_truth_enc.append(self._wikiid2nnid["<u>"])
                    except KeyError: pass
                    else: 
                        begin_gm.append(sample.begin_gm[i])
                        end_gm.append(sample.end_gm[i])
                    finally: ground_truth_errors_cnt += 1
#TODO        Remove Obsolete Bloc
#            try:
#                ground_truth_enc = [self._wikiid2nnid[gt] if gt in self._wikiid2nnid else self._wikiid2nnid["<u>"] for gt in sample.ground_truth]
#                ground_truth_errors_cnt += ground_truth_enc.count(self._wikiid2nnid["<u>"])   # it is always zero
#                total_gt += len(ground_truth_enc)# - ground_truth_enc.count(self._wikiid2nnid["<u>"])
#            except KeyError: 
#                ground_truth_enc = [self._wikiid2nnid[gt] for gt in sample.ground_truth if gt in self._wikiid2nnid]
#                ground_truth_errors_cnt += (len(sample.ground_truth) - len(ground_truth_enc))
#                total_gt += len(sample.ground_truth) #to prevent error in final print even if no real stats are compute
#            begin_gm = sample.begin_gm
#            end_gm = sample.end_gm
            
            total_sample += 1
            if len(begin_gm) != len(end_gm) or \
                len(begin_gm) != len(ground_truth_enc):
                samples_with_errors_1 += 1
                continue
            if isinstance(sample, GmonlySample):
                cand_entities, cand_entities_scores, cand_entities_labels, not_in_universe_cnt, total_cand_temp = \
                    self._encode_cand_entities_and_labels(
                        sample.cand_entities, sample.cand_entities_scores, sample.ground_truth)
                total_cand += total_cand_temp
                if(args.context_bert):
                    yield SampleEncoded2(chunk_id=sample.chunk_id,
                                    words=words, words_len=len(words),
                                    context=context,
                                    chars=chars, chars_len=chars_len,
                                    begin_spans=begin_gm, end_spans=end_gm, spans_len=len(begin_gm),
                                    cand_entities=cand_entities, cand_entities_scores=cand_entities_scores,
                                    cand_entities_labels=cand_entities_labels,
                                    cand_entities_len=[len(t) for t in cand_entities],
                                    ground_truth=ground_truth_enc, ground_truth_len=len(ground_truth_enc),
                                    begin_gm=[], end_gm=[])
                else:
                    yield SampleEncoded(chunk_id=sample.chunk_id,
                                    words=words, words_len=len(words),
                                    chars=chars, chars_len=chars_len,
                                    begin_spans=begin_gm, end_spans=end_gm, spans_len=len(begin_gm),
                                    cand_entities=cand_entities, cand_entities_scores=cand_entities_scores,
                                    cand_entities_labels=cand_entities_labels,
                                    cand_entities_len=[len(t) for t in cand_entities],
                                    ground_truth=ground_truth_enc, ground_truth_len=len(ground_truth_enc),
                                    begin_gm=[], end_gm=[])

            elif isinstance(sample, AllspansSample):
                if len(sample.begin_spans) != len(sample.end_spans):
                    samples_with_errors_2 += 1
                    continue
                # for each span i have the gt or the value -1 if this span is not a gm
                # and then i work in the same way as above
                span_ground_truth = []
                gm_spans = list(zip(begin_gm, end_gm))   # [(3, 5), (10, 11), (15, 18)]
                for left, right in zip(sample.begin_spans, sample.end_spans):
                    if (left, right) in gm_spans:
                        span_ground_truth.append(sample.ground_truth[gm_spans.index((left, right))])
                    else:
                        span_ground_truth.append(-1)   # this span is not a gm
                        #continue #TODO remove
                cand_entities, cand_entities_scores, cand_entities_labels, not_in_universe_cnt, total_cand_temp = \
                    self._encode_cand_entities_and_labels(
                        sample.cand_entities, sample.cand_entities_scores, span_ground_truth)
                total_cand += total_cand_temp
                if(args.context_bert):
                    yield SampleEncoded2(chunk_id=sample.chunk_id,
                                    words=words, words_len=len(words),
                                    context=context,
                                    chars=chars, chars_len=chars_len,
                                    begin_spans=sample.begin_spans, end_spans=sample.end_spans, spans_len=len(sample.begin_spans),
                                    cand_entities=cand_entities, cand_entities_scores=cand_entities_scores,
                                    cand_entities_labels=cand_entities_labels,
                                    cand_entities_len=[len(t) for t in cand_entities],
                                    ground_truth=ground_truth_enc, ground_truth_len=len(ground_truth_enc),
                                    begin_gm=begin_gm, end_gm=end_gm)
                else:
                    yield SampleEncoded(chunk_id=sample.chunk_id,
                                    words=words, words_len=len(words),
                                    chars=chars, chars_len=chars_len,
                                    begin_spans=sample.begin_spans, end_spans=sample.end_spans, spans_len=len(sample.begin_spans),
                                    cand_entities=cand_entities, cand_entities_scores=cand_entities_scores,
                                    cand_entities_labels=cand_entities_labels,
                                    cand_entities_len=[len(t) for t in cand_entities],
                                    ground_truth=ground_truth_enc, ground_truth_len=len(ground_truth_enc),
                                    begin_gm=begin_gm, end_gm=end_gm)

            cand_entities_not_in_universe_cnt += not_in_universe_cnt
        print("ground_truth_errors_cnt = {}/{} ({:.2f}%)".format(ground_truth_errors_cnt, total_gt, 100*(ground_truth_errors_cnt/total_gt)))
        print("cand_entities_not_in_universe_cnt = {}/{} ({:.2f}%)".format(cand_entities_not_in_universe_cnt, total_cand, 100*(cand_entities_not_in_universe_cnt/total_cand)))
        print("encoder samples_with_errors = {}/{} ({:.2f}%)".format(samples_with_errors_1 + samples_with_errors_2, total_sample, 100*((samples_with_errors_1+samples_with_errors_2)/total_sample)))
        if samples_with_errors_1 + samples_with_errors_2 > 0 :
            print("\t- caused by entities spans : {} ({:.2f}%)".format(samples_with_errors_1, 100*(samples_with_errors_1/(samples_with_errors_1+samples_with_errors_2))))
            print("\t- caused by mentions spans : {} ({:.2f}%)".format(samples_with_errors_2, 100*(samples_with_errors_2/(samples_with_errors_1+samples_with_errors_2))))



    def _encode_cand_entities_and_labels(self, cand_entities_p, cand_entities_scores_p,
                                        ground_truth_p):
        """receives cand_entities (list of lists), and ground_truth (list) and does the following:
        1) removes cand ent that are not in our universe
        2) creates a label 0, 1 if this candidate is correct or not (i.e. if the span is indeed a
         gold mention (row of candidate entities array) and this specific candidate entity (column
         of candidate entities array) is correct. Returns the filtered cand_entities
        and the corresponding label (they have the same shape)"""
        cand_entities = []
        cand_entities_scores = []
        cand_entities_labels = []
        not_in_universe_cnt = 0
        total = 0
        for cand_ent_l, cand_scores_l, gt in zip(cand_entities_p, cand_entities_scores_p, ground_truth_p):
            ent_l = []
            score_l = []
            label_l = []
            for cand_ent, score in zip(cand_ent_l, cand_scores_l):
                if cand_ent in self._wikiid2nnid:  # else continue, this entity not in our universe
                    ent_l.append(self._wikiid2nnid[cand_ent])
                    score_l.append(score)
                    label_l.append(1 if cand_ent == gt else 0)
                else:
                    not_in_universe_cnt += 1
                total += 1
            cand_entities.append(ent_l)
            cand_entities_scores.append(score_l)
            cand_entities_labels.append(label_l)
        return cand_entities, cand_entities_scores, cand_entities_labels, not_in_universe_cnt, total


class TFRecordsGenerator(object):
    def __init__(self):
        self._generator = EncoderGenerator()

    def set_gmonly_mode(self):
        self._generator.set_gmonly_mode()

    def set_allspans_mode(self):
        self._generator.set_allspans_mode()

    def is_gmonly_mode(self):
        return self._generator.is_gmonly_mode()

    def is_allspans_mode(self):
        return self._generator.is_allspans_mode()

    @staticmethod
    def _to_sequence_example(sample):
        def _bytes_feature(value):
            return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
        # Those two create a simple feature. The first a simple feature with one integer, whereas the second a simple
        # list of integers as one feature.
        def _int64_feature(value):
            """value is a simple integer."""
            return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

        def _int64list_feature(value):
            """value is a list of integers."""
            return tf.train.Feature(int64_list=tf.train.Int64List(value=value))
            
        def _int64_feature_list(values):
            """ values is a list of integers like the words (words = [2,4,6,8,10])
            a feature list where each feature has only one number (a list with fixed
            number of elements, specifically only one)"""
            return tf.train.FeatureList(feature=[_int64_feature(v) for v in values])

        def _int64list_feature_list(values):
            """ like the chars = [[1,2,3], [4,5], [6], [7,8], [9,10,11,12]] a feature list where each feature can have variable
            number of ements"""
            return tf.train.FeatureList(feature=[_int64list_feature(v) for v in values])

        def _floatlist_feature_list(values):
            """ like the chars = [[0.1,0.2,0.3], [0.4,0.5]] a feature list where each feature can have variable
            number of ements"""
            def _floatlist_feature(value):
                """value is a list of float."""
                return tf.train.Feature(float_list=tf.train.FloatList(value=value))
            return tf.train.FeatureList(feature=[_floatlist_feature(v) for v in values])
            
        def _floatlist_feature_tensor(values):
            """ like the chars = [[0.1,0.2,0.3], [0.4,0.5]] a feature list where each feature can have variable
            number of ements"""
            def _floatlist_feature(value):
                """value is a list of float."""
                return tf.train.Feature(float_list=tf.train.FloatList(value=value))
            # appliquer sur un tenseur
            return tf.train.FeatureList(feature=[[_floatlist_feature(v) for v in x] for x in values])

        context = tf.train.Features(feature={
                "chunk_id": _bytes_feature(sample.chunk_id.encode('utf-8')),
                "words_len": _int64_feature(sample.words_len),
                "spans_len": _int64_feature(sample.spans_len),
                "ground_truth_len": _int64_feature(sample.ground_truth_len)
        })
        if(args.context_bert):
            feature_list = {
                    "words": _int64_feature_list(sample.words),
                    "chars": _int64list_feature_list(sample.chars),
                    "context": _int64_feature_list(sample.context),
                    "chars_len": _int64_feature_list(sample.chars_len),
                    "begin_span": _int64_feature_list(sample.begin_spans),
                    "end_span": _int64_feature_list(sample.end_spans),
                    "cand_entities": _int64list_feature_list(sample.cand_entities),
                    "cand_entities_scores": _floatlist_feature_list(sample.cand_entities_scores),
                    "cand_entities_labels": _int64list_feature_list(sample.cand_entities_labels),
                    "cand_entities_len": _int64_feature_list(sample.cand_entities_len),
                    "ground_truth": _int64_feature_list(sample.ground_truth)
            }
        else:
            feature_list = {
                    "words": _int64_feature_list(sample.words),
                    "chars": _int64list_feature_list(sample.chars),
                    "chars_len": _int64_feature_list(sample.chars_len),
                    "begin_span": _int64_feature_list(sample.begin_spans),
                    "end_span": _int64_feature_list(sample.end_spans),
                    "cand_entities": _int64list_feature_list(sample.cand_entities),
                    "cand_entities_scores": _floatlist_feature_list(sample.cand_entities_scores),
                    "cand_entities_labels": _int64list_feature_list(sample.cand_entities_labels),
                    "cand_entities_len": _int64_feature_list(sample.cand_entities_len),
                    "ground_truth": _int64_feature_list(sample.ground_truth)
            }
        if isinstance(sample, SampleEncoded) or isinstance(sample, SampleEncoded2):
            feature_list["begin_gm"] = _int64_feature_list(sample.begin_gm)
            feature_list["end_gm"] = _int64_feature_list(sample.end_gm)
        #print("{}|-|".format(len(feature_list)+len(list(context.keys()))),end="")
        feature_lists = tf.train.FeatureLists(feature_list=feature_list)
        sequence_example = tf.train.SequenceExample(context=context, feature_lists=feature_lists)
        return sequence_example


    def process(self, filepath):
        print("processing file: ", filepath)
        #the name of the dataset. just extract the last part of path
        filename = os.path.basename(os.path.normpath(filepath))[:-4]  # omit the '.txt'
        output_folder = config.base_folder+"data/tfrecords/"+args.experiment_name+"/"
        output_folder += "gmonly/" if self.is_gmonly_mode() else "allspans/"
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        writer = tf.python_io.TFRecordWriter(output_folder+filename)
        records_cnt = 0
        total_sample = 0
        for sample in self._generator.process(filepath):
            #print("{} ".format(len(sample)),end="")
            sequence_example = self._to_sequence_example(sample)
            #print("{}|-|".format(sequence_example.SerializeToString()),end="")
            # write it to file
            total_sample += 1
            if sequence_example is not None : #and total_sample < 101 :
                writer.write(sequence_example.SerializeToString())
                records_cnt += 1
        writer.close()
        print("records_cnt = {}/{} ({:.2f}%)".format(records_cnt, total_sample, 100*(records_cnt/total_sample)))


def create_tfrecords():
    #if args.chunking=="bert_token": new_dataset_folder = config.base_folder+"data/new_datasets2/"
    new_dataset_folder = config.base_folder+"data/"+args.datasets_folder+"/"
    datasets = [os.path.basename(os.path.normpath(d)) for d in util.get_immediate_files(new_dataset_folder) if d[-4:] == ".txt"] #["aida_dev.txt","aida_test.txt","aida_train.txt"] #
    print("datasets: ", datasets)
    print(20*"/"+"\n///Initialized TFRecordsGenerator///\n"+20*"/")
    tfrecords_generator = TFRecordsGenerator()
#    print(20*"/"+"\n///GM Only Mode///\n"+20*"/")
#    tfrecords_generator.set_gmonly_mode()
#    for file in datasets:
#        tfrecords_generator.process(filepath=new_dataset_folder+file)
    print(20*"/"+"\n///All Spans Mode///\n"+20*"/")
    tfrecords_generator.set_allspans_mode()
    for file in datasets:
        tfrecords_generator.process(filepath=new_dataset_folder+file)


class PrintSamples(object):
    def __init__(self, only_misses=True):
        _, self.wiki_id_name_map = util.load_wiki_name_id_map(filepath=args.wiki_id_file)
        self.only_misses = only_misses

    def print_candidates(self, ent_ids_list):
        """takes as input a list of ent_id and returns a string. This string has each ent_id
        together with the corresponding name (in the name withspaces are replaced by underscore)
        and candidates are separated with a single space. e.g.  ent_id,Barack_Obama ent_id2,US_President"""
        acc = []
        for ent_id in ent_ids_list:
            acc.append(ent_id + "," + self.wiki_id_name_map[ent_id].replace(' ', '_'))
        return ' '.join(acc)

    def print_sample(self, sample):
        chunk_words, begin_gm, end_gm, ground_truth, cand_entities = \
            sample.chunk_words, sample.begin_gm, sample.end_gm, sample.ground_truth, sample.cand_entities
        if isinstance(sample, GmonlySample):
            misses_idx = []
            for i, (gt, cand_ent) in enumerate(zip(ground_truth, cand_entities)):
                if gt not in cand_ent:
                    misses_idx.append(i)  # miss detected

            if self.only_misses and misses_idx:
                print(colored("New sample", 'red'))
                print(' '.join(chunk_words))
                for i in misses_idx:
                    message = ' '.join(chunk_words[begin_gm[i]:end_gm[i]]) + "\tgt=" + \
                              self.print_candidates([ground_truth[i]]) + \
                              "\tCandidates: " + self.print_candidates(cand_entities[i])
                    print(colored(message, 'yellow'))
            if self.only_misses == False:
                print(colored("New sample", 'red'))
                print(' '.join(chunk_words))
                for i in range(len(begin_gm)):
                    message = ' '.join(chunk_words[begin_gm[i]:end_gm[i]]) + "\tgt=" + \
                              self.print_candidates([ground_truth[i]]) + \
                              "\tCandidates: " + self.print_candidates(cand_entities[i])
                    print(colored(message, 'yellow' if i in misses_idx else 'white'))
        elif isinstance(sample, AllspansSample):
            begin_spans, end_spans = sample.begin_spans, sample.end_spans
            gm_spans = list(zip(begin_gm, end_gm))   # [(3, 5), (10, 11), (15, 18)]
            all_spans = list(zip(begin_spans, end_spans))
            print(colored("New sample", 'red'))
            print(' '.join(chunk_words))
            for i, gm_span in enumerate(gm_spans):
                if gm_span not in all_spans:
                    message = ' '.join(chunk_words[begin_gm[i]:end_gm[i]]) + "\tgt=" + \
                              self.print_candidates([ground_truth[i]]) + "\tgm_miss"
                    print(colored(message, 'magenta'))
                elif ground_truth[i] not in cand_entities[all_spans.index(gm_span)]:
                    message = ' '.join(chunk_words[begin_gm[i]:end_gm[i]]) + "\tgt=" + \
                              self.print_candidates([ground_truth[i]]) + "\tgt_miss Candidates: " + \
                              self.print_candidates(cand_entities[all_spans.index(gm_span)])
                    print(colored(message, 'yellow'))

            if self.only_misses == False:
                # then also print all the spans and their candidate entities
                for left, right, cand_ent in zip(begin_spans, end_spans, cand_entities):
                    # if span is a mention and includes gt then green color, otherwise white
                    if (left, right) in gm_spans and ground_truth[gm_spans.index((left, right))] in cand_ent:
                        message = ' '.join(chunk_words[left:right]) + "\tgt=" + \
                                  self.print_candidates([ground_truth[gm_spans.index((left, right))]]) + \
                                  "\tgm_gt_hit Candidates: " + \
                                  self.print_candidates(cand_ent)
                        print(colored(message, 'green'))
                    else:
                        message = ' '.join(chunk_words[left:right]) + \
                                  "\t not a mention Candidates: " + \
                                  self.print_candidates(cand_ent)
                        print(colored(message, 'white'))


def create_entity_universe(gmonly_files=None, allspans_files=None, printSamples=None):
    new_dataset_folder = config.base_folder+"data/"+args.datasets_folder+"/"
    if gmonly_files is None:
        gmonly_files = []
    if allspans_files is None:
        allspans_files = ['aida_train.txt', 'aida_dev.txt', 'aida_test.txt', 'ace2004.txt',
                          'aquaint.txt', 'clueweb.txt', 'msnbc.txt', 'wikipedia.txt']
    print("gmonly_files: ", gmonly_files)
    print("allspans_files: ", allspans_files)

    def create_entity_universe_aux(generator, datasets):
        entities_universe = set()
        for dataset in datasets:
            print("Processing dataset: ", dataset)
            for sample in generator.process(filepath=new_dataset_folder+dataset):
                entities_universe.update(*sample.cand_entities)
                entities_universe.update(sample.ground_truth)
                if printSamples:
                    printSamples.print_sample(sample)

        print("Overall statistics: ")
        print("all_gm_misses: ", generator.all_gm_misses)
        print("all_gt_misses: ", generator.all_gt_misses)
        print("all_gm: ", generator.all_gm)
        print("recall %     : ", (1 - (generator.all_gm_misses+generator.all_gt_misses)/generator.all_gm)*100, " %")
        print("len(entities_universe):\t\t\t", colored(len(entities_universe), 'red'))
        return entities_universe

    gmonly_entities, allspans_entities = set(), set()
    samplesGenerator = SamplesGenerator()
    if gmonly_files:
        print("gmonly files statistics: ")
        samplesGenerator.set_gmonly_mode()
        gmonly_entities = create_entity_universe_aux(samplesGenerator, gmonly_files)
    if allspans_files:
        print("Test files statistics: ")
        samplesGenerator.set_allspans_mode()
        allspans_entities = create_entity_universe_aux(samplesGenerator, allspans_files)

    all_entities = gmonly_entities | allspans_entities
    print("len(all_entities) = ", len(all_entities))

    # print the entities of our universe to a file together with the name
    with open(config.base_folder+"data/entities/entities_universe.txt", "w") as fout:
        _, wiki_id_name_map = util.load_wiki_name_id_map(filepath=args.wiki_id_file)
        for ent_id in all_entities:
            fout.write(ent_id + "\t" + wiki_id_name_map[ent_id].replace(' ', '_') + "\n")

    return all_entities


def create_necessary_folders():
    if not os.path.exists(config.base_folder+"data/tfrecords/"):
        os.makedirs(config.base_folder+"data/tfrecords/")


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunking", default="per_document",
                        help="per_sentence or per_paragraph or per_article or bert_token"
                             "per_document: each document is processed as one example"
                             "per_paragraph: each paragraph is processed as a separate example"
                             "bert_token: use preprocess chunks tokenise with BERT instead of usual chunking")

    parser.add_argument("--p_e_m_choice", default="yago",
                        help="'wiki'  p(e|m) constructed only from wikipedia dump (prob_wikipedia_p_e_m.txt file),\
                             'crosswiki' constructed from wikipedia dump + crosswiki  (prob_crosswikis_wikipedia_p_e_m.txt),\
                             'yago' (prob_yago_crosswikis_wikipedia_p_e_m.txt)")
    parser.add_argument("--cand_ent_num", type=int, default=30,
                        help="how many candidate entities to keep for each mention")
    parser.add_argument("--wercase_p_e_m", type=bool, default=False)
    parser.add_argument("--lowercase_spans", type=bool, default=False)
    parser.add_argument("--lowercase_p_e_m", type=bool, default=False)
    parser.add_argument("--calculate_stats", type=bool, default=True)

    parser.add_argument("--word_bert", dest="word_bert", action='store_true')
    parser.add_argument("--no_word_bert", dest="word_bert", action='store_false')
    parser.set_defaults(word_bert=False)
    
    parser.add_argument("--context_bert", dest="context_bert", action='store_true')
    parser.add_argument("--no_context_bert", dest="context_bert", action='store_false')
    parser.set_defaults(context_bert=False)
    
    parser.add_argument("--experiment_name", default="corefmerge",
                        help="under folder data/tfrecords/")
    parser.add_argument("--include_wikidumpRLTD", type=bool, default=False)
    parser.add_argument("--word_freq_thr", type=int, default=1,
                        help="words that have freq less than this are not included in our"
                             "vocabulary.")
    parser.add_argument("--char_freq_thr", type=int, default=1)

    parser.add_argument("--max_mention_width", type=int, default=10, help="in allspans mode consider all spans with"
                                              "length <= to this value as candidate entities to be linked")
    parser.add_argument("--entity_extension", default=None, help="extension_entities or extension_entities_all etc")
    parser.add_argument("--persons_coreference", type=bool, default=True)
    parser.add_argument("--persons_coreference_merge", type=bool, default=True)
    parser.add_argument("--create_entity_universe", type=bool, default=False)
    parser.add_argument("--word2vec", default="google", 
                        help="loading Word2vec vectors styles. Choice between 'google', 'google_FR', 'google768', 'glove100', 'glove300'")
    
    parser.add_argument("--prob_p_e_m", default="prob_yago_crosswikis_wikipedia_p_e_m.txt")
    parser.add_argument("--wikiid2nnid_file", default="wikiid2nnid.txt")
    parser.add_argument("--wiki_id_file", default="wiki_name_id_map.txt")
    parser.add_argument("--vocab_file", default="vocab_freq.pickle")
    
    parser.add_argument("--datasets_folder", default="new_datasets")
    return parser.parse_args()


def log_args(folderpath):
    if not os.path.exists(folderpath):
        os.makedirs(folderpath)
    with open(folderpath+"prepro_args.txt", "w") as fout:
        attrs = vars(args)
        fout.write('\n'.join("%s: %s" % item for item in attrs.items()))
    with open(folderpath+"prepro_args.pickle", 'wb') as handle:
        pickle.dump(args, handle)


if __name__ == "__main__":
    args = _parse_args()
    print(args)
    create_necessary_folders()
    log_args(config.base_folder+"data/tfrecords/"+args.experiment_name+"/")
    if(args.word2vec == "google"):
        word2vec = "GoogleNews-vectors-negative300_EN.bin"
    elif(args.word2vec == "google_FR"):
        word2vec = "GoogleNews-vectors-negative300_FR.bin"
    elif(args.word2vec == "google768"):
        word2vec = "embed_Skipgramwiki_en_data_iter5_vec_size_768.bin"    
    elif(args.word2vec == "glove100"):
        word2vec = "wiki_en_glove_Embed_100V_mincount5.bin"
    elif(args.word2vec == "glove300"):
        word2vec = "wiki_en_glove_Embed_300V_mincount5.bin"
    else:
        print("ERROR : option word2vec incorrect -- {}".format(args.word2vec))
        raise Exception("option word2vec incorrect")
    if(not args.word_bert):
        if not os.path.isfile(config.base_folder+"data/vocabulary/{}".format(args.vocab_file)):
            vocabularyCounter = VocabularyCounter(vocab_name=args.vocab_file, new_datasets=args.datasets_folder)
            vocabularyCounter.count_datasets_vocabulary()
        else:
            print("use existing vocaulary : '{}'".format(args.vocab_file))
    if args.create_entity_universe:
        create_entity_universe(gmonly_files=[], allspans_files=['aida_train.txt', 'aida_dev.txt', 'aida_test.txt' # ])
                                                                , 'ace2004.txt', 'aquaint.txt', 'msnbc.txt'])
    else:
        create_tfrecords()













