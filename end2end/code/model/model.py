
# b9d87f7  on Mar 21 Nikolaos Kolitsas ffnn dropout and some minor modif in evaluate to accept entity extension
# ed_model_21_march
import numpy as np
import pickle
try: 
    import tensorflow.compat.v1 as tf
    import tensorflow as tf2
except ImportError: import tensorflow as tf
import model.config as config
from .base_model import BaseModel
import model.util as util
try: import model.transformers_model as Transformer
except ImportError: print("unable to import Transformer")
#from tensorflow.contrib.tensorboard.plugins import projector


class Model(BaseModel):

    def __init__(self, args, next_element):
        super().__init__(args)
        if(self.args.context_bert_lstm):
            self.chunk_id, self.words, self.words_len, self.context_bert, self.chars, self.chars_len,\
            self.begin_span, self.end_span, self.spans_len,\
            self.cand_entities, self.cand_entities_scores, self.cand_entities_labels,\
            self.cand_entities_len, self.ground_truth, self.ground_truth_len,\
            self.begin_gm, self.end_gm = next_element

        else:
            self.chunk_id, self.words, self.words_len, self.chars, self.chars_len,\
            self.begin_span, self.end_span, self.spans_len,\
            self.cand_entities, self.cand_entities_scores, self.cand_entities_labels,\
            self.cand_entities_len, self.ground_truth, self.ground_truth_len,\
            self.begin_gm, self.end_gm = next_element

        self.begin_span = tf.cast(self.begin_span, tf.int32)
        self.end_span = tf.cast(self.end_span, tf.int32)
        self.words_len = tf.cast(self.words_len, tf.int32)
        self.name = self.args.experiment_name
        """
        self.words:  tf.int64, shape=[None, None]   # shape = (batch size, max length of sentence in batch)
        self.words_len: tf.int32, shape=[None],     #   shape = (batch size)
        self.chars: tf.int64, shape=[None, None, None], # shape = (batch size, max length of sentence, max length of word)
        self.chars_len: tf.int64, shape=[None, None],   # shape = (batch_size, max_length of sentence)
        self.begin_span: tf.int32, shape=[None, None],  # shape = (batch_size, max number of candidate spans in one of the batch sentences)
        self.end_span: tf.int32, shape=[None, None],
        self.spans_len: tf.int64, shape=[None],     # shape = (batch size)
        self.cand_entities: tf.int64, shape=[None, None, None],  # shape = (batch size, max number of candidate spans, max number of cand entitites)
        self.cand_entities_scores: tf.float32, shape=[None, None, None],
        self.cand_entities_labels: tf.int64, shape=[None, None, None],
        # shape = (batch_size, max number of candidate spans)
        self.cand_entities_len: tf.int64, shape=[None, None],
        self.ground_truth: tf.int64, shape=[None, None],  # shape = (batch_size, max number of candidate spans)
        self.ground_truth_len: tf.int64, shape=[None],    # shape = (batch_size)
        self.begin_gm: tf.int64, shape=[None, None],  # shape = (batch_size, max number of gold mentions)
        self.end_gm = tf.placeholder(tf.int64, shape=[None, None],
        """
        with open(config.base_folder +"data/tfrecords/" + self.args.experiment_name +
                          "/word_char_maps.pickle", 'rb') as handle:
            _, id2word, _, id2char, _, _ = pickle.load(handle)
            self.nwords = len(id2word)
            self.nchars = len(id2char)
        if(self.args.context_bert_lstm):
            with open(config.base_folder +"data/tfrecords/" + self.args.experiment_name +
                          "/word_char_maps_bert.pickle", 'rb') as handle:
                _, id2bert, _, _, _, _ = pickle.load(handle)
                self.nbert = len(id2bert)

        self.loss_mask = self._sequence_mask_v13(self.cand_entities_len, tf.shape(self.cand_entities_scores)[2])

    def add_placeholders(self):
        """Define placeholders = entries to computational graph"""
        self.dropout = tf.placeholder(dtype=tf.float32, shape=[], name="dropout")
        self.lr = tf.placeholder(dtype=tf.float32, shape=[], name="lr")

    def init_embeddings(self):
        print("\n!!!! init embeddings !!!!\n")
        # read the numpy file
        # WORD EMBEDDINGS
        embeddings_nparray = np.load(config.base_folder +"data/tfrecords/" + self.args.experiment_name +
                                        "/embeddings_array.npy")
        # print("word shape : {}\nplaceholder : {}".format(np.shape(embeddings_nparray),np.shape(self.word_embeddings_placeholder)))
        # print("pure word embedding : {}".format(np.shape(self.pure_word_embeddings)))
        # print("word embedding : {}".format(np.shape(self.word_embeddings)))
        # print("context shape : {}\nplaceholder : {}".format(np.shape(self.context_emb),np.shape(self.context_embeddings_placeholder)))
        # print("context reshape : {}".format(np.shape(self.context_emb_reshape)))
        # print("word shape : {}".format(np.shape(self.words)))
        self.sess.run(self.word_embedding_init, feed_dict={self.word_embeddings_placeholder: embeddings_nparray})
        # BERT EMBEDDINGS
        if(self.args.context_bert_lstm):
            embeddings_bert_nparray = np.load(config.base_folder +"data/tfrecords/" + self.args.experiment_name +
                                        "/embedding_bert.npy")
            self.sess.run(self.bert_embedding_init, feed_dict={self.bert_embeddings_placeholder: embeddings_bert_nparray})
        # ENTITIES EMBEDDINGS
        entity_embeddings_nparray = util.load_ent_vecs(self.args)
        #require_pad = self.nentities - np.shape(entity_embeddings_nparray)[0]
        #print("entity un-padded : {}\npad size : {}".format(np.shape(entity_embeddings_nparray),require_pad))
        #entity_embeddings_nparray = np.pad(entity_embeddings_nparray, pad_width=((0,require_pad),(0,0)), mode="constant")
        #print("entity padded : {}".format(np.shape(entity_embeddings_nparray)))
        print("nentities : {}\nentities shape : {}".format(self.nentities, np.shape(entity_embeddings_nparray)))
        self.sess.run(self.entity_embedding_init, feed_dict={self.entity_embeddings_placeholder: entity_embeddings_nparray})
        # assign the variables for tensorboard
        # self.sess.run(self.context_embedding_init) #, feed_dict={self.context_embeddings_placeholder: self.context_emb})

    def add_embeddings_op(self):
        """Defines self.word_embeddings"""
        with tf.variable_scope("words"): #{}_words".format(self.name)):
            _word_embeddings = tf.get_variable(
                    "variable",
                    #tf.constant(0.0,  ),
                    shape=[self.nwords, self.args.dim_word_emb],
                    dtype=tf.float32,
                    trainable=False)

            self.word_embeddings_placeholder = tf.placeholder(tf.float32, [self.nwords, self.args.dim_word_emb])
            self.word_embedding_init = _word_embeddings.assign(self.word_embeddings_placeholder)

            word_embeddings = tf.nn.embedding_lookup(_word_embeddings,
                                                         self.words, name="words_embeddings")

            # if(self.args.word_bert or self.args.context_bert): word_embeddings = tf.layers.dense(word_embeddings,self.args.dim_ent_emb)
            self.pure_word_embeddings =  word_embeddings
            #print("word_embeddings (after lookup) ", word_embeddings)

        with tf.variable_scope("chars"): #{}_chars".format(self.name)):
            if self.args.use_chars:
                # get char embeddings matrix
                _char_embeddings = tf.get_variable(
                        name="variable",
                        dtype=tf.float32,
                        shape=[self.nchars, self.args.dim_char], trainable=True)
                char_embeddings = tf.nn.embedding_lookup(_char_embeddings,
                                                         self.chars, name="chars_embeddings")

                # char_embeddings: tf.float32, shape=[None, None, None, dim_char],
                # shape = (batch size, max length of sentence, max length of word, dim_char)
                # put the time dimension on axis=1
                s = tf.shape(char_embeddings)
                char_embeddings = tf.reshape(char_embeddings, shape=[s[0] * s[1], s[-2], self.args.dim_char])
                # (batch*sent_length, characters of word, dim_char)

                char_lengths = tf.reshape(self.chars_len, shape=[s[0] * s[1]])
                # shape = (batch_size*max_length of sentence)

                # bi lstm on chars
                cell_fw = tf.nn.rnn_cell.LSTMCell(self.args.hidden_size_char, state_is_tuple=True) #tf.contrib.rnn.LSTMCell(self.args.hidden_size_char, state_is_tuple=True)
                cell_bw = tf.nn.rnn_cell.LSTMCell(self.args.hidden_size_char, state_is_tuple=True) #tf.contrib.rnn.LSTMCell(self.args.hidden_size_char, state_is_tuple=True)
                _output = tf.nn.bidirectional_dynamic_rnn(
                        cell_fw, cell_bw, char_embeddings,
                        sequence_length=char_lengths, dtype=tf.float32)

                # read and concat output
                _, ((_, output_fw), (_, output_bw)) = _output
                output = tf.concat([output_fw, output_bw], axis=-1)

                # shape = (batch size, max sentence length, char hidden size)
                output = tf.reshape(output, shape=[s[0], s[1], 2 * self.args.hidden_size_char])
                #print("output after char lstm ", output)
                word_embeddings = tf.concat([word_embeddings, output], axis=-1)  # concatenate word and char embeddings
                #print("word_embeddings with char after concatenation ", word_embeddings)
                # (batch, words, 100+2*300)
        self.word_embeddings = tf.nn.dropout(word_embeddings, self.dropout)

        with tf.variable_scope("entities"): #{}_entities".format(self.name)):
            from preprocessing.util import load_wikiid2nnid, load_wiki_name_id_map
            self.nentities = len(load_wikiid2nnid(extension_name=self.args.entity_extension, txt_file=self.args.wikiid2nnid_name))
            _entity_embeddings = tf.get_variable(
                "variable",
                #tf.constant(0.0,  ),
                shape=[self.nentities, self.args.dim_ent_emb],
                dtype=tf.float32,
                trainable=self.args.train_ent_vecs)

            self.entity_embeddings_placeholder = tf.placeholder(tf.float32, [self.nentities, self.args.dim_ent_emb])
            self.entity_embedding_init = _entity_embeddings.assign(self.entity_embeddings_placeholder)

            self.entity_embeddings = tf.nn.embedding_lookup(_entity_embeddings, self.cand_entities,
                                                       name="entities_embeddings")
            self.pure_entity_embeddings = self.entity_embeddings
            if self.args.ent_vecs_regularization.startswith("l2"):  # 'l2' or 'l2dropout'
                self.entity_embeddings = tf.nn.l2_normalize(self.entity_embeddings, dim=3)
                # not necessary since i do normalization in the entity embed creation as well, just for safety
            if self.args.ent_vecs_regularization == "dropout" or \
                            self.args.ent_vecs_regularization == "l2dropout":
                self.entity_embeddings = tf.nn.dropout(self.entity_embeddings, self.dropout)
            #print("entity_embeddings = ", self.entity_embeddings)

    def add_context_emb_op(self):
        """this method creates the bidirectional LSTM layer (takes input the v_k vectors and outputs the
        context-aware word embeddings x_k)"""
        with tf.variable_scope("context-bi-lstm"): #{}_context-bi-lstm".format(self.name)):
            dictargs = vars(self.args)
            if(("transformer" in dictargs) and (self.args.transformer)):
                #Si "self.args.transformer" existe, les autres arguments existent aussi
                num_layers = self.args.transformer_size # number of encoder/decoder in each bloc (6 is defaut in article)
                num_neurons = int(np.shape(self.word_embeddings)[-1]) # size of MultiHeadAttention
                num_hidden_neurons = self.args.transformer_neurons # 2*(self.args.hidden_size_lstm) # size of FeedFoward
                num_heads = 8 # Pas compris à quoi ça sert
                num_output = 2*(self.args.hidden_size_lstm)
                enc_size = tf.shape(self.word_embeddings)[1] #int(np.shape(self.word_embeddings)[1]) # word embeddings dimension (defaut : 300 + char dimension)
                dec_size = tf.shape(self.word_embeddings)[1] #2*(self.args.hidden_size_lstm) # context embeddings dimension (default : same size than the bi-LSTM)
                print("Transformer param :\nlayers : {}\nneurons : {}\nhidden neurons : {}\nheads : {}\nenc size : {}\ndec size : {}".format(num_layers, num_neurons, num_hidden_neurons, num_heads, enc_size, dec_size))
                megatron = Transformer.Transformer(num_layers, num_neurons, num_hidden_neurons, num_heads, enc_size, dec_size, num_output=num_output)
                megaMask = Transformer.MaskHandler()
                padding_mask = megaMask.padding_mask(self.word_embeddings)
                lookhead_mask = megaMask.look_ahead_mask(num_hidden_neurons)
                #combined_mask = tf2.maximum(padding_mask, lookhead_mask)
                
                output, _ = megatron(self.word_embeddings, None, (self.dropout != 1), None, None, None) #(self.dropout != 1)
                #output, _ = megatron(self.word_embeddings, None, True, None, lookhead_mask, None) #TODO argument pour différencier entraînement ou non
                
                #output = tf2.convert_to_tensor(output)
                word_shape = tf.shape(self.word_embeddings)
                context_shape = np.shape(output)
                self.context_emb = output #tf.reshape(output, shape=[word_shape[0], word_shape[1], context_shape[2]])
            else:
                cell_fw = tf.nn.rnn_cell.LSTMCell(self.args.hidden_size_lstm) #tf.contrib.rnn.LSTMCell(self.args.hidden_size_lstm)
                cell_bw = tf.nn.rnn_cell.LSTMCell(self.args.hidden_size_lstm) #tf.contrib.rnn.LSTMCell(self.args.hidden_size_lstm)
                (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(
                        cell_fw, cell_bw, self.word_embeddings,
                        sequence_length=self.words_len, dtype=tf.float32)
                output = tf.concat([output_fw, output_bw], axis=-1)
                self.context_emb = tf.nn.dropout(output, self.dropout,name="context_bi_lstm")
                
            cs = tf.shape(self.context_emb)
            csi = tf.size(self.context_emb)
            csn = np.shape(self.context_emb)
            ct = type(self.context_emb)
            print("type ct : {}\nshape cs : {}\nsize cs : {}\nnp shape cs : {}".format(ct,cs,csi,csn))
            #print("context_emb = ", self.context_emb)  # [batch, words, 300]
                    
            #self.context_embeddings_placeholder = tf.placeholder(tf.float32, [self.nwords, dim_context])            
            #self.context_embedding_init = _context_emb.assign(self.context_emb) #self.context_embeddings_placeholder)
            #self.context_emb_lookup = tf.nn.embedding_lookup(_context_emb, self.words) #output)
            
            #self.context_emb_reshape = tf.reshape(self.context_emb, shape=[cs[0]*cs[1],cs[2]])
            #self.image_context_emb = tf.reshape(self.context_emb,shape=[csn[0],csn[1],csn[3],1])
            #print("shape img : {}".format(np.shape(self.image_context_emb)))
    
        if(self.args.context_bert_lstm):
            with tf.variable_scope("embeddings_bert"): #{}_embeddings_bert".format(self.name)):
                _bert_embeddings = tf.get_variable(
                    "variable",
                    #tf.constant(0.0,  ),
                    shape=[self.nbert, 768],#self.args.dim_word_emb]),
                    dtype=tf.float32,
                    trainable=False)

                self.bert_embeddings_placeholder = tf.placeholder(tf.float32, [self.nbert, 768])#self.args.dim_word_emb])
                self.bert_embedding_init = _bert_embeddings.assign(self.bert_embeddings_placeholder)

                self.bert_embeddings = tf.nn.embedding_lookup(_bert_embeddings,
                                                             self.context_bert, name="bert_embeddings")
                                                             
                self.context_emb = tf.concat([self.bert_embeddings,self.context_emb],axis=2) #tf.layers.dense((),self.args.dim_ent_emb)

    def add_span_emb_op(self):
        mention_emb_list = []
        # span embedding based on boundaries (start, end) and head mechanism. but do that on top of contextual bilistm
        # output or on top of original word+char embeddings. this flag determines that. The parer reports results when
        # using the contextual lstm emb as it achieves better score. Used for ablation studies.
        boundaries_input_vecs = self.word_embeddings if self.args.span_boundaries_from_wordemb else self.context_emb

        # the span embedding is modeled by g^m = [x_q; x_r; \hat(x)^m]  (formula (2) of paper)
        # "boundaries" mean use x_q and x_r.   "head" means use also the head mechanism \hat(x)^m (formula (3))
        if self.args.span_emb.find("boundaries") != -1:
            # shape (batch, num_of_cand_spans, emb)
            mention_start_emb = tf.gather_nd(boundaries_input_vecs, tf.stack(
                [tf.tile(tf.expand_dims(tf.range(tf.shape(self.begin_span)[0]), 1), [1, tf.shape(self.begin_span)[1]]),
                 self.begin_span], 2))  # extracts the x_q embedding for each candidate span
            # the tile command creates a 2d tensor with the batch information. first lines contains only zeros, second
            # line ones etc...  because the begin_span tensor has the information which word inside this sentence is the
            # beginning of the candidate span.
            mention_emb_list.append(mention_start_emb)

            mention_end_emb = tf.gather_nd(boundaries_input_vecs, tf.stack(
                [tf.tile(tf.expand_dims(tf.range(tf.shape(self.begin_span)[0]), 1), [1, tf.shape(self.begin_span)[1]]),
                 tf.nn.relu(self.end_span-1)], 2))   # -1 because the end of span in exclusive  [start, end)
            # relu so that the 0 doesn't become -1 of course no valid candidate span end index is zero since [0,0) is empty
            mention_emb_list.append(mention_end_emb)
            #print("mention_start_emb = ", mention_start_emb)
            #print("mention_end_emb = ", mention_end_emb)

        mention_width = self.end_span - self.begin_span  # [batch, num_mentions]     the width of each candidate span

        if self.args.span_emb.find("head") != -1:   # here the attention is computed
            # here the \hat(x)^m is computed (formula (2) and (3))
            self.max_mention_width = tf.minimum(self.args.max_mention_width,
                                                tf.reduce_max(self.end_span - self.begin_span))
            mention_indices = tf.range(self.max_mention_width) + \
                              tf.expand_dims(self.begin_span, 2)  # [batch, num_mentions, max_mention_width]
            mention_indices = tf.minimum(tf.shape(self.entity_embeddings)[1] - 1, #tf.shape(self.word_embeddings)[1] - 1,
                                         mention_indices)  # [batch, num_mentions, max_mention_width]
            #print("mention_indices = ", mention_indices)
            batch_index = tf.tile(tf.expand_dims(tf.expand_dims(tf.range(tf.shape(mention_indices)[0]), 1), 2),
                                  [1, tf.shape(mention_indices)[1], tf.shape(mention_indices)[2]])
            mention_indices = tf.stack([batch_index, mention_indices], 3)
            # [batch, num_mentions, max_mention_width, [row,col] ]    4d tensor

            # for the boundaries we had the option to take them either from x_k (output of bilstm) or from v_k
            # the head is derived either from the same option as boundaries or from the v_k.
            head_input_vecs = boundaries_input_vecs if self.args.model_heads_from_bilstm else self.word_embeddings
            mention_text_emb = tf.gather_nd(head_input_vecs, mention_indices)
            # [batch, num_mentions, max_mention_width, 500 ]    4d tensor
            #print("mention_text_emb = ", mention_text_emb)

            with tf.variable_scope("head_scores"): #{}_head_scores".format(self.name)):
                # from [batch, max_sent_len, 300] to [batch, max_sent_len, 1]
                self.head_scores = util.projection(boundaries_input_vecs, 1)
            # [batch, num_mentions, max_mention_width, 1]
            mention_head_scores = tf.gather_nd(self.head_scores, mention_indices)
            # print("mention_head_scores = ", mention_head_scores)

            # depending on tensorflow version we do the same with different operations (since each candidate span is not
            # of the same length we mask out the invalid indices created above (mention_indices)).
            temp_mask = self._sequence_mask_v13(mention_width, self.max_mention_width)
            # still code for masking invalid indices for the head computation
            mention_mask = tf.expand_dims(temp_mask, 3)  # [batch, num_mentions, max_mention_width, 1]
            mention_mask = tf.minimum(1.0, tf.maximum(self.args.zero, mention_mask))  # 1e-3
            # formula (3) computation
            mention_attention = tf.nn.softmax(mention_head_scores + tf.log(mention_mask),
                                              dim=2)  # [batch, num_mentions, max_mention_width, 1]
            mention_head_emb = tf.reduce_sum(mention_attention * mention_text_emb, 2)  # [batch, num_mentions, emb]
            #print("mention_head_emb = ", mention_head_emb)
            mention_emb_list.append(mention_head_emb)

        self.span_emb = tf.concat(mention_emb_list, 2) # [batch, num_mentions, emb i.e. 1700] formula (2) concatenation
        #print("span_emb = ", self.span_emb)

    def add_lstm_score_op(self):
        with tf.variable_scope("span_emb_ffnn"): #{}_span_emb_ffnn".format(self.name)):
            # [batch, num_mentions, self.args.dim_ent_emb]
            # the span embedding can have different size depending on the chosen hyperparameters. We project it to 300 (default)
            # dims to match the entity embeddings  (formula 4)
            if self.args.span_emb_ffnn[0] == 0:
                span_emb_projected = util.projection(self.span_emb, self.args.dim_ent_emb)
            else:
                hidden_layers, hidden_size = self.args.span_emb_ffnn[0], self.args.span_emb_ffnn[1]
                span_emb_projected = util.ffnn(self.span_emb, hidden_layers, hidden_size, self.args.dim_ent_emb,
                                               self.dropout if self.args.ffnn_dropout else None)
                #print("span_emb_projected = ", span_emb_projected)
        # formula (6) <x^m, y_j>   computation. this is the lstm score
        scores = tf.matmul(tf.expand_dims(span_emb_projected, 2), self.entity_embeddings, transpose_b=True)
        #print("scores = ", scores)
        self.similarity_scores = tf.squeeze(scores, axis=2)  # [batch, num_mentions, 1, 30]
        #print("scores = ", self.similarity_scores)   # [batch, num_mentions, 30]

    def add_local_attention_op(self):
        attention_entity_emb = self.pure_entity_embeddings if self.args.attention_ent_vecs_no_regularization else self.entity_embeddings
        with tf.variable_scope("attention"): #{}_attention".format(self.name)):
            K = self.args.attention_K
            left_mask = self._sequence_mask_v13(self.begin_span, K)   # number of words on the left (left window)
            right_mask = self._sequence_mask_v13(tf.expand_dims(self.words_len, 1) - self.end_span, K)
            # number of words on the right. of course i don't get more than K even if more words exist.
            ctxt_mask = tf.concat([left_mask, right_mask], 2)  # [batch, num_of_spans, 2*K]
            ctxt_mask = tf.log(tf.minimum(1.0, tf.maximum(self.args.zero, ctxt_mask)))
               #  T,   T,  T, F,  F | T,  T,  F,  F,  F
               # -1, -2, -3, -4, -5  +0, +1, +2, +3, +4

            leftctxt_indices = tf.maximum(0, tf.range(-1, -K - 1, -1) +
                                          tf.expand_dims(self.begin_span, 2))  # [batch, num_mentions, K]
            rightctxt_indices = tf.minimum(tf.shape(self.pure_word_embeddings)[1] - 1, tf.range(K) +
                                           tf.expand_dims(self.end_span, 2))  # [batch, num_mentions, K]
            ctxt_indices = tf.concat([leftctxt_indices, rightctxt_indices], 2)  # [batch, num_mentions, 2*K]

            batch_index = tf.tile(tf.expand_dims(tf.expand_dims(tf.range(tf.shape(ctxt_indices)[0]), 1), 2),
                                  [1, tf.shape(ctxt_indices)[1], tf.shape(ctxt_indices)[2]])
            ctxt_indices = tf.stack([batch_index, ctxt_indices], 3)
            # [batch, num_of_spans, 2*K, 2]   the last dimension is row,col for gather_nd
            # [batch, num_of_spans, 2*K, [row,col]]

            att_x_w = self.pure_word_embeddings  # [batch, max_sent_len, 300]
            if self.args.attention_on_lstm and self.args.nn_components.find("lstm") != -1:
                # ablation: here the attention is computed on the output of the lstm layer x_k instead of using the
                # pure word2vec vectors. (word2vec used in paper).
                att_x_w = util.projection(self.context_emb, self.args.dim_ent_emb)  # if tf.shape(self.context_emb)[-1] != 300 else self.context_emb

            ctxt_word_emb = tf.gather_nd(att_x_w, ctxt_indices)
            # [batch, num_of_spans, 2K, emb_size]    emb_size = 300  only pure word emb used  (word2vec)
            #  and not after we add char emb and dropout

            # in this implementation we don't use the diagonal A and B arrays that are mentioned in
            # Ganea and Hoffmann 2017 (only used in the ablations)
            temp = attention_entity_emb
            if self.args.attention_use_AB:
                att_A = tf.get_variable("att_A", [self.args.dim_ent_emb])
                temp = att_A * attention_entity_emb
            scores = tf.matmul(ctxt_word_emb, temp, transpose_b=True)
            scores = tf.reduce_max(scores, reduction_indices=[-1])  # max score of each word for each span acquired from any cand entity
            scores = scores + ctxt_mask   # some words are not valid out of window so we assign to them very low score
            top_values, _ = tf.nn.top_k(scores, self.args.attention_R)
            # [batch, num_of_spans, R]
            R_value = top_values[:, :, -1]    # [batch, num_of_spans]
            R_value = tf.maximum(self.args.zero, R_value)  # so to avoid keeping words that
            # have max score with any of the entities <=0 (also score = 0 can have words with
            # padding candidate entities)

            threshold = tf.tile(tf.expand_dims(R_value, 2), [1, 1, 2 * K])
            # [batch, num_of_spans, 2K]
            scores = scores - tf.to_float(((scores - threshold) < 0)) * 50  # 50 where score<thr, 0 where score>=thr
            scores = tf.nn.softmax(scores, dim=2)  # [batch, num_of_spans, 2K]
            scores = tf.expand_dims(scores, 3)  # [batch, num_of_spans, 2K, 1]
            #    [batch, num_of_spans, 2K, 1]  *  [batch, num_of_spans, 2K, emb_size]
            # =  [batch, num_of_spans, 2K, emb_size]
            x_c = tf.reduce_sum(scores * ctxt_word_emb, 2)  # =  [batch, num_of_spans, emb_size]
            if self.args.attention_use_AB:
                att_B = tf.get_variable("att_B", [self.args.dim_ent_emb])
                x_c = att_B * x_c
            x_c = tf.expand_dims(x_c, 3)   # [batch, num_of_spans, emb_size, 1]
            # [batch, num_of_spans, 30, emb_size=300]  mul with  [batch, num_of_spans, emb_size, 1]
            x_e__x_c = tf.matmul(attention_entity_emb, x_c)  # [batch, num_of_spans, 30, 1]
            x_e__x_c = tf.squeeze(x_e__x_c, axis=3)  # [batch, num_of_spans, 30]
            self.attention_scores = x_e__x_c

    def add_cand_ent_scores_op(self):
        self.log_cand_entities_scores = tf.log(tf.minimum(1.0, tf.maximum(self.args.zero, self.cand_entities_scores)))
        stack_values = []
        if self.args.nn_components.find("lstm") != -1:
            stack_values.append(self.similarity_scores)
        if self.args.nn_components.find("pem") != -1:
            stack_values.append(self.log_cand_entities_scores)
        if self.args.nn_components.find("attention") != -1:
            stack_values.append(self.attention_scores)

        scalar_predictors = tf.stack(stack_values, 3)
        #print("scalar_predictors = ", scalar_predictors)   # [batch, num_mentions, 30, 3]

        with tf.variable_scope("similarity_and_prior_ffnn"): #{}_similarity_and_prior_ffnn".format(self.name)):
            if self.args.final_score_ffnn[0] == 0:
                self.final_scores = util.projection(scalar_predictors, 1)  # [batch, num_mentions, 30, 1]
            else:
                hidden_layers, hidden_size = self.args.final_score_ffnn[0], self.args.final_score_ffnn[1]
                self.final_scores = util.ffnn(scalar_predictors, hidden_layers, hidden_size, 1,
                                              self.dropout if self.args.ffnn_dropout else None)
            self.final_scores = tf.squeeze(self.final_scores, axis=3)  # squeeze to [batch, num_mentions, 30]
            #print("final_scores = ", self.final_scores)

    def add_global_voting_op(self):
        with tf.variable_scope("global_voting"): #{}_global_voting".format(self.name)):
            self.final_scores_before_global = - (1 - self.loss_mask) * 50 + self.final_scores
            gmask = tf.to_float(((self.final_scores_before_global - self.args.global_thr) >= 0))  # [b,s,30]

            masked_entity_emb = self.pure_entity_embeddings * tf.expand_dims(gmask, axis=3)  # [b,s,30,300] * [b,s,30,1]
            batch_size = tf.shape(masked_entity_emb)[0]
            all_voters_emb = tf.reduce_sum(tf.reshape(masked_entity_emb, [batch_size, -1, self.args.dim_ent_emb]), axis=1,
                                           keep_dims=True)  # [b, 1, 300]
            span_voters_emb = tf.reduce_sum(masked_entity_emb, axis=2)  # [batch, num_of_spans, 300]
            valid_voters_emb = all_voters_emb - span_voters_emb
            # [b, 1, 300] - [batch, spans, 300] = [batch, spans, 300]  (broadcasting)
            # [300] - [batch, spans, 300]  = [batch, spans, 300]  (broadcasting)
            valid_voters_emb = tf.nn.l2_normalize(valid_voters_emb, dim=2)

            self.global_voting_scores = tf.squeeze(tf.matmul(self.pure_entity_embeddings, tf.expand_dims(valid_voters_emb, axis=3)), axis=3)
            # [b,s,30,300] matmul [b,s,300,1] --> [b,s,30,1]-->[b,s,30]

            scalar_predictors = tf.stack([self.final_scores_before_global, self.global_voting_scores], 3)
            #print("scalar_predictors = ", scalar_predictors)   #[b, s, 30, 2]
            with tf.variable_scope("psi_and_global_ffnn"): #{}_psi_and_global_ffnn".format(self.name)):
                if self.args.global_score_ffnn[0] == 0:
                    self.final_scores = util.projection(scalar_predictors, 1)
                else:
                    hidden_layers, hidden_size = self.args.global_score_ffnn[0], self.args.global_score_ffnn[1]
                    self.final_scores = util.ffnn(scalar_predictors, hidden_layers, hidden_size, 1,
                                                  self.dropout if self.args.ffnn_dropout else None)
                # [batch, num_mentions, 30, 1] squeeze to [batch, num_mentions, 30]
                self.final_scores = tf.squeeze(self.final_scores, axis=3)
                #print("final_scores = ", self.final_scores)

    def add_loss_op(self):
        cand_entities_labels = tf.cast(self.cand_entities_labels, tf.float32)
        loss1 = cand_entities_labels * tf.nn.relu(self.args.gamma_thr - self.final_scores)
        loss2 = (1 - cand_entities_labels) * tf.nn.relu(self.final_scores)
        self.loss = loss1 + loss2
        if self.args.nn_components.find("global") != -1 and not self.args.global_one_loss:
            loss3 = cand_entities_labels * tf.nn.relu(self.args.gamma_thr - self.final_scores_before_global)
            loss4 = (1 - cand_entities_labels) * tf.nn.relu(self.final_scores_before_global)
            self.loss = loss1 + loss2 + loss3 + loss4
        #print("loss_mask = ", loss_mask)
        self.loss = self.loss_mask * self.loss
        self.loss = tf.reduce_sum(self.loss)
        # for tensorboard
        #tf.summary.scalar("loss", self.loss)

    def def_param_tensorboard(self):  
        # DEFINE WHAT SAVE          
        layer_ids = [("word_embeddings",self.word_embeddings),
                      ("entities_embeddings",self.entity_embeddings),
                      ("context_bi_lstm",self.context_emb),
                      ("mention_embeddings",self.span_emb)]
        layers_fails = []
        nb_fails = 0
        all_summaries = []
        if(self.args.context_bert_lstm): layer_ids.append(("bert_embeddings",self.bert_embeddings))
        # DEFINE HISTOGRAM
        for lid,tensor in layer_ids:
            try :
                # Create a scalar summary object for the loss so it can be displayed
                tf_w_hist = tf.summary.histogram('{}'.format(lid), tensor)
                all_summaries.append(tf_w_hist)
            except ValueError:
                nb_fails+=1
                layers_fails.append(lid) 
                
        # DEFINE IMAGES
        #tf_w_img = tf.summary.image("image context_emb",self.image_context_emb)
        #all_summaries.append(tf_w_img) 
        
        # DEFINE PROJECTORS
        # configs = []
        # for id, tensor in layer_ids:
            # config = projector.ProjectorConfig()
            # embedding = config.embeddings.add()
            # embedding.tensor_name = tensor.name
            # configs.append(config)
        
        # MERGE ALL
        print("param non récupérés : {}/{}\nparam échoués : {}".format(nb_fails,len(layer_ids),layers_fails))
        if len(layer_ids) > len(layers_fails): self.tf_param_summaries = tf.summary.merge(all_summaries)
        self.projector_embeddings = [x[1] for x in layer_ids]
        self.layers = dict([(v.name, v) for v in tf.trainable_variables()])
        
    def build(self):
        self.add_placeholders()
        self.add_embeddings_op()
        if self.args.nn_components.find("lstm") != -1:
            self.add_context_emb_op()
            self.add_span_emb_op()
            self.add_lstm_score_op()
        if self.args.nn_components.find("attention") != -1:
            self.add_local_attention_op()
        self.add_cand_ent_scores_op()
        if self.args.nn_components.find("global") != -1:
            self.add_global_voting_op()
        if self.args.running_mode.startswith("tr"): # "train", "train_continue", "transfert_learning"
            self.add_loss_op()
            # Generic functions that add training op
            self.add_train_op(self.args.lr_method, self.lr, self.loss, self.args.clip)
            
        self.def_param_tensorboard()
        self.merged_summary_op = tf.summary.merge_all()

        var_name_list = [v.name for v in tf.trainable_variables()]
        print(50*"/"+"\nvar_name_list (self.layers) :")
        for v in var_name_list:
            print("- {}".format(v))
        print(50*"/")
        
        print("building mode : {}".format(self.args.running_mode))
        
        if self.args.running_mode == "train_continue" :
            self.restore_session("el")
            
        elif self.args.running_mode == "transfert_learning" :
            self.load_pretrained_model(config.base_folder+self.args.pretrained_model) # "../data/final_model.pkl"
         
        elif self.args.running_mode == "evaluation":
            final_model_path = self.args.checkpoints_folder+"final_model.pkl"
            print("load model from : {}".format(final_model_path))
            self.load_pretrained_model(final_model_path)
            
        elif self.args.running_mode == "train":
            self.initialize_session()  # now self.sess is defined and vars are init
            self.init_embeddings()
        
        # if we run the evaluate.py script then we should call explicitly the model.restore("ed")
        # or model.restore("el"). here it doesn't initialize or restore values for the evaluate.py
        # case.

    def _sequence_mask_v13(self, mytensor, max_width):
        """mytensor is a 2d tensor"""
        if not tf.__version__.startswith("1.4"):
            temp_shape = tf.shape(mytensor)
            temp = tf.sequence_mask(tf.reshape(mytensor, [-1]), max_width, dtype=tf.float32)
            temp_mask = tf.reshape(temp, [temp_shape[0], temp_shape[1], tf.shape(temp)[-1]])
        else:
            temp_mask = tf.sequence_mask(mytensor, max_width, dtype=tf.float32)
        return temp_mask


