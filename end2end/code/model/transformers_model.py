################################################ TRANSFORMER MODEL ################################################
try:
    import tensorflow.compat.v1 as tf
    import tensorflow as tf2
except ImportError:
    import tensorflow as tf
    import tensorflow as tf2

from tensorflow.keras.layers import Layer, Dense, LayerNormalization, Embedding, Dropout
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers.schedules import LearningRateSchedule
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import Mean, SparseCategoricalAccuracy

import numpy as np




################################################# GLOBAL CLASS ##############################################################
class Transformer(Model):
    def __init__(self, num_layers, num_neurons, num_hidden_neurons, num_heads, input_vocabular_size, target_vocabular_size, num_output=-1):
        print("Building Transformer")
        if(num_output == -1): num_output = target_vocabular_size
        super(Transformer, self).__init__()
        #with tf.variable_scope("transformer_encoders"): 
        self.encoder = Encoder(num_neurons, num_hidden_neurons, num_heads, input_vocabular_size, num_layers)
        #with tf.variable_scope("transformer_decoders"):
        self.decoder = Decoder(num_neurons, num_hidden_neurons, num_heads, target_vocabular_size, num_layers)
        #with tf.variable_scope("transformer_softmax"):
        self.linear_layer = Dense(num_output)

    def call(self, transformer_input, tar, training, encoder_padding_mask, look_ahead_mask, decoder_padding_mask):
        print("########## SHAPE ##########")
        print("init : {}".format(np.shape(transformer_input)))
        encoder_output = self.encoder(transformer_input, training, encoder_padding_mask)
        print("encoder : {}".format(np.shape(encoder_output)))
        decoder_output, attention_weights = self.decoder(tar, encoder_output, training, look_ahead_mask, decoder_padding_mask)
        print("decoder : {}".format(np.shape(decoder_output)))
        output = self.linear_layer(decoder_output)
        print("dense : {}".format(np.shape(output)))
        print("###########################")

        return output, attention_weights
##############################################################################################################################
################################################# ENCODER CLASS ############################################################################
class EncoderLayer(Layer):
    def __init__(self, num_neurons, num_hidden_neurons, num_heads):
        print("\tInit Encoder Layer")
        super(EncoderLayer, self).__init__()

        # Build multi head attention layer and necessary additional layers
        self.multi_head_attention_layer, self.attention_dropout, self.attention_normalization = \
        build_multi_head_attention_layers(num_neurons, num_heads)   
            
        # Build feed-forward neural network and necessary additional layers
        self.feed_forward_layer, self.feed_forward_dropout, self.feed_forward_normalization = \
        build_feed_forward_layers(num_neurons, num_hidden_neurons)
       
    def call(self, sequence, training, mask):

        # Calculate attention output
        attnention_output, _ = self.multi_head_attention_layer(sequence, sequence, sequence, mask)
        attnention_output = self.attention_dropout(attnention_output, training=training)
        attnention_output = self.attention_normalization(sequence + attnention_output)
        
        # Calculate output of feed forward network
        output = self.feed_forward_layer(attnention_output)
        output = self.feed_forward_dropout(output, training=training)
        
        # Combine two outputs
        output = self.feed_forward_normalization(attnention_output + output)

        return output
        
class Encoder(Layer):
    def __init__(self, num_neurons, num_hidden_neurons, num_heads, vocabular_size, num_enc_layers = 6):
        print("Building Encoder size {}".format(num_enc_layers))
        super(Encoder, self).__init__()
        
        self.num_enc_layers = num_enc_layers
        
        self.pre_processing_layer = PreProcessingLayer(num_neurons, vocabular_size)
        self.encoder_layers = [EncoderLayer(num_neurons, num_hidden_neurons, num_heads) for _ in range(num_enc_layers)]

    def call(self, sequence, training, mask):
        
        sequence = self.pre_processing_layer(sequence, training, mask)
        for i in range(self.num_enc_layers):
            sequence = self.encoder_layers[i](sequence, training, mask)

        return sequence
#########################################################################################################################################
################################################# DECODER CLASS #########################################################################
class DecoderLayer(Layer):
    def __init__(self, num_neurons, num_hidden_neurons, num_heads):
        print("\tInit Decoder Layer")
        super(DecoderLayer, self).__init__()

        # Build multi head attention layers and necessary additional layers
        self.multi_head_attention_layer1, self.attention_dropout1, self.attention_normalization1 =\
        build_multi_head_attention_layers(num_neurons, num_heads)   
        
        self.multi_head_attention_layer2, self.attention_dropout2, self.attention_normalization2 =\
        build_multi_head_attention_layers(num_neurons, num_heads)           

        # Build feed-forward neural network and necessary additional layers
        self.feed_forward_layer, self.feed_forward_dropout, self.feed_forward_normalization = \
        build_feed_forward_layers(num_neurons, num_hidden_neurons)

    def call(self, sequence, enconder_output, training, look_ahead_mask, padding_mask):

        attnention_output1, attnention_weights1 = self.multi_head_attention_layer1(sequence, sequence, sequence, look_ahead_mask)
        #attnention_output1, attnention_weights1 = self.multi_head_attention_layer1(enconder_output, enconder_output, enconder_output, look_ahead_mask)
        attnention_output1 = self.attention_dropout1(attnention_output1, training=training)
        attnention_output1 = self.attention_normalization1(sequence + attnention_output1)
        
        attnention_output2, attnention_weights2 = self.multi_head_attention_layer2(enconder_output, enconder_output, attnention_output1, padding_mask)
        #attnention_output2, attnention_weights2 = self.multi_head_attention_layer2(attnention_output1, attnention_output1, attnention_output1, padding_mask)
        attnention_output2 = self.attention_dropout1(attnention_output2, training=training)
        attnention_output2 = self.attention_normalization1(attnention_output1 + attnention_output2)

        output = self.feed_forward_layer(attnention_output2)
        output = self.feed_forward_dropout(output, training=training)
        output = self.feed_forward_normalization(attnention_output2 + output)

        return output, attnention_weights1, attnention_weights2
        
class Decoder(Layer):
    def __init__(self, num_neurons, num_hidden_neurons, num_heads, vocabular_size, num_dec_layers=6):
        print("Building Decoder size {}".format(num_dec_layers))
        super(Decoder, self).__init__()

        self.num_dec_layers = num_dec_layers
        
        self.pre_processing_layer = PreProcessingLayer(num_neurons, vocabular_size)
        self.decoder_layers = [DecoderLayer(num_neurons, num_hidden_neurons, num_heads) for _ in range(num_dec_layers)]

    def call(self, sequence, encoder_output, training, look_ahead_mask, padding_mask):
            
        #sequence = self.pre_processing_layer(sequence, training, look_ahead_mask)
        sequence = self.pre_processing_layer(encoder_output, training, look_ahead_mask)
        attention_weights = dict()
        
        for i in range(self.num_dec_layers):

            sequence, attention_weights1, attention_weights2 = self.decoder_layers[i](sequence, encoder_output, training, look_ahead_mask, padding_mask)
            attention_weights['decoder_layer{}_attention_weights1'.format(i+1)] = attention_weights1
            attention_weights['decoder_layer{}_attention_weights2'.format(i+1)] = attention_weights2

        return sequence, attention_weights
#########################################################################################################################################
############################################# MULTIHEAD ATTENTION #######################################################################
class MultiHeadAttentionLayer(Layer):
    def __init__(self, num_neurons, num_heads):
        print("\tInit Multi Head Attention")
        super(MultiHeadAttentionLayer, self).__init__()
        
        self.num_heads = num_heads
        self.num_neurons = num_neurons
        self.depth = num_neurons // self.num_heads
        self.attention_layer = ScaledDotProductAttentionLayer()
        
        self.q_layer = Dense(num_neurons)
        self.k_layer = Dense(num_neurons)
        self.v_layer = Dense(num_neurons)

        self.linear_layer = Dense(num_neurons)

    def split(self, x, batch_size):
        x = tf2.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf2.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask):
        batch_size = tf2.shape(q)[0]
        dim_size = tf2.shape(q)[1]
        dim_size2 = tf2.shape(q)[2]
        len_size = len(tf2.shape(q))
        print("batch_size : \n\tnp : {}\n\ttf[0] : {}\n\ttf[1] : {}\n\ttf[2] : {}\n\tlen : {}\n".format(np.shape(q),batch_size,dim_size,dim_size2,len_size))
        # Run through linear layers
        q = self.q_layer(q)
        k = self.k_layer(k)
        v = self.v_layer(v)

        # Split the heads
        q = self.split(q, batch_size)
        k = self.split(k, batch_size)
        v = self.split(v, batch_size)

        # Run through attention
        attention_output, weights = self.attention_layer.calculate_output_weights(q, k, v, mask)
        
        # Prepare for the rest of processing
        output = tf2.transpose(attention_output, perm=[0, 2, 1, 3])
        concat_attention = tf2.reshape(output, (batch_size, -1, self.num_neurons))
        
        # Run through final linear layer
        output = self.linear_layer(concat_attention)

        return output, weights
        
class ScaledDotProductAttentionLayer():
    def calculate_output_weights(self, q, k, v, mask):
        qk = tf2.matmul(q, k, transpose_b=True)
        dk = tf2.cast(tf2.shape(k)[-1], tf2.float32)
        scaled_attention = qk / tf2.math.sqrt(dk)

        if mask is not None:
            scaled_attention += (mask * -1e9)  

        weights = tf2.nn.softmax(scaled_attention, axis=-1)
        output = tf2.matmul(weights, v)

        return output, weights
        
def build_multi_head_attention_layers(num_neurons, num_heads):
    multi_head_attention_layer = MultiHeadAttentionLayer(num_neurons, num_heads)   
    dropout = tf2.keras.layers.Dropout(0.1)
    normalization = LayerNormalization(epsilon=1e-6)
    return multi_head_attention_layer, dropout, normalization

def build_feed_forward_layers(num_neurons, num_hidden_neurons):
    feed_forward_layer = tf2.keras.Sequential()
    feed_forward_layer.add(Dense(num_hidden_neurons, activation='relu'))
    feed_forward_layer.add(Dense(num_neurons))
        
    dropout = Dropout(0.1)
    normalization = LayerNormalization(epsilon=1e-6)
    return feed_forward_layer, dropout, normalization
#########################################################################################################################################
########################################## POSITIONNAL LAYER ############################################################################
class PreProcessingLayer(Layer):
    def __init__(self, num_neurons, vocabular_size):
        print("\tInit Positionnal Layer")
        super(PreProcessingLayer, self).__init__()
        
        # Initialize
        self.num_neurons = num_neurons

        # Add embedings and positional encoding
        self.embedding = Embedding(vocabular_size, self.num_neurons)
        positional_encoding = PositionalEncoding(vocabular_size, self.num_neurons)
        self.positional_encoding = positional_encoding.get_positional_encoding()

        # Add embedings and positional encoding
        self.dropout = Dropout(0.1)
    
    def call(self, sequence, training, mask):
        sequence_lenght = tf2.shape(sequence)[1]
        #print("\t\tpre process init : {}".format(np.shape(sequence)))
        #sequence = self.embedding(sequence)
        #print("\t\tpre process step 0 : {}".format(np.shape(sequence)))
        sequence *= tf2.math.sqrt(tf2.cast(self.num_neurons, tf2.float32))
        sequence += self.positional_encoding[:, :sequence_lenght, :]
        sequence = self.dropout(sequence, training=training)
        
        return sequence
        
class PositionalEncoding(object):
    def __init__(self, position, d):
        angle_rads = self._get_angles(tf.range(position)[:, np.newaxis], tf.range(d)[np.newaxis, :], d)
        #self._get_angles(np.arange(position)[:, np.newaxis], np.arange(d)[np.newaxis, :], d)

        sines = tf.math.sin(angle_rads[:, 0::2]) #np.sin(angle_rads[:, 0::2])
        cosines = tf.math.cos(angle_rads[:, 0::2]) #tf.cos(angle_rads[:, 1::2])
        self._encoding = tf.concat([sines, cosines], axis=-1) #np.concatenate([sines, cosines], axis=-1)
        self._encoding = self._encoding[np.newaxis, ...]
    
    def _get_angles(self, position, i, d):
        #print("position : {} {}".format(type(position),np.shape(position)))
        ftposition = tf.cast(position, tf.float64)
        angle_rates = 1 / tf.math.pow(np.float64(10000), (2 * (i//2)) / d) #np.power | np.float32(d)
        #print("position : {} {}\nangle : {} {}".format(type(ftposition),np.shape(ftposition),type(angle_rates),tf.shape(angle_rates)))
        return tf2.multiply(ftposition,angle_rates) #np.float64(position)
    
    def get_positional_encoding(self):
        return tf2.cast(self._encoding, dtype=tf2.float32)
#########################################################################################################################################

####################################### DATASET HANDLER FOR TEST ########################################################################
class MaskHandler(object):
    def padding_mask(self, sequence):
        sequence = tf2.cast(tf2.math.equal(sequence, 0), tf2.float32)
        return sequence[:, tf2.newaxis, tf2.newaxis, :]

    def look_ahead_mask(self, size):
        mask = 1 - tf2.linalg.band_part(tf2.ones((size, size)), -1, 0)
        return mask

class DataHandler(object):         
    def __init__(self, word_max_length = 30, batch_size = 64, buffer_size = 20000):
        
        train_data, test_data = self._load_data()
        
        self.tokenizer_ru = tfds.features.text.SubwordTextEncoder.build_from_corpus((ru.numpy() for ru, en in train_data), target_vocab_size=2**13)
        self.tokenizer_en = tfds.features.text.SubwordTextEncoder.build_from_corpus((en.numpy() for ru, en in train_data), target_vocab_size=2**13)
        
        self.train_data = self._prepare_training_data(train_data, word_max_length, batch_size, buffer_size)
        self.test_data = self._prepare_testing_data(test_data, word_max_length, batch_size)
       
    def _load_data(self):
        data, info = tfds.load('ted_hrlr_translate/ru_to_en', with_info=True, as_supervised=True)
        return data['train'], data['validation']
    
    def _prepare_training_data(self, data, word_max_length, batch_size, buffer_size):
        data = data.map(self._encode_tf_wrapper)
        data.filter(lambda x, y: tf2.logical_and(tf2.size(x) <= word_max_length, tf2.size(y) <= word_max_length))
        data = data.cache()
        data = data.shuffle(buffer_size).padded_batch(batch_size, padded_shapes=([-1], [-1]))
        data = data.prefetch(tf2.data.experimental.AUTOTUNE)
        return data
        
    def _prepare_testing_data(self, data, word_max_length, batch_size):
        data = data.map(self._encode_tf_wrapper)
        data = data.filter(lambda x, y: tf2.logical_and(tf2.size(x) <= word_max_length, tf2.size(y) <= word_max_length)).padded_batch(batch_size, padded_shapes=([-1], [-1]))
        
    
    def _encode(self, english, russian):
        russian = [self.tokenizer_ru.vocab_size] + self.tokenizer_ru.encode(russian.numpy()) + [self.tokenizer_ru.vocab_size+1]
        english = [self.tokenizer_en.vocab_size] + self.tokenizer_en.encode(english.numpy()) + [self.tokenizer_en.vocab_size+1]

        return russian, english
    
    def _encode_tf_wrapper(self, pt, en):
        return tf2.py_function(self._encode, [pt, en], [tf2.int64, tf2.int64])
#########################################################################################################################################


if __name__ == "__main__":
    print("BEGIN TRANSFORMER TESTING")
    # Import spécifique
    import tensorflow_datasets as tfds
    from tqdm import tqdm
    from time import time
    
    # Fonctions
    class Schedule(LearningRateSchedule):
        def __init__(self, num_neurons, warmup_steps=4000):
            super(Schedule, self).__init__()

            self.num_neurons = tf2.cast(num_neurons, tf2.float32)
            self.warmup_steps = warmup_steps

        def __call__(self, step):
            arg1 = tf2.math.rsqrt(step)
            arg2 = step * (self.warmup_steps ** -1.5)

            return tf2.math.rsqrt(self.num_neurons) * tf2.math.minimum(arg1, arg2)

    train_step_signature = [
        tf2.TensorSpec(shape=(None, None), dtype=tf2.int64),
        tf2.TensorSpec(shape=(None, None), dtype=tf2.int64),
    ]    
    @tf.function(input_signature=train_step_signature)
    def train_step(input_language, target_language):
        target_input = target_language[:, :-1]
        tartet_output = target_language[:, 1:]
        
        # Create masks
        encoder_padding_mask = maskHandler.padding_mask(input_language)
        decoder_padding_mask = maskHandler.padding_mask(input_language)
        
        look_ahead_mask = maskHandler.look_ahead_mask(tf2.shape(target_language)[1])
        decoder_target_padding_mask = maskHandler.padding_mask(target_language)
        combined_mask = tf2.maximum(decoder_target_padding_mask, look_ahead_mask)
        
        # Run training step
        with tf2.GradientTape() as tape:
            predictions, _ = optimus_prime(input_language, target_input,  True, encoder_padding_mask, combined_mask, decoder_padding_mask)
            total_loss = padded_loss_function(tartet_output, predictions)


        gradients = tape.gradient(total_loss, optimus_prime.trainable_variables)    
        optimizer.apply_gradients(zip(gradients, optimus_prime.trainable_variables))
        training_loss(total_loss)
        training_accuracy(tartet_output, predictions)

    def padded_loss_function(real, prediction):
        mask = tf2.math.logical_not(tf2.math.equal(real, 0))
        loss = loss_objective_function(real, prediction)

        mask = tf2.cast(mask, dtype=loss.dtype)
        loss *= mask

        return tf2.reduce_mean(loss)        
            
    # Initialize helpers
    top = time()
    print(">>> data & mask handler")
    data_container = DataHandler()
    maskHandler = MaskHandler()
    lendata = 0
    for data in enumerate(data_container.train_data):
        lendata += 1
    print("\tdata_container : {}".format(lendata))
    print(">>> done in {:.1f}s".format(time()-top))
    
    # Initialize parameters
    num_layers = 4
    num_neurons = 128
    num_hidden_neurons = 512
    num_heads = 8
    epochs = 10

    # Initialize vocabular size
    input_vocabular_size = data_container.tokenizer_ru.vocab_size + 2
    target_vocabular_size = data_container.tokenizer_en.vocab_size + 2
    print("input : {}\ntarget : {}".format(input_vocabular_size,target_vocabular_size))
    
    # Initialize learning rate
    learning_rate = Schedule(num_neurons)
    optimizer = Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
    
    print(">>> create transformer")
    top2 = time()
    optimus_prime = Transformer(num_layers, num_neurons, num_hidden_neurons, num_heads, input_vocabular_size, target_vocabular_size)
    print(">>> done in {:.1f}s".format(time()-top2))
    
    print(">>> prepare training")
    
    earning_rate = Schedule(num_neurons)
    optimizer = Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
    loss_objective_function = SparseCategoricalCrossentropy(from_logits=True, reduction='none')

    training_loss = Mean(name='training_loss')
    training_accuracy = SparseCategoricalAccuracy(name='training_accuracy')
    
    print(">>> training")
    top3 = time()
    for epoch in tqdm(range(epochs)):
        training_loss.reset_states()
        training_accuracy.reset_states()
        print("Epoch {}".format(epoch))
        i = 0
        for (batch, (input_language, target_language)) in enumerate(data_container.train_data):
            train_step(input_language, target_language)
            print("\tbatch loss {} : {:.4f}".format(i, training_loss.result()))
            i += 1
            if i >= 100: break
        print ("\tFinal Loss {:.4f}\n\tFinal Accuracy {:.4f}".format(training_loss.result(), training_accuracy.result()))
        print(50*"#")
    
    print(">>> done in {:.1f}s".format(time()-top3))    
    print(">>> ALL DONE IN {}s".format(time()-top))
