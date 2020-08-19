import tensorflow as tf

from tensorflow.keras.layers import *

class LocationLayer(Layer):
    def __init__(self, attention_dim, attention_filters, 
                 attention_kernel_size, **kwargs):
        super(LocationLayer, self).__init__(**kwargs)
        
        self.attention_dim      = attention_dim
        self.attention_filters  = attention_filters
        self.attention_kernel_size  = attention_kernel_size
        
        self.location_conv = Conv1D(filters     = attention_filters,
                                    kernel_size = attention_kernel_size, 
                                    use_bias    = False,
                                    strides     = 1,
                                    padding     = "same",
                                    name = "location_conv"
                                   )
        self.location_dense = Dense(attention_dim, use_bias = False, 
                                    name = "location_dense")
        
    def call(self, inputs):
        processed = self.location_conv(inputs)
        processed = self.location_dense(processed)
        return processed
    
    def get_config(self):
        config = super(LocationLayer, self).get_config()
        config['attention_dim'] = self.attention_dim
        config['attention_filters'] = self.attention_filters
        config['attention_kernel_size'] = self.attention_kernel_size
        return config

class LocationAttention(Layer):
    def __init__(self, attention_dim, attention_filters, attention_kernel_size, **kwargs):
        super(LocationAttention, self).__init__(**kwargs)
        self.attention_dim      = attention_dim
        self.attention_filters  = attention_filters
        self.attention_kernel_size  = attention_kernel_size
        
        self.query_layer    = Dense(attention_dim, use_bias = False, name = "query_layer")
        
        self.memory_layer   = Dense(attention_dim, use_bias = False, name = "memory_layer")
        
        self.v              = Dense(1, use_bias = False, name = "value_layer")
        
        self.location_layer = LocationLayer(attention_dim, 
                                            attention_filters, 
                                            attention_kernel_size, 
                                            name = "location_layer"
                                           )
        
    def process_memory(self, memory):
        return self.memory_layer(memory)
        
    def get_alignment_energies(self, query, processed_memory, attention_weights_cat):
        """
            inputs :
                - query : decoder output (batch, n_mel_channels * n_frames_per_step)
                - processed_memory : processed encoder outputs(batch, T_in, attention_dim)
                - attention_weights_cat : cumulative and pref attention weights (batch, 2, max_time)
            
            return : 
                - alignment (batch, time_steps, max_time)
        """
        
        processed_query = self.query_layer(tf.expand_dims(query, axis = 1))
        processed_attention_weights = self.location_layer(attention_weights_cat)
        energies = self.v(tf.nn.tanh(
            processed_query + processed_attention_weights + processed_memory
        ))
        energies = tf.squeeze(energies, axis = -1)
        return energies
        
    def call(self, inputs):
        """
            inputs = [attention_hidden_state, memory, processed_memory, attention_weights_cat]
            
            - attention_hidden_state : attention rnn last output
            - memory : encoder outputs
            - processed_memory : processed encoder outputs
            attention_weights_cat : previous and cumulative attention weights
            - mask : binary mask for padded data
        """
        attention_hidden_state, memory, processed_memory, attention_weights_cat = inputs[:4]
        mask = inputs[4] if len(inputs) == 5 else None

        alignment = self.get_alignment_energies(attention_hidden_state, 
                                                processed_memory, attention_weights_cat)
        
        #print("Alignment shape : {}".format(alignment.shape))
        attention_weights = tf.nn.softmax(alignment, axis = 1)
        #print("attention_weights shape : {}".format(attention_weights.shape))
        attention_context = tf.matmul(tf.expand_dims(attention_weights, 1), memory)
        #print("attention_context shape : {}".format(attention_context.shape))
        #print("memory shape : {}".format(memory.shape))
        attention_context = tf.squeeze(attention_context, axis = 1)
        
        return attention_context, attention_weights
    
    def get_config(self):
        config = super(LocationAttention, self).get_config()
        config['attention_dim'] = self.attention_dim
        config['attention_filters'] = self.attention_filters
        config['attention_kernel_size'] = self.attention_kernel_size
        return config
        
