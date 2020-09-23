import tensorflow as tf
import tensorflow.keras.backend as K

from tensorflow.keras.layers import *

from location_attention import LocationAttention
from current_blocks import Conv1DBN

_rnn_impl = 2

def _get_var(_vars, i):
    if callable(_vars): return _vars(i)
    elif isinstance(_vars, list): return _vars[i]
    else: return _vars

class Prenet(tf.keras.Model):
    def __init__(self, sizes = [256, 256], use_bias = False, activation = 'relu', 
                 drop_rate = 0.5, deterministic = False, **kwargs):
        super(Prenet, self).__init__(**kwargs)
        
        self.sizes      = sizes
        self.use_bias   = use_bias
        self.activation = activation
        self.drop_rate  = drop_rate
        self.deterministic  = deterministic
        
        self.denses = [
            Dense(size, use_bias = use_bias, activation = activation, 
                  name = 'prenet_layer_{}'.format(i+1)) 
            for i, size in enumerate(sizes)
        ]
        self.dropout = Dropout(drop_rate)
        
    def call(self, inputs):
        x = inputs
        for layer in self.denses:
            x = layer(x)
            x = self.dropout(x, training = not self.deterministic)
        return x
    
    def get_config(self):
        config = {}
        config['sizes']     = self.sizes
        config['use_bias']  = self.use_bias
        config['activation']    = self.activation
        config['drop_rate']     = self.drop_rate
        config['deterministic'] = self.deterministic
        return config

def Postnet(n_mel_channels  = 80, 
            n_conv          = 5, 
            filters         = 512, 
            use_bias        = True,
            kernel_size     = 5,
            
            bnorm           = 'after',
            epsilon         = 1e-5,
            momentum        = 0.1,
            
            activation          = 'tanh',
            drop_rate           = 0.5,
            final_activation    = None,
            layer_name          = 'postnet_conv_{}',
            name    = 'Tacotron2_postnet',
            ** kwargs
           ):
    model = tf.keras.Sequential(name = name)

    for i in range(n_conv):
        config = {
            'filters'        : _get_var(filters, i),
            'kernel_size'    : _get_var(kernel_size, i),
            'use_bias'       : _get_var(use_bias, i),
            'strides'        : 1,
            'padding'        : 'same',
            'activation'     : _get_var(activation, i),
            'bnorm'          : _get_var(bnorm, i),
            'momentum'       : _get_var(momentum, i),
            'epsilon'        : _get_var(epsilon, i),
            'drop_rate'      : _get_var(drop_rate, i),
            'name'           : layer_name.format(i+1)
        }
        if i == n_conv - 1: # last layer
            config['filters']       = n_mel_channels
            config['activation']    = final_activation
            config['drop_rate']     = 0.
        
        Conv1DBN(model, ** config)
        
    return model
    

def TacotronEncoder(vocab_size, 
                    embedding_dims  = 512, 
                    n_conv          = 3, 
                    kernel_size     = 5,
                    use_bias        = True,
                    
                    bnorm           = 'after',
                    epsilon         = 1e-5,
                    momentum        = 0.1,
                    
                    activation      = 'relu',
                    drop_rate       = 0.5,
                    
                    layer_name  = 'encoder_conv_{}',
                    name    = "Tacotron2_encoder"
                   ):
    model = tf.keras.Sequential(name = name)
    
    model.add(Embedding(vocab_size, embedding_dims, 
                        name = "encoder_embeddings"))
    
    for i in range(n_conv):
        config = {
            'filters'        : embedding_dims,
            'kernel_size'    : _get_var(kernel_size, i),
            'use_bias'       : _get_var(use_bias, i),
            'strides'        : 1,
            'padding'        : 'same',
            'activation'     : _get_var(activation, i),
            'bnorm'          : _get_var(bnorm, i),
            'momentum'       : _get_var(momentum, i),
            'epsilon'        : _get_var(epsilon, i),
            'drop_rate'      : _get_var(drop_rate, i),
            'name'           : layer_name.format(i+1)
        }
        Conv1DBN(model, ** config)
    
    model.add(Bidirectional(LSTM(
        embedding_dims // 2, return_sequences = True, implementation = _rnn_impl
    ), name = "encoder_lstm"))
    
    model.build((None, None))
    
    return model

class TacotronDecoder(tf.keras.Model):
    def __init__(self, 
                 n_mel_channels     = 80, 
                 n_frames_per_step  = 1,
                 with_logits        = True,
                 # Prenet config variables
                 prenet_sizes       = [256, 256], 
                 prenet_use_bias    = False,
                 prenet_activation  = 'relu',
                 prenet_drop_rate   = 0.5,
                 # Attention variables
                 attention_rnn_dim  = 1024, 
                 p_attention_dropout    = 0.5,
                 
                 attention_dim      = 128, 
                 attention_filters  = 32, 
                 attention_kernel_size  = 31, 
                 
                 decoder_rnn_dim    = 1024,
                 p_decoder_dropout  = 0.5,
                 # Postnet variables
                 postnet_n_conv      = 5, 
                 postnet_filters     = 512, 
                 postnet_kernel_size = 5, 
                 postnet_drop_rate = 0.5, 
                 postnet_epsilon     = 1e-5,
                 postnet_momentum    = 0.1,
                 # Inference config
                 gate_threshold     = 0.5,
                 max_decoder_steps  = 1000,
                 early_stopping     = True,
                 ** kwargs
                ):
        super().__init__(**kwargs)
        self.kwargs = kwargs
        self.n_mel_channels     = n_mel_channels
        self.n_frames_per_step  = n_frames_per_step
        self.with_logits        = with_logits
        
        self.prenet_sizes       = prenet_sizes
        self.prenet_use_bias    = prenet_use_bias
        self.prenet_activation  = prenet_activation
        self.prenet_drop_rate   = prenet_drop_rate
        self.attention_rnn_dim  = attention_rnn_dim
        self.decoder_rnn_dim    = decoder_rnn_dim
        self.p_attention_dropout    = p_attention_dropout
        self.p_decoder_dropout  = p_decoder_dropout
        
        self.postnet_n_conv     = postnet_n_conv
        self.postnet_filters    = postnet_filters
        self.postnet_kernel_size    = postnet_kernel_size
        self.postnet_drop_rate  = postnet_drop_rate
        self.postnet_epsilon    = postnet_epsilon
        self.postnet_momentum   = postnet_momentum
        
        self.gate_threshold     = gate_threshold
        self.max_decoder_steps  = max_decoder_steps
        self.early_stopping     = early_stopping
                
        self.prenet = Prenet(
            sizes       = prenet_sizes, 
            use_bias    = prenet_use_bias,
            activation  = prenet_activation,
            drop_rate   = prenet_drop_rate,
            name        = 'prenet'
        )
        
        self.attention_rnn = LSTMCell(
            attention_rnn_dim,
            dropout             = p_attention_dropout,
            recurrent_dropout   = p_attention_dropout,
            implementation      = _rnn_impl,
            name = 'attention_rnn'
        )
        
        self.attention_layer = LocationAttention(
            attention_dim, 
            attention_filters, 
            attention_kernel_size,
            name = 'location_attention'
        )
        
        self.decoder_rnn = LSTMCell(
            decoder_rnn_dim,
            dropout             = p_decoder_dropout,
            recurrent_dropout   = p_decoder_dropout,
            implementation      = _rnn_impl,
            name = 'decoder_rnn'
        )
        
        self.linear_projection = Dense(
            units   = n_mel_channels * n_frames_per_step, 
            name    = 'linear_projection'
        )
        
        self.gate_layer = Dense(
            units       = 1 * n_frames_per_step, 
            activation  = 'sigmoid' if with_logits else None,
            name        = 'gate_output'
        )
        
        self.postnet = Postnet(
            n_conv      = postnet_n_conv, 
            filters     = postnet_filters, 
            kernel_size = postnet_kernel_size, 
            drop_rate   = postnet_drop_rate, 
            epsilon     = postnet_epsilon,
            momentum    = postnet_momentum,
            name        = 'postnet'
        )
        
        
    def get_go_frame(self, memory):
        B = tf.shape(memory)[0]
        return tf.zeros((B, self.n_mel_channels * self.n_frames_per_step))
        
    def get_initial_state(self, memory_shape = None, memory = None):
        if memory_shape is None:
            memory_shape = (tf.shape(memory)[0], tf.shape(memory)[1], tf.shape(memory)[-1])
        B, enc_seq_len, enc_hidden_size = memory_shape
        
        self.attention_rnn.reset_dropout_mask()
        self.attention_rnn.reset_recurrent_dropout_mask()
        
        self.decoder_rnn.reset_dropout_mask()
        self.decoder_rnn.reset_recurrent_dropout_mask()
        
        attn_rnn_states = [tf.zeros((B, size), dtype = tf.float32) 
                           for size in self.attention_rnn.state_size]

        decoder_rnn_states = [tf.zeros((B, size), dtype = tf.float32) 
                              for size in self.decoder_rnn.state_size]

        attn_weights = tf.zeros((B, enc_seq_len))
        attn_weights_cum = tf.zeros((B, enc_seq_len))

        attn_context = tf.zeros((B, enc_hidden_size))
        
        return [attn_rnn_states, decoder_rnn_states, attn_weights, attn_weights_cum, attn_context]
    
    def decode_step(self, decoder_input, memory, processed_memory, states, 
                    debug = False, training = False):
        attn_rnn_states, decoder_rnn_states, attn_weights, attn_weights_cum, attn_context = states
            
        cell_input = tf.concat([decoder_input, attn_context], axis = -1)
            
        if debug:
            print("Decoder input : {}".format(decoder_input))
            print("Decoder_input shape : {}".format(tuple(decoder_input.shape)))
            print("Cell input : {}".format(cell_input))
            print("Cell_input shape : {}".format(tuple(cell_input.shape)))
            
        cell_out, new_attn_rnn_states = self.attention_rnn(
            cell_input, attn_rnn_states, training = training
        )
            
        attn_weights_cat = tf.concat([
            tf.expand_dims(attn_weights, -1),
            tf.expand_dims(attn_weights_cum, -1)
        ], axis = -1)
        new_attn_context, new_attn_weights = self.attention_layer([
            cell_out, memory, processed_memory, attn_weights_cat
        ], training = training)
            
        new_attn_weights_cum = attn_weights_cum + new_attn_weights
            
        if debug:
            print("cell_out : {}".format(tuple(cell_out)))
            print("new_attn_context : {}".format(tuple(new_attn_context.shape)))
        
        decoder_rnn_input = tf.concat([
            cell_out, new_attn_context
        ], axis = -1)
        decoder_rnn_out, new_decoder_rnn_state = self.decoder_rnn(
            decoder_rnn_input, decoder_rnn_states, training = training
        )
        
        decoder_rnn_out_cat = tf.concat([
            decoder_rnn_out, new_attn_context
        ], axis = -1)
        
        new_states = [
            new_attn_rnn_states,
            new_decoder_rnn_state,
            new_attn_weights,
            new_attn_weights_cum,
            new_attn_context
        ]
                        
        return [decoder_rnn_out_cat, attn_weights], new_states
    
    def project(self, decoder_output):        
        mel_outputs = self.linear_projection(decoder_output)
        
        gate_outputs = self.gate_layer(decoder_output)
        
        return mel_outputs, gate_outputs
    
    @tf.function(experimental_relax_shapes = True)
    def call(self, inputs, initial_states = None, 
             debug = False, training = False):
        memory, mel_inputs = inputs
        
        processed_memory = self.attention_layer.process_memory(memory)
        
        decoder_inputs = self.prenet(mel_inputs, training = training)
        
        if debug:
            print("decoder_inputs shape : {}".format(decoder_inputs))
            print("memory shape : {}".format(memory))
        
        
        if initial_states is None:
            initial_states = self.get_initial_state(memory = memory)
        
        def step(decoder_input, states):
            return self.decode_step(
                decoder_input,
                memory,
                processed_memory,
                states,
                debug       = debug,
                training    = training
            )
        
        _, decoder_step_outputs, last_state = K.rnn(
            step,
            decoder_inputs,
            initial_states
        )
        decoder_output, attn_weights = decoder_step_outputs
        
        mel_outputs, gate_outputs = self.project(decoder_output)
        
        mel_outputs_postnet = self.postnet(mel_outputs, training = training)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet
        
        outputs = [mel_outputs, mel_outputs_postnet, gate_outputs, attn_weights]
        
        return outputs, last_state
    
    def infer(self, memory, 
              initial_states    = None, 
              go_frame          = None,
              
              max_length        = None,
              early_stopping    = None, 
              gate_threshold    = None, 
              
              training  = False,
              debug     = False
             ):
        if max_length is None: max_length = self.max_decoder_steps
        if early_stopping is None: early_stopping = self.early_stopping
        if gate_threshold is None: gate_threshold = self.gate_threshold
        
        processed_memory = self.attention_layer.process_memory(memory)
        
        last_mel_output = self.get_go_frame(memory) if go_frame is None else go_frame
        
        states = initial_states
        if states is None:
            states = self.get_initial_state(memory = memory)
        
        not_finished = tf.ones((tf.shape(memory)[0],), dtype = tf.int32)
        
        mel_outputs, gate_outputs, attn_weights = [], [], []
        
        run = True
        while run:
            decoder_input = self.prenet(last_mel_output)
            
            step_outputs, next_states = self.decode_step(
                decoder_input,
                memory, 
                processed_memory, 
                states, 
                training   = training,
                debug      = debug
            )
            decoder_output, attn_weight = step_outputs
            mel_output, gate_output = self.project(decoder_output)
            
            mel_outputs.append(tf.expand_dims(mel_output, axis = 1))
            gate_outputs.append(tf.expand_dims(gate_output, axis = 1))
            attn_weights.append(tf.expand_dims(attn_weight, axis = 1))
            
            if early_stopping:
                if not self.with_logits:
                    gate_output = tf.nn.sigmoid(gate_output)

                not_stop = tf.squeeze(tf.cast(tf.math.less_equal(
                    gate_output, gate_threshold
                ), dtype = tf.int32), axis = 1)

                not_finished = not_finished * not_stop

                if tf.reduce_sum(not_finished) == 0:
                    run = False
            
            if len(mel_outputs) >= max_length:
                run = False
                if early_stopping:
                    print("Max_decoder_steps atteint !")
            
            states = next_states
            last_mel_output = mel_output
            
        mel_outputs = tf.concat(mel_outputs, axis = 1)
        gate_outputs = tf.concat(gate_outputs, axis = 1)
        attn_weights = tf.concat(attn_weights, axis = 1)
        
        mel_outputs_postnet = self.postnet(mel_outputs, training = training)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet
        
        outputs = [mel_outputs, mel_outputs_postnet, gate_outputs, attn_weights]
        
        return outputs, states

    def get_config(self):
        config = {** self.kwargs}
        config['n_mel_channels']    = self.n_mel_channels
        config['n_frames_per_step'] = self.n_frames_per_step
        config['with_logits']       = self.with_logits
        
        config['prenet_sizes']      = self.prenet_sizes
        config['prenet_use_bias']   = self.prenet_use_bias
        config['prenet_activation'] = self.prenet_activation
        config['prenet_drop_rate']  = self.prenet_drop_rate
        config['attention_rnn_dim'] = self.attention_rnn_dim
        config['decoder_rnn_dim']   = self.decoder_rnn_dim
        config['p_attention_dropout']   = self.p_attention_dropout
        config['p_decoder_dropout'] = self.p_decoder_dropout
        
        config['attention_dim']     = self.attention_dim
        config['attention_filters'] = self.attention_filters
        config['attention_kernel_size'] = self.attention_kernel_size
        
        config['postnet_n_conv']    = self.postnet_n_conv
        config['postnet_filters']   = self.postnet_filters
        config['postnet_kernel_size']   = self.postnet_kernel_size
        config['postnet_drop_rate'] = self.postnet_drop_rate
        config['postnet_epsilon']   = self.postnet_epsilon
        config['postnet_momentum']  = self.postnet_momentum
        
        config['gate_threshold']    = self.gate_threshold
        config['max_decoder_steps'] = self.max_decoder_steps
        config['early_stopping']    = self.early_stopping
        
        return config
        
class Tacotron2(tf.keras.Model):
    def __init__(self, 
                 vocab_size,
                 n_mel_channels     = 80, 
                 n_frames_per_step  = 1,
                 with_logits        = True,
                 
                 encoder_embedding_dims = 512, 
                 encoder_n_convolutions = 3, 
                 encoder_kernel_size    = 5,
                 encoder_epsilon        = 1e-5,
                 encoder_momentum       = 0.1,
                 encoder_drop_rate      = 0.5,
                 encoder_activation     = 'relu',
                
                 prenet_sizes       = [256, 256], 
                 
                 attention_rnn_dim  = 1024, 
                 attention_dim      = 128, 
                 attention_filters  = 32, 
                 attention_kernel_size  = 31, 
                 decoder_rnn_dim    = 1024,
                 
                 postnet_n_conv      = 5, 
                 postnet_filters     = 512, 
                 postnet_kernel_size = 5, 
                 postnet_drop_rate = 0.5, 
                 postnet_epsilon     = 1e-5,
                 postnet_momentum    = 0.1,
                 
                 gate_threshold     = 0.5,
                 max_decoder_steps  = 1000,
                 early_stopping     = True,
                 return_state       = True,
                 **kwargs
                ):
        super().__init__(**kwargs)
        
        self.vocab_size         = vocab_size
        self.encoder_embedding_dims = encoder_embedding_dims
        self.encoder_n_convolutions = encoder_n_convolutions
        self.encoder_kernel_size    = encoder_kernel_size
        self.encoder_epsilon        = encoder_epsilon
        self.encoder_momentum       = encoder_momentum
        self.encoder_drop_rate      = encoder_drop_rate
        self.encoder_activation     = encoder_activation
        
        self.with_logits        = with_logits
        self.n_mel_channels     = n_mel_channels
        self.n_frames_per_step  = n_frames_per_step
        self.prenet_sizes       = prenet_sizes
        
        self.attention_rnn_dim  = attention_rnn_dim
        self.attention_dim      = attention_dim
        self.attention_filters  = attention_filters
        self.attention_kernel_size  = attention_kernel_size
        self.decoder_rnn_dim    = decoder_rnn_dim
                 
        self.postnet_n_conv     = postnet_n_conv
        self.postnet_filters    = postnet_filters
        self.postnet_kernel_size    = postnet_kernel_size
        self.postnet_drop_rate  = postnet_drop_rate
        self.postnet_epsilon    = postnet_epsilon
        self.postnet_momentum   = postnet_momentum
                 
        self.gate_threshold     = gate_threshold
        self.max_decoder_steps  = max_decoder_steps
        self.early_stopping     = early_stopping
        self.return_state       = return_state
        
        self.encoder    = None
        self.decoder    = None
        self._init_encoder()
        self._init_decoder()
        
    def _init_encoder(self):
        self.encoder = TacotronEncoder(
            self.vocab_size, 
            embedding_dims  = self.encoder_embedding_dims, 
            n_conv          = self.encoder_n_convolutions,
            kernel_size     = self.encoder_kernel_size,
            epsilon         = self.encoder_epsilon,
            momentum        = self.encoder_momentum,
            drop_rate       = self.encoder_drop_rate,
            activation      = self.encoder_activation,
            name = "encoder"
        )
    
    def _init_decoder(self):
        self.decoder = TacotronDecoder(
            n_mel_channels       = self.n_mel_channels,
            n_frames_per_step    = self.n_frames_per_step,
            with_logits          = self.with_logits,
                 
            prenet_sizes         = self.prenet_sizes,
                 
            attention_rnn_dim    = self.attention_rnn_dim,
            attention_dim        = self.attention_dim,
            attention_filters    = self.attention_filters, 
            attention_kernel_size    = self.attention_kernel_size, 
            decoder_rnn_dim      = self.decoder_rnn_dim,
                 
            postnet_n_conv       = self.postnet_n_conv,
            postnet_filters      = self.postnet_filters,
            postnet_kernel_size  = self.postnet_kernel_size,
            postnet_drop_rate    = self.postnet_drop_rate,
            postnet_epsilon      = self.postnet_epsilon,
            postnet_momentum     = self.postnet_momentum,
                 
            gate_threshold       = self.gate_threshold,
            max_decoder_steps    = self.max_decoder_steps,
            early_stopping       = self.early_stopping,
            name = "decoder"
        )
        
    @property
    def encoder_embedding_dim(self):
        return self.encoder.output_shape[-1]
    
    def get_initial_state(self, *args, **kwargs):
        return self.decoder.get_initial_state(*args, **kwargs)
    
    #@tf.function(experimental_relax_shapes = True)
    def call(self, inputs, * args, training = False, ** kwargs):
        """
            Forward pass with theacher forcing (decoder_mel is provided)
            Arguments : 
                - inputs : [encoder_inputs, decoder_inputs]
                    - encoder_inputs    : encoder text input
                        shape : (batch_size, text_length) (currently (None, None))
                    - decoder_inputs : mel spectrogram input (first frame is go_frame)
                        shape : (batch_size, mel_length, n_mel_channels)
        """
        encoder_inputs, decoder_inputs = inputs
        
        encoder_embedding = self.encoder(encoder_inputs, training = training)
        
        outputs, state = self.decoder(
            [encoder_embedding, decoder_inputs], * args, training = training, ** kwargs
        )
        if self.return_state:
            return outputs, state
        return outputs
        
    def infer(self, inputs, * args, training = False, ** kwargs):
        encoder_embedding = self.encoder(inputs, training = training)
        
        outputs, state = self.decoder.infer(
            encoder_embedding, *args, training = training, **kwargs
        )
        
        if self.return_state:
            return outputs, state
        return outputs
        
    def get_config(self):
        config = {}
        config['vocab_size']                = self.vocab_size
        config['encoder_embedding_dims']    = self.encoder_embedding_dims
        config['encoder_n_convolutions']    = self.encoder_n_convolutions
        config['encoder_kernel_size']       = self.encoder_kernel_size
        config['encoder_epsilon']           = self.encoder_epsilon
        config['encoder_momentum']          = self.encoder_momentum
        config['encoder_drop_rate']         = self.encoder_drop_rate
        config['encoder_activation']        = self.encoder_activation
        
        config['n_mel_channels']    = self.n_mel_channels
        config['n_frames_per_step'] = self.n_frames_per_step
        config['with_logits']       = self.with_logits
        config['prenet_sizes']      = self.prenet_sizes
        
        config['attention_rnn_dim'] = self.attention_rnn_dim
        config['attention_dim']     = self.attention_dim
        config['attention_filters'] = self.attention_filters
        config['attention_kernel_size'] = self.attention_kernel_size
        config['decoder_rnn_dim']   = self.decoder_rnn_dim
                 
        config['postnet_n_conv']    = self.postnet_n_conv
        config['postnet_filters']   = self.postnet_filters
        config['postnet_kernel_size']   = self.postnet_kernel_size
        config['postnet_drop_rate'] = self.postnet_drop_rate
        config['postnet_epsilon']   = self.postnet_epsilon
        config['postnet_momentum']  = self.postnet_momentum
                 
        config['gate_threshold']    = self.gate_threshold
        config['max_decoder_steps'] = self.max_decoder_steps
        config['early_stopping']    = self.early_stopping
        config['return_state']      = self.return_state
        return config
    
    @classmethod
    def from_config(cls, config, custom_objects = None):
        return cls(**config)
        
