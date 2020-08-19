import tensorflow as tf

class TacotronLoss(tf.keras.losses.Loss):
    def __init__(self, mask_mel_padding = True, from_logits = False, 
                 name = 'tacotron_loss', reduction = 'none', **kwargs):
        super(TacotronLoss, self).__init__(name = name, reduction = 'none', **kwargs)
        self.mask_mel_padding   = mask_mel_padding
        self.from_logits        = from_logits
        
    @property
    def loss_names(self):
        return ['loss', 'mel_loss', 'mel_postnet_loss', 'gate_loss']
        
    def call(self, y_true, y_pred):
        mel_target, gate_target = y_true
        mel_pred, mel_postnet_pred, gate_pred = y_pred[:3]
        
        reshaped_gate_target = tf.reshape(gate_target, [-1, 1])
        reshaped_gate_pred   = tf.reshape(gate_pred, [-1, 1])
        
        mel_loss            = tf.square(mel_target - mel_pred) 
        mel_postnet_loss    = tf.square(mel_target - mel_postnet_pred)
        
        gate_loss = tf.keras.losses.binary_crossentropy(
            reshaped_gate_target, reshaped_gate_pred, from_logits = self.from_logits
        )

        if self.mask_mel_padding:
            mask = 1. - tf.expand_dims(gate_target, -1)
            mask = tf.cast(mask, mel_loss.dtype)
            mel_loss *= mask
            mel_postnet_loss *= mask
            
            nb_values = tf.reduce_sum(mask) * tf.cast(tf.shape(mel_loss)[-1], tf.float32)
            mel_loss            = tf.reduce_sum(mel_loss) / nb_values
            mel_postnet_loss    = tf.reduce_sum(mel_postnet_loss) / nb_values
        else:
            mel_loss            = tf.reduce_mean(mel_loss)
            mel_postnet_loss    = tf.reduce_mean(mel_postnet_loss)
            
        gate_loss   = tf.reduce_mean(gate_loss)
        total_loss  = mel_loss + mel_postnet_loss + gate_loss
        return total_loss, mel_loss, mel_postnet_loss, gate_loss
    
    def get_config(self):
        config = super(TacotronLoss, self).get_config()
        config['mask_mel_padding']  = self.mask_mel_padding
        config['from_logits']   = self.from_logits
        return config

