import numpy as np
import tensorflow as tf

from tacotron2 import Tacotron2
from tacotron_loss import TacotronLoss

def train_step(model, loss_fn, optimizer, batch, max_train_frames = 25):
    def to_np(tensor):
        if isinstance(tensor, (list, tuple)): return [to_np(t) for t in tensor]
        return tensor.numpy() if hasattr(tensor, 'numpy') else tensor
    variables   = model.trainable_variables
        
    def optimize_step(sub_inputs, state, sub_target):
        with tf.GradientTape() as tape:
            pred, new_state = model(sub_inputs, state)

            loss, mel_loss, mel_post_loss, gate_loss = loss_fn(sub_target, pred)
                        
        gradients = tape.gradient(loss, variables)
            
        optimizer.apply_gradients(zip(gradients, variables))
            
        losses = {
            'loss'  : loss,
            'mel_loss'  : mel_loss,
            'mel_postnet_loss'  : mel_post_loss,
            'gate_loss' : gate_loss
        }
        
        return new_state, losses
        
    (text, mel), (mel_out, gate) = batch
        
    n_frames = tf.shape(mel)[1]
    memory_shape = [tf.shape(text)[0], tf.shape(text)[1], 512]
        
    total_loss = {}
    
    state = model.get_initial_state(memory_shape)
    for i in range(n_frames // max_train_frames + 1):
        start = i * max_train_frames
        end = tf.minimum((i+1) * max_train_frames, n_frames)
        
        if end - start <= 0: continue

        sub_inputs = [text, mel[:, start : end, :]]
        sub_target = [mel_out[:, start : end, :], gate[:, start : end]]
            
        state, losses = optimize_step(sub_inputs, state, sub_target)
        state = to_np(state)
        
        for name, value in losses.items():
            total_loss.setdefault(name, 0.)
            total_loss[name] += value * tf.cast((end - start), tf.float32)
                    
    total_loss = {k : v / tf.cast(n_frames, tf.float32) for k, v in total_loss.items()}

    return total_loss

model = Tacotron2(vocab_size = 148)
loss_fn = TacotronLoss()
optimizer = tf.keras.optimizers.Adam()

batch_size = 64
length = 400

txt_inp = tf.ones((batch_size, 150), dtype = tf.int32)
mel_inp = tf.ones((batch_size, length, 80), dtype = tf.float32)
gate_inp = tf.ones((batch_size, length), dtype = tf.float32)

_ = model((txt_inp, mel_inp))

batch = ((txt_inp, mel_inp), (mel_inp, gate_inp))

loss = train_step(model, loss_fn, optimizer, batch)

print("Loss = {}".format(loss))
