import numpy as np
import tensorflow as tf

def train_step(model, loss_fn, optimizer, batch, max_train_frames = 50):
    variables   = model.trainable_variables
        
        def optimize_step(sub_inputs, state, target):
            with tf.GradientTape() as tape:
                pred, new_state = self.tts_model(sub_inputs, state)

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
        
        (text, embedded_speaker, mel), (mel_out, gate) = batch
        
        n_frames = tf.shape(mel)[1]
        memory_shape = [tf.shape(text)[0], tf.shape(text)[1], self.memory_length]
        
        total_loss = {}
        
        state = model.get_initial_state(memory_shape)
        for i in range(n_frames // max_train_frames + 1):
            start = i * max_train_frames
            end = tf.minimum((i+1) * max_train_frames, n_frames)
            
            if end - start <= 5: continue

            sub_inputs = [text, embedded_speaker, mel[:, start : end, :]]
            sub_target = [mel_out[:, start : end, :], gate[:, start : end]]
            
            state, losses = optimize_step(sub_inputs, state, sub_target)
            
            for name, value in losses.items():
                total_loss.setdefault(name, 0.)
                total_loss[name] += value * tf.cast((end - start), tf.float32)
                    
        total_loss = {k : v / tf.cast(n_frames, tf.float32) for k, v in total_loss.items()}

        return total_loss

model = SV2TTSTacotron2()
loss_fn = TacotronLoss()
optimizer = tf.keras.optimizers.Adam()

batch_size = 16
length = 100

txt_inp = tf.ones((batch_size, text_length), dtype = tf.int32)
spk_inp = tf.ones((batch_size, embedding_dim), dtype = tf.float32)
mel_inp = tf.ones((batch_size, length, 80), dtype = tf.float32)
gate_inp = tf.ones((batch_size, length), dtype = tf.float32)

batch = ((txt_inp, spk_inp, mel_inp), (mel_inp, gate_inp))

loss = train_step(model, loss_fn, optimizer, batch)

print("Loss = {}".format(loss))
