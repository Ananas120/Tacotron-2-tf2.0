# Tacotron-2-tf2.0

Tensorflow 2.x implementation of the Tacotron-2 model

This repo is highly inspired from the https://github.com/NVIDIA/tacotron2 implementation.

Note : in fact this repo implements the SV2TTS Tacotron-2 but you can easily use it as Tacotron-2 by removing the speaker_embedding in the SV2TTSTacotron2 call() and infer() method

You can also use it as pretrained model with the NVIDIA's pretrained model by creating the 2 models and passing them to my convertor script (other repo on my github). 

## Known limitations : 

- The model can't be trained with tensorflow 2.2 or higher (only 2.1). The training crashes with "InternalError : failed to launch ThenRNBackward with model config" at random step (most of time around step 200 for me)
- The model training is highly memory efficient and i can only run it on 50 frames on my GeForce GTX 1070 with 6.2Gb of RAM this is why i made a custom training step where i split the input in "sub-blocks" of N frames (here N = 50)
- The model can't run in graph mode

If anyone can help me to improve / optimize the code, you are welcome !
