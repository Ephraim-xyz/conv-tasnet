# Conv-TasNet: Surpassing Ideal Time-Frequency Magnitude Masking for Speech Separation

# Dataset:
1. I use the wsj0 dataset as the training and testing dataset, the website link is
https://catalog.ldc.upenn.edu/LDC93s6a (free one in the kaggle)
2. TIMI and libri2mix dataset can replace the dataset wsj0.

# preprocessing the data:
1. convert the dataset from .wv1 form to .wav form, in order to use the torchaudio to read， see the file sph2pipe_v2.5.tar.gz
2. See the tools_for_dataset folder I uploaded, it contains the audio mixture method with both python and matlab
3. Generate scp file to read the file path by using script file of create_scp.py

# trainning data:
In train.yml file, feel free to adjust the network parameters or using Multi-GPU to train, and it also include the path of the training file.

# Inferencing the model:
1. Separation.py - use this python file to test a large number of audio files
2. Separation_wav.py - use this python file to test a single file
3. separate_new.pu - this can support mutiple channel(still in testing)

# Pretrained model 
the pretrained model I upload in google drive, see link
https://drive.google.com/file/d/1YZAhYiCx-LncfaNHNoC9VUsapm7bU3JZ/view?usp=sharing
by trainning with half of the wsj0 dataset, batch size is 8

# Project Summary !!!
1. show-case power point link:
https://docs.google.com/presentation/d/1nrX5GsaZ-AoUcMKAukDBb2ufFqr6Va_z/edit?usp=sharing&ouid=116493363062780510716&rtpof=true&sd=true

2. A demo video for how to use the deployed model in the google cloud https://drive.google.com/file/d/1bKvL9LqjfBk9G6_cRVsFN4b26PSExHYm/view?usp=sharing

# fulture planning
1. First of all, depending on different scenars， conv-tasnet also has the ability to separate audio in noisy environment and do speech enhancement. However, we can see  wsj0 is a clean dataset(without any noise), if we straightly use it to separate some blindsource with lots of noise, the result will be bad, because the generalization ability of noise environment for this model is not good. In this case, I also found a dataset called "wham", which is a pure noise dataset. If necessary, I can mix it with the wsj0 dataset and retrain the model.
2. Currently, the model is single-channel separation, I am still trying to make multi-channel on this model.
3. Though conv-tasnet is a great one among several separation models, there still had some new models released recent years, the performance will be better. So if conv-tasnet in realworld scenarios cannot be satisfied, I will do more research on other new one.

# reference
the paper for conv-tasnet: https://arxiv.org/abs/1809.07454.

the model reference: https://github.com/JusperLee/Conv-TasNet.


