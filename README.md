# Conv-TasNet: Surpassing Ideal Time-Frequency Magnitude Masking for Speech Separation

# Dataset:
I use the wsj0 dataset as the tt,cv,tr dataset, the website link is below:
https://catalog.ldc.upenn.edu/LDC93s6a (free one in the kaggle)

# preprocessing the data:
1. convert the dataset from .wv1 form to .wav form
2. See the tools_for_dataset folder I uploaded, it contains the audio mixture method with both python and matlab
3. Generate scp file to read the file path by using script file of create_scp.py
For inferencing the model, 

# the pretrained model I upload in google drive:
https://drive.google.com/file/d/1YZAhYiCx-LncfaNHNoC9VUsapm7bU3JZ/view?usp=sharing
bb trainning with the wsj0 dataset

# test the separation effect by using separation_wav.py(sigle file),  Separation.py (large number of file)
