# Speech-to-Text-WaveNet
A tensorflow implementation of speech-to-text recognizer using DeepMind's [WaveNet: A Generative Model for Raw Audio](https://arxiv.org/abs/1609.03499).

Although [ibab](https://github.com/ibab/tensorflow-wavenet) and [tomlepaine](https://github.com/tomlepaine/fast-wavenet) have already implemented WaveNet with tensorflow, they did not implement voice recognition, so I implemented it myself.  Recently, it is difficult to reproduce the paper of DeepMind. This paper also omits specific details about the implementation, and I implemented the empty area in my own way.

First, in the paper, they used TIMIT dataset for speech recognition test, I replace TIMIT dataset with free VTCK dataset.

Second, they added a mean-pooling layer to the dilated convolution layer for down-sampling. I transformed the wave file to [MFCC](https://en.wikipedia.org/wiki/Mel-frequency_cepstrum) feature and removed the final mean-pooling layer. Without it, the model cannot be computable on my TitanX GPU.

Third, since TIMIT has phoneme labels, they train the model with losses of phoneme classification and next phoneme prediction. I used one CTC loss because VCTK provides sentence-level labels. Therefore, there was no reason to use causal conv1d, so only dilated conv1d was used.

Finally, quantitative analysis such as BLEU score and language model are omitted by my time constraint.

The final architecture is shown in the following figure.
<p align="center">
  <img src="https://raw.githubusercontent.com/buriburisuri/speech-to-text-wavenet/master/png/architecture.png" width="1024"/>
</p>
(Some images are cropped from [WaveNet: A Generative Model for Raw Audio](https://arxiv.org/abs/1609.03499) and [Neural Machine Translation in Linear Time](https://arxiv.org/abs/1610.10099))  


## Dependencies

1. [tensorflow](https://www.tensorflow.org/versions/r0.11/get_started/os_setup.html#pip-installation) >= 0.11
1. [sugartensor](https://github.com/buriburisuri/sugartensor) >= 0.0.1.9
1. [pandas](http://pandas.pydata.org/pandas-docs/stable/install.html) >= 0.19.1
1. [librosa](https://github.com/librosa/librosa) >= 0.4.3

## Dataset

I used the VCTK corpus mentioned in the original paper. In particular, we used only 36,395 sentences in the VCTK corpus with a length of more than 5 seconds to prevent CTC loss errors. VCTK corpus can be downloaded from [http://homepages.inf.ed.ac.uk/jyamagis/release/VCTK-Corpus.tar.gz](http://homepages.inf.ed.ac.uk/jyamagis/release/VCTK-Corpus.tar.gz). After downloading, extract the 'VCTK-Corpus.tar.gz' file to the 'asset/data/' directory.

## Training the network

Execute
<pre><code>
python train.py
</code></pre>
to train the network. You can see the result ckpt files and log files in the 'asset/train' directory.
Launch tensorboard --logdir asset/train/log to monitor training process.

I've trained this model on a single Titan X GPU during 45 hours until 30 epochs and the model ended with 10.8 ctc loss. If you don't have a Titan X GPU, reduce batch_size in the train.py file from 16 to 4.  
<p align="center">
  <img src="https://raw.githubusercontent.com/buriburisuri/speech-to-text-wavenet/master/png/loss.png" width="1024"/>
</p>

## Speech wave file to text
 
Execute
<pre><code>
python recognize.py --file <wave_file path>
</code></pre>
to transform speech wave file to English sentence. The result will be printed on the console. 

For example, try the following command.
<pre><code>
python recognize.py --file asset/data/wav48/p225/p225_003.wav
</code></pre>

The result will be as follows:
<pre><code>
six spons of fresh snow peas five thick slbs of blue these and maybe a stack for her brother bob
</code></pre>

The ground truth is as follows:
<pre><code>
Six spoons of fresh snow peas, five thick slabs of blue cheese, and maybe a snack for her brother Bob.
</code></pre>

As mentioned earlier, there is no language model, so there are some cases where capital letters, punctuations, and words are misspelled.

## pre-trained models

You can transform speech waves file to English sentences with the pre-trained model on the VCTK corpus. 
Extract [the following zip file](https://drive.google.com/file/d/0B3ILZKxzcrUydklJTXgyRzRwUzQ/view?usp=sharing) in 'asset/train/ckpt'.

## Other resources

1. [ibab's WaveNet(speech synthesis) tensorflow implementation](https://github.com/ibab/tensorflow-wavenet)
1. [tomlepaine's Fast WaveNet(speech synthesis) tensorflow implementation](https://github.com/ibab/tensorflow-wavenet)

## My other repositories

1. [SugarTensor](https://github.com/buriburisuri/sugartensor)
1. [EBGAN tensorflow implementation](https://github.com/buriburisuri/ebgan)
1. [Timeseries gan tensorflow implementation](https://github.com/buriburisuri/timeseries_gan)
1. [Supervised InfoGAN tensorflow implementation](https://github.com/buriburisuri/supervised_infogan)
1. [AC-GAN tensorflow implementation](https://github.com/buriburisuri/ac-gan)
1. [SRGAN tensorflow implementation](https://github.com/buriburisuri/SRGAN)
1. [ByteNet-Fast Neural Machine Translation](https://github.com/buriburisuri/ByteNet)

# Authors

Namju Kim (buriburisuri@gmail.com) at Jamonglabs Co., Ltd.
Kyubyong Park (kbpark@jamonglab.com) at Jamonglabs Co., Ltd.
