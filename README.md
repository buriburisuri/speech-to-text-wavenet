# Speech-to-Text-WaveNet : End-to-end sentence level English speech recognition using DeepMind's WaveNet
A tensorflow implementation of speech recognition based on DeepMind's [WaveNet: A Generative Model for Raw Audio](https://arxiv.org/abs/1609.03499). (Hereafter the Paper)

Although [ibab](https://github.com/ibab/tensorflow-wavenet) and [tomlepaine](https://github.com/tomlepaine/fast-wavenet) have already implemented WaveNet with tensorflow, they did not implement speech recognition. That's why we decided to implement it ourselves. 

Some of Deepmind's recent papers are tricky to reproduce. The Paper also omitted specific details about the implementation, and we had to fill the gaps in our own way.

Here are a few important notes.

First, while the Paper used the TIMIT dataset for the speech recognition experiment, we used the free VTCK dataset.

Second, the Paper added a mean-pooling layer after the dilated convolution layer for down-sampling. We extracted [MFCC](https://en.wikipedia.org/wiki/Mel-frequency_cepstrum) from wav files and removed the final mean-pooling layer because the original setting was impossible to run on our TitanX GPU.

Third, since the TIMIT dataset has phoneme labels, the Paper trained the model with two loss terms, phoneme classification and next phoneme prediction. We, instead, used a single CTC loss because VCTK provides sentence-level labels. As a result, we used only dilated conv1d layers without any causal conv1d layers.

Finally, we didn't do quantitative analyses such as WER/CER/PER and post-processing by combining a language model due to the time constraints.

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
1. [tqdm] (https://github.com/tqdm/tqdm) >= 4.10.0

To install the required python packages ( except TensorFlow ), run

```
pip install -r requirements.txt
```

## Dataset

We used only 36,395 sentences in the VCTK corpus with a length of more than 5 seconds to prevent CTC loss errors. VCTK corpus can be downloaded from [http://homepages.inf.ed.ac.uk/jyamagis/release/VCTK-Corpus.tar.gz](http://homepages.inf.ed.ac.uk/jyamagis/release/VCTK-Corpus.tar.gz). After downloading, extract the 'VCTK-Corpus.tar.gz' file to the 'asset/data/' directory.

## Training the network

Execute
<pre><code>
python train.py
</code></pre>
to train the network. You can see the result ckpt files and log files in the 'asset/train' directory.
Launch tensorboard --logdir asset/train/log to monitor training process.

We've trained this model on a single Titan X GPU during 30 hours until 20 epochs and the model stopped at 13.4 ctc loss. If you don't have a Titan X GPU, reduce batch_size in the train.py file from 16 to 4.  
<p align="center">
  <img src="https://raw.githubusercontent.com/buriburisuri/speech-to-text-wavenet/master/png/loss.png" width="1024"/>
</p>

## Transforming speech wave file to English text 
 
Execute
<pre><code>
python recognize.py --file wave_file_path
</code></pre>
to transform a speech wave file to the English sentence. The result will be printed on the console. 

For example, try the following command.
<pre><code>
python recognize.py --file asset/data/wav48/p225/p225_003.wav
</code></pre>

The result will be as follows:
<pre><code>
six spoons of fresh now peas five thick slbs of blue cheese and maybe a snack for her brother bob
</code></pre>

The ground truth is as follows:
<pre><code>
Six spoons of fresh snow peas, five thick slabs of blue cheese, and maybe a snack for her brother Bob.
</code></pre>

As mentioned earlier, there is no language model, so there are some cases where capital letters, punctuations, and words are misspelled.

## Pre-trained models

You can transform a speech wave file to English text with the pre-trained model on the VCTK corpus. 
Extract [the following zip file](https://drive.google.com/open?id=0B3ILZKxzcrUyVWwtT25FemZEZ1k) to the 'asset/train/ckpt/' directory.

## Other resources

1. [ibab's WaveNet(speech synthesis) tensorflow implementation](https://github.com/ibab/tensorflow-wavenet)
1. [tomlepaine's Fast WaveNet(speech synthesis) tensorflow implementation](https://github.com/tomlepaine/fast-wavenet)

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

## Citation

If you find this code useful please cite us in your work:

```
@inproceedings{Namju2016SpeechToTextWaveNet,
  title={Speech-to-Text-WaveNet : End-to-end sentence level English speech recognition using DeepMind's WaveNet},
  author={Namju Kim and Kyubyong Park},
  year={2016}
}
```
