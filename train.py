# -*- coding: utf-8 -*-
import sugartensor as tf
from data import VCTK


__author__ = 'buriburisuri@gmail.com'


# set log level to debug
tf.sg_verbosity(10)


#
# hyper parameters
#

batch_size = 16    # batch size
num_blocks = 3     # dilated blocks
num_dim = 128      # latent dimension

#
# inputs
#

# VCTK corpus input tensor ( with QueueRunner )
data = VCTK(batch_size=batch_size)

# vocabulary size
voca_size = data.voca_size

# mfcc feature of audio
x = data.mfcc

# sequence length except zero-padding
seq_len = tf.not_equal(x.sg_sum(dims=2), 0.).sg_int().sg_sum(dims=1)

# target sentence label
y = data.label


#
# encode graph ( atrous convolution )
#

# residual block
def res_block(tensor, size, rate, dim=num_dim):

    # filter convolution
    conv_filter = tensor.sg_aconv1d(size=size, rate=rate, act='tanh', bn=True)

    # gate convolution
    conv_gate = tensor.sg_aconv1d(size=size, rate=rate,  act='sigmoid', bn=True)

    # output by gate multiplying
    out = conv_filter * conv_gate

    # final output
    out = out.sg_conv1d(size=1, dim=dim, act='tanh', bn=True)

    # residual and skip output
    return out + tensor, out

# expand dimension
z = x.sg_conv1d(size=1, dim=num_dim, act='tanh', bn=True)

# dilated conv block loop
skip = 0  # skip connections
for i in range(num_blocks):
    for r in [1, 2, 4, 8, 16]:
        z, s = res_block(z, size=7, rate=r)
        skip += s

# final logit layers
logit = (skip
         .sg_conv1d(size=1, act='tanh', bn=True)
         .sg_conv1d(size=1, dim=voca_size))

# CTC loss
loss = logit.sg_ctc(target=y, seq_len=seq_len)

# train
tf.sg_train(log_interval=30, lr=0.0001, loss=loss,
            ep_size=data.num_batch, max_ep=20, early_stop=False)
