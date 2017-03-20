import sugartensor as tf


num_blocks = 3     # dilated blocks
num_dim = 128      # latent dimension

#
# encode graph ( atrous convolution )
#

def encode(x, voca_size):

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

    return logit
