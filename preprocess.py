import numpy as np
import pandas as pd
import glob
import os
import string
import itertools
import pickle


__author__ = 'namju.kim@kakaobrain.com'


# data path
_data_path = "asset/data/"

# read meta-info
df = pd.read_table(_data_path + 'speaker-info.txt', usecols=['ID', 'AGE', 'GENDER', 'ACCENTS'],
                   index_col=False, delim_whitespace=True)

# make file ID
file_ids = []
for d in [_data_path + 'txt/p%d/' % uid for uid in df.ID.values]:
    file_ids.extend([f[-12:-4] for f in sorted(glob.glob(d + '*.txt'))])

# make wave file list
wav_files = [_data_path + 'wav48/%s/' % f[:4] + f + '.wav' for f in file_ids]

# exclude extremely short wave files
file_id, wav_file = [], []
for i, w in zip(file_ids, wav_files):
    if os.stat(w).st_size > 240000:  # at least 5 seconds
        file_id.append(i)
        wav_file.append(w)

# read label sentence
sents = []
for f in file_id:
    # remove punctuation, to lower, clean white space
    s = ' '.join(open(_data_path + 'txt/%s/' % f[:4] + f + '.txt').read()
                 .translate(None, string.punctuation).lower().split())
    # append byte code
    sents.append([ord(ch) for ch in s])

# make vocabulary index
index2byte = [0] + list(np.unique(list(itertools.chain(*sents))))  # add <EMP> token
byte2index = {}
for i, b in enumerate(index2byte):
    byte2index[b] = i

# save vocabulary
vocabulary_file = _data_path + 'index2byte.pickle'
with open(vocabulary_file, 'wb') as f:
    pickle.dump([index2byte, byte2index], f, protocol=pickle.HIGHEST_PROTOCOL)

