from itertools import groupby
from sklearn.model_selection import train_test_split
import numpy as np
import os.path
import pickle
import os
import random

def fasta_iter(fasta_name):

    fh = open(fasta_name)

    faiter = (x[1] for x in groupby(fh, lambda line: line[0] == ">"))

    for header in faiter:
        # drop the ">"
        headerStr = header.__next__()[1:].strip()

        # join all sequence lines to one.
        seq = "".join(s.strip() for s in faiter.__next__())


        for i, s in enumerate(seq):
            if(s == "N" or s == "n"):
                seq = seq[:i] + random.choices(population=['G', 'C', 'T', 'A'])[0] + seq[i+1:]

        yield (headerStr, seq)


def labels_iter(labels_name):

    lh = open(labels_name)
    for line in lh:
        line = line.strip().split("\t")
        header = line[0]
        label = list(map(int, line[1:]))
        yield header, label


#Loding the sequences.

if not os.path.exists('data/Sequences.npy'):

    fiter = fasta_iter('data/encode_roadmap.fa')
    seqs = []
    count = 0
    for ff in fiter:
        headerStr, seq = ff
        seqs.append(seq)
        count += 1

    seqs = np.array(seqs).reshape((-1, 1))

    np.save("data/Sequences", seqs)

    print(str(count) + " examples of shape " + str(seqs.shape) + " loaded into Sequences.npy file")

else:

    print("Sequences.npy file already exists.")
    seqs = np.load('data/Sequences.npy')

idxs = np.random.permutation(seqs.shape[0])
train_idxs, test_idxs = idxs[:int(len(idxs)*0.8)], idxs[int(len(idxs)*0.8):]
train_features, test_features = seqs[train_idxs, 0], seqs[test_idxs, 0]
seqs = None

os.remove("data/Sequences.npy")
print("Removing sequences.npy file.")

np.save("data/train_features", train_features)
np.save("data/test_features", test_features)
train_features = None
test_features = None
print("Saving Train, Test features...")




#Loading the labels.
if not os.path.exists('data/Labels.npy'):

    fiter = labels_iter('data/encode_roadmap_act.txt')
    labels = []
    count = 0
    for ff in fiter:
        headerStr, seq = ff
        labels.append(seq)
        count += 1

    labels = np.array(labels)

    np.save("data/Labels", labels)

    print(str(count) + " examples of shape " + str(labels.shape) + " loaded into Labels.npy file")

else:

    print("Labels.npy file already exists.")
    labels = np.load('data/Labels.npy')

train_labels, test_labels = labels[train_idxs, :], labels[test_idxs, :]
labels = None

os.remove("data/Labels.npy")
print("Removing labels.npy file.")

np.save("data/train_labels", train_labels)
np.save("data/test_labels", test_labels)

train_labels = None
test_labels = None
print("Saving Train, Test labels...")
