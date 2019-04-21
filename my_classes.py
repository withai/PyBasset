from torch.utils.data import Dataset
import numpy as np

class BassetDataset(Dataset):
    def __init__(self, sequences, labels):
        'Initialization'
        self.labels = labels
        self.sequences = sequences

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.sequences)

    def onehot(self, X):
        X_onehot = np.zeros((4, 600), dtype=np.float32)

        all_letters = ['a', 't', 'g', 'c']
        all_cap_letters = ['A', 'T', 'G', 'C']

        for li in range(len(X)):
            letter = X[li]
            try:
                X_onehot[all_letters.index(str(letter))][li] = 1
            except ValueError:
                X_onehot[all_cap_letters.index(str(letter))][li] = 1

        return X_onehot

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        X_onehot = self.onehot(self.sequences[index])
        y = self.labels[index, :]

        return X_onehot, y
