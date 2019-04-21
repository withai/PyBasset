import torch
from torch.utils import data
import torch.nn as nn
import torch.optim as optim

from my_classes import BassetDataset
from torch.backends import cudnn
import numpy as np
from sklearn import metrics


# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
#cudnn.benchmark = True

# Parameters
params = {'batch_size': 256,
          'shuffle': True,
          'num_workers': 6}

params_train = {'batch_size': 4096,
          'shuffle': True,
          'num_workers': 6}

params_test = {'batch_size': 4096,
          'shuffle': True,
          'num_workers': 6}
max_epochs = 100


# Generators
training_set = BassetDataset(np.load("data/train_features.npy"), np.load("data/train_labels.npy"))
training_generator = data.DataLoader(training_set, **params)
training_acc_generator = data.DataLoader(training_set, **params_train)
print("Train features and labels loaded to memory...")

validation_set = BassetDataset(np.load("data/test_features.npy"), np.load("data/test_labels.npy"))
validation_acc_generator = data.DataLoader(validation_set, **params_test)
print("Test features and labels loaded to memory...")
print("##############################################")
print()

class Basset(nn.Module):

    def __init__(self):
        super(Basset, self).__init__()

        self.layer1  = nn.Sequential(
            nn.Conv1d(in_channels=4, out_channels=300, kernel_size=19),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3))

        #self.layer1 = nn.GRU(input_size=4, hidden_size=50, num_layers=3, batch_first=True, dropout=0.3, bidirectional=False)

        self.layer2 = nn.Sequential(
            nn.Conv1d(in_channels=300, out_channels=200, kernel_size=11),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4))

        #self.layer2 = nn.GRU(input_size=300, hidden_size=200, num_layer=11, dropout=0.3, bidirectional=True)
        #self.layer2 = nn.GRU(input_size=300, hidden_size=200, num_layers=3, batch_first=True, dropout=0.3, bidirectional=False)


        self.layer3 = nn.Sequential(
            nn.Conv1d(in_channels=200, out_channels=200, kernel_size=7),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4))

        self.layer4 = nn.GRU(input_size=10, hidden_size=10, num_layers=2, batch_first=True, dropout=0.3, bidirectional=False)

        self.fc1 = nn.Linear(in_features=2000, out_features=1000)
        self.relu4 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=0.3)

        self.fc2 = nn.Linear(in_features=1000, out_features=1000)
        self.relu5 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=0.3)

        self.fc3 = nn.Linear(in_features=1000, out_features=164)
        self.sig3 = nn.Sigmoid()


    def forward(self, inputs):

        output = self.layer1(inputs)
        #output = output.permute(1, 0, 2)
        output = self.layer2(output)
        #
        output = self.layer3(output)
        output = output.permute(1, 0, 2)

        output = self.layer4(output)[0]
        output = output.permute(1, 0, 2)

        output = output.reshape(output.size(0), -1)

        output = self.fc1(output)
        output = self.relu4(output)
        output = self.dropout1(output)

        output = self.fc2(output)
        output = self.relu5(output)
        output = self.dropout2(output)

        output = self.fc3(output)
        #output = self.sig3(output)

        return output


def main():
    net = Basset().to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(net.parameters())

    #Training.
    net.train()
    for epoch in range(max_epochs):
        #Training

        running_loss = 0.0
        net.train()
        for i, (local_batch, local_labels) in enumerate(training_generator):

            local_batch, local_labels = torch.tensor(local_batch), torch.tensor(local_labels, dtype=torch.float32)
            # Transfer to GPU
            local_batch, local_labels = local_batch.to(device), local_labels.to(device)

            #local_batch = local_batch.permute(2, 0, 1)
            #print(local_batch.size())

            outputs = net(local_batch)
            loss = criterion(outputs, local_labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        #Compute training accuracy.
        net.eval()
        acc_count = 0
        tot_count = 0
        auc_train = []
        with torch.set_grad_enabled(False):

            for j, (local_batch, local_labels) in enumerate(training_acc_generator):

                local_batch, local_labels = torch.tensor(local_batch), torch.tensor(local_labels, dtype=torch.uint8)
                # Transfer to GPU
                local_batch, local_labels = local_batch.to(device), local_labels.to(device)

                #local_batch = local_batch.permute(2, 0, 1)

                # Model computations
                outputs = net(local_batch)
                pred = torch.sigmoid(outputs)

                pred=pred.cpu().detach().numpy()
                labels=local_labels.cpu().numpy()

                auc_train.append(metrics.roc_auc_score(labels, pred))

            print('[%d] loss: %.3f, Train accuracy: %.3f' %(epoch + 1, running_loss / (i+1), np.mean(auc_train)), end=", ")
        running_loss = 0.0

        # Testing.
        net.eval()
        acc_count = 0
        tot_count = 0
        auc_test = []
        with torch.set_grad_enabled(False):

            for i, (local_batch, local_labels) in enumerate(validation_acc_generator):

                local_batch, local_labels = torch.tensor(local_batch), torch.tensor(local_labels, dtype=torch.uint8)
                # Transfer to GPU
                local_batch, local_labels = local_batch.to(device), local_labels.to(device)

                #local_labels[local_labels==0] = 2
                #local_batch = local_batch.permute(2, 0, 1)

                # Model computations
                outputs = net(local_batch)
                pred = torch.sigmoid(outputs)

                pred=pred.cpu().detach().numpy()
                labels=local_labels.cpu().numpy()

                # acc = (outputs == local_labels)
                #
                # local_labels[local_labels==2] = 0

                auc_test.append(metrics.roc_auc_score(labels, pred))

                # acc_count += acc.float().sum().item()
                # tot_count += local_labels.float().sum().item()

            print('Test accuracy: %.3f' %(np.mean(auc_test)))


        torch.save({
            'epoch': epoch,
            'model_state_dict': net.state_dict(),
            'model': net,
            'auc_train': auc_train,
            'auc_test': auc_test,
            'running_loss': running_loss}, 'models_maserati_no_bn/pybasset_{0:03d}.pwf'.format(epoch))

    print("finished training.")

    # Validation
    # with torch.set_grad_enabled(False):
    #     for local_batch, local_labels in validation_generator:
    #         # Transfer to GPU
    #         local_batch, local_labels = local_batch.to(device), local_labels.to(device)
    #
    #         # Model computations
    #         [...]


################################################################################
# __main__
################################################################################
if __name__ == '__main__':
    main()
