# models.py

from sentiment_data import *
from typing import List

import torch
import torch.nn as nn
import itertools    
import pdb
def pad_to_length(np_arr, length):
    """
    Forces np_arr to length by either truncation (if longer) or zero-padding (if shorter)
    :param np_arr:
    :param length: Length to pad to
    :return: a new numpy array with the data from np_arr padded to be of length length. If length is less than the
    length of the base array, truncates instead.
    """
    result = np.zeros(length)
    result[0:np_arr.shape[0]] = np_arr
    return result

class SentimentLSTM(nn.Module):
    

    def __init__(self, output_size, embedding_dim, hidden_dim, n_layers,word_vectors, drop_prob=0.3):
        """
        Initialize the model by setting up the layers.
        """
        super(SentimentLSTM, self).__init__()

        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim

        
        # embedding and LSTM layers
        self.embedding_dim = embedding_dim
        self.embedding = word_vectors
        self.ffinit = nn.Linear(embedding_dim,hidden_dim)
        nn.init.xavier_uniform_(self.ffinit.weight)

        self.lstm = nn.LSTM(embedding_dim, hidden_dim, dropout=drop_prob, batch_first=True)#, bidirectional=True)
        for name, param in self.lstm.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_normal_(param)
        #nn.init.xavier_uniform(self.lstm.weight)

        
        self.dropout = nn.Dropout(0.3)
        self.ffmiddle = nn.Linear(hidden_dim, hidden_dim)
        nn.init.xavier_uniform_(self.ffmiddle.weight)

        # linear and sigmoid layer

        self.fc = nn.Linear(hidden_dim, output_size)
        nn.init.xavier_uniform_(self.fc.weight)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        
        batch_size = x.size(0)
        embeds = torch.ones([x.size(0), x.size(1), self.embedding_dim], dtype=torch.float64)
        #len_sum = 0
        
        # embeddings and lstm_out
        for i in range(x.size(0)):
            for j in range(x.size(1)):

                embeds[i,j]= torch.from_numpy(self.embedding.get_embedding_from_index(int(x[i,j])))

        out, hidden = self.lstm(embeds.float())
        out = self.ffmiddle(out)
        out = self.fc(out)

        #pdb.set_trace()
        
        # sigmoid function
        sig_out = self.sig(out)
        sig_out = sig_out.view(batch_size, -1)
        sig_out = sig_out.mean(dim=1)
        #sig_out = sig_out[:, -1] # get last batch of labels
        
        return sig_out, hidden


def train_evaluate_fancy(train_exs: List[SentimentExample], dev_exs: List[SentimentExample], test_exs: List[SentimentExample], word_vectors: WordEmbeddings) -> List[SentimentExample]:
    """
    Train a feedforward neural network on the given training examples, using dev_exs for development, and returns
    predictions on the *blind* test examples passed in. Returned predictions should be SentimentExample objects with
    predicted labels and the same sentences as input (but these won't be read by the external code). The code is set
    up to go all the way to test predictions so you have more freedom to handle example processing as you see fit.
    :param train_exs:
    :param dev_exs:
    :param test_exs:
    :param word_vectors:
    :return:
    """
    # 59 is the max sentence length in the corpus, so let's set this to 60
    seq_max_len = 60
    # To get you started off, we'll pad the training input to 60 words to make it a square matrix.
    train_mat = np.asarray([pad_to_length(np.array(ex.indexed_words), seq_max_len) for ex in train_exs])
    # Also store the sequence lengths -- this could be useful for training LSTMs
    train_seq_lens = np.array([len(ex.indexed_words) for ex in train_exs])
    # Labels
    train_labels_arr = np.array([ex.label for ex in train_exs])

    # get numpy array with sequence length for dev and test
    dev_mat = np.asarray([pad_to_length(np.array(ex.indexed_words), seq_max_len) for ex in dev_exs])
    # Also store the sequence lengths -- this could be useful for training LSTMs
    dev_seq_lens = np.array([len(ex.indexed_words) for ex in dev_exs])
    # Labels
    dev_labels_arr = np.array([ex.label for ex in dev_exs])

    test_mat = np.asarray([pad_to_length(np.array(ex.indexed_words), seq_max_len) for ex in test_exs])
    # Also store the sequence lengths -- this could be useful for training LSTMs
    test_seq_lens = np.array([len(ex.indexed_words) for ex in test_exs])
    # Labels
    test_labels_arr = np.array([ex.label for ex in test_exs])
    #print(test_labels_arr)



    from torch.utils.data import TensorDataset, DataLoader
    # create Tensor datasets
    train_data = TensorDataset(torch.from_numpy(train_mat), torch.from_numpy(train_labels_arr))
    dev_data = TensorDataset(torch.from_numpy(dev_mat), torch.from_numpy(dev_labels_arr))
    test_data = TensorDataset(torch.from_numpy(test_mat), torch.from_numpy(test_labels_arr))
    # dataloaders
    batch_size = 100 #10
    # make sure to SHUFFLE your data
    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
    dev_loader = DataLoader(dev_data, shuffle=True, batch_size=batch_size)
    

    # Instantiate the model w/ hyperparams
    output_size = 1
    embedding_dim = 300
    hidden_dim = 256
    n_layers = 2
    net = SentimentLSTM(output_size, embedding_dim, hidden_dim, n_layers,word_vectors)
    print(net)

    # loss and optimization functions
    lr=0.001 #0.001
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    # training params

    epochs = 5 #20 

    counter = 0
    print_every = 500
    
    num_correct = 0


    net.train()
    for epoch_iter in range(epochs):
        
        for inputs, labels in train_loader:
            counter += 1


            net.zero_grad()
            output, h = net.forward(inputs)#, h)
            #print(len(output))
            #print(len(labels))

            # calculate the loss and perform backprop
            loss = criterion(output.squeeze(), labels.float())
            loss.backward()
            
            optimizer.step()

            # loss stats
            if counter % print_every == 0:
                val_losses = []
                net.eval()
                for inputs, labels in train_loader:
                    output, val_h = net(inputs)
                    val_loss = criterion(output.squeeze(), labels.float())

                    val_losses.append(val_loss.item())
                    
                net.train()
                print("Epoch: {}/{}...".format(epoch_iter+1, epochs),
                      "Step: {}...".format(counter),
                      "Loss: {:.6f}...".format(loss.item()),
                      "Train Loss: {:.6f}".format(np.mean(val_losses)))
                pred = torch.round(output.squeeze())  
                       
                # compare predictions to true label
                correct_tensor = pred.eq(labels.float().view_as(pred))
                correct = np.squeeze(correct_tensor.numpy())
                num_correct += np.sum(correct)
                train_acc = num_correct/batch_size
                print("train accuracy:",train_acc)
                num_correct = 0


    dev_losses = []
    num_correct = 0

    
    net.eval()
    for inputs, labels in dev_loader:
        output, val_h = net(inputs)        
        dev_loss = criterion(output.squeeze(), labels.float())
        dev_losses.append(dev_loss.item())       
        pred = torch.round(output.squeeze())         
        correct_tensor = pred.eq(labels.float().view_as(pred))
        correct = np.squeeze(correct_tensor.numpy())
        num_correct += np.sum(correct)

    print("dev final loss: {:.3f}".format(np.mean(dev_losses)))
    print("num_correct", num_correct)
    print("len(dev_loader.dataset):",len(dev_loader.dataset))

    dev_acc = num_correct/len(dev_loader.dataset)
    print("dev final accuracy: {:.3f}".format(dev_acc))

    test_loader = DataLoader(test_data, shuffle=False, batch_size=1)
    pred_test = []
    i = 0
    for inputs, lables in test_loader:
        output, h = net(inputs)
        pred = torch.round(output.squeeze())
        inputs = inputs.numpy()
        #print(inputs.tolist())
        inputs = list(itertools.chain(*inputs))
        pred_test.append(SentimentExample(inputs[:test_seq_lens[i]], int(pred.detach().numpy())))
        i +=1
    return pred_test

class SentimentFNN(nn.Module):

    def __init__(self, output_size, embedding_dim, hidden_dim, n_layers,word_vectors, drop_prob=0.5):
        
        super(SentimentFNN, self).__init__()

        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim

        
        self.embedding_dim = embedding_dim
        self.embedding = word_vectors
        self.ffinit = nn.Linear(embedding_dim,hidden_dim)
        nn.init.xavier_uniform(self.ffinit.weight)

        self.ffmiddle = nn.Linear(hidden_dim,hidden_dim)
        nn.init.xavier_uniform(self.ffmiddle.weight)
        self.dropout = nn.Dropout(0.3)
        

        self.fc = nn.Linear(hidden_dim, output_size)
        nn.init.xavier_uniform(self.fc.weight)
        self.sig = nn.Sigmoid()

    def forward(self, x):
       
        batch_size = x.size(0)
        embeds = torch.ones([x.size(0), self.embedding_dim], dtype=torch.float64)
        len_sum = 0
        
        for i in range(x.size(0)):
            for j in range(x.size(1)):
                len_sum += self.embedding.get_embedding_from_index(int(x[i,j]))
            embeds[i] = torch.from_numpy(len_sum)/60
            len_sum = 0

        fc_out = self.ffinit(embeds.float())

        fc_out = self.ffmiddle(fc_out)
        fc_out = self.ffmiddle(fc_out)
        
        
        out = self.fc(fc_out)
        
        sig_out = self.sig(out)
        
        return sig_out
    
    
    
# , using dev_exs for development and returning
# predictions on the *blind* test_exs (all test_exs have label 0 as a dummy placeholder value).
def train_evaluate_ffnn(train_exs: List[SentimentExample], dev_exs: List[SentimentExample], test_exs: List[SentimentExample], word_vectors: WordEmbeddings) -> List[SentimentExample]:
    """
    Train a feedforward neural network on the given training examples, using dev_exs for development, and returns
    predictions on the *blind* test examples passed in. Returned predictions should be SentimentExample objects with
    predicted labels and the same sentences as input (but these won't be read by the external code). The code is set
    up to go all the way to test predictions so you have more freedom to handle example processing as you see fit.
    :param train_exs:
    :param dev_exs:
    :param test_exs:
    :param word_vectors:
    :return:
    """
    # 59 is the max sentence length in the corpus, so let's set this to 60
    seq_max_len = 60
    # To get you started off, we'll pad the training input to 60 words to make it a square matrix.
    train_mat = np.asarray([pad_to_length(np.array(ex.indexed_words), seq_max_len) for ex in train_exs])
    # Also store the sequence lengths -- this could be useful for training LSTMs
    train_seq_lens = np.array([len(ex.indexed_words) for ex in train_exs])
    # Labels
    train_labels_arr = np.array([ex.label for ex in train_exs])

    # get numpy array with sequence length for dev and test
    dev_mat = np.asarray([pad_to_length(np.array(ex.indexed_words), seq_max_len) for ex in dev_exs])
    # Also store the sequence lengths -- this could be useful for training LSTMs
    dev_seq_lens = np.array([len(ex.indexed_words) for ex in dev_exs])
    # Labels
    dev_labels_arr = np.array([ex.label for ex in dev_exs])

    test_mat = np.asarray([pad_to_length(np.array(ex.indexed_words), seq_max_len) for ex in test_exs])
    # Also store the sequence lengths -- this could be useful for training LSTMs
    test_seq_lens = np.array([len(ex.indexed_words) for ex in test_exs])
    # Labels
    test_labels_arr = np.array([ex.label for ex in test_exs])
    #print(test_labels_arr)



    from torch.utils.data import TensorDataset, DataLoader
    # create Tensor datasets
    train_data = TensorDataset(torch.from_numpy(train_mat), torch.from_numpy(train_labels_arr))
    dev_data = TensorDataset(torch.from_numpy(dev_mat), torch.from_numpy(dev_labels_arr))
    test_data = TensorDataset(torch.from_numpy(test_mat), torch.from_numpy(test_labels_arr))
    # dataloaders
    batch_size = 5
    # make sure to SHUFFLE your data
    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
    dev_loader = DataLoader(dev_data, shuffle=True, batch_size=batch_size)
    

    # Instantiate the model w/ hyperparams
    output_size = 1
    embedding_dim = 300 
    hidden_dim = 100
    n_layers = 1
    net = SentimentFNN(output_size, embedding_dim, hidden_dim, n_layers,word_vectors)
    print(net)

    # loss and optimization functions
    lr=0.01
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    # training params

    epochs = 10 #20 

    counter = 0
    print_every = 500
    
    num_correct = 0


    net.train()
    # train for some number of epochs
    for epoch_iter in range(epochs):
        
        for inputs, labels in train_loader:
            counter += 1

            net.zero_grad()
            output= net.forward(inputs)

            loss = criterion(output.squeeze(), labels.float())
            loss.backward()
            
            optimizer.step()

            # loss stats
            if counter % print_every == 0:
                val_losses = []
                net.eval()
                for inputs, labels in train_loader:
                    output = net(inputs)
                    val_loss = criterion(output.squeeze(), labels.float())

                    val_losses.append(val_loss.item())
                    
                net.train()
                print("Epoch: {}/{}...".format(epoch_iter+1, epochs),
                      "Step: {}...".format(counter),
                      "Loss: {:.6f}...".format(loss.item()),
                      "Train Loss: {:.6f}".format(np.mean(val_losses)))
                pred = torch.round(output.squeeze())  
                       
                # compare predictions to true label
                correct_tensor = pred.eq(labels.float().view_as(pred))
                correct = np.squeeze(correct_tensor.numpy())
                num_correct += np.sum(correct)
                train_acc = num_correct/batch_size
                print("train accuracy:",train_acc)
                num_correct = 0


    dev_losses = []
    num_correct = 0


    net.eval()
    for inputs, labels in dev_loader:
        output = net(inputs)        
        dev_loss = criterion(output.squeeze(), labels.float())
        dev_losses.append(dev_loss.item())       
        pred = torch.round(output.squeeze())         
        correct_tensor = pred.eq(labels.float().view_as(pred))
        correct = np.squeeze(correct_tensor.numpy())
        num_correct += np.sum(correct)

    print("dev final loss: {:.3f}".format(np.mean(dev_losses)))
    print("num_correct", num_correct)
    print("len(dev_loader.dataset):",len(dev_loader.dataset))

    dev_acc = num_correct/len(dev_loader.dataset)
    print("dev final accuracy: {:.3f}".format(dev_acc))

    test_loader = DataLoader(test_data, shuffle=False, batch_size=1)
    pred_test = []
    i = 0
    for inputs, lables in test_loader:
        output = net(inputs)
        pred = torch.round(output.squeeze())
        inputs = inputs.numpy()
        #print(inputs.tolist())
        inputs = list(itertools.chain(*inputs))
        pred_test.append(SentimentExample(inputs[:test_seq_lens[i]], int(pred.detach().numpy())))
        i +=1
    return pred_test




# # Analogous to train_ffnn, but trains your fancier model.
# def train_evaluate_fancy(train_exs: List[SentimentExample], dev_exs: List[SentimentExample], test_exs: List[SentimentExample], word_vectors: WordEmbeddings) -> List[SentimentExample]:
#     raise Exception("Not implemented")