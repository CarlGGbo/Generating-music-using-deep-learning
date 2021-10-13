import Data_preprocess

import torch
from torch import nn, optim
from torch.utils.data import DataLoader,Dataset
from sklearn.model_selection import train_test_split
import numpy as np
import os



# args = Namespace(
#     seed=1234,
#     cuda=True,
#     path="data",
#     train_batch_size=32,
#     val_batch_size=32,
#     num_workers=1,
#     myembedding_dim = 100,
#     learning_rate = 0.001
# )


#train_file_path ->  midi_file
def read_midi(file_path, num):
    files = [file_path + i for i in os.listdir(file_path) if i.endswith(".midi")]
    files = files[0:num]
    return files

#preprocess data  -> array_like data
def preprocess(midi_files, max_length,save = False):

    data_X = np.zeros((1,max_length))
    data_Y = np.zeros((1,max_length))
    for file in midi_files:
        print(file)
        file_array = Data_preprocess.Encode(file)
        len_array = len(file_array)
        file_array = file_array[0:min(len_array,1000)]
        for j in range(len(file_array)-max_length-1):
            data_X = np.append(data_X, [file_array[j: j+max_length]], axis=0)
            data_Y = np.append(data_Y, [file_array[j+1: j + max_length+1]], axis=0)

    data_X = data_X[1:]
    data_Y = data_Y[1:]
    if save:
        np.save("data_X_{}.npy".format(max_length), data_X)
        np.save("data_Y_{}.npy".format(max_length), data_Y)

    return data_X, data_Y

def preprocess2(midi_files, max_length,save = False):
    data_X = np.zeros((600000, max_length))
    data_Y = np.zeros((600000, max_length))
    i = 0
    for file in midi_files:
        print(file)
        file_array = Data_preprocess.Encode(file)
        len_array = len(file_array)
        file_array = file_array[0:min(len_array,1000)]
        for j in range(len(file_array)-max_length-1):
            data_X[i] = file_array[j : j+max_length]
            data_Y[i] = file_array[j + 1 : j + max_length + 1]
            i += 1

    data_X = data_X[:i]
    data_Y = data_Y[:i]
    if save:
        np.save("data_X_{}.npy".format(max_length), data_X)
        np.save("data_Y_{}.npy".format(max_length), data_Y)

    return data_X, data_Y

class MyDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return (torch.LongTensor(self.x[idx]),torch.LongTensor(self.y[idx]))

class RNNModel(nn.Module):
    def __init__(self):
        super(RNNModel, self).__init__()
        self.n_notes = 276
        self.emb_dim = 32
        self.lstm_hidden_size = 128
        self.lstm_layer = 10
        self.emb = nn.Embedding(self.n_notes,self.emb_dim)
        self.lstm = nn.LSTM(
            input_size = self.emb_dim,
            hidden_size = self.lstm_hidden_size,
            num_layers = self.lstm_layer,
            dropout=0.3
        )
        self.linear = nn.Linear(self.lstm_hidden_size, self.n_notes)

    def forward(self, x, h = None):
        x = self.emb(x)
        x, h = self.lstm(x, h)
        x = self.linear(x)

        return x, h

    # def init_state(self, sequence_length):
    #     return (torch.zeros(self.num_layers, sequence_length, self.lstm_size),
    #             torch.zeros(self.num_layers, sequence_length, self.lstm_size))

def train(model, n_epoch, dataloader):

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print(len(dataloader))

    print('training----')
    model.train()
    loss_his = []
    for i in range(n_epoch):

        total_loss = 0
        for step, batch in enumerate(dataloader):
            if step % 50 == 0 and not step == 0:
                print('batch:{}/{}'.format(step,len(dataloader)))

            x = batch[0].to(device)
            y = batch[1].to(device)

            model.zero_grad()
            pre_y,_ = model(x)

            loss = loss_fn(pre_y.permute(0, 2, 1), y)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()

        avg_loss = total_loss / len(train_dataloader)
        loss_his.append(avg_loss)
        print('EPOCH:{}/{}------LOSS:{}'.format(i, n_epoch, avg_loss))

    return loss_his

#seq_begin list
def generate_midi(model, seq_begin, max_length):
    print('Generating------')
    model.eval()
    h = torch.zeros(model.lstm_layer, 256, model.lstm_hidden_size).to(device)
    c = torch.zeros(model.lstm_layer, 256, model.lstm_hidden_size).to(device)
    x = torch.LongTensor(seq_begin)
    x = torch.unsqueeze(x, dim=0).to(device)

    for i in range(max_length):
        print(i)
        new, (h, c) = model(x, (h, c))
        new = torch.squeeze(new, dim = 0)
        new_note = nn.Softmax(dim=1)(new)
        new_note = np.argmax(new_note[-1].cpu().detach().numpy())

        seq_begin.append(new_note)
        x = torch.unsqueeze(torch.LongTensor(seq_begin[i+1:]),dim=0).to(device)


    return seq_begin

midi_files = read_midi('dataset/data/train/',700)

#data_x,data_y = preprocess2(midi_files,256,True)

data_x = np.load('data_X_256.npy')
data_y = np.load('data_Y_256.npy')

data_x = data_x[:80000]
data_y =data_y[:80000]
#split

train_x, val_x, train_y, val_y = train_test_split(data_x, data_y, random_state=2021, test_size=0.2)

#dataset
train_dataset = MyDataset(train_x,train_y)
val_dataset = MyDataset(val_x,val_y)

#dataloader
train_dataloader = DataLoader(train_dataset, batch_size=32)
val_dataloader = DataLoader(val_dataset, batch_size=32)

#parameter
#n_epoch = 100

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

#model
model = RNNModel()
model.to(device)

#train
#train(model, 10, train_dataloader)

#torch.save(model.state_dict(), 'weight.pt')
path = 'weight.pt'
model.load_state_dict(torch.load(path))
#generate
first = Data_preprocess.Encode('dataset/data/train/000.midi')
first = first[0:256]
pred = generate_midi(model, first, 600)
print(pred)