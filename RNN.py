import Data_preprocess
import torch
from torch import nn, optim
from torch.utils.data import DataLoader,Dataset
from sklearn.model_selection import train_test_split
import numpy as np
import os


#train_file_path ->  midi_file_name
def read_midi(file_path, num):
    files = [file_path + i for i in os.listdir(file_path) if i.endswith(".midi")]
    files = files[0:num]
    return files


#midi_file_name  -> data with array structure
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

#midi_file_name  -> data with array structure   (save the data to  .npy file)
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

# class Mydataset
class MyDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return (torch.LongTensor(self.x[idx]),torch.LongTensor(self.y[idx]))

#class RNNModel
#structue:
# embedding (276,128)
# lstm(128,256)
# fc(256,300)
# relu()
# fc(300,276)
class RNNModel(nn.Module):
    def __init__(self):
        super(RNNModel, self).__init__()
        self.n_notes = 276
        self.emb_dim = 128
        self.lstm_hidden_size = 256
        self.lstm_layer = 2
        self.emb = nn.Embedding(self.n_notes,self.emb_dim)
        self.lstm = nn.LSTM(
            input_size = self.emb_dim,
            hidden_size = self.lstm_hidden_size,
            num_layers = self.lstm_layer,
            batch_first=True
        )
        self.linear = nn.Linear(300, self.n_notes)
        self.fc = nn.Linear(self.lstm_hidden_size, 300)
        self.relu = nn.ReLU()

    def forward(self, x, h = None):
        x = self.emb(x)
        x, h = self.lstm(x, h)
        x = x[:,-1,:]
        x = self.fc(x)
        x = self.relu(x)
        x = self.linear(x)
        return x, h


#train method
def train(model, n_epoch, dataloader, save):

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0002)

    #print('Shape of dataloader:'.format(len(dataloader)))

    print('training----')
    model.train()
    loss_his = []
    for i in range(n_epoch):

        total_loss = 0
        for step, batch in enumerate(dataloader):
            if step % 500 == 0 and not step == 0:
                print('batch:{}/{}-------------{}%'.format(step,len(dataloader),int((step/len(dataloader)*100))))
                print(test[-1])

            x = batch[0].to(device)
            y = batch[1].to(device)

            model.zero_grad()
            pre_y,_ = model(x)
            new_note = nn.Softmax(dim=1)(pre_y)
            test = np.argmax(new_note.cpu().detach().numpy(), axis=1)

            loss = loss_fn(pre_y, y[:,-1])
            total_loss += loss.item()
            loss.backward()
            optimizer.step()

        avg_loss = total_loss / len(train_dataloader)
        loss_his.append(avg_loss)
        print('EPOCH:{}/{}------LOSS:{}'.format(i, n_epoch, avg_loss))

        if save:
            torch.save(model.state_dict(), 'weight/RNN_weight_test.pt')
    return loss_his


#generate midi file method
def generate_midi(model, seq_begin, max_length):
    print('Generating------')
    model.eval()
    h = torch.zeros(model.lstm_layer, 1, model.lstm_hidden_size).to(device)
    c = torch.zeros(model.lstm_layer, 1, model.lstm_hidden_size).to(device)
    x = torch.LongTensor(seq_begin)
    x = torch.unsqueeze(x, dim=0).to(device)

    for i in range(max_length):
        new, (h, c) = model(x, (h, c))
        new = torch.squeeze(new, dim = 0)
        new_note = nn.Softmax(dim=0)(new)
        new_note_id = np.argmax(new_note.cpu().detach().numpy(),axis=0)
        #print(new_note_id)
        #new_note = np.argmax(new_note[-1].cpu().detach().numpy())
        seq_begin.append(new_note_id)
        x = torch.unsqueeze(torch.LongTensor(seq_begin[i+1:]),dim=0).to(device)
    return seq_begin



# main function

#parameter
If_train = False
If_generate = True
If_save_weight = False
If_load_weight_for_training = True
If_load_weight_for_generating = True
Start_midi_file = 'dataset/data/train/235.midi'    # use first 16 tokens to generate
Generated_midi_length = 1000                       # how long you want to generate
Generated_midi_name = 'generated_example/test_RNN_001.midi'              # generated files name


# get device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

#model
model = RNNModel()
model.to(device)


# used in training
if If_train:
    midi_files = read_midi('dataset/data/train/',700)  #use 700 midi files

    #data_x,data_y = preprocess2(midi_files,256,True)   # save the array to npy

    if os.path.exists('data_X_256.npy') and os.path.exists('data_Y_256.npy'):
        data_x = np.load('data_X_256.npy')
        data_y = np.load('data_Y_256.npy')
    else:
        raise FileNotFoundError('There is not any npy files, you should run the 181 line')

    print(data_x.shape)

    data_x = data_x[:410000,:16]   #sequece length = 16
    data_y =data_y[:410000,:16]

    print(data_x.shape)
    #split
    train_x, val_x, train_y, val_y = train_test_split(data_x, data_y, random_state=2021, test_size=0.2)

    #dataset
    train_dataset = MyDataset(train_x,train_y)
    val_dataset = MyDataset(val_x,val_y)

    #dataloader
    train_dataloader = DataLoader(train_dataset, batch_size=32)
    val_dataloader = DataLoader(val_dataset, batch_size=32)

    #parameter
    n_epoch = 50

    if If_load_weight_for_training:

        path = 'weight/RNN_weight.pt'
    model.load_state_dict(torch.load(path))
    #train
    train(model, n_epoch, train_dataloader,If_save_weight)

    #torch.save(model.state_dict(), 'weight4.pt')


#generate
if If_generate:

    if If_load_weight_for_generating:
        path = 'weight/RNN_weight.pt'
        model.load_state_dict(torch.load(path))

    first = Data_preprocess.Encode(Start_midi_file)
    # print(first[0:16])
    # print(first[16:128])
    # print(first[256:364])

    first = first[0:16]

    pred = generate_midi(model, first, Generated_midi_length)

    Data_preprocess.Decode(pred,Generated_midi_name,True)

    print('finished------')

    # print(pred[0:16])
    # print(pred[16:128])
    # print(pred[256:364])