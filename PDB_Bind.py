import pandas as pd 
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import Dataset, DataLoader

#-------------- Model ----------------:
class ConvNet(nn.Module): 
    
    def __init__(self): 
        super(ConvNet, self).__init__() 
        input_size = 2048
        hidden_size1 = 45
        hidden_size2 = 20
        hidden_size3 = 7
        output_size = 1

        self.fl1 = nn.Linear(input_size, hidden_size1) 
        self.fl2 = nn.Linear(hidden_size1, hidden_size2) 
        self.fl3 = nn.Linear(hidden_size2, hidden_size3) 
        self.fl4 = nn.Linear(hidden_size3, output_size) 
        
    def forward(self, x):
        out = self.fl1(x)
        out = F.relu(out) 
        out = self.fl2(out)
        out = F.relu(out) 
        out = self.fl3(out)
        out = F.relu(out) 
        out = self.fl4(out)
        
        return out 

#-------------- Train routine ----------------:
def train(model, device, train_dataloader, optim, epoch): 
    model.train()

    loss_func = torch.nn.L1Loss(reduction='sum') 
    for b_i, (fp, y) in enumerate(train_dataloader):

        fp, y = fp.to(device), y.to(device)

        optim.zero_grad() 
        pred_prob = model(fp.float())

        loss = loss_func(pred_prob, y)
        loss.backward() 

        optim.step() 
        
        if b_i % 10 == 0: #"Cada 10 pasos"
            print('epoch: {} [{}/{} ({:.0f}%)]\t training loss: {:.6f}'.format(
                epoch, b_i * len(fp), len(train_dataloader.dataset),
                100. * b_i / len(train_dataloader), loss.item()))

#-------------- Validation Routine ----------------:

def test(model, device, test_dataloader): 

    model.eval()
    loss = 0 
    loss_func = torch.nn.L1Loss(reduction='sum') 

    with torch.no_grad(): 
        for fp, y in test_dataloader:
            fp, y = fp.to(device), y.to(device) 
            pred_prob = model(fp)
            loss += loss_func(pred_prob, y).item()  
            pred = pred_prob.argmax(dim=1, keepdim=True) 
            

    loss /= len(test_dataloader.dataset)

    print('\nTest dataset: Overall Loss: {:.4f},  ({:.0f}%)\n'.format(
        loss, len(test_dataloader.dataset)))

#-------------- Dataset ----------------:
class PDB(Dataset):
    def __init__(self,path):

        self.df = pd.read_csv(path)

        self.fp= self.df[self.df.columns[0:-1]].values
        
        self.y = self.df[self.df.columns[-1]].values

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index): 
        fp = self.fp[index]
        y = self.y[index]
        
        return torch.tensor(fp, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

data_set = PDB('pdbind_full_fp2.csv') 

#-------------- Data Loader ----------------:
batch_size = 3

data_loader = DataLoader(data_set, batch_size=batch_size,shuffle=True ) 

train_loader = torch.utils.data.DataLoader(dataset = data_set, 
                                           batch_size = batch_size, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset = data_set, 
                                          batch_size = batch_size, 
                                          shuffle=True)

#-------------- Run training loop ----------------:
learning_rate = 0.5
torch.manual_seed(0)
device = torch.device("cpu")

model = ConvNet() 
optimizer = optim.Adadelta(model.parameters(), lr = learning_rate) 

for epoch in range(1, 3): 
    train(model, device, train_loader, optimizer, epoch)
    test(model, device, test_loader)