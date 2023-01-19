import pandas as pd 
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import Dataset, DataLoader


#-------------- Model ----------------:
class Linear_Net(nn.Module): 
    
    def __init__(self): 
        super(Linear_Net, self).__init__() 
        input_size = 2048
        hidden_size1 = 60
        hidden_size2 = 30
        hidden_size3 = 3
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
def train(model, device, train_dataloader, optim): 
    model.train()

    loss_func = torch.nn.L1Loss(reduction='sum') 
    loss_collect = 0

    for b_i, (fp, y) in enumerate(train_dataloader):

        fp, y = fp.to(device), y.to(device)

        optim.zero_grad() 
        pred_prob = model(fp.float())

        loss = loss_func(pred_prob, y)
        loss.backward() 

        optim.step() 
        loss_collect += loss.item() 
    

        if b_i % 10 == 0:
            print('epoch: {} [{}/{} ({:.0f}%)]\t training loss: {:.6f}'.format(
                epoch, b_i * len(fp), len(train_dataloader.dataset),
                100. * b_i / len(train_dataloader), loss.item()))
    loss_collect /= len(train_dataloader.dataset)

    return loss_collect

#-------------- Validation Routine ----------------:

def test(model, device, test_dataloader, epoch): 

    model.eval()
    loss_collect = 0 
    loss_func = torch.nn.L1Loss(reduction='sum') 

    with torch.no_grad(): 
        for fp, y in test_dataloader:
            fp, y = fp.to(device), y.to(device) 
            pred_prob = model(fp)
            loss_collect += loss_func(pred_prob, y).item()
  
    loss_collect /= len(test_dataloader.dataset)
    
    print('\nTest dataset: Overall Loss: {:.4f},  ({:.0f}%)\n'.format(
        loss_collect, len(test_dataloader.dataset)))

    return loss_collect

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
batch_size = 2

data_loader = DataLoader(data_set, batch_size=batch_size,shuffle=True ) 

train_loader = torch.utils.data.DataLoader(dataset = data_set, 
                                           batch_size = batch_size, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset = data_set, 
                                          batch_size = batch_size, 
                                          shuffle=True)

#-------------- Run training loop ----------------:
learning_rate = 0.05
torch.manual_seed(0)
device = torch.device("cpu")

model = Linear_Net() 
optimizer = optim.Adadelta(model.parameters(), lr = learning_rate) 

losses_train = []
losses_test = []

for epoch in range(1, 11): 
    train_loss = train(model, device, train_loader, optimizer)
    losses_train.append(train_loss)

    test_loss = test(model, device, test_loader, epoch)
    losses_test.append(test_loss)

plt.plot(losses_train, label ='train losses')
plt.legend()
plt.xlabel('time')
plt.ylabel('train losses')

plt.plot(losses_test, label ='test losses')
plt.legend()
plt.xlabel('time')
plt.ylabel('test losses')

plt.show()


#-------------- Testing ----------------:

def predict(model, device, dataloader):

    model.eval()

    fp_all = []
    y_all = []
    pred_prob_all = []

    # Remove gradients:
    with torch.no_grad():

        # Looping over the dataloader allows us to pull out or input/output data:
        for fp, y in dataloader:

            # Assign X and y to the appropriate device:
            fp, y = fp.to(device), y.to(device)

            # Make a prediction:
            pred_prob = model(fp)

            fp_all.append(fp)
            y_all.append(y)
            pred_prob_all.append(pred_prob)
    return (
        torch.concat(fp_all), 
        torch.concat(y_all), 
        torch.concat(pred_prob_all).view(-1),
        )

fp_all, y_all, pred_prob_all = predict(model, device, test_loader)


#----------------------------------------------------------------
r2 = r2_score(y_all, pred_prob_all)
mae = mean_absolute_error(y_all, pred_prob_all)
rmse = mean_squared_error(y_all, pred_prob_all, squared=False)

print("R2 Score: {}".format(r2))
print("MAE: {}".format(mae))
print("RMSE: {}".format(rmse))
#----------------------------------------------------------------

plt.figure(figsize=(4, 4), dpi=100)
plt.scatter(y_all, pred_prob_all, alpha=0.3)
plt.plot([min(y_all), max(y_all)], [min(y_all), max(y_all)], color="k", ls="--")
plt.xlim([min(y_all), max(y_all)])
plt.xlabel("True Values")
plt.ylabel("Predicted Values")

plt.show()

#----------------------------------------------------------------

plt.figure(figsize=(4, 3), dpi=100)
plt.hist(test_loader.dataset.df["-logKd/Ki"])
plt.show()
