import pandas as pd 
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import Dataset, DataLoader


class ConvNet(nn.Module): #Las clases deben empezar con MAYUS, la clase padre es nn.Modulo y la clase hijo es ConvNet
    
    def __init__(self): #defines the core architecture of the model that is, all the layers with the number of neurons at each layer. CONSTRUCTOR
        super(ConvNet, self).__init__() 
        input_size = 2048
        hidden_size1 = 20
        hidden_size2 = 10
        hidden_size3 = 3
        output_size = 1

        self.fl1 = nn.Linear(input_size, hidden_size1) 
        self.fl2 = nn.Linear(hidden_size1, hidden_size2) 
        self.fl3 = nn.Linear(hidden_size2, hidden_size3) 
        self.fl4 = nn.Linear(hidden_size3, output_size) 
    def forward(self, x): #the forward function does a forward pass in the network. It includes all the activation functions at each layer. ACCIONES
        out = self.fl1(x)
        out = F.relu(out) #activation function
        out = self.fl2(out)
        out = F.relu(out) #activation function
        out = self.fl3(out)
        out = F.relu(out) #activation function
        out = self.fl4(out)
        
        return out 

def train(model, device, train_dataloader, optim, epoch): #the actual backpropagation step
    model.train()

    loss_func = torch.nn.L1Loss(reduction='sum') 
    for b_i, (fp, y) in enumerate(train_dataloader):

        fp, y = fp.to(device), y.to(device)
        optim.zero_grad() #despues de la optimizacion hay que reiniciar el gradiente, esto lo hace esta funcion
        pred_prob = model(fp.float())
        loss = loss_func(pred_prob, y)
        loss.backward() #calcula la perdida con respecto a la entrada, backward propagation
        optim.step() #optimization step, update weights
        
        if b_i % 10 == 0: #"Cada 10 pasos"
            print('epoch: {} [{}/{} ({:.0f}%)]\t training loss: {:.6f}'.format(
                epoch, b_i * len(fp), len(train_dataloader.dataset),
                100. * b_i / len(train_dataloader), loss.item()))

def test(model, device, test_dataloader): #Para evaluar el desempeño de la NN

    model.eval()
    loss = 0 #inicialización de la perdida
    loss_func = torch.nn.L1Loss(reduction='sum') 

    with torch.no_grad(): #El with torch.no_grad() es para hacer operaciones SIN la gradient function
        for fp, y in test_dataloader:
            fp, y = fp.to(device), y.to(device) 
            pred_prob = model(fp)
            loss += loss_func(pred_prob, y).item()  # loss summed across the batch
            pred = pred_prob.argmax(dim=1, keepdim=True)  # us argmax to get the most likely prediction
            

    loss /= len(test_dataloader.dataset)

    print('\nTest dataset: Overall Loss: {:.4f},  ({:.0f}%)\n'.format(
        loss, len(test_dataloader.dataset)))

class PDB(Dataset):
    def __init__(self, fp, y):
        self.fp = fp
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx): #idx means index
        fp = self.fp[idx]
        y = self.y[idx]
        
        return fp, y

df = pd.read_csv('pdbind_full_fp2.csv')


fp = torch.tensor(np.array(df[df.columns[0:-1]])).type(torch.float32)
y = torch.tensor(np.array(df['-logKd/Ki'])).type(torch.float32)

#DataSet
data_set = PDB(fp, y) #llama la clase PDB

batch_size = 3

#DataLoader
data_loader = DataLoader(data_set, batch_size=batch_size,shuffle=True ) #With collate_fn then the output are tensors. el tamaño del lote se establece en 2. Esto significa que cuando recorre el conjunto de datos, DataLoader generará 2 instancias de datos en lugar de una

train_loader = torch.utils.data.DataLoader(dataset = data_set, 
                                           batch_size = batch_size, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset = data_set, 
                                          batch_size = batch_size, 
                                          shuffle=True)

learning_rate = 0.5
torch.manual_seed(0)
device = torch.device("cpu")

model = ConvNet() #Model es el objeto que tiene la clase ConvNet
optimizer = optim.Adadelta(model.parameters(), lr = learning_rate) #lr = learning rate , model.parameters llama a los parametros del modelo 

for epoch in range(1, 3): #For only two epochs (1 an 2) , training loop 
    train(model, device, train_loader, optimizer, epoch)
    test(model, device, test_loader)