""" LSTM model from 
Hecker, L., Rupprecht, R., Elst, L. T. van, & Kornmeier, J. (2022). 
Long-Short Term Memory Networks for Electric Source Imaging with Distributed Dipole Models (p. 2022.04.13.488148).
bioRxiv. https://doi.org/10.1101/2022.04.13.488148
"""

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F


class HeckerLSTM(nn.Module): 
    def __init__(self, n_electrodes=61, hidden_size=85, n_sources=1284, bias=False) -> None:
        super().__init__()

        self.lstm = torch.nn.LSTM(input_size = n_electrodes, hidden_size = hidden_size, num_layers = 2, dropout = 0.2, bidirectional = True)
        self.fc = torch.nn.Linear( hidden_size*2, n_sources, bias=bias )
        
    def forward(self, x): 
        out, _ = self.lstm( torch.permute( x , (2,0,1) ) )
        out = F.relu(out)
        out = self.fc( torch.permute( out, (1,0,2) ) )
        
        return torch.permute(  out, (0,2,1) )

##############################################
###### attempt to put drop out in lstm
class DPlstm(nn.Module): 
    def __init__(self, n_electrodes=61, hidden_size=85, n_sources=1284, bias=False, dropout_rate=0.2) -> None:
        super().__init__()

        self.layer1 = torch.nn.LSTM( input_size = n_electrodes, hidden_size = hidden_size, num_layers = 1, dropout = 0., bidirectional = True)
        self.dropout = torch.nn.Dropout1d( p = dropout_rate )
        self.layer2 = torch.nn.LSTM(input_size = 2*hidden_size, hidden_size = hidden_size, num_layers = 1, dropout = 0., bidirectional = True)
        self.fc = torch.nn.Linear( hidden_size*2, n_sources, bias=bias )
    
    def forward(self, x): 
        out, _ = self.layer1( torch.permute( x , (2,0,1) ) )
        out = self.dropout(out)
        out, _ = self.layer2(out)
        out = F.relu(out)
        out = self.fc( torch.permute( out, (1,0,2) ) )
        
        return torch.permute(  out, (0,2,1) )

################################################################
######## en cours: pytorch lightning
class HeckerLSTMpl( pl.LightningModule ): 
    def __init__(self, n_electrodes=61, hidden_size=85, 
            n_sources=1284, bias=False, 
            optimizer = torch.optim.Adam, lr = 0.001,  criterion = nn.MSELoss(), 
            mc_dropout_rate=0) -> None:
        super().__init__()
        self.mc_dropout_rate = mc_dropout_rate
        
        if mc_dropout_rate != 0 : 
            # use mc dropout
            self.model = DPlstm(n_electrodes=n_electrodes, hidden_size=hidden_size, n_sources=n_sources, bias=bias, dropout_rate=mc_dropout_rate)
        else :
            self.model = HeckerLSTM(n_electrodes=n_electrodes, hidden_size=hidden_size, n_sources=n_sources ,bias=bias)

        self.criterion = criterion
        self.optimizer = optimizer
        self.lr = lr


    def forward(self, x): 
        return self.model(x)
    
    def configure_optimizers(self):
        return self.optimizer(self.model.parameters(), lr=self.lr)

    def training_step(self, batch, batch_idx):
        eeg, src = batch
        eeg, src = eeg.float(), src.float()
        src_hat = self.forward(eeg)

        #torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1)

        # compute loss
        loss = self.criterion(src_hat, src)
        
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss
    
    
    def validation_step(self, batch, batch_idx):
        eeg, src = batch
        eeg, src = eeg.float(), src.float()
        src_hat = self.forward(eeg)

        # compute loss
        loss = self.criterion(src_hat, src)

        self.log("validation_loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        return loss
    
    def predict_step(self, batch) :
        self.model.dropout.train()

        preds = self.model(batch)
        return preds
