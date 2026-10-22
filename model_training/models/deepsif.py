"""
feb. 2023 - network from the article: 
Sun, R., Sohrabpour, A., Worrell, G. A., & He, B. (2022). 
Deep neural networks constrained by neural mass models improve electrophysiological source imaging of spatiotemporal brain dynamics.
Proceedings of the National Academy of Sciences, 119(31), e2201128119. https://doi.org/10.1073/pnas.2201128119 

code from the github repo https://github.com/bfinl/DeepSIF/

"""
import torch
from torch import nn


class MLPSpatialFilter(nn.Module):

    def __init__(self, num_sensor, num_hidden, activation):
        super(MLPSpatialFilter, self).__init__()
        self.fc11 = nn.Linear(num_sensor, num_sensor)
        self.fc12 = nn.Linear(num_sensor, num_sensor)
        self.fc21 = nn.Linear(num_sensor, num_hidden)
        self.fc22 = nn.Linear(num_hidden, num_hidden)
        self.fc23 = nn.Linear(num_sensor, num_hidden)
        self.value = nn.Linear(num_hidden, num_hidden)
        self.activation = nn.__dict__[activation]()

    def forward(self, x):
        out = dict()
        x = self.activation(self.fc12(self.activation(self.fc11(x))) + x)
        x = self.activation(self.fc22(self.activation(self.fc21(x))) + self.fc23(x))
        out['value'] = self.value(x)
        out['value_activation'] = self.activation(out['value'])
        return out


class TemporalFilter(nn.Module):

    def __init__(self, input_size, num_source, num_layer, activation):
        super(TemporalFilter, self).__init__()
        self.rnns = nn.ModuleList()
        self.rnns.append(nn.LSTM(input_size, num_source, batch_first=True, num_layers=num_layer))
        self.num_layer = num_layer
        self.input_size = input_size
        self.activation = nn.__dict__[activation]()

    def forward(self, x):
        out = dict()
        # c0/h0 : num_layer, T, num_out
        for l in self.rnns:
            l.flatten_parameters()
            x, _ = l(x)

        out['rnn'] = x  # seq_len, batch, num_directions * hidden_size
        return out


class TemporalInverseNet(nn.Module):

    def __init__(self, num_sensor=64, num_source=994, rnn_layer=3,
                 spatial_model=MLPSpatialFilter, temporal_model=TemporalFilter,
                 spatial_output='value_activation', temporal_output='rnn',
                 spatial_activation='ReLU', temporal_activation='ReLU', temporal_input_size=500):
        super(TemporalInverseNet, self).__init__()
        self.attribute_list = [num_sensor, num_source, rnn_layer,
                               spatial_model, temporal_model, spatial_output, temporal_output,
                               spatial_activation, temporal_activation, temporal_input_size]
        self.spatial_output = spatial_output
        self.temporal_output = temporal_output
        # Spatial filtering
        self.spatial = spatial_model(num_sensor, temporal_input_size, spatial_activation)
        # Temporal filtering
        self.temporal = temporal_model(temporal_input_size, num_source, rnn_layer, temporal_activation)

    def forward(self, x):
        out = dict()
        out['fc2'] = self.spatial(x)[self.spatial_output]
        x = out['fc2']
        out['last'] = self.temporal(x)[self.temporal_output]
        return out

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    

import pytorch_lightning as pl


class DeepSIFpl(pl.LightningModule): 
    def __init__(self, 
        num_sensor = 64, num_source = 994, temporal_input_size = 500, 
        optimizer = torch.optim.Adam, 
        lr = 0.001, 
        criterion = torch.nn.MSELoss(), rnn_layer=3) -> None:
        super().__init__()

        self.deep_sif_params = dict( 
            num_sensor = num_sensor, 
            num_source =num_source, 
            temporal_input_size = temporal_input_size,
            rnn_layer = rnn_layer,
            spatial_output='value_activation', temporal_output='rnn',
            spatial_activation='ReLU', temporal_activation='ReLU'
        )
        
        self.optimizer = optimizer 
        self.lr = lr 
        self.criterion = criterion 

        self.model = TemporalInverseNet( 
            **self.deep_sif_params
        )


    def forward(self, x): 
        out = self.model(torch.permute(x, (0,2,1)))["last"]
        return torch.permute( out, (0,2,1))
    
    def configure_optimizers(self):
        return self.optimizer(self.model.parameters(), lr=self.lr)
    
    def training_step(self, batch, batch_idx):
        eeg, src = batch
        eeg = eeg.float()
        src = src.float()
        src_hat = self.forward(eeg)

        loss = self.criterion(src_hat, src)
        
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        eeg, src = batch
        eeg = eeg.float()
        src = src.float()
        src_hat = self.forward(eeg)

        # compute loss
        loss = self.criterion(src_hat, src)

        self.log("validation_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss