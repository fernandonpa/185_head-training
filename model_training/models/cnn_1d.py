### One dimensional convolutional neural network

import torch
from torch import nn 
import pytorch_lightning as pl
import torch.nn.functional as F


class simple_1dCNN_v2(nn.Module):
  """
  First CNN :
  2 convolutional layers (1d) + non linear activation function btwn them 
  (= minimum to be able to approximate a non linear function)

  activation = Relu

  inputs: 
    channels : list of values of input and output channels for the different layers (ex: n_electrodes, 4*n_electrodes, n_sources)
     if there are N layers, channels should be N+1 long (2 layers: 2+1=3 values indeed)
    kernel_size: size of the kernel for convolutions
    bias : bias for convolutions

  """ 
  def __init__(self, channels, kernel_size = 3, bias = False, sum_xai=False):
        super().__init__()
        self.sum_xai = sum_xai
        self.conv1  = nn.Conv1d( 
          in_channels   = channels[0], 
          out_channels  = channels[1], 
          kernel_size   = kernel_size, 
          dilation      = 1, 
          padding       = 'same', 
          bias          = bias)
        self.fc     = nn.Linear(
          in_features   = channels[1], 
          out_features  = channels[2], 
          bias = bias )

  def forward(self, x):
    x = self.conv1(x)
    output = self.fc( F.relu( torch.permute(x, (0,2,1)) ) )
    #output = F.relu(x)
    if self.sum_xai : 
      return torch.permute( output, (0,2,1) ).sum(dim=2)
    else : 
      return torch.permute( output, (0,2,1) )
    

## Lighting module
class CNN1Dpl(pl.LightningModule ): 
    def __init__(self, channels, kernel_size = 5, bias=False,  optimizer = torch.optim.Adam, lr = 0.001, 
          criterion = torch.nn.MSELoss(), sum_xai=False) -> None:

        super().__init__()
        self.model = simple_1dCNN_v2(channels, kernel_size, bias, sum_xai=sum_xai)
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

        # compute loss
        loss_train = self.criterion(src_hat, src)
        self.log("train_loss", loss_train, prog_bar=True, on_step=False, on_epoch=True)    

        return loss_train
      
    def validation_step(self, batch, batch_idx):
        eeg, src = batch
        eeg, src = eeg.float(), src.float()
        src_hat = self.forward(eeg)

        # compute loss
        loss_val = self.criterion(src_hat, src)

        self.log("validation_loss", loss_val, prog_bar=True, on_step=False, on_epoch=True)

        return loss_val

