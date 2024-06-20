import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq import utils


class Adapter(nn.Module):
    def __init__(self, 
        in_dim=512, 
        bottleneck_dim=64, 
        activation="relu", 
        dropout=0., 
        ln_before=True, 
    ):
        super().__init__()
        self.in_dim = in_dim
        self.bottleneck_dim = bottleneck_dim

        self.ln_before = ln_before
        self.layer_norm = nn.LayerNorm(self.in_dim)
        self.dropout_module = nn.Dropout(p=dropout)

        self.down = Linear(self.in_dim, self.bottleneck_dim)
        self.up = Linear(self.bottleneck_dim, self.in_dim)
        self.activation_fn = utils.get_activation_fn(activation)  
            
    def residual_connection(self, x, residual):
        return residual + x

    def forward(self, x):
        residual = x
        if self.ln_before:
            x = self.layer_norm(x)

        x = self.up(self.activation_fn(self.down(x)))

        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.ln_before:
            x = self.layer_norm(x)

        return x
        

class AdapterBlock(nn.Module):
    def __init__(self, 
                 adapter_ids, 
                 in_dim=512,
                 bottleneck_dim=64,
                 activation="relu",
                 dropout=0,
                 ln_before=True):

        super().__init__()
        
        self.adapters = nn.ModuleDict({
            str(id_): Adapter(
                in_dim=in_dim,
                bottleneck_dim=bottleneck_dim,
                activation=activation,
                dropout=dropout,
                ln_before=ln_before,
            ) for id_ in adapter_ids
        })
        self.adapter_ids = adapter_ids
            
    def forward(self, x, adapter_lagging):    
        for i in self.adapter_ids:
            if adapter_lagging >= i:
                adapter_id = str(i)
        return self.adapters[adapter_id](x)


def Linear(in_features, out_features, bias=False):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.0)
    return m


