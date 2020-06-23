import numpy as np
class SimpleModel(nn.Module):
    '''
    A really simple network arcchitecture
    '''
    def __init__(self,input_size, output_size, n_layers,  layer_size, skip_connection=False):
        super(SimpleModel,self).__init__()
        
        self.input_lay=nn.Linear(input_size,layer_size)
        self.intermediate_layers:[nn.Sequential]=[]
        for n in range(n_layers):
            self.intermediate_layers.append(nn.Sequential((nn.Linear(layer_size,layer_size)),nn.ELU()))
        
        self.output_lay=nn.Linear(layer_size,output_size)
    
    def forward(self,x):
   
        x=self.input_lay(x)
        x=F.elu(x)
        for n in self.intermediate_layers:
            x=n(x)
        x=self.output_lay(x)
        return x
    
    
    