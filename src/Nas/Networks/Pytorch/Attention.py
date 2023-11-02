import torch
import torch.nn as nn

from src.Nas.Layers.Pytorch.SelfAttention import SelfAttention

class Pytorch_Attention(nn.Module):
    def __init__(
        self,
        input_shape ,
        number_class , 
        filter_num = 32,
        filter_size = 5,
        nb_conv_layers = 4,
        dropout = 0.2,
        activation = "ReLU",
        sa_div= 1,
    ):
        super().__init__()
        
        # PART 1 , Channel wise Feature Extraction
        
        layers_conv = []
        for i in range(nb_conv_layers):
        
            if i == 0:
                in_channel = 1
            else:
                in_channel = filter_num
    
            layers_conv.append(nn.Sequential(
                nn.Conv2d(in_channel, filter_num, (filter_size, 1),(2,1)),#(2,1)
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(filter_num),

            ))
        
        self.layers_conv = nn.ModuleList(layers_conv)

        # PART2 , Cross Channel Fusion through Attention
        self.dropout = nn.Dropout(dropout)

        self.sa = SelfAttention(filter_num, sa_div)
        
        shape = self.get_the_shape(input_shape)

        # PART 3 , Prediction 
        
        self.activation = nn.ReLU() 
        self.fc1 = nn.Linear(input_shape[2]*filter_num ,filter_num)
        self.flatten = nn.Flatten()
        self.fc2 = nn.Linear(shape[1]*filter_num ,filter_num)
        self.fc3 = nn.Linear(filter_num ,number_class)

    def get_the_shape(self, input_shape):
        x = torch.rand(input_shape)
        x = x.unsqueeze(1)
        for layer in self.layers_conv:
            x = layer(x)    
        atten_x = torch.cat(
            [self.sa(torch.unsqueeze(x[:, :, t, :], dim=3)) for t in range(x.shape[2])],
            dim=-1,
        )
        atten_x = atten_x.permute(0, 3, 1, 2)
        return atten_x.shape

    def forward(self, x):
        # B L C

        print("inputShape: ", x.shape)
        x = x.unsqueeze(1)
        print("unsqueezed: ", x.shape)
        
        
        for layer in self.layers_conv:
            x = layer(x)      


        batch, filter, length, channel = x.shape 

        print("before_att: ", x.shape)
        print("before_conv_002: ", torch.unsqueeze(x[:, :, 0, :], dim=3).shape)

        # apply self-attention on each temporal dimension (along sensor and feature dimensions)
        refined = torch.cat(
            [self.sa(torch.unsqueeze(x[:, :, t, :], dim=3)) for t in range(x.shape[2])],
            dim=-1,
        )
        print("one_att: ", self.sa(torch.unsqueeze(x[:, :, 0, :], dim=3)).shape)



        print("after att: ", refined.shape)

        x = refined.permute(0, 3, 1, 2)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        print("after reshape: ", x.shape)
        x = self.dropout(x)
        x = self.activation(self.fc1(x)) # B L C
        x = self.flatten(x)
        x = self.activation(self.fc2(x)) # B L C
        y = self.fc3(x)    
        return y