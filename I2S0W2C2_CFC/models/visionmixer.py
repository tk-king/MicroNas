import torch
from torch import nn
from torch.nn import functional as F



def pad_image_to_divisible(image, p):
    # 获取图片的高度和宽度
    _, _, h, w = image.shape
    
    # 计算需要的 padding 大小
    pad_h = (p - h % p) % p
    pad_w = (p - w % p) % p
    
    # 对图片进行 padding，使用 `torch.nn.functional.pad`
    # `pad` 参数的格式是 (pad_left, pad_right, pad_top, pad_bottom)
    padding = (0, pad_w, 0, pad_h)
    padded_image = F.pad(image, padding, mode='constant', value=0)
    
    return padded_image




class MLP(nn.Module):
    def __init__(self, num_features, num_hidden, dropout):
        super().__init__()

        self.fc1 = nn.Linear(num_features, num_hidden)
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(num_hidden, num_features)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout1(F.gelu(self.fc1(x)))
        x = self.dropout2(self.fc2(x))
        return x





class TokenMixer(nn.Module):
    def __init__(self, num_patches, embedding_dim, patch_dim, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(embedding_dim)
        self.mlp = MLP(num_patches, patch_dim, dropout)

    def forward(self, x):

        residual = x
        x = self.norm(x)
        x = x.transpose(1, 2)

        x = self.mlp(x)
        x = x.transpose(1, 2)

        out = x + residual
        return out


class ChannelMixer(nn.Module):
    def __init__(self, embedding_dim, filter_dim, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(embedding_dim)
        self.mlp = MLP(embedding_dim, filter_dim, dropout)

    def forward(self, x):
        # x.shape == (batch_size, num_patches, num_features)
        residual = x
        x = self.norm(x)
        x = self.mlp(x)
        # x.shape == (batch_size, num_patches, num_features)
        out = x + residual
        return out



class MixerLayer(nn.Module):
    def __init__(self, num_patches, embedding_dim, patch_dim, filter_dim, dropout):
        super().__init__()
        self.token_mixer = TokenMixer(
            num_patches, embedding_dim, patch_dim, dropout
        )
        self.channel_mixer = ChannelMixer(
            embedding_dim, filter_dim, dropout
        )

    def forward(self, x):

        x = self.token_mixer(x)
        x = self.channel_mixer(x)

        return x
    
class Vision_MIXER(nn.Module):
    def __init__(
        self,
        input_shape,
        patch_size,
        number_class,
        embedding_dim = 256,
        patch_dim = 64,
        filter_dim = 256,
        layer_nr = 5,
        dropout = 0.1
    ):
        super(Vision_MIXER,self).__init__()
        batch,_,window_length,sensor_channel_nr = input_shape
        temp_x = torch.randn(1, 1, window_length, sensor_channel_nr)
        y = pad_image_to_divisible(temp_x,patch_size)
        padded_window_length = y.shape[2]
        padded_sensor_channel = y.shape[3]
        self.num_patches = int(padded_window_length/patch_size)*int(padded_sensor_channel/patch_size)
        self.padded_window_length     = padded_window_length
        self.padded_sensor_channel    = padded_sensor_channel
        self.patch_size               = patch_size
        self.number_class             = number_class
        self.embedding_dim            = embedding_dim
        self.patch_dim                = patch_dim
        self.filter_dim               = filter_dim
        self.layer_nr                 = layer_nr

        self.patcher = nn.Conv2d(
            1, self.embedding_dim, kernel_size=self.patch_size, stride=self.patch_size
        )

        self.mixers = nn.Sequential(
            *[
                MixerLayer(self.num_patches, self.embedding_dim, self.patch_dim, self.filter_dim , dropout)
                for _ in range(self.layer_nr)
            ]
        )

        self.classifier = nn.Linear(embedding_dim, number_class)
    
    def forward(self, x):
        x = pad_image_to_divisible(x,self.patch_size)
        x = self.patcher(x)
        batch_size, num_features, _, _ = x.shape
        x = x.permute(0, 2, 3, 1)
        x = x.view(batch_size, -1, num_features)

        x = self.mixers(x)
        # embedding.shape == (batch_size, num_patches, num_features)
        x = x.mean(dim=1)
        logits = self.classifier(x)
        return logits

