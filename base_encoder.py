import numpy.random as npr
import flax.linen as nn

class BaseEncoder(nn.Module):
    d_hidden: list
    n_latents: int

    def encode(self, x):
        raise NotImplementedError("Subclasses must implement this method.")

    def hidden_layers(self, x):
        for d_hidden in self.d_hidden:
            x = nn.sigmoid(nn.Dense(d_hidden)(x))
        return x

class Sigmoid_Encoder(BaseEncoder):
    @nn.compact
    def encode(self, x):
        x = self.hidden_layers(x)
        x = nn.Dense(self.n_latents)(x)
        return x

class Sigmoid_Dropout_Encoder(BaseEncoder):
    latents: int
    dropout_rates: list
    
    @nn.compact
    def encode(self, x):
        for i in range(len(self.d_hidden)):
            x = nn.sigmoid(nn.Dense(self.d_hidden[i])(x))
            x = nn.Dropout(rate=self.dropout_rates[i])(x, deterministic=True)
        x = nn.Dense(self.latents, name='f5')(x)
        return x

class Softmax_Sigmoid_Encoder(BaseEncoder):
    @nn.compact
    def encode(self, x):
        x = self.hidden_layers(x)
        x = nn.softmax(nn.Dense(self.n_latents)(x))
        return x