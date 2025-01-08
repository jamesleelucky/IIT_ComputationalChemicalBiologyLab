import flax.linen as nn

# Abstract Class 
class BaseEncoder(nn.Module):
    d_hidden: list
    n_latents: int
    activation: callable = nn.sigmoid
    dropout_rates: list = None

    def hidden_layers(self, x):
        for i, d_hidden in enumerate(self.d_hidden):
            x = self.activation(nn.Dense(d_hidden)(x))
            if self.dropout_rates and i < len(self.dropout_rates):
                x = nn.Dropout(rate=self.dropout_rates[i])(x, deterministic=True)
        return x

    # Abstract Method 
    def encode(self):
        raise NotImplementedError("Subclasses must implement this method.")

# Inheritance 
class CustomEncoder(BaseEncoder):
    @nn.compact
    def encode(self, x):
        x = self.hidden_layers(x)
        x = nn.Dense(self.n_latents)(x)
        return x

class CustomAutoEncoder(nn.Module):
    input_size: int
    hidden_layers: tuple
    n_latents: int
    activation: callable = nn.sigmoid
    dropout_rates: list = None

    # choosing activation functions(initialized to sigmoid) and dropout layers by arguments 
    def setup(self):
        self.encoder = CustomEncoder(list(self.hidden_layers), self.n_latents, self.activation, self.dropout_rates)
        
    def __call__(self, x):
        z_latent = self.encoder(x)
        return z_latent

