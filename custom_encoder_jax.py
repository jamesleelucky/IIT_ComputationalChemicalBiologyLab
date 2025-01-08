from abc import ABC, abstractmethod
import jax

class BaseEncoder(ABC):
    def __init__(self, d_hidden, n_latents, activation=jax.nn.sigmoid, dropout_rates=None):
        self.d_hidden = d_hidden
        self.n_latents = n_latents
        self.activation = activation
        self.dropout_rates = dropout_rates

    @abstractmethod
    def encode(self, x):
        pass

    def hidden_layers(self, x):
        for i, hidden_size in enumerate(self.d_hidden):
            x = self.activation(jax.nn.dense(x, hidden_size))
            if self.dropout_rates and i < len(self.dropout_rates):
                x = jax.nn.dropout(x, rate=self.dropout_rates[i], deterministic=True)
        return x

class CustomEncoder(BaseEncoder):
    def encode(self, x):
        x = self.hidden_layers(x)
        x = jax.nn.dense(x, self.n_latents)
        return x

class CustomAutoEncoder:
    def __init__(self, input_size, hidden_layers, n_latents, activation=jax.nn.sigmoid, dropout_rates=None):
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.n_latents = n_latents
        self.activation = activation
        self.dropout_rates = dropout_rates
        self.encoder = CustomEncoder(hidden_layers, n_latents, activation, dropout_rates)

    def __call__(self, x):
        z_latent = self.encoder.encode(x)
        return z_latent
