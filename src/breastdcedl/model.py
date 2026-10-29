"""Una sola CNN configurable; sin arquitecturas de catálogo ni preentrenados."""
import torch
from torch import nn


def input_to_device(x, device):
    return tuple(v.to(device) for v in x) if isinstance(x, (tuple, list)) else x.to(device)


class ConfigurableCNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        activations = {'relu': nn.ReLU, 'leaky_relu': nn.LeakyReLU, 'gelu': nn.GELU}
        activation = activations[config['activation']]
        channels = 3
        self.bloques = nn.ModuleList()
        self.minimum_side = 2 ** sum(block['pool'] != 'none' for block in config['blocks'])
        for block in config['blocks']:
            layers = []
            for _ in range(block['convolutions']):
                out = block['channels']
                k = block['kernel_size']
                layers.append(nn.Conv2d(channels, out, k, padding=k//2))
                if block['normalization'] == 'batch':
                    layers.append(nn.BatchNorm2d(out))
                elif block['normalization'] == 'group':
                    layers.append(nn.GroupNorm(8, out))
                layers.append(activation())
                channels = out
            if block['pool'] != 'none':
                layers.append(nn.MaxPool2d(2) if block['pool'] == 'max' else nn.AvgPool2d(2))
            self.bloques.append(nn.Sequential(*layers))
        side = config['adaptive_pool_size']
        self.promedio = nn.AdaptiveAvgPool2d(side)
        features = channels * side**2
        hidden = []
        for out in config['hidden_features']:
            hidden.extend([nn.Linear(features, out), activation()])
            features = out
        self.hidden = nn.Sequential(*hidden)
        self.dropout = nn.Dropout(config['dropout'])
        self.clinical = config.get('clinical', 'none') == 'age_menopause'
        self.salida = nn.Linear(features + (4 if self.clinical else 0), 1)
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
                nn.init.zeros_(layer.bias)
            elif isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=1.0)
                nn.init.zeros_(layer.bias)

        self.attention = None
        if config.get('attention', 'none') == 'spatial':
            # Preserve the base initialization and RNG; initial gate is identity.
            with torch.random.fork_rng(devices=[]):
                self.attention = nn.Conv2d(channels, 1, 1)
                nn.init.zeros_(self.attention.weight)
                nn.init.zeros_(self.attention.bias)

    def forward(self, x, return_attention=False):
        clinical = None
        if self.clinical:
            if not isinstance(x, (tuple, list)) or len(x) != 2:
                raise ValueError('Se requieren imagen y cuatro entradas clínicas.')
            x, clinical = x
            if clinical.ndim != 2 or clinical.shape != (x.shape[0], 4):
                raise ValueError('Entrada clínica esperada N,4.')
        if x.ndim != 4 or x.shape[1] != 3 or min(x.shape[2:]) < self.minimum_side:
            raise ValueError(f'Entrada esperada N,3,H,W: PRE/EARLY/LATE; H y W >= {self.minimum_side}.')
        for block in self.bloques:
            x = block(x)
        attention_logits = None
        if self.attention is not None:
            attention_logits = self.attention(x)
            x = x * (2 * torch.sigmoid(attention_logits))
        x = self.dropout(self.hidden(self.promedio(x).flatten(1)))
        if self.clinical:
            x = torch.cat((x, clinical), dim=1)
        logits = self.salida(x)
        return (logits, attention_logits) if return_attention else logits
