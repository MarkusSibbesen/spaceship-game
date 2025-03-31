import torch
from torch import nn
import einops

class SAE(nn.Module):

    def __init__(
            self,
            input_size,
            hidden_size,
            k):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.k = k

        initial_W = torch.randn(self.hidden_size, self.input_size) * 0.01
        with torch.no_grad():
            self.W = nn.Parameter(initial_W.clone())
            self.WT = nn.Parameter(initial_W.T.clone())
        
        self.pre_encode_b = nn.Parameter(torch.randn(self.input_size)*0.1)
        self.b1 = nn.Parameter(torch.randn(self.hidden_size)*0.1)  # Bias for encoder

        self.activations = None

    def forward(self, x):
        x = x - self.pre_encode_b
        h = torch.topk(torch.matmul(x, self.WT) + self.b1, k=self.k, dim=-1)
        x_hat = einops.einsum(h.values, self.W[h.indices], 'token topk, token topk out -> token out')
        x_hat += self.pre_encode_b

        return x_hat



class SaeTrainer():
    def __init__(self, input_size, hidden_size, k, learning_rate, device):

        self.device = device

        self.model = SAE(
            input_size=input_size,
            hidden_size=hidden_size,
            k=k
        ).to(self.device)
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()

        self.batches = 0
        self.losses = []


    def train_step(self, input_, labels):
        outputs = self.model(input_).to(self.device)
        loss = self.loss_fn(outputs, labels)

        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.losses.append(loss.item())
        self.batches += 1
        return loss
