"""
model.py -- the reconstructed residual MLP as a loadable nn.Module.

The reassembled network is:  proj(784->16) -> K residual blocks -> last(16->10),
each block computing  x -> x + W_out @ relu(W_in @ x + b_in) + b_out.

final_model.pth holds this module's state_dict. To load it:

    import torch
    from model import ReconstructedResNet
    net = ReconstructedResNet()
    net.load_state_dict(torch.load("final_model.pth", map_location="cpu"))
    net.eval()
    logits = net(X)            # X: (N, 784) float32  ->  (N, 10) logits
"""
import torch
import torch.nn as nn

INPUT_DIM, LATENT_DIM, HIDDEN_DIM, NUM_CLASSES, NUM_BLOCKS = 784, 16, 32, 10, 32


class ResidualBlock(nn.Module):
    """x -> x + lin_out(relu(lin_in(x)));  lin_in/lin_out carry b_in/b_out as bias."""
    def __init__(self):
        super().__init__()
        self.lin_in = nn.Linear(LATENT_DIM, HIDDEN_DIM)    # W_in (32x16), b_in (32,)
        self.lin_out = nn.Linear(HIDDEN_DIM, LATENT_DIM)   # W_out (16x32), b_out (16,)

    def forward(self, x):
        return x + self.lin_out(torch.relu(self.lin_in(x)))


class ReconstructedResNet(nn.Module):
    def __init__(self, num_blocks: int = NUM_BLOCKS):
        super().__init__()
        self.proj = nn.Linear(INPUT_DIM, LATENT_DIM)
        self.blocks = nn.ModuleList(ResidualBlock() for _ in range(num_blocks))
        self.last = nn.Linear(LATENT_DIM, NUM_CLASSES)

    def forward(self, x):
        x = self.proj(x)
        for blk in self.blocks:
            x = blk(x)
        return self.last(x)

    @classmethod
    def from_reconstruction(cls, S, order, pair):
        """Build the model from the recovered depth `order` (list of W_in files) and
        `pair` (W_in file -> W_out file), using the loaded fragments in S."""
        net = cls(len(order))
        with torch.no_grad():
            net.proj.weight.copy_(S["proj"]["w"]); net.proj.bias.copy_(S["proj"]["b"])
            net.last.weight.copy_(S["last"]["w"]); net.last.bias.copy_(S["last"]["b"])
            for blk, fi in zip(net.blocks, order):
                fo = pair[fi]
                blk.lin_in.weight.copy_(S["inp"][fi]["w"]); blk.lin_in.bias.copy_(S["inp"][fi]["b"])
                blk.lin_out.weight.copy_(S["out"][fo]["w"]); blk.lin_out.bias.copy_(S["out"][fo]["b"])
        return net
