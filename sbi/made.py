import numpy as np

import torch 
import torch.nn as nn
import torch.nn.functional as F

class MaskedLinear(nn.Linear):
    """ same as Linear except has a configurable mask on the weights """
    
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias)        
        self.register_buffer('mask', torch.ones(out_features, in_features))
        
    def set_mask(self, mask):
        self.mask.data.copy_(torch.from_numpy(mask.astype(np.uint8).T))
        
    def forward(self, input):
        return F.linear(input, self.mask * self.weight, self.bias)

class MADE(nn.Module):
    def __init__(self, n_inputs, n_hidden, n_outputs, n_masks=1, ordering=False):
        super().__init__()
        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.n_outputs = n_outputs
        self.n_masks = n_masks
        self.ordering = ordering

        self.net = []
        hs = [n_inputs] + n_hidden + [n_outputs]
        for h0, h1 in zip(hs, hs[1:]):
            self.net.extend([
                MaskedLinear(h0, h1),
                nn.ReLU(),
            ])

        self.net.pop() # pop the last ReLU for the output layer
        self.net = nn.Sequential(*self.net)

        self.seed = 0

        self.m = {}
        self.update_masks() # builds the initial self.m connectivity
        # note, we could also precompute the masks and cache them, but this
        # could get memory expensive for large number of masks.

    def update_masks(self):
        if self.m and self.n_masks == 1: return # only a single mask, skip for efficiency
        L = len(self.n_hidden)

        # fetch the next seed and construct a random stream
        rng = np.random.RandomState(self.seed)
        self.seed = (self.seed + 1) % self.n_masks

        # sample the order of the inputs and the connectivity of all neurons
        self.m[-1] = np.arange(self.n_inputs) if self.ordering else rng.permutation(self.n_inputs)
        for l in range(L):
            self.m[l] = rng.randint(self.m[l-1].min(), self.n_inputs-1, size=self.n_hidden[l])

        # construct the mask matrices
        masks = [self.m[l-1][:, None] <= self.m[l][None, :] for l in range(L)]
        masks.append(self.m[L-1][:, None] < self.m[-1][None, :])

        # handle the case where n_outpus = n_inputs*k, for integer k > 1
        if self.n_outputs > self.n_inputs:
            k = self.n_outpus // self.n_inputs
            # replicate the mask across the other outputs
            masks[-1] = np.concatenate([masks[-1]] * k, axis=1)

        # set the masks in all MaskedLinear layers
        layers = [l for l in self.net.modules() if isinstance(l, MaskedLinear)]
        for l, m in zip(layers, masks):
            l.set_mask(m)

    def forward(self, x):
        return self.net(x)
    
if __name__=="__main__":

    import IPython

    from torch.autograd import Variable

    n_inputs = 10
    n_hidden = [100, 100, 100]
    n_outputs = 10
    ordering = False

    rng = np.random.RandomState(14)
    x = (rng.rand(1, n_inputs) > 0.5).astype(np.float32)

    model = MADE(n_inputs, n_hidden, n_outputs, 3, ordering)

    # run backpropagation for each dimension to compute what other dimensions it depends on
    res = []

    for k in range(n_outputs):
        xtr = Variable(torch.from_numpy(x), requires_grad=True)
        xtrhat = model(xtr)
        loss = xtrhat[0, k]
        loss.backward()

        depends = (xtr.grad[0].numpy() != 0).astype(np.uint8)
        depends_ix = list(np.where(depends)[0])
        isok = k % n_outputs not in depends_ix
        
        res.append((len(depends_ix), k, depends_ix, isok))
    
    # pretty print the dependencies
    res.sort()
    for nl, k, ix, isok in res:
        print("output %2d depends on inputs: %30s : %s" % (k, ix, "OK" if isok else "NOTOK"))

    IPython.embed()