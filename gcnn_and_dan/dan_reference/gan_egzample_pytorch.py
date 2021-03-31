from sklearn.datasets import make_moons, make_circles
import numpy as np

def make_dataset(test_config={}):
   mode = test_config.get('mode', 'moons')
   noise = test_config.get('noise', 0.15)
   size = test_config.get('size',10000)
   if mode=='moons':
      return make_moons(size, noise=noise)
   elif mode=='circles':
      factor = test_config.get('factor',0.5)
      return make_circles(size, noise=noise, factor=factor)
   else:
      raise KeyError('Unknown mode: %s'%str(mode))


def relabel_positive(y, fraction=0.1):
   positive_idx = np.where(y==1)[0]
   sample = int(len(positive_idx)*fraction)
   to_change = np.random.choice(positive_idx, sample, replace=False)
   y2 = np.ones(y.shape).astype(int)*y
   y2[to_change] = 0
   return y2


import torch
import torch.nn.functional as nn
import torch.autograd as autograd
import torch.optim as optim
from torch.autograd import Variable


mnist = input_data.read_data_sets('../../MNIST_data', one_hot=True)
mb_size = 64
Z_dim = 100
X_dim = mnist.train.images.shape[1]
y_dim = mnist.train.labels.shape[1]
h_dim = 128
c = 0
lr = 1e-3


def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / np.sqrt(in_dim / 2.)
    return Variable(torch.randn(*size) * xavier_stddev, requires_grad=True)


""" ==================== GENERATOR ======================== """
Wzh = xavier_init(size=[Z_dim, h_dim])
bzh = Variable(torch.zeros(h_dim), requires_grad=True)

Whx = xavier_init(size=[h_dim, X_dim])
bhx = Variable(torch.zeros(X_dim), requires_grad=True)


def G(z):
   h = nn.relu(z @ Wzh + bzh.repeat(z.size(0), 1))
   X = nn.sigmoid(h @ Whx + bhx.repeat(h.size(0), 1))
   return X


""" ==================== DISCRIMINATOR ======================== """

Wxh = xavier_init(size=[X_dim, h_dim])
bxh = Variable(torch.zeros(h_dim), requires_grad=True)

Why = xavier_init(size=[h_dim, 1])
bhy = Variable(torch.zeros(1), requires_grad=True)


def D(X):
    h = nn.relu(X @ Wxh + bxh.repeat(X.size(0), 1))
    y = nn.sigmoid(h @ Why + bhy.repeat(h.size(0), 1))
    return y


G_params = [Wzh, bzh, Whx, bhx]
D_params = [Wxh, bxh, Why, bhy]
params = G_params + D_params

""" ===================== TRAINING ======================== """


def reset_grad():
    for p in params:
       if p.grad is not None:
          data = p.grad.data
          p.grad = Variable(data.new().resize_as_(data).zero_())


G_solver = optim.Adam(G_params, lr=1e-3)
D_solver = optim.Adam(D_params, lr=1e-3)

ones_label = Variable(torch.ones(mb_size, 1))
zeros_label = Variable(torch.zeros(mb_size, 1))

for it in range(100000):
   # Sample data
   z = Variable(torch.randn(mb_size, Z_dim))
   X, _ = mnist.train.next_batch(mb_size)
   X = Variable(torch.from_numpy(X))

   # Dicriminator forward-loss-backward-update
   G_sample = G(z)
   D_real = D(X)
   D_fake = D(G_sample)

   D_loss_real = nn.binary_cross_entropy(D_real, ones_label)
   D_loss_fake = nn.binary_cross_entropy(D_fake, zeros_label)
   D_loss = D_loss_real + D_loss_fake

   D_loss.backward()
   D_solver.step()

   # Housekeeping - reset gradient
   reset_grad()

   # Generator forward-loss-backward-update
   z = Variable(torch.randn(mb_size, Z_dim))
   G_sample = G(z)
   D_fake = D(G_sample)

   G_loss = nn.binary_cross_entropy(D_fake, ones_label)

   G_loss.backward()
   G_solver.step()

   # Housekeeping - reset gradient
   reset_grad()

   # Print and plot every now and then
   if it % 1000 == 0:
      print('Iter-{}; D_loss: {}; G_loss: {}'.format(it, D_loss.data.numpy(), G_loss.data.numpy()))


