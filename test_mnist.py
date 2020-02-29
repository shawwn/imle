import os

import numpy as np
import torch
import torchvision
from torch import nn
from torch.nn import Parameter
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import MNIST
from torchvision.datasets import CIFAR10
from torchvision import transforms
import tqdm

import pytorch_lightning as pl

import sys
sys.path.append('./dci_code_mac')
from dci import DCI

import collections
Hyperparams = collections.namedtuple('Hyperarams', 'base_lr batch_size num_epochs decay_step decay_rate staleness num_samples_factor train_percent')
Hyperparams.__new__.__defaults__ = (None, None, None, None, None, None, None, None)

class Namespace():
  pass

me = Namespace()
#me.batch_size = 32
me.z_dim = 128
me.batch_size = 64
me.max_step = 2.0/me.batch_size
#me.radius = 0.1
#me.radius = 0.5
me.radius = 2.0
me.interp_every = 10


def one_hot_encode(x, n_class):
  from torch.autograd import Variable
  #class_emb = self.linear(class_id)  # 128
  # https://discuss.pytorch.org/t/convert-int-into-one-hot-format/507/11
  e = nn.Embedding(n_class, n_class) 
  e.weight.data = torch.eye(n_class)
  class_emb = e(Variable(torch.LongTensor(x)))
  return class_emb

def latent_and_labels(z, class_id, n_class):
  class_emb = one_hot_encode(class_id, n_class)
  z = torch.cat([class_emb.reshape(class_emb.shape[0], class_emb.shape[1], 1, 1), z], 1)
  return z

# https://stackoverflow.com/questions/29831489/convert-array-of-indices-to-1-hot-encoded-numpy-array
def np_one_hot(a, n_class):
  #b = np.zeros((a.size, a.max()+1))
  b = np.zeros((a.size, n_class))
  b[np.arange(a.size),a] = 1
  return b

def np_latent_and_labels(z, class_id, n_class):
  class_emb = np_one_hot(np.array(class_id), n_class=n_class)
  #z = np.concatenate([class_emb.reshape(class_emb.shape[0], class_emb.shape[1], 1, 1), z], 1)
  z = np.concatenate([class_emb, z], 1)
  return z

def rand_latent(class_id=None,z_dim=None,n_class=10):
  if z_dim is None:
    z_dim = me.z_dim
  z = np.random.randn(1, z_dim)#.reshape([1,-1,1,1])
  if class_id is not None:
    z = np_latent_and_labels(z, class_id, n_class=n_class)
  return z

#Gs = Gs_network
#rnd = np.random
#shape = [128, 128, 1, 1]
#latents_a = rnd.randn(1, shape[1])
#latents_b = rnd.randn(1, shape[1])
#latents_c = rnd.randn(1, shape[1])
#import pdb; pdb.set_trace()
me.latents_a = rand_latent()
me.latents_b = rand_latent()
me.latents_c = rand_latent()
import pdb; pdb.set_trace()

import math

def vdist(v):
  v = v.flatten()
  return np.dot(v,v)**0.5

def vnorm(v):
    return v/vdist(v)

def circ_generator(latents_interpolate):
    #radius = 40.0
    #radius = 0.1
    radius = me.radius
    latents_axis_x = (me.latents_a - me.latents_b).flatten() / vdist(me.latents_a - me.latents_b)
    latents_axis_y = (me.latents_a - me.latents_c).flatten() / vdist(me.latents_a - me.latents_c)
    latents_x = math.sin(math.pi * 2.0 * latents_interpolate) * radius
    latents_y = math.cos(math.pi * 2.0 * latents_interpolate) * radius
    latents = me.latents_a + latents_x * latents_axis_x + latents_y * latents_axis_y
    class_id=1
    n_class=10
    latents = np_latent_and_labels(latents, class_id, n_class=n_class)
    return latents

me.circ_generator = circ_generator

def circle_interpolation(model, count, gen_func=None, max_step=None, change_min=10.0, change_max=11.0):
    if gen_func is None:
      gen_func = me.circ_generator
    if max_step is None:
      max_step = me.max_step

    def gen_latent(pos):
      z = gen_func(pos)
      z = z.reshape([1,-1,1,1])
      return z

    def generate(current_latent):
        #fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
        #current_image = Gs.run(current_latent, None, truncation_psi=0.5, randomize_noise=False, output_transform=fmt)[0]
        z = np.array(current_latent)
        z = torch.from_numpy(z).float().cpu()
        imgs = model(z)
        return imgs

    def build_latent(count, current_pos=0.0):
      current_latent = gen_latent(current_pos)
      yield current_latent
      for i in range(count - 1):
          lower = current_pos
          upper = current_pos + max_step
          current_pos = (upper + lower) / 2.0
          current_latent = gen_latent(current_pos)
          yield current_latent

    def get_latents(count):
      z = list(build_latent(count))
      z = np.concatenate(z, axis=0)
      return z

    z = get_latents(count)
    return generate(z)

me.circle_interpolation = circle_interpolation

def mse(x, y):
    return (np.square(x - y)).mean()

def l2normalize(v, eps=1e-4):
  return v / (v.norm() + eps)

class SpectralNorm(nn.Module):
  def __init__(self, module, name='weight', power_iterations=1):
    super(SpectralNorm, self).__init__()
    self.module = module
    self.name = name
    self.power_iterations = power_iterations
    if not self._made_params():
      self._make_params()

  def _update_u_v(self):
    u = getattr(self.module, self.name + "_u")
    v = getattr(self.module, self.name + "_v")
    w = getattr(self.module, self.name + "_bar")

    height = w.data.shape[0]
    _w = w.view(height, -1)
    for _ in range(self.power_iterations):
      v = l2normalize(torch.matmul(_w.t(), u))
      u = l2normalize(torch.matmul(_w, v))

    sigma = u.dot((_w).mv(v))
    setattr(self.module, self.name, w / sigma.expand_as(w))

  def _made_params(self):
    try:
      getattr(self.module, self.name + "_u")
      getattr(self.module, self.name + "_v")
      getattr(self.module, self.name + "_bar")
      return True
    except AttributeError:
      return False

  def _make_params(self):
    w = getattr(self.module, self.name)

    height = w.data.shape[0]
    width = w.view(height, -1).data.shape[1]

    u = Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
    v = Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
    u.data = l2normalize(u.data)
    v.data = l2normalize(v.data)
    w_bar = Parameter(w.data)

    del self.module._parameters[self.name]
    self.module.register_parameter(self.name + "_u", u)
    self.module.register_parameter(self.name + "_v", v)
    self.module.register_parameter(self.name + "_bar", w_bar)

  def forward(self, *args):
    self._update_u_v()
    return self.module.forward(*args)

class SelfAttention(nn.Module):
  """ Self Attention Layer"""

  def __init__(self, in_dim, activation=F.relu):
    super().__init__()
    self.chanel_in = in_dim
    self.activation = activation

    self.theta = SpectralNorm(nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1, bias=False))
    self.phi = SpectralNorm(nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1, bias=False))
    self.pool = nn.MaxPool2d(2, 2)
    self.g = SpectralNorm(nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 2, kernel_size=1, bias=False))
    self.o_conv = SpectralNorm(nn.Conv2d(in_channels=in_dim // 2, out_channels=in_dim, kernel_size=1, bias=False))
    self.gamma = nn.Parameter(torch.zeros(1))

    self.softmax = nn.Softmax(dim=-1)

  def forward(self, x):
    m_batchsize, C, width, height = x.size()
    N = height * width

    theta = self.theta(x)
    phi = self.phi(x)
    phi = self.pool(phi)
    phi = phi.view(m_batchsize, -1, N // 4)
    theta = theta.view(m_batchsize, -1, N)
    theta = theta.permute(0, 2, 1)
    attention = self.softmax(torch.bmm(theta, phi))
    g = self.pool(self.g(x)).view(m_batchsize, -1, N // 4)
    attn_g = torch.bmm(g, attention.permute(0, 2, 1)).view(m_batchsize, -1, width, height)
    out = self.o_conv(attn_g)
    return self.gamma * out + x


class ConditionalBatchNorm2d(nn.Module):
  def __init__(self, num_features, num_classes, eps=1e-4, momentum=0.1):
    super().__init__()
    self.num_features = num_features
    self.bn = nn.BatchNorm2d(num_features, affine=False, eps=eps, momentum=momentum)
    self.gamma_embed = SpectralNorm(nn.Linear(num_classes, num_features, bias=False))
    self.beta_embed = SpectralNorm(nn.Linear(num_classes, num_features, bias=False))

  def forward(self, x, y):
    out = self.bn(x)
    gamma = self.gamma_embed(y) + 1
    beta = self.beta_embed(y)
    out = gamma.view(-1, self.num_features, 1, 1) * out + beta.view(-1, self.num_features, 1, 1)
    return out

class CBatchNorm2d(nn.Module):
  def __init__(self, num_features, z_dim=148):
    super().__init__()
    self.num_features = num_features
    self.HyperBN = ConditionalBatchNorm2d(in_channel, z_dim)
    self.HyperBN_1 = ConditionalBatchNorm2d(out_channel, z_dim)


class GBlock(nn.Module):
  def __init__(
    self,
    in_channel,
    out_channel,
    kernel_size=[3, 3],
    padding=1,
    stride=1,
    n_class=None,
    bn=True,
    activation=F.relu,
    upsample=True,
    downsample=False,
    z_dim=148,
  ):
    super().__init__()

    self.conv0 = SpectralNorm(
      nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding, bias=True if bn else True)
    )
    self.conv1 = SpectralNorm(
      nn.Conv2d(out_channel, out_channel, kernel_size, stride, padding, bias=True if bn else True)
    )

    self.skip_proj = False
    if in_channel != out_channel or upsample or downsample:
      self.conv_sc = SpectralNorm(nn.Conv2d(in_channel, out_channel, 1, 1, 0))
      self.skip_proj = True

    self.upsample = upsample
    self.downsample = downsample
    self.activation = activation
    self.bn = bn
    if bn:
      self.HyperBN = ConditionalBatchNorm2d(in_channel, z_dim)
      self.HyperBN_1 = ConditionalBatchNorm2d(out_channel, z_dim)

  def forward(self, input, condition=None):
    out = input

    if self.bn:
      out = self.HyperBN(out, condition)
    out = self.activation(out)
    if self.upsample:
      out = F.interpolate(out, scale_factor=2)
    out = self.conv0(out)
    if self.bn:
      out = self.HyperBN_1(out, condition)
    out = self.activation(out)
    out = self.conv1(out)

    if self.downsample:
      out = F.avg_pool2d(out, 2)

    if self.skip_proj:
      skip = input
      if self.upsample:
        skip = F.interpolate(skip, scale_factor=2)
      skip = self.conv_sc(skip)
      if self.downsample:
        skip = F.avg_pool2d(skip, 2)
    else:
      skip = input
    return out + skip

class Generator128(nn.Module):
  def __init__(self, code_dim=120, n_class=1000, chn=96, debug=False):
    super().__init__()

    self.linear = nn.Linear(n_class, 128, bias=False)

    if debug:
      chn = 8

    self.first_view = 16 * chn

    self.G_linear = SpectralNorm(nn.Linear(20, 4 * 4 * 16 * chn))

    z_dim = code_dim + 28

    self.GBlock = nn.ModuleList([
      GBlock(16 * chn, 16 * chn, n_class=n_class, z_dim=z_dim),
      GBlock(16 * chn, 8 * chn, n_class=n_class, z_dim=z_dim),
      GBlock(8 * chn, 4 * chn, n_class=n_class, z_dim=z_dim),
      GBlock(4 * chn, 2 * chn, n_class=n_class, z_dim=z_dim),
      GBlock(2 * chn, 1 * chn, n_class=n_class, z_dim=z_dim),
    ])

    self.sa_id = 4
    self.num_split = len(self.GBlock) + 1
    self.attention = SelfAttention(2 * chn)
    self.ScaledCrossReplicaBN = nn.BatchNorm2d(1 * chn, eps=1e-4)
    self.colorize = SpectralNorm(nn.Conv2d(1 * chn, 3, [3, 3], padding=1))

    #self.embed = nn.Embedding(n_class, 16 * chn)
    #self.embed.weight.data.uniform_(-0.1, 0.1)
    #self.embed = SpectralNorm(self.embed)
    

  def forward(self, input, class_id):
    codes = torch.chunk(input, self.num_split, 1)
    #classes = torch.chunk(class_id, self.num_split, 0)
    #emb = self.embed(class_id)  # 128
    from torch.autograd import Variable
    #class_emb = self.linear(class_id)  # 128
    # https://discuss.pytorch.org/t/convert-int-into-one-hot-format/507/11
    e = nn.Embedding(10, 10) 
    e.weight.data = torch.eye(10)
    class_emb = e(Variable(torch.LongTensor(class_id)))
    import pdb; pdb.set_trace()

    out = self.G_linear(codes[0])
    out = out.view(-1, 4, 4, self.first_view).permute(0, 3, 1, 2)
    for i, (code, GBlock) in enumerate(zip(codes[1:], self.GBlock)):
      if i == self.sa_id:
        out = self.attention(out)
      condition = torch.cat([code, class_emb], 1)
      out = GBlock(out, condition)

    out = self.ScaledCrossReplicaBN(out)
    out = F.relu(out)
    out = self.colorize(out)
    return torch.tanh(out)

class ConvolutionalImplicitModel(nn.Module):
    def __init__(self, z_dim, n_class=10):
        super(ConvolutionalImplicitModel, self).__init__()
        self.n_class = n_class
        #self.tconv0 = nn.ConvTranspose2d(z_dim+n_class, 4096, 1, 1, bias=False)
        #self.bn0 = nn.BatchNorm2d(4096)
        self.tconv0 = None
        self.bn0 = None
        self.tconv1 = nn.ConvTranspose2d(z_dim+n_class, 1024, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(1024)
        self.tconv2 = nn.ConvTranspose2d(1024, 128, 8, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(128)
        self.tconv3 = nn.ConvTranspose2d(128, 64, 4, 2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(64)
        self.tconv4 = nn.ConvTranspose2d(64, 3, 4, 2, padding=1, bias=False)
        self.bn4 = None
        self.relu = nn.ReLU(True)
        chn = 32
        self.attention0 = None # SelfAttention(2 * chn)
        self.attention1 = None # SelfAttention(2 * chn)
        self.attention2 = None # SelfAttention(2 * chn)
        self.attention3 = SelfAttention(2 * chn)
        self.attention4 = None # SelfAttention(2 * chn)
        #self.linear = nn.Linear(n_class, 128, bias=False)
        
    def forward(self, z, class_id=None):
        #class_emb = self.linear(class_id)  # 128
        if class_id is not None:
          z = latent_and_labels(z, class_id, self.n_class)
        #import pdb; pdb.set_trace()
        if self.bn0:
          z = self.relu(self.bn0(self.tconv0(z)))
        if self.attention0: z = self.attention0(z)
        z = self.relu(self.bn1(self.tconv1(z)))
        if self.attention1: z = self.attention1(z)
        z = self.relu(self.bn2(self.tconv2(z)))
        if self.attention2: z = self.attention2(z)
        z = self.relu(self.bn3(self.tconv3(z)))
        if self.attention3: z = self.attention3(z)
        z = self.tconv4(z)
        if self.attention4: z = self.attention4(z)
        z = torch.sigmoid(z)
        return z



def pearsonr(x, y):
  mean_x = torch.mean(x)
  mean_y = torch.mean(y)
  xm = x.sub(mean_x)
  ym = y.sub(mean_y)
  r_num = xm.dot(ym)
  r_den = torch.norm(xm, 2) * torch.norm(ym, 2)
  r_val = r_num / r_den
  return r_val


class NetPerLayer(nn.Module):
  def __init__(self):
    super(NetPerLayer, self).__init__()

    net = torchvision.models.resnet18(pretrained=True)
    net.eval()

    self.nodeLevels = nn.ModuleList()
    self.nodeLevels.append(nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool))
    self.nodeLevels.append(net.layer1)
    self.nodeLevels.append(net.layer2)
    self.nodeLevels.append(net.layer3)
    self.nodeLevels.append(net.layer4)
    #self.nodeLevels.append(nn.MaxPool2d(8))

  def forward(self, x):
    activations = []
    for m in self.nodeLevels:
      x = m(x)
      activations.append(x)
    return activations

class CoolSystem(pl.LightningModule):

    def __init__(self, z_dim, hyperparams, shuffle_data=True):
        super(CoolSystem, self).__init__()
        # not the best model...
        #self.l1 = torch.nn.Linear(28 * 28, 10)
        self.z_dim = z_dim
        self.z_grid = None
        self.hyperparams = hyperparams
        self.shuffle_data = shuffle_data
        self.dci_db = None
        self.model = ConvolutionalImplicitModel(z_dim)
        #self.model = Generator128(z_dim, n_class=10)
        #self.squeezeNet = NetPerLayer()
        self.squeezeNet = None
        self.loss_fn = nn.MSELoss()
        self._step = 0

    def regen(self, batch):
        #import pdb; pdb.set_trace()

        imgs, labels = batch
        data_np = imgs.numpy()
        #import pdb;  pdb.set_trace()
        data_flat_np = np.reshape(data_np, (data_np.shape[0], np.prod(data_np.shape[1:])))
        
        self.data_np = data_np
        self.data_flat_np = data_flat_np
        hyperparams = self.hyperparams
        
        batch_size = hyperparams.batch_size
        #num_batches = data_np.shape[0] // batch_size
        num_batches = 1
        num_samples = num_batches * hyperparams.num_samples_factor
        #import pdb; pdb.set_trace()
      
        z_np = np.empty((num_samples * batch_size, self.z_dim, 1, 1))
        samples_np = np.empty((num_samples * batch_size,)+data_np.shape[1:])
        for i in range(num_samples):
          z = torch.randn(batch_size, self.z_dim, 1, 1).cpu()
          samples = self.model(z, labels)
          z_np[i*batch_size:(i+1)*batch_size] = z.cpu().data.numpy()
          samples_np[i*batch_size:(i+1)*batch_size] = samples.cpu().data.numpy()
        
        samples_flat_np = np.reshape(samples_np, (samples_np.shape[0], np.prod(samples_np.shape[1:]))).copy()
        
        if self.dci_db is None:
          self.dci_db = DCI(np.prod(data_np.shape[1:]), num_comp_indices = 2, num_simp_indices = 7)
        
        self.dci_db.reset()
        self.dci_db.add(samples_flat_np, num_levels = 2, field_of_view = 10, prop_to_retrieve = 0.002)
        nearest_indices, _ = self.dci_db.query(data_flat_np, num_neighbours = 1, field_of_view = 20, prop_to_retrieve = 0.02)
        nearest_indices = np.array(nearest_indices)[:,0]
        
        z_np = z_np[nearest_indices]
        z_np += 0.01*np.random.randn(*z_np.shape)
        self.z_np = z_np
        if self.z_grid is None:
          self.z_grid = self.z_np.copy() * 0.7
        
        del samples_np, samples_flat_np

        return z_np

    # def forward(self, x):
    #     return torch.relu(self.l1(x.view(x.size(0), -1)))

    def forward(self, z, class_id=None):
        cur_samples = self.model(z, class_id)
        return cur_samples

    def training_step(self, batch, batch_idx):
        # REQUIRED
        #x, y = batch
        #y_hat = self.forward(x)
        #loss = F.cross_entropy(y_hat, y)

        #batch_size = batch.shape[0]
        hyperparams = self.hyperparams
        batch_size = hyperparams.batch_size
        i = batch_idx
        data_np = self.data_np

        #z_np = self.regen()
        z_np = self.z_np
        z_grid = self.z_grid
        #cur_z = torch.from_numpy(z_np[i*batch_size:(i+1)*batch_size]).float().cpu()
        #imgTarget = torch.from_numpy(data_np[i*batch_size:(i+1)*batch_size]).float().cpu()
        cur_z = torch.from_numpy(z_np).float().cpu()
        imgTarget = torch.from_numpy(data_np).float().cpu()
        print(cur_z.shape, batch_idx)
        def nimg(img):
          #img = img + 1
          #img = img / 2
          #import pdb; pdb.set_trace()
          img = torch.clamp(img, 0, 1)
          return img
        imgInput = batch[0]
        imgLabels = batch[1]
        self.logger.experiment.add_image('imgInput', torchvision.utils.make_grid(nimg(imgInput)), self._step)
        imgOutput = self.forward(cur_z, imgLabels)
        self.logger.experiment.add_image('imgOutput', torchvision.utils.make_grid(nimg(imgOutput)), self._step)
        if self._step % me.interp_every == 0 and self._step > -1 and True:
          cur_zgrid = torch.from_numpy(z_grid).float().cpu()
          imgGrid = self.forward(cur_zgrid, imgLabels)
          self.logger.experiment.add_image('imgGrid', torchvision.utils.make_grid(nimg(imgGrid)), self._step)
          #print('circle...')
          #import pdb; pdb.set_trace()
          imgs = me.circle_interpolation(self.model, imgInput.shape[0])
          #imgs = np.array(imgs)
          #imgs = torch.from_numpy(imgs).float().cpu()
          self.logger.experiment.add_image('imgInterp', torchvision.utils.make_grid(nimg(imgs)), self._step)
          #import pdb; pdb.set_trace()
        if self.squeezeNet is not None:
          #print('squeezeNet(imgTarget)...')
          activationsTarget = self.squeezeNet(imgTarget.repeat(1,3,1,1))
          #print('squeezeNet(imgOutput)...')
          activationsOutput = self.squeezeNet(imgOutput.repeat(1,3,1,1))
          #print('loss...')
          featLoss = None
          #for actTarget, actOutput in zip(activationsTarget[1:3], activationsOutput[1:3]):
          #for actTarget, actOutput in tqdm.tqdm(list(zip(activationsTarget, activationsOutput))):
          for actTarget, actOutput in zip(activationsTarget, activationsOutput):
            #l = F.mse_loss(actTarget, actOutput)
            #l = torch.abs(actTarget - actOutput).sum()
            #l = F.l1_loss(actTarget, actOutput)

            l = -pearsonr(actTarget.view(-1), actOutput.view(-1))
            if featLoss is None:
              featLoss = l
            else:
              featLoss += l
        else:
          featLoss = 0.0
        #pixelLoss = F.mse_loss(imgOutput, imgTarget)
        pixelLoss = self.loss_fn(imgOutput, imgTarget)
        loss = featLoss + pixelLoss
        lr = self.lr_fn(self.current_epoch)
        tensorboard_logs = {'train_loss': loss, 'lr': lr, 'epoch': self.current_epoch}
        self._step += 1
        return {'loss': loss, 'log': tensorboard_logs}

    def number_interpolation(self, count, lo, hi):
        hyperparams = self.hyperparams
        batch_size = hyperparams.batch_size

        def gen_latent(pos):
          z = gen_func(pos)
          z = z.reshape([1,-1,1,1])
          return z

#     def validation_step(self, batch, batch_idx):
#         # OPTIONAL
#         x, y = batch
#         y_hat = self.forward(x)
#         return {'val_loss': F.cross_entropy(y_hat, y)}

#     def validation_end(self, outputs):
#         # OPTIONAL
#         avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
#         tensorboard_logs = {'val_loss': avg_loss}
#         return {'avg_val_loss': avg_loss, 'log': tensorboard_logs}
        
#     def test_step(self, batch, batch_idx):
#         # OPTIONAL
#         x, y = batch
#         y_hat = self.forward(x)
#         return {'test_loss': F.cross_entropy(y_hat, y)}

#     def test_end(self, outputs):
#         # OPTIONAL
#         avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
#         tensorboard_logs = {'test_loss': avg_loss}
#         return {'avg_test_loss': avg_loss, 'log': tensorboard_logs}

    def configure_optimizers(self):
        # REQUIRED
        # can return multiple optimizers and learning_rate schedulers
        # (LBFGS it is automatically supported, no need for closure function)
        #return torch.optim.Adam(self.parameters(), lr=0.0004)
        #epoch = 0
        hyperparams = self.hyperparams
        lr = hyperparams.base_lr # * hyperparams.decay_rate ** (epoch // hyperparams.decay_step)
        self.lr_fn = lambda epoch: hyperparams.decay_rate ** (epoch // hyperparams.decay_step)
        #optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=1e-5)
        #optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, amsgrad=True)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, betas=(0.5, 0.999), amsgrad=True, weight_decay=1e-5)
        #optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, betas=(0.5, 0.999), amsgrad=True)
        #optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, betas=(0.0, 0.999))
        from torch.optim.lr_scheduler import LambdaLR
        scheduler = LambdaLR(optimizer, lr_lambda=[self.lr_fn])
        return [optimizer], [scheduler]
        

    def prepare_data(self):
        #MNIST(os.getcwd(), train=True, download=True, transform=transforms.ToTensor())
        #MNIST(os.getcwd(), train=False, download=True, transform=transforms.ToTensor())
        CIFAR10(os.getcwd(), train=True, download=True, transform=transforms.ToTensor())
        CIFAR10(os.getcwd(), train=False, download=True, transform=transforms.ToTensor())

        #self.data_np = np.random.randn(128, 1, 28, 28)
        # self.data_np = np.array([img.numpy() for img, label in dataset])
        # print('ready')
        # data_np = self.data_np
        # hyperparams = self.hyperparams

        # if self.shuffle_data:
        #     data_ordering = np.random.permutation(data_np.shape[0])
        #     data_np = data_np[data_ordering]

        # batch_size = hyperparams.batch_size
        # num_batches = data_np.shape[0] // batch_size
        # num_samples = num_batches * hyperparams.num_samples_factor
        
        # self.data_flat_np = np.reshape(data_np, (data_np.shape[0], np.prod(data_np.shape[1:])))

    def on_batch_start(self, batch):
        self.regen(batch)
        

    def train_dataloader(self):
        # REQUIRED
        #dataset = MNIST(os.getcwd(), train=True, download=False, transform=transforms.ToTensor())
        #dataset = CIFAR10(os.getcwd(), train=True, download=False, transform=transforms.ToTensor())
        #loader = DataLoader(dataset, batch_size=32)

        # Downloading/Louding CIFAR10 data
        trainset  = CIFAR10(root=os.getcwd(), train=True , download=False)#, transform = transform_with_aug)
        testset   = CIFAR10(root=os.getcwd(), train=False, download=False)#, transform = transform_no_aug)
        classDict = {'plane':0, 'car':1, 'bird':2, 'cat':3, 'deer':4, 'dog':5, 'frog':6, 'horse':7, 'ship':8, 'truck':9}

        # Separating trainset/testset data/label
        x_train  = trainset.data
        x_test   = testset.data
        y_train  = trainset.targets
        y_test   = testset.targets

        # Define a function to separate CIFAR classes by class index

        def get_class_i(x, y, i):
            """
            x: trainset.train_data or testset.test_data
            y: trainset.train_labels or testset.test_labels
            i: class label, a number between 0 to 9
            return: x_i
            """
            # Convert to a numpy array
            y = np.array(y)
            # Locate position of labels that equal to i
            pos_i = np.argwhere(y == i)
            # Convert the result into a 1-D list
            pos_i = list(pos_i[:,0])
            # Collect all data that match the desired label
            x_i = [x[j] for j in pos_i]
            
            return x_i

        class DatasetMaker(Dataset):
            def __init__(self, datasets, transformFunc = transforms.ToTensor()):
                """
                datasets: a list of get_class_i outputs, i.e. a list of list of images for selected classes
                """
                self.datasets = datasets
                self.lengths  = [len(d) for d in self.datasets]
                self.transformFunc = transformFunc
            def __getitem__(self, i):
                class_label, index_wrt_class = self.index_of_which_bin(self.lengths, i)
                img = self.datasets[class_label][index_wrt_class]
                if self.transformFunc:
                  img = self.transformFunc(img)
                return img, class_label

            def __len__(self):
                return sum(self.lengths)
            
            def index_of_which_bin(self, bin_sizes, absolute_index, verbose=False):
                """
                Given the absolute index, returns which bin it falls in and which element of that bin it corresponds to.
                """
                # Which class/bin does i fall into?
                accum = np.add.accumulate(bin_sizes)
                if verbose:
                    print("accum =", accum)
                bin_index  = len(np.argwhere(accum <= absolute_index))
                if verbose:
                    print("class_label =", bin_index)
                # Which element of the fallent class/bin does i correspond to?
                index_wrt_class = absolute_index - np.insert(accum, 0, 0)[bin_index]
                if verbose:
                    print("index_wrt_class =", index_wrt_class)

                return bin_index, index_wrt_class

        # ================== Usage ================== #

        # Let's choose cats (class 3 of CIFAR) and dogs (class 5 of CIFAR) as trainset/testset
        cat_dog_trainset = \
            DatasetMaker(
                [get_class_i(x_train, y_train, classDict['cat']), get_class_i(x_train, y_train, classDict['dog'])]
                #transform_with_aug
            )
        cat_dog_testset  = \
            DatasetMaker(
                [get_class_i(x_test , y_test , classDict['cat']), get_class_i(x_test , y_test , classDict['dog'])]
                #transform_no_aug
            )

        kwargs = {'num_workers': 2, 'pin_memory': False}
        hyperparams = self.hyperparams
        batch_size = hyperparams.batch_size

        # Create datasetLoaders from trainset and testset
        trainsetLoader   = DataLoader(cat_dog_trainset, batch_size=batch_size, shuffle=True , **kwargs)
        testsetLoader    = DataLoader(cat_dog_testset , batch_size=batch_size, shuffle=False, **kwargs)

        #loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        # self.data_np = np.array([img.numpy() for img, label in dataset])
        #self._train_loader = loader
        #return loader
        return trainsetLoader

    # def val_dataloader(self):
    #     dataset = MNIST(os.getcwd(), train=True, download=False, transform=transforms.ToTensor())
    #     loader = DataLoader(dataset, batch_size=32)
    #     return loader

    # def test_dataloader(self):
    #     dataset = MNIST(os.getcwd(), train=False, download=False, transform=transforms.ToTensor())
    #     loader = DataLoader(dataset, batch_size=32)
    #     return loader

def main():
  from pytorch_lightning import Trainer

  hparams = Hyperparams(base_lr=1e-3, batch_size=me.batch_size, num_epochs=100000, decay_step=25, decay_rate=0.4, staleness=5, num_samples_factor=160, train_percent=0.01)
  #hparams = Hyperparams(base_lr=1e-3, batch_size=me.batch_size, num_epochs=100000, decay_step=25, decay_rate=1.0, staleness=5, num_samples_factor=40, train_percent=0.01)
  #hparams = Hyperparams(base_lr=1e-2, batch_size=32, num_epochs=1000, decay_step=25, decay_rate=1.0, staleness=5, num_samples_factor=40, train_percent=0.01)
  #hparams = Hyperparams(base_lr=1e-3, batch_size=64, num_epochs=10, decay_step=25, decay_rate=1.0, staleness=5, num_samples_factor=10, train_percent=0.1)
  #hparams = Hyperparams(base_lr=1e-3, batch_size=64, num_epochs=10, decay_step=25, decay_rate=1.0, staleness=5, num_samples_factor=10, train_percent=1.0)
  #z_dim = 64
  #z_dim = 128
  model = CoolSystem(me.z_dim, hparams)

  # most basic trainer, uses good defaults
  #trainer = Trainer(num_tpu_cores=8)
  #trainer = Trainer()
  # train on cpu using only 10% of the data (for demo purposes)
  trainer = Trainer(max_epochs=hparams.num_epochs, train_percent_check=hparams.train_percent)
  trainer.fit(model)   
  import pdb; pdb.set_trace()
  #trainer.test(model)

if __name__ == '__main__':
  main()
