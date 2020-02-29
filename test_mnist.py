import os

import numpy as np
import torch
import torchvision
from torch import nn
from torch.nn import Parameter
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms
import tqdm

import pytorch_lightning as pl

import sys
sys.path.append('./dci_code_mac')
from dci import DCI

import collections
Hyperparams = collections.namedtuple('Hyperarams', 'base_lr batch_size num_epochs decay_step decay_rate staleness num_samples_factor train_percent')
Hyperparams.__new__.__defaults__ = (None, None, None, None, None, None, None, None)

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

def rand_latent(class_id,z_dim=64,n_class=10):
  latents_a = np.random.randn(1, z_dim)#.reshape([1,-1,1,1])
  return np_latent_and_labels(latents_a, class_id, n_class=n_class)


#Gs = Gs_network
#rnd = np.random
#shape = [64, 64, 1, 1]
#latents_a = rnd.randn(1, shape[1])
#latents_b = rnd.randn(1, shape[1])
#latents_c = rnd.randn(1, shape[1])
#import pdb; pdb.set_trace()
latents_a = rand_latent(4)
latents_b = rand_latent(4)
latents_c = rand_latent(4)
import pdb; pdb.set_trace()

import math

def vdist(v):
  v = v.flatten()
  return np.dot(v,v)**0.5

def vnorm(v):
    return v/vdist(v)

def circ_generator(latents_interpolate):
    #radius = 40.0
    radius = 0.1
    latents_axis_x = (latents_a - latents_b).flatten() / vdist(latents_a - latents_b)
    latents_axis_y = (latents_a - latents_c).flatten() / vdist(latents_a - latents_c)
    latents_x = math.sin(math.pi * 2.0 * latents_interpolate) * radius
    latents_y = math.cos(math.pi * 2.0 * latents_interpolate) * radius
    latents = latents_a + latents_x * latents_axis_x + latents_y * latents_axis_y
    return latents


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
        self.tconv1 = nn.ConvTranspose2d(64+n_class, 1024, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(1024)
        self.tconv2 = nn.ConvTranspose2d(1024, 128, 7, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(128)
        self.tconv3 = nn.ConvTranspose2d(128, 64, 4, 2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(64)
        self.tconv4 = nn.ConvTranspose2d(64, 1, 4, 2, padding=1, bias=False)
        self.bn4 = None
        #self.bn4 = nn.BatchNorm2d(1)
        self.relu = nn.ReLU(True)
        chn = 32
        self.attention = None
        self.attention = SelfAttention(2 * chn)
        #self.linear = nn.Linear(n_class, 128, bias=False)
        
    def forward(self, z, class_id=None):
        #class_emb = self.linear(class_id)  # 128
        if class_id is not None:
          z = latent_and_labels(z, class_id, self.n_class)
        z = self.relu(self.bn1(self.tconv1(z)))
        z = self.relu(self.bn2(self.tconv2(z)))
        z = self.relu(self.bn3(self.tconv3(z)))
        if self.attention:
          z = self.attention(z)
        z = self.tconv4(z)
        if self.bn4 is not None:
          z = self.bn4(z)
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
        self.hyperparams = hyperparams
        self.shuffle_data = shuffle_data
        self.dci_db = None
        self.model = ConvolutionalImplicitModel(z_dim)
        #self.model = Generator128(z_dim, n_class=10)
        #self.squeezeNet = NetPerLayer()
        self.squeezeNet = None
        #self.loss_fn = nn.MSELoss()
        self._step = 0

    def regen(self, batch):

        imgs, labels = batch
        data_np = imgs.numpy()
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
        if self._step % 20 == 0 and True:
          #print('circle...')
          #import pdb; pdb.set_trace()
          imgs = self.circle_interpolation(imgInput.shape[0])
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
        pixelLoss = F.mse_loss(imgOutput, imgTarget)
        loss = featLoss + pixelLoss
        tensorboard_logs = {'train_loss': loss}
        self._step += 1
        return {'loss': loss, 'log': tensorboard_logs}

    def number_interpolation(self, count, lo, hi):
        hyperparams = self.hyperparams
        batch_size = hyperparams.batch_size

        def gen_latent(pos):
          z = gen_func(pos)
          z = z.reshape([1,-1,1,1])
          return z


    def circle_interpolation(self, count, gen_func=circ_generator, max_step=1.0/64.0, change_min=10.0, change_max=11.0):
        hyperparams = self.hyperparams
        batch_size = hyperparams.batch_size

        def gen_latent(pos):
          z = gen_func(pos)
          z = z.reshape([1,-1,1,1])
          return z

        def generate(current_latent):
            #fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
            #current_image = Gs.run(current_latent, None, truncation_psi=0.5, randomize_noise=False, output_transform=fmt)[0]
            z = np.array(current_latent)
            #z = z.reshape([batch_size,-1,1,1])
            z = torch.from_numpy(z).float().cpu()
            imgs = self.model(z)
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

        current_pos = 0.0
        current_latent = gen_func(current_pos)
        current_image = generate(current_latent)
        array_list = []
        with tqdm.tqdm(total=count) as pbar:
          while current_pos < 1.0 and len(array_list) < count:
              array_list.append(current_image)
              pbar.update(1)
              lower = current_pos
              upper = current_pos + max_step
              current_pos = (upper + lower) / 2.0
              current_latent = gen_func(current_pos)
              current_image = generate(current_latent)
              if False:
                  current_mse = mse(array_list[-1], current_image)
                  prev_pos = current_pos
                  while current_mse < change_min or current_mse > change_max:
                      if current_mse < change_min:
                          lower = current_pos
                          current_pos = (upper + lower) / 2.0
                      if current_mse > change_max:
                          upper = current_pos
                          current_pos = (upper + lower) / 2.0
                      current_latent = gen_func(current_pos)
                      current_image = generate(current_latent)
                      current_mse = mse(array_list[-1], current_image)
                      diff = abs(prev_pos - current_pos)
                      if diff < 0.001:
                        break
                      #print('inner', current_pos, 'delta', diff, 'mse', current_mse)
                      prev_pos = current_pos
                  #print('outer', current_pos, current_mse)
        return array_list

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
        epoch = 0
        hyperparams = self.hyperparams
        lr = hyperparams.base_lr * hyperparams.decay_rate ** (epoch // hyperparams.decay_step)
        #optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=1e-5)
        #optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, amsgrad=True)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, betas=(0.5, 0.999), amsgrad=True)
        return optimizer
        

    def prepare_data(self):
        MNIST(os.getcwd(), train=True, download=True, transform=transforms.ToTensor())
        MNIST(os.getcwd(), train=False, download=True, transform=transforms.ToTensor())

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
        dataset = MNIST(os.getcwd(), train=True, download=False, transform=transforms.ToTensor())
        #loader = DataLoader(dataset, batch_size=32)

        hyperparams = self.hyperparams
        batch_size = hyperparams.batch_size
        loader = DataLoader(dataset, batch_size=batch_size)
        # self.data_np = np.array([img.numpy() for img, label in dataset])
        #self._train_loader = loader
        return loader

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

  hparams = Hyperparams(base_lr=1e-3, batch_size=64, num_epochs=10, decay_step=25, decay_rate=1.0, staleness=5, num_samples_factor=10, train_percent=0.1)
  #hparams = Hyperparams(base_lr=1e-3, batch_size=64, num_epochs=10, decay_step=25, decay_rate=1.0, staleness=5, num_samples_factor=10, train_percent=1.0)
  z_dim = 64
  model = CoolSystem(z_dim, hparams)

  # most basic trainer, uses good defaults
  #trainer = Trainer(num_tpu_cores=8)
  #trainer = Trainer()
  # train on cpu using only 10% of the data (for demo purposes)
  trainer = Trainer(max_epochs=hparams.num_epochs, train_percent_check=hparams.train_percent)
  trainer.fit(model)   
  #trainer.test(model)

if __name__ == '__main__':
  main()
