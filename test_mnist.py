import os

import numpy as np
import torch
import torchvision
from torch import nn
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


#Gs = Gs_network
rnd = np.random
shape = [64, 64, 1, 1]
latents_a = rnd.randn(1, shape[1])
latents_b = rnd.randn(1, shape[1])
latents_c = rnd.randn(1, shape[1])

import math

def vdist(v):
  v = v.flatten()
  return np.dot(v,v)**0.5

def vnorm(v):
    return v/vdist(v)

def circ_generator(latents_interpolate):
    radius = 40.0
    latents_axis_x = (latents_a - latents_b).flatten() / vdist(latents_a - latents_b)
    latents_axis_y = (latents_a - latents_c).flatten() / vdist(latents_a - latents_c)
    latents_x = math.sin(math.pi * 2.0 * latents_interpolate) * radius
    latents_y = math.cos(math.pi * 2.0 * latents_interpolate) * radius
    latents = latents_a + latents_x * latents_axis_x + latents_y * latents_axis_y
    return latents


def mse(x, y):
    return (np.square(x - y)).mean()


class ConvolutionalImplicitModel(nn.Module):
    def __init__(self, z_dim):
        super(ConvolutionalImplicitModel, self).__init__()
        self.z_dim = z_dim
        self.tconv1 = nn.ConvTranspose2d(64, 1024, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(1024)
        self.tconv2 = nn.ConvTranspose2d(1024, 128, 7, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(128)
        self.tconv3 = nn.ConvTranspose2d(128, 64, 4, 2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(64)
        self.tconv4 = nn.ConvTranspose2d(64, 1, 4, 2, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(1)
        self.relu = nn.ReLU(True)
        
    def forward(self, z):
        z = self.relu(self.bn1(self.tconv1(z)))
        z = self.relu(self.bn2(self.tconv2(z)))
        z = self.relu(self.bn3(self.tconv3(z)))
        z = torch.sigmoid(self.bn4(self.tconv4(z)))
        return z



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
        self.loss_fn = nn.MSELoss()
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
          samples = self.model(z)
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

    def forward(self, z):
        cur_samples = self.model(z)
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
        #cur_data = torch.from_numpy(data_np[i*batch_size:(i+1)*batch_size]).float().cpu()
        cur_z = torch.from_numpy(z_np).float().cpu()
        cur_data = torch.from_numpy(data_np).float().cpu()
        print(cur_z.shape, batch_idx)
        def nimg(img):
          #img = img + 1
          #img = img / 2
          #import pdb; pdb.set_trace()
          img = torch.clamp(img, 0, 1)
          return img
        imgInput = batch[0]
        self.logger.experiment.add_image('imgInput', torchvision.utils.make_grid(nimg(imgInput)), self._step)
        cur_samples = self.forward(cur_z)
        self.logger.experiment.add_image('imgOutput', torchvision.utils.make_grid(nimg(cur_samples)), self._step)
        if self._step % 1 == 0:
          #print('circle...')
          imgs = self.circle_interpolation(imgInput.shape[0])
          #imgs = np.array(imgs)
          #imgs = torch.from_numpy(imgs).float().cpu()
          self.logger.experiment.add_image('imgInterp', torchvision.utils.make_grid(nimg(imgs)), self._step)
          #import pdb; pdb.set_trace()
        #print('done')
        loss = self.loss_fn(cur_samples, cur_data)
        tensorboard_logs = {'train_loss': loss}
        self._step += 1
        return {'loss': loss, 'log': tensorboard_logs}

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
            z = z.reshape([batch_size,self.z_dim,1,1])
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
        z = list(build_latent(count))
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
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=1e-5)
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

  #hparams = Hyperparams(base_lr=1e-3, batch_size=64, num_epochs=10, decay_step=25, decay_rate=1.0, staleness=5, num_samples_factor=10, train_percent=0.1)
  hparams = Hyperparams(base_lr=1e-3, batch_size=64, num_epochs=10, decay_step=25, decay_rate=1.0, staleness=5, num_samples_factor=10, train_percent=1.0)
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
