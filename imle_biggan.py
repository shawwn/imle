'''
Code for Implicit Maximum Likelihood Estimation

This code implements the method described in the Implicit Maximum Likelihood 
Estimation paper, which can be found at https://arxiv.org/abs/1809.09087

Copyright (C) 2018    Ke Li


This file is part of the Implicit Maximum Likelihood Estimation reference 
implementation.

The Implicit Maximum Likelihood Estimation reference implementation is free 
software: you can redistribute it and/or modify it under the terms of the GNU 
Affero General Public License as published by the Free Software Foundation, 
either version 3 of the License, or (at your option) any later version.

The Implicit Maximum Likelihood Estimation reference implementation is 
distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; 
without even the implied warranty of MERCHANTABILITY or FITNESS FOR A 
PARTICULAR PURPOSE.  See the GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with the Dynamic Continuous Indexing reference implementation.  If 
not, see <http://www.gnu.org/licenses/>.
'''

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import sys
sys.path.append('./dci_code_mac')
from dci import DCI
import collections
import os

class Namespace():
  pass

me = Namespace()

Hyperparams = collections.namedtuple('Hyperarams', 'base_lr batch_size num_epochs decay_step decay_rate staleness num_samples_factor')
Hyperparams.__new__.__defaults__ = (None, None, None, None, None, None, None)

Labels = [0, 1]
Resolution = 32
#Resolution = 128
#z_dim = 128
#Images = 128
Images = 16
z_dim = me.z_dim = 64
Classes = len(Labels)
Channels = 3
BatchSize = 16
#me.radius = 4.0
me.radius = 40.0
#me.radius = 10.0
me.batch_size = 512
me.max_step = 2.0/me.batch_size

def rand_latent(class_id=None,z_dim=None,n_class=10):
  if z_dim is None:
    z_dim = me.z_dim
  batch_size = 1
  z = torch.randn(z_dim, requires_grad=False).cpu()
  #y = torch.randint(low=0, high=Classes, size=(1,), dtype=torch.int64, requires_grad=False).cpu()
  y = torch.randint(low=0, high=1, size=(1,), dtype=torch.int64, requires_grad=False).cpu()
  #z = np.random.randn(1, z_dim)#.reshape([1,-1,1,1])
  #if class_id is not None:
  #  z = np_latent_and_labels(z, class_id, n_class=n_class)
  return z, y

#Gs = Gs_network
#rnd = np.random
#shape = [128, 128, 1, 1]
#latents_a = rnd.randn(1, shape[1])
#latents_b = rnd.randn(1, shape[1])
#latents_c = rnd.randn(1, shape[1])
#import pdb; pdb.set_trace()
me.latents_a, me.labels_a = rand_latent()
me.latents_b, me.labels_b = rand_latent()
me.latents_c, me.labels_c = rand_latent()

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
    #class_id=1
    #n_class=10
    #latents = np_latent_and_labels(latents, class_id, n_class=n_class)
    return latents

me.circ_generator = circ_generator

def circle_interpolation(model, count, gen_func=None, max_step=None, change_min=10.0, change_max=11.0):
    if gen_func is None:
      gen_func = me.circ_generator
    if max_step is None:
      max_step = me.max_step

    def gen_latent(pos):
      z = gen_func(pos)
      z = z.reshape([1,-1])
      return z

    def generate(current_latent):
        #fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
        #current_image = Gs.run(current_latent, None, truncation_psi=0.5, randomize_noise=False, output_transform=fmt)[0]
        z = np.array(current_latent)
        z = torch.from_numpy(z).float().cpu()
        y = me.labels_a
        y = torch.stack((y,)*z.shape[0], axis=0)
        #import pdb; pdb.set_trace()
        imgs = model(z, model.shared(y))
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
        self.relu = nn.ReLU(True)
        
    def forward(self, z):
        z = self.relu(self.bn1(self.tconv1(z)))
        z = self.relu(self.bn2(self.tconv2(z)))
        z = self.relu(self.bn3(self.tconv3(z)))
        z = torch.sigmoid(self.tconv4(z))
        return z

import BigGAN

class IMLE():
    def __init__(self, z_dim):
        self.z_dim = z_dim
        #self.model = ConvolutionalImplicitModel(z_dim).cpu()
        self.model = BigGAN.Generator(resolution=Resolution, dim_z=z_dim, n_classes=Classes).cpu()
        self.dci_db = None
        
    def train(self, data_np, label_np, hyperparams, shuffle_data=True, path='results'):
        loss_fn = nn.MSELoss().cpu()
        self.model.train()
        
        batch_size = hyperparams.batch_size
        num_batches = data_np.shape[0] // batch_size
        num_samples = num_batches * hyperparams.num_samples_factor

        grid_size = (5, 5)
        grid_z = torch.randn(np.prod(grid_size), self.z_dim).cpu()
        grid_y = torch.randint(low=0, high=Classes, size=(np.prod(grid_size),), dtype=torch.int64).cpu()
        
        if shuffle_data:
            data_ordering = np.random.permutation(data_np.shape[0])
            data_np = data_np[data_ordering]
            label_np = label_np[data_ordering]
        
        data_flat_np = np.reshape(data_np, (data_np.shape[0], np.prod(data_np.shape[1:])))
        
        if self.dci_db is None:
            self.dci_db = DCI(np.prod(data_np.shape[1:]), num_comp_indices = 2, num_simp_indices = 7)
            
        for epoch in range(hyperparams.num_epochs):
            if epoch % 10 == 0:
                print('Saving net_weights.pth...')
                torch.save(self.model.state_dict(), os.path.join(path, 'net_weights.pth'))
            if epoch % 5 == 0:
                print('Saving grid...')
                save_grid(path=path, index=epoch, count=np.prod(grid_size), imle=self, z=grid_z, y=grid_y, z_dim=self.z_dim)
                print('Saved')
            
            if epoch % hyperparams.decay_step == 0:
                lr = hyperparams.base_lr * hyperparams.decay_rate ** (epoch // hyperparams.decay_step)
                optimizer = optim.Adam(self.model.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=1e-5)
            
            if epoch % hyperparams.staleness == 0:
                z_np = np.empty((num_samples * batch_size, self.z_dim))
                y_np = np.empty((num_samples * batch_size, 1), dtype=np.int64)
                samples_np = np.empty((num_samples * batch_size,)+data_np.shape[1:])
                for i in range(num_samples):
                    z = torch.randn(batch_size, self.z_dim, requires_grad=False).cpu()
                    y = torch.randint(low=0, high=Classes, size=(batch_size, 1), dtype=torch.int64, requires_grad=False).cpu()
                    samples = self.model(z, self.model.shared(y))
                    #import pdb; pdb.set_trace()
                    z_np[i*batch_size:(i+1)*batch_size] = z.cpu().data.numpy()
                    y_np[i*batch_size:(i+1)*batch_size] = y.cpu().data.numpy()
                    samples_np[i*batch_size:(i+1)*batch_size] = samples.cpu().data.numpy()
                
                samples_flat_np = np.reshape(samples_np, (samples_np.shape[0], np.prod(samples_np.shape[1:]))).copy()
                
                self.dci_db.reset()
                self.dci_db.add(samples_flat_np, num_levels = 2, field_of_view = 10, prop_to_retrieve = 0.002)
                nearest_indices, _ = self.dci_db.query(data_flat_np, num_neighbours = 1, field_of_view = 20, prop_to_retrieve = 0.02)
                nearest_indices = np.array(nearest_indices)[:,0]
                
                z_np = z_np[nearest_indices]
                z_np += 0.01*np.random.randn(*z_np.shape)
                y_np = y_np[nearest_indices]
                
                del samples_np, samples_flat_np
            
            err = 0.
            for i in range(num_batches):
                self.model.zero_grad()
                cur_z = torch.from_numpy(z_np[i*batch_size:(i+1)*batch_size]).float().cpu()
                cur_y = torch.from_numpy(y_np[i*batch_size:(i+1)*batch_size]).long().cpu()
                cur_data = torch.from_numpy(data_np[i*batch_size:(i+1)*batch_size]).float().cpu()
                cur_samples = self.model(cur_z, self.model.shared(cur_y))
                loss = loss_fn(cur_samples, cur_data)
                loss.backward()
                err += loss.item()
                optimizer.step()
            
            print("Epoch %d: Error: %f" % (epoch, err / num_batches))

import mnist_dataset
import datasets

def main(*args):
    
    if len(args) > 0:
        device_id = int(args[0])
    else:
        device_id = 0
    
    #torch.cuda.set_device(device_id)

    path = 'results'
    
    # train_data is of shape N x C x H x W, where N is the number of examples, C is the number of channels, H is the height and W is the width
    #train_data = np.random.randn(128, 1, 28, 28)
    #train_data = mnist_dataset.get_mnist_images(Images, only_labels=Labels, resolution=Resolution, channels=Channels)
    train_data, label_data = datasets.get_samples(Images, only_labels=Labels, resolution=Resolution, channels=Channels)
    
    imle = IMLE(z_dim)

    if os.path.isfile(os.path.join(path, 'net_weights.pth')):
        imle.model.load_state_dict(torch.load(os.path.join(path, 'net_weights.pth')))

    if False:
        imgs = me.circle_interpolation(imle.model.eval(), me.batch_size)
        print('Saving...')
        save_image_grid(imgs.detach().numpy(), "samples/samples.jpg", "samples/frame_%04d.jpg")
        #import pdb; pdb.set_trace()
        import sys
        sys.exit(0)
        #import pdb; pdb.set_trace()
    
    # Hyperparameters:
    
    # base_lr: Base learning rate
    # batch_size: Batch size
    # num_epochs: Number of epochs
    # decay_step: Number of epochs before learning rate decay
    # decay_rate: Rate of learning rate decay
    # staleness: Number of times to re-use nearest samples
    # num_samples_factor: Ratio of the number of generated samples to the number of real data examples
    imle.train(train_data, label_data, Hyperparams(base_lr=1e-3, batch_size=BatchSize, num_epochs=10000, decay_step=25, decay_rate=1.0, staleness=5, num_samples_factor=10))
    
    torch.save(imle.model.state_dict(), os.path.join(path, 'net_weights.pth'))
    
def load(path='results'):
    imle = IMLE(z_dim)
    imle.model.load_state_dict(torch.load(os.path.join(path, 'net_weights.pth')))
    return imle

def sample(batch_size, z_dim, imle=None, z=None, y=None):
    if imle is None:
        imle = load()
    if z is None:
        z = torch.randn(batch_size, z_dim, 1, 1).cpu()
    if y is None:
        z = torch.randn(batch_size, z_dim, 1, 1).cpu()
    ev = imle.model.eval()
    samples = ev(z, ev.shared(y)).detach().numpy()
    return samples


def create_image_grid(images, grid_size=None):
    assert images.ndim == 3 or images.ndim == 4
    num, img_w, img_h = images.shape[0], images.shape[-1], images.shape[-2]

    if grid_size is not None:
        grid_w, grid_h = tuple(grid_size)
    else:
        grid_w = max(int(np.ceil(np.sqrt(num))), 1)
        grid_h = max((num - 1) // grid_w + 1, 1)

    grid = np.zeros(list(images.shape[1:-2]) + [grid_h * img_h, grid_w * img_w], dtype=images.dtype)
    for idx in range(num):
        x = (idx % grid_w) * img_w
        y = (idx // grid_w) * img_h
        grid[..., y : y + img_h, x : x + img_w] = np.clip(images[idx], 0, 255)
    return grid

def save_image_grid(images, outfile, outspec=None):
    images = images*255
    grid = create_image_grid(images.copy()).transpose([1,2,0])
    mnist_dataset.save_mnist_image(grid, outfile)
    if outspec:
      for i, img in enumerate(images):
        #import pdb; pdb.set_trace()
        img = img.transpose([1,2,0])
        #mnist_dataset.save_mnist_image(img, outspec % i)
        from PIL import Image
        Image.fromarray(img.clip(0,255).astype(np.uint8)).save(outspec % i)

def save_grid(path, index, count, *args, **kws):
    try:
      os.mkdir(path)
    except:
      pass
    mnist_dataset.save_mnist_image(create_image_grid(sample(count, *args, **kws)*255).transpose([1,2,0]), os.path.join(path, 'fakes_%06d.jpg' % index))
    #reals = mnist_dataset.get_mnist_images(count, only_labels=Labels, resolution=Resolution, channels=Channels)
    reals, labels = datasets.get_samples(count, only_labels=Labels, resolution=Resolution, channels=Channels)
    mnist_dataset.save_mnist_image(create_image_grid(reals*255).transpose([1,2,0]), os.path.join(path, 'reals_%06d.jpg' % index))

if __name__ == '__main__':
    main(*sys.argv[1:])
