import numpy as np
import torch


class ReplayBuffer(object):
    def __init__(
        self, image_size, action_dim, hidden_size,
        max_size=int(5e3), recurrent=True, device='cpu'
    ):
        self.max_size = int(max_size)
        self.ptr = 0
        self.size = 0
        self.recurrent = recurrent
        self.image_size = image_size

        self.image = np.zeros((self.max_size, image_size * image_size * 3), dtype=np.float32)
        self.action = np.zeros((self.max_size, action_dim))
        self.next_image = np.zeros((self.max_size, image_size * image_size * 3), dtype=np.float32)
        self.reward = np.zeros((self.max_size, 1))
        self.not_done = np.zeros((self.max_size, 1))

        if self.recurrent:
            self.h = np.zeros((self.max_size, hidden_size))
            self.nh = np.zeros((self.max_size, hidden_size))

            self.c = np.zeros((self.max_size, hidden_size))
            self.nc = np.zeros((self.max_size, hidden_size))

        self.device = torch.device(device)

    def add(
        self, image, action, next_image, 
              reward, done, hiddens, next_hiddens
    ):
        self.image[self.ptr] = image
        self.action[self.ptr] = action
        self.next_image[self.ptr] = next_image
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done

        if self.recurrent:
            h, c = hiddens
            nh, nc = next_hiddens

            # Detach the hidden state so that BPTT only goes through 1 timestep
            self.h[self.ptr] = h.detach().cpu()
            self.c[self.ptr] = c.detach().cpu()
            self.nh[self.ptr] = nh.detach().cpu()
            self.nc[self.ptr] = nc.detach().cpu()

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size=100):
        ind = np.random.randint(0, self.size, size=int(batch_size))


        h = torch.tensor(self.h[ind][None, ...],
                         requires_grad=True,
                         dtype=torch.float).to(self.device)
        c = torch.tensor(self.c[ind][None, ...],
                         requires_grad=True,
                         dtype=torch.float).to(self.device)
        nh = torch.tensor(self.nh[ind][None, ...],
                          requires_grad=True,
                          dtype=torch.float).to(self.device)
        nc = torch.tensor(self.nc[ind][None, ...],
                          requires_grad=True,
                          dtype=torch.float).to(self.device)

        hidden = (h, c)
        next_hidden = (nh, nc)

        images = torch.FloatTensor(
            self.image[ind][:, None, :]).to(self.device)
        images = torch.reshape(images, (-1, self.image_size, self.image_size, 3))
        images = images.permute(0, 3, 1, 2)
        a = torch.FloatTensor(
            self.action[ind][:, None, :]).to(self.device)
        n_images = torch.FloatTensor(
            self.next_image[ind][:, None, :]).to(self.device)
        n_images = torch.reshape(n_images, (-1, self.image_size, self.image_size, 3))
        n_images = n_images.permute(0, 3, 1, 2)
        r = torch.FloatTensor(
            self.reward[ind][:, None, :]).to(self.device)
        d = torch.FloatTensor(
            self.not_done[ind][:, None, :]).to(self.device)

        return images, a, n_images, r, d, hidden, next_hidden

    def clear_memory(self):
        self.ptr = 0
        self.size = 0
