import copy
import torch
import torch.nn as nn
import torch.nn.functional as F


print(torch.cuda.is_available(), torch.backends.cudnn.enabled)
# Implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)
# Paper: https://arxiv.org/abs/1802.09477


class Actor(nn.Module):
    def __init__(
        self, image_size, image_feature_dim, action_dim, hidden_dim, max_action, is_recurrent=False
    ):
        super(Actor, self).__init__()
        self.recurrent = is_recurrent
        self.image_size = image_size
        self.image_feature_dim = image_feature_dim

        self.image_compress_net = nn.Sequential(nn.Conv2d(3, 32, 4, stride=2),
                                     nn.ReLU(), nn.Conv2d(32, 32 * 2, 4, stride=2),
                                     nn.ReLU(), nn.Conv2d(32 * 2, 32 * 4, 4, stride=2),
                                     nn.ReLU(), nn.Conv2d(32 * 4, 32 * 8, 4, stride=2),
                                     nn.ReLU(), nn.Conv2d(32 * 8, 32 * 8, 4, stride=2),
                                     nn.ReLU())

        if self.recurrent:
            self.l1 = nn.LSTM(image_feature_dim, hidden_dim, batch_first=True)

        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, action_dim)

        self.max_action = max_action

    def forward(self, image, hidden):
        with torch.no_grad():
            image_feature = self.image_compress_net(image)
            image_feature = torch.reshape(image_feature, (-1, 1, self.image_feature_dim))
      
        if self.recurrent:
            self.l1.flatten_parameters()
            a, h = self.l1(image_feature, hidden)


        a = F.relu(self.l2(a))
        a = torch.tanh(self.l3(a))
        return self.max_action * a, h


class Critic(nn.Module):
    def __init__(
        self, image_size, image_feature_dim, action_dim, hidden_dim, is_recurrent=False
    ):
        super(Critic, self).__init__()
        self.recurrent = is_recurrent
        self.image_size = image_size
        self.image_feature_dim = image_feature_dim

        self.image_compress_net = nn.Sequential(nn.Conv2d(3, 32, 4, stride=2),
                                     nn.ReLU(), nn.Conv2d(32, 32 * 2, 4, stride=2),
                                     nn.ReLU(), nn.Conv2d(32 * 2, 32 * 4, 4, stride=2),
                                     nn.ReLU(), nn.Conv2d(32 * 4, 32 * 8, 4, stride=2),
                                     nn.ReLU(), nn.Conv2d(32 * 8, 32 * 8, 4, stride=2),
                                     nn.ReLU())


        if self.recurrent:
            self.l1 = nn.LSTM(
                image_feature_dim + action_dim, hidden_dim, batch_first=True)
            self.l4 = nn.LSTM(
                image_feature_dim + action_dim, hidden_dim, batch_first=True)

       

        # Q1 architecture
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, 1)

        # Q2 architecture
        self.l5 = nn.Linear(hidden_dim, hidden_dim)
        self.l6 = nn.Linear(hidden_dim, 1)

    def forward(self, image, action, hidden1, hidden2):
        image_feature = self.image_compress_net(image)
        image_feature = torch.reshape(image_feature, (-1, 1, self.image_feature_dim))
        sa = torch.cat([image_feature, action], -1)
        if self.recurrent:
            self.l1.flatten_parameters()
            self.l4.flatten_parameters()
            q1, hidden1 = self.l1(sa, hidden1)
            q2, hidden2 = self.l4(sa, hidden2)
        else:
            q1, hidden1 = F.relu(self.l1(sa)), None
            q2, hidden2 = F.relu(self.l4(sa)), None

        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)

        return q1, q2

    def Q1(self, image, action, hidden1):
        image_feature = self.image_compress_net(image)
        image_feature = torch.reshape(image_feature, (-1, 1, self.image_feature_dim))
        sa = torch.cat([image_feature, action], -1)
        if self.recurrent:
            self.l1.flatten_parameters()
            q1, hidden1 = self.l1(sa, hidden1)
        else:
            q1, hidden1 = F.relu(self.l1(sa)), None

        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        return q1


class TD3(object):
    def __init__(
            self,
            image_size,
            image_feature_dim,
            action_dim,
            max_action,
            hidden_dim,
            discount=0.99,
            tau=0.005,
            policy_noise=0.2,
            noise_clip=0.5,
            policy_freq=2,
            lr=3e-4,
            recurrent_actor=False,
            recurrent_critic=False,
            device='cpu'
    ):
        self.hidden_dim = hidden_dim
        self.image_size = image_size
        self.device = torch.device(device)
        self.on_policy = False
        self.recurrent = recurrent_actor
        self.actor = Actor(
            image_size, image_feature_dim, action_dim, hidden_dim, max_action,
            is_recurrent=recurrent_actor
        ).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=lr)

        self.critic = Critic(
            image_size, image_feature_dim, action_dim, hidden_dim,
            is_recurrent=recurrent_critic
        ).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)

        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=lr)

        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq

        self.total_it = 0

    def get_initial_states(self):
        h_0, c_0 = None, None
        if self.actor.recurrent:
            h_0 = torch.zeros((
                self.actor.l1.num_layers,
                1,
                self.actor.l1.hidden_size),
                dtype=torch.float)
            h_0 = h_0.to(device=self.device)

            c_0 = torch.zeros((
                self.actor.l1.num_layers,
                1,
                self.actor.l1.hidden_size),
                dtype=torch.float)
            c_0 = c_0.to(device=self.device)
        return (h_0, c_0)

    def select_action(self, image, hidden, test=True):
        
        if self.recurrent:
            image = torch.FloatTensor(
                image.reshape(1, -1)).to(self.device)
            image = torch.reshape(image, (-1, self.image_size, self.image_size, 3))
            image = image.permute(0, 3, 1, 2)
        else:
            print('please set recurrent as True')
            raise NotImplementedError(self.recurrent)

        
        action, hidden = self.actor(image, hidden)
        return action.cpu().data.numpy().flatten(), hidden

    def train(self, replay_buffer, batch_size=100):
        self.total_it += 1

        # Sample replay buffer
        image, action, next_image, reward, not_done, hidden, next_hidden = \
            replay_buffer.sample(batch_size)

        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (
                torch.randn_like(action) * self.policy_noise
            ).clamp(-self.noise_clip, self.noise_clip)

            next_action = (
                self.actor_target(next_image, next_hidden)[0] + noise
            ).clamp(-self.max_action, self.max_action)

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(
                next_image, next_action, next_hidden, next_hidden)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + not_done * self.discount * target_Q

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(image, action, hidden, hidden)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + \
            F.mse_loss(current_Q2, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        for param, target_param in zip(
                self.critic.image_compress_net.parameters(), self.actor.image_compress_net.parameters()
            ):
            target_param.data.copy_(param.data)

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:

            # Compute actor losse
            actor_loss = -self.critic.Q1(
                image.detach(), 
                self.actor(image, hidden)[0], hidden).mean()

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(
                self.critic.parameters(), self.critic_target.parameters()
            ):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(
                self.actor.parameters(), self.actor_target.parameters()
            ):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(),
                   filename + "_critic_optimizer")
        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(),
                   filename + "_actor_optimizer")

    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(
            torch.load(filename + "_critic_optimizer"))
        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(
            torch.load(filename + "_actor_optimizer"))

    def eval_mode(self):
        self.actor.eval()
        self.critic.eval()

    def train_mode(self):
        self.actor.train()
        self.critic.train()
