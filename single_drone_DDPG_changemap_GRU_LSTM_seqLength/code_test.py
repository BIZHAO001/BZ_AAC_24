import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
import random


class RandomNetwork(nn.Module):
    def __init__(self, input_dim, alpha):
        super(RandomNetwork, self).__init__()
        self.fc = nn.Linear(input_dim, input_dim)
        self.reset_parameters(alpha)

    def reset_parameters(self, alpha):
        with torch.no_grad():
            # Identity kernel initialization
            I = torch.eye(self.fc.weight.shape[0], self.fc.weight.shape[1])
            if I.shape != self.fc.weight.shape:
                I = I.expand(self.fc.weight.shape[0], self.fc.weight.shape[1])
            # Xavier normal distribution initialization
            std = math.sqrt(2.0 / (self.fc.weight.shape[1] + self.fc.weight.shape[0]))
            normal_dist = torch.randn_like(self.fc.weight) * std

            # Mixture of identity kernel and normal distribution
            self.fc.weight.copy_(alpha * I + (1-alpha) * normal_dist)
            self.fc.bias.fill_(0)

    def forward(self, x):
        return self.fc(x)

# Define the GRU-based Actor and Critic Networks
class ActorGRU(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, feature_dim):
        super(ActorGRU, self).__init__()
        self.gru = nn.GRU(state_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(5, action_dim)
        self.feature_extractor = FeatureExtractor(hidden_dim, feature_dim)

    def forward(self, x):
        x, _ = self.gru(x)
        # features = self.feature_extractor(x[:, -1, :])
        features = self.feature_extractor(x)
        out = torch.tanh(self.fc(features))
        return out, features

class CriticGRU(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, feature_dim):
        super(CriticGRU, self).__init__()
        self.gru = nn.GRU(state_dim + action_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(5, 1)
        self.feature_extractor = FeatureExtractor(hidden_dim, feature_dim)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)  # Concatenate along the last dimension
        x, _ = self.gru(x)
        # features = self.feature_extractor(x[:, -1, :])
        features = self.feature_extractor(x)
        q_value = self.fc(features)
        return q_value, features

# Define the Feature Extractor
class FeatureExtractor(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super(FeatureExtractor, self).__init__()
        self.fc1 = nn.Linear(input_dim, feature_dim)
        self.fc2 = nn.Linear(feature_dim, feature_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        features = F.relu(self.fc2(x))
        return features

# Define the Feature Matching Loss
def feature_matching_loss(original_features, randomized_features):
    return F.mse_loss(original_features, randomized_features)

# Training Loop with Feature Matching
def train_with_feature_matching(num_epochs, replay_buffer, state_dim, hidden_dim, action_dim, feature_dim, learning_rate, beta, gamma):
    # Initialize networks
    actor = ActorGRU(state_dim, action_dim, hidden_dim, feature_dim)
    critic = CriticGRU(state_dim, action_dim, hidden_dim, feature_dim)
    random_network = RandomNetwork(state_dim)

    # Optimizers
    actor_optimizer = optim.Adam(actor.parameters(), lr=learning_rate)
    critic_optimizer = optim.Adam(critic.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        for state, action, reward, next_state, done in replay_buffer:
            state = torch.FloatTensor(state).unsqueeze(0)  # Add batch dimension
            action = torch.FloatTensor(action).unsqueeze(0)  # Add batch and sequence dimensions
            reward = torch.FloatTensor(reward).unsqueeze(0)
            next_state = torch.FloatTensor(next_state).unsqueeze(0)  # Add batch dimension
            done = torch.FloatTensor(done).unsqueeze(0)


            random_state = random_network(state)  # Perturb the input state


            q_value, original_features = critic(state, action)
            _, randomized_features = critic(random_state, action)

            # Compute critic loss
            with torch.no_grad():
                next_action, _ = actor(next_state)
                next_q_value, _ = critic(next_state, next_action)
                target_q_value = reward + (1 - done) * gamma * next_q_value

            critic_loss = F.mse_loss(q_value, target_q_value)
            fm_loss = feature_matching_loss(original_features, randomized_features)
            total_critic_loss = critic_loss + beta * fm_loss

            # Optimize the critic
            critic_optimizer.zero_grad()
            total_critic_loss.backward()
            critic_optimizer.step()

            # Compute actor loss
            policy_loss = -critic(state, actor(state)[0])[0].mean()

            # Optimize the actor
            actor_optimizer.zero_grad()
            policy_loss.backward()
            actor_optimizer.step()

        if epoch % 100 == 0:
            print(f'Epoch {epoch}, Critic Loss: {total_critic_loss.item()}, Actor Loss: {policy_loss.item()}')

# Hyperparameters
state_dim = 10
hidden_dim = 20
action_dim = 2
feature_dim = 5
learning_rate = 0.001
beta = 0.002  # Weight for feature matching loss
gamma = 0.99  # Discount factor
num_epochs = 1000  # Number of epochs for training

# Example replay buffer (for illustration purposes)
replay_buffer = [
    (torch.randn(state_dim), torch.randn(action_dim), torch.tensor(1.0), torch.randn(state_dim), torch.tensor(0.0))
    for _ in range(100)
]

# Train the networks with feature matching
train_with_feature_matching(num_epochs, replay_buffer, state_dim, hidden_dim, action_dim, feature_dim, learning_rate, beta, gamma)