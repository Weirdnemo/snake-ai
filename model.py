import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import numpy as np


class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, hidden_size_2, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size_2)
        self.linear3 = nn.Linear(hidden_size_2, output_size)

        # Proper weight initialization is crucial for learning
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # He initialization for ReLU activation
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Handle different input types
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float)

        # Add batch dimension if needed
        if len(x.shape) == 1:
            x = x.unsqueeze(0)

        # Simple architecture with ReLU activations
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        return self.linear3(x)  # No activation on output layer for Q-values

    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)

    def load(self, file_name='model.pth'):
        model_folder_path = './model'
        file_name = os.path.join(model_folder_path, file_name)

        if os.path.exists(file_name):
            self.load_state_dict(torch.load(file_name))
            self.eval()  # Set to evaluation mode
            print(f"Loaded model from {file_name}")
            return True
        return False


class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        # Use RMSprop instead of Adam - often better for RL tasks
        self.optimizer = optim.RMSprop(model.parameters(), lr=self.lr, alpha=0.99, eps=1e-08)
        # Huber loss is more robust to outliers
        self.criterion = nn.SmoothL1Loss()

    def train_step(self, state, action, reward, next_state, done):
        # Convert inputs to tensors if they aren't already
        if isinstance(state, tuple) or isinstance(state, list):
            state = torch.tensor(np.array(state), dtype=torch.float)
            next_state = torch.tensor(np.array(next_state), dtype=torch.float)
            action = torch.tensor(np.array(action), dtype=torch.long)
            reward = torch.tensor(np.array(reward), dtype=torch.float)
            done = torch.tensor(np.array(done), dtype=torch.bool)
        else:
            state = torch.tensor(state, dtype=torch.float)
            next_state = torch.tensor(next_state, dtype=torch.float)
            action = torch.tensor(action, dtype=torch.long)
            reward = torch.tensor(reward, dtype=torch.float)
            # Check if we have a single boolean or a list
            if isinstance(done, bool):
                done = torch.tensor([done], dtype=torch.bool)
            else:
                done = torch.tensor(done, dtype=torch.bool)

        # Handle both single samples and batches
        is_single_sample = len(state.shape) == 1
        if is_single_sample:
            state = state.unsqueeze(0)
            next_state = next_state.unsqueeze(0)
            action = action.unsqueeze(0)
            reward = reward.unsqueeze(0)
            done = done.unsqueeze(0) if len(done.shape) == 0 else done

        # 1: Get predicted Q values with current state
        pred = self.model(state)

        # 2: Calculate target Q values
        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                # Only consider future rewards if not done
                next_q = self.model(next_state[idx])
                Q_new = reward[idx] + self.gamma * torch.max(next_q)

            # Update only the Q value for the action taken
            action_idx = torch.argmax(action[idx]).item()
            target[idx][action_idx] = Q_new

        # Zero gradients, perform backprop and update weights
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()

        # Clip gradients to prevent exploding gradients
        torch.nn.utils.clip_grad_value_(self.model.parameters(), 1)

        self.optimizer.step()
        return loss.item()