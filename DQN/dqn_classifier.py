import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import logging  
from collections import deque
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from typing import Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('dqn_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DQN(nn.Module):
    def __init__(self, input_size, hidden_size=32):  # Reduced from 64 to 32
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.dropout2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(hidden_size, 2)  # 2 actions: 0 (non-fraud) or 1 (fraud)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        return self.fc3(x)

class DQNClassifier:
    def __init__(self, input_size, learning_rate=0.0001, gamma=0.99, 
                 epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DQN(input_size).to(self.device)
        self.target_model = DQN(input_size).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        
        # Add L2 regularization with weight_decay
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=learning_rate, 
            weight_decay=1e-5
        )
        self.criterion = nn.MSELoss()
        
        # Training parameters
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.memory = deque(maxlen=10000)
        self.batch_size = 64
        
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        
    def act(self, state, training=True):
        if training and np.random.rand() <= self.epsilon:
            return random.randrange(2)  # Random action (0 or 1)
        
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.model(state)
            return torch.argmax(q_values).item()
    
    def replay(self):
        if len(self.memory) < self.batch_size:
            return
            
        minibatch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)
        
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Current Q values
        current_q = self.model(states).gather(1, actions)
        
        # Compute target Q values
        with torch.no_grad():
            next_q = self.target_model(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * self.gamma * next_q
        
        # Compute loss and optimize
        loss = self.criterion(current_q.squeeze(), target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())
        
    def predict(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.model(state)
            return torch.argmax(q_values).item()
    
    def evaluate(self, X, y) -> Dict[str, float]:
        """Evaluate the model and return metrics including False Alarm Ratio"""
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            outputs = self.model(X_tensor)
            predictions = outputs.argmax(dim=1).cpu().numpy()
            probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
        
        # Calculate metrics
        accuracy = accuracy_score(y, predictions)
        precision = precision_score(y, predictions, zero_division=0)
        recall = recall_score(y, predictions, zero_division=0)
        f1 = f1_score(y, predictions, zero_division=0)
        auc = roc_auc_score(y, probs)
        
        # Calculate False Alarm Ratio (FAR)
        tn, fp, fn, tp = confusion_matrix(y, predictions).ravel()
        far = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc,
            'far': far
        }

    def save(self, path='dqn_model.pth'):
        """Save the model weights and optimizer state"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'target_state_dict': self.target_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'input_size': self.model.fc1.in_features
        }, path)
        logger.info(f"Model saved to {path}")

    @classmethod
    def load(cls, path='dqn_model.pth', **kwargs):
        """Load a saved model"""
        checkpoint = torch.load(path)
        model = cls(input_size=checkpoint['input_size'], **kwargs)
        model.model.load_state_dict(checkpoint['model_state_dict'])
        model.target_model.load_state_dict(checkpoint['target_state_dict'])
        model.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        model.epsilon = checkpoint['epsilon']
        return model