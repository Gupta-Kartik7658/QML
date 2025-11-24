import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from DQN.dqn_classifier import DQNClassifier, logger
import matplotlib.pyplot as plt
import os
from datetime import datetime
import logging 
import torch  

# Configuration
EPISODES = 200  # Increased to account for early stopping
TARGET_UPDATE_FREQ = 10
PATIENCE = 15  # For early stopping
SELECTED_FEATURES = ['V14', 'V7', 'V4', 'V19', 'V20', 'V17']

def setup_logging():
    """Set up logging configuration"""
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"training_{timestamp}.log")
    
    # Clear existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Add new handlers
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(console_handler)
    
    logger.info(f"Logging to {os.path.abspath(log_file)}")
    return log_file

def load_and_prepare_data():
    """Load and preprocess the credit card fraud dataset"""
    try:
        data = pd.read_csv('creditcard.csv')
        logger.info(f"Loaded dataset with shape: {data.shape}")
        
        # Select features and target
        X = data[SELECTED_FEATURES].values
        y = data['Class'].values
        
        # Balance the dataset (undersampling majority class)
        fraud_indices = np.where(y == 1)[0]
        non_fraud_indices = np.where(y == 0)[0]
        n_samples = min(len(fraud_indices), len(non_fraud_indices))
        
        np.random.seed(42)
        fraud_sample = np.random.choice(fraud_indices, n_samples, replace=False)
        non_fraud_sample = np.random.choice(non_fraud_indices, n_samples, replace=False)
        
        X_balanced = np.vstack((X[fraud_sample], X[non_fraud_sample]))
        y_balanced = np.hstack((y[fraud_sample], y[non_fraud_sample]))
        
        # Split into train and test
        X_train, X_test, y_train, y_test = train_test_split(
            X_balanced, y_balanced, test_size=0.3, random_state=42, stratify=y_balanced
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        logger.info(f"Training set: {len(X_train)} samples, Test set: {len(X_test)} samples")
        logger.info(f"Fraud samples in train: {np.sum(y_train)}, Test: {np.sum(y_test)}")
        
        return X_train, X_test, y_train, y_test, scaler
        
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}", exc_info=True)
        raise

def train_dqn():
    """Train the DQN model with early stopping and logging"""
    # Setup logging
    log_file = setup_logging()
    logger.info("Starting DQN training...")
    logger.info(f"Using features: {SELECTED_FEATURES}")
    
    try:
        # Load and prepare data
        X_train, X_test, y_train, y_test, scaler = load_and_prepare_data()
        input_size = X_train.shape[1]
        
        # Initialize DQN agent with updated parameters
        agent = DQNClassifier(
            input_size=input_size,
            learning_rate=0.0001,  # Reduced learning rate
            epsilon_decay=0.995    # Slower epsilon decay
        )
        
        # Training
        best_test_auc = 0
        patience_counter = 0
        rewards_history = []
        test_auc_history = []
        
        for episode in range(EPISODES):
            episode_rewards = []
            
            # Shuffle training data
            indices = np.arange(len(X_train))
            np.random.shuffle(indices)
            
            for idx in indices:
                state = X_train[idx]
                action = agent.act(state)
                
                # Get reward (higher penalty for missing fraud cases)
                true_label = y_train[idx]
                if action == true_label:
                    reward = 1.0  # Correct classification
                else:
                    if true_label == 1:  # Missed fraud
                        reward = -5.0
                    else:  # False positive
                        reward = -1.0
                
                # Store experience
                next_state = X_train[(idx + 1) % len(X_train)]
                done = (idx == len(X_train) - 1)
                
                agent.remember(state, action, reward, next_state, done)
                episode_rewards.append(reward)
                
                # Train on batch
                agent.replay()
            
            # Update target network
            if episode % TARGET_UPDATE_FREQ == 0:
                agent.update_target_model()
            
            # Evaluate on test set
            test_metrics = agent.evaluate(X_test, y_test)
            test_auc = test_metrics['auc']
            test_auc_history.append(test_auc)
            
            # Early stopping
            if test_auc > best_test_auc:
                best_test_auc = test_auc
                patience_counter = 0
                # Save best model
                agent.save('best_dqn_model.pth')
                logger.info(f"New best model saved with AUC: {best_test_auc:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= PATIENCE:
                    logger.info(f"Early stopping triggered at episode {episode+1}")
                    break
            
            # Log progress
            avg_reward = np.mean(episode_rewards)
            rewards_history.append(avg_reward)
            
            if (episode + 1) % 5 == 0 or episode == 0:
                train_metrics = agent.evaluate(X_train, y_train)
                logger.info(
                    f"Episode {episode+1}/{EPISODES} - "
                    f"Train Acc: {train_metrics['accuracy']:.4f}, "
                    f"Test Acc: {test_metrics['accuracy']:.4f}, "
                    f"Test AUC: {test_metrics['auc']:.4f}, "
                    f"Test FAR: {test_metrics['far']:.4f}, "
                    f"Epsilon: {agent.epsilon:.3f}"
                )
        
        # Final evaluation
        logger.info("\n=== Training Complete ===")
        
        # Load best model
        best_agent = DQNClassifier.load('best_dqn_model.pth')
        
        # Evaluate on both sets
        logger.info("\n=== Best Model Evaluation ===")
        
        train_metrics = best_agent.evaluate(X_train, y_train)
        logger.info("\nTraining Set Metrics:")
        for metric, value in train_metrics.items():
            logger.info(f"{metric}: {value:.4f}")
        
        test_metrics = best_agent.evaluate(X_test, y_test)
        logger.info("\nTest Set Metrics:")
        for metric, value in test_metrics.items():
            logger.info(f"{metric}: {value:.4f}")
        
        # Plot training rewards
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(rewards_history)
        plt.title('Training Rewards')
        plt.xlabel('Episode')
        plt.ylabel('Average Reward')
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(test_auc_history)
        plt.title('Test AUC Over Time')
        plt.xlabel('Episode')
        plt.ylabel('AUC')
        plt.grid(True)
        
        plot_path = 'training_metrics.png'
        plt.tight_layout()
        plt.savefig(plot_path)
        logger.info(f"\nTraining plots saved to {os.path.abspath(plot_path)}")
        plt.close()
        
        return best_agent, (X_test, y_test, scaler)
        
    except Exception as e:
        logger.error(f"Error during training: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    try:
        agent, test_data = train_dqn()
        X_test, y_test, scaler = test_data
        
        # Example predictions
        logger.info("\n=== Example Predictions ===")
        for i in range(3):  # Show first 3 examples
            sample = X_test[i]
            prediction = agent.predict(sample)
            prob = torch.softmax(
                agent.model(torch.FloatTensor(sample).unsqueeze(0).to(agent.device)),
                dim=1
            )[0, 1].item()
            
            logger.info(
                f"Sample {i+1}: "
                f"Features: {sample.round(2)}, "
                f"True: {y_test[i]}, "
                f"Pred: {prediction}, "
                f"Prob(Fraud): {prob:.4f}"
            )
            
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)