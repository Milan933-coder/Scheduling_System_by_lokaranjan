"""
Training Pipeline for Enhanced Hybrid Timetable Generator

This pipeline implements:
1. Self-play data generation using MCTS
2. Neural network training with policy and value losses
3. Model evaluation and comparison
4. Iterative improvement cycles
5. Comprehensive logging and monitoring

Fixed version with proper GPU support and error handling.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
import os
import time
import pickle
from datetime import datetime
from collections import deque, defaultdict
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional
import random

# Set device globally
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🚀 Training Pipeline using device: {DEVICE}")

# Import from the fixed UCTP module
try:
    from UCTP_fixed import *
except ImportError:
    print("Please ensure UCTP_fixed.py is in the same directory")
    raise

# ============================================================================
# TRAINING DATA STRUCTURES
# ============================================================================

class TimetableTrainingExample:
    """Single training example containing state, action probabilities, and value"""
    def __init__(self, state_data, action_probs, value, reward=0.0):
        self.state_data = state_data  # Graph data
        self.action_probs = action_probs  # Target policy
        self.value = value  # Target value
        self.reward = reward  # Immediate reward

class TimetableDataset(Dataset):
    """PyTorch dataset for timetable training examples"""
    def __init__(self, examples: List[TimetableTrainingExample]):
        self.examples = examples
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return self.examples[idx]

# ============================================================================
# TRAINING PIPELINE
# ============================================================================

class TimetableTrainingPipeline:
    def __init__(self, 
                 generator: 'HybridTimetableGenerator',
                 config: Dict = None):
        """
        Initialize training pipeline
        
        Args:
            generator: The hybrid timetable generator
            config: Training configuration dictionary
        """
        self.generator = generator
        self.network = generator.network
        self.mcts = generator.mcts
        
        # Default training configuration
        self.config = {
            'learning_rate': 1e-3,
            'weight_decay': 1e-4,
            'batch_size': 32,
            'epochs_per_iteration': 10,
            'num_training_iterations': 50,
            'mcts_simulations': 200,
            'self_play_games': 10,
            'evaluation_games': 5,
            'data_buffer_size': 10000,
            'min_buffer_size': 1000,
            'temperature': 1.0,
            'value_loss_weight': 1.0,
            'policy_loss_weight': 1.0,
            'save_interval': 5,
            'evaluation_interval': 2,
            'use_wandb': False
        }
        
        if config:
            self.config.update(config)
        
        # Training components
        self.optimizer = optim.Adam(
            self.network.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )
        
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=10,
            gamma=0.9
        )
        
        # Data storage
        self.training_buffer = deque(maxlen=self.config['data_buffer_size'])
        
        # Metrics tracking
        self.training_history = {
            'policy_loss': [],
            'value_loss': [],
            'total_loss': [],
            'learning_rate': [],
            'fitness_scores': [],
            'completion_rates': [],
            'training_time': []
        }
        
        # Create directories
        self.checkpoint_dir = "checkpoints"
        self.logs_dir = "logs"
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)
        
        # Initialize wandb if requested
        if self.config['use_wandb']:
            try:
                import wandb
                wandb.init(
                    project="timetable-generator",
                    config=self.config,
                    name=f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                )
            except ImportError:
                print("⚠ wandb not installed, skipping wandb logging")
                self.config['use_wandb'] = False
        
        print("🚀 Training Pipeline Initialized")
        print(f"📊 Configuration: {json.dumps(self.config, indent=2)}")
    
    def generate_self_play_data(self, num_games: int) -> List[TimetableTrainingExample]:
        """Generate training data through self-play"""
        print(f"🎮 Generating {num_games} self-play games...")
        training_examples = []
        
        for game in range(num_games):
            print(f"  Game {game + 1}/{num_games}")
            
            # Create fresh state
            current_state = EnhancedGraphTimetableState(
                self.generator.teachers,
                self.generator.classrooms,
                self.generator.subjects,
                self.generator.class_groups
            )
            
            game_examples = []
            move_count = 0
            max_moves = 100  # Prevent infinite games
            
            while not current_state.is_complete() and move_count < max_moves:
                move_count += 1
                
                # Get valid actions
                valid_actions = current_state.get_possible_actions()
                if not valid_actions:
                    break
                
                # Run MCTS to get action probabilities
                try:
                    action_probs, root = self.mcts.search(
                        current_state,
                        self.generator.action_mapping,
                        num_simulations=self.config['mcts_simulations']
                    )
                except Exception as e:
                    print(f"    Error in MCTS search: {e}")
                    break
                
                if not action_probs:
                    break
                
                # Store current state and action probabilities
                try:
                    graph_data = current_state.to_graph_data()
                    # Get value estimate from root
                    value_estimate = root.mean_value if root.visit_count > 0 else 0.0
                    
                    # Create training example
                    example = TimetableTrainingExample(
                        state_data=graph_data,
                        action_probs=action_probs.copy(),
                        value=value_estimate
                    )
                    
                    game_examples.append(example)
                except Exception as e:
                    print(f"    Error creating training example: {e}")
                    continue
                
                # Select action based on temperature
                try:
                    if self.config['temperature'] > 0:
                        actions = list(action_probs.keys())
                        probs = list(action_probs.values())
                        if sum(probs) > 0:
                            probs = np.array(probs) ** (1.0 / self.config['temperature'])
                            probs = probs / probs.sum()
                            selected_action = np.random.choice(actions, p=probs)
                        else:
                            selected_action = random.choice(actions)
                    else:
                        selected_action = max(action_probs.keys(), key=action_probs.get)
                    
                    # Apply action
                    time_slot, class_id, subject_id, teacher_id, room_id = selected_action
                    current_state.make_assignment(time_slot, class_id, subject_id, teacher_id, room_id)
                    
                except (ValueError, Exception) as e:
                    print(f"    Invalid action or assignment error: {e}")
                    break
            
            # Calculate final game value
            try:
                final_fitness = current_state.calculate_fitness()
                final_value = final_fitness / 1000.0  # Normalize
                
                # Update all examples with final game outcome
                for example in game_examples:
                    example.value = final_value
                
                training_examples.extend(game_examples)
                print(f"    Game {game + 1}: {len(game_examples)} moves, fitness={final_fitness:.1f}")
                
            except Exception as e:
                print(f"    Error calculating final fitness: {e}")
                continue
        
        print(f"✅ Generated {len(training_examples)} training examples")
        return training_examples
    
    def train_network(self, training_examples: List[TimetableTrainingExample]) -> Dict:
        """Train the neural network on collected examples"""
        print(f"🧠 Training network on {len(training_examples)} examples...")
        
        # Create dataset and dataloader
        dataset = TimetableDataset(training_examples)
        dataloader = DataLoader(
            dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            collate_fn=self._collate_batch
        )
        
        self.network.train()
        epoch_losses = {'policy': [], 'value': [], 'total': []}
        
        for epoch in range(self.config['epochs_per_iteration']):
            epoch_policy_loss = 0.0
            epoch_value_loss = 0.0
            epoch_total_loss = 0.0
            batch_count = 0
            
            for batch in dataloader:
                batch_count += 1
                self.optimizer.zero_grad()
                
                # Forward pass
                batch_graphs, batch_action_probs, batch_values = batch
                total_policy_loss = 0.0
                total_value_loss = 0.0
                
                for graph_data, target_action_probs, target_value in zip(
                    batch_graphs, batch_action_probs, batch_values
                ):
                    try:
                        # Ensure graph data is on correct device
                        graph_data = graph_data.to(DEVICE)
                        
                        # Get network predictions
                        node_embeddings, predicted_value = self.network(graph_data)
                        
                        # Value loss
                        target_value_tensor = torch.tensor([target_value], dtype=torch.float, device=DEVICE)
                        value_loss = nn.MSELoss()(
                            predicted_value.unsqueeze(0),
                            target_value_tensor
                        )
                        
                        # Policy loss (only for valid actions)
                        if target_action_probs:
                            valid_actions = list(target_action_probs.keys())
                            try:
                                action_scores = self.network.compute_action_scores(
                                    node_embeddings,
                                    valid_actions,
                                    self.generator.action_mapping
                                )
                                
                                if len(action_scores) > 0:
                                    predicted_probs = torch.softmax(action_scores, dim=0)
                                    target_probs = torch.tensor(
                                        [target_action_probs[action] for action in valid_actions],
                                        dtype=torch.float,
                                        device=DEVICE
                                    )
                                    
                                    # Normalize target probabilities
                                    if target_probs.sum() > 0:
                                        target_probs = target_probs / target_probs.sum()
                                    
                                    # Cross-entropy loss
                                    policy_loss = -torch.sum(target_probs * torch.log(predicted_probs + 1e-8))
                                else:
                                    policy_loss = torch.tensor(0.0, device=DEVICE)
                            except Exception as e:
                                print(f"      Error computing policy loss: {e}")
                                policy_loss = torch.tensor(0.0, device=DEVICE)
                        else:
                            policy_loss = torch.tensor(0.0, device=DEVICE)
                        
                        total_policy_loss += policy_loss
                        total_value_loss += value_loss
                        
                    except Exception as e:
                        print(f"      Error in forward pass: {e}")
                        continue
                
                if batch_count > 0:
                    # Combined loss
                    policy_loss_avg = total_policy_loss / len(batch_graphs)
                    value_loss_avg = total_value_loss / len(batch_graphs)
                    total_loss = (
                        self.config['policy_loss_weight'] * policy_loss_avg +
                        self.config['value_loss_weight'] * value_loss_avg
                    )
                    
                    # Backward pass
                    total_loss.backward()
                    
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
                    
                    self.optimizer.step()
                    
                    # Track losses
                    epoch_policy_loss += policy_loss_avg.item()
                    epoch_value_loss += value_loss_avg.item()
                    epoch_total_loss += total_loss.item()
            
            # Average losses for epoch
            if batch_count > 0:
                epoch_policy_loss /= batch_count
                epoch_value_loss /= batch_count
                epoch_total_loss /= batch_count
                
                epoch_losses['policy'].append(epoch_policy_loss)
                epoch_losses['value'].append(epoch_value_loss)
                epoch_losses['total'].append(epoch_total_loss)
                
                if epoch % 2 == 0:
                    print(f"    Epoch {epoch + 1}: Policy={epoch_policy_loss:.4f}, "
                          f"Value={epoch_value_loss:.4f}, Total={epoch_total_loss:.4f}")
        
        # Update learning rate
        self.scheduler.step()
        
        # Calculate average losses
        avg_losses = {
            'policy_loss': np.mean(epoch_losses['policy']) if epoch_losses['policy'] else 0.0,
            'value_loss': np.mean(epoch_losses['value']) if epoch_losses['value'] else 0.0,
            'total_loss': np.mean(epoch_losses['total']) if epoch_losses['total'] else 0.0,
            'learning_rate': self.optimizer.param_groups[0]['lr']
        }
        
        print(f"✅ Training completed: Avg loss={avg_losses['total_loss']:.4f}")
        return avg_losses
    
    def _collate_batch(self, batch):
        """Custom collate function for batching graph data"""
        graphs = []
        action_probs = []
        values = []
        
        for example in batch:
            graphs.append(example.state_data)
            action_probs.append(example.action_probs)
            values.append(example.value)
        
        return graphs, action_probs, values
    
    def evaluate_model(self, num_games: int = None) -> Dict:
        """Evaluate current model performance"""
        if num_games is None:
            num_games = self.config['evaluation_games']
        
        print(f"📊 Evaluating model on {num_games} games...")
        
        self.network.eval()
        fitness_scores = []
        completion_rates = []
        
        with torch.no_grad():
            for i in range(num_games):
                try:
                    # Generate a complete solution
                    solution = self.generator.generate_seeding_solution(
                        max_iterations=50,
                        mcts_sims=100
                    )
                    
                    fitness = solution.calculate_fitness()
                    completion = 1.0 if solution.is_complete() else 0.0
                    
                    fitness_scores.append(fitness)
                    completion_rates.append(completion)
                    
                    print(f"    Game {i + 1}: Fitness={fitness:.1f}, Complete={completion}")
                    
                except Exception as e:
                    print(f"    Error in evaluation game {i + 1}: {e}")
                    continue
        
        if fitness_scores:
            metrics = {
                'avg_fitness': np.mean(fitness_scores),
                'std_fitness': np.std(fitness_scores),
                'max_fitness': np.max(fitness_scores),
                'min_fitness': np.min(fitness_scores),
                'avg_completion': np.mean(completion_rates),
                'completion_rate': np.sum(completion_rates) / len(completion_rates)
            }
        else:
            metrics = {
                'avg_fitness': 0.0,
                'std_fitness': 0.0,
                'max_fitness': 0.0,
                'min_fitness': 0.0,
                'avg_completion': 0.0,
                'completion_rate': 0.0
            }
        
        print(f"📊 Evaluation results: Avg fitness={metrics['avg_fitness']:.1f}, "
              f"Completion rate={metrics['completion_rate']:.2%}")
        
        return metrics
    
    def save_checkpoint(self, iteration: int, metrics: Dict = None):
        """Save model checkpoint"""
        checkpoint = {
            'iteration': iteration,
            'model_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'training_history': self.training_history,
            'config': self.config,
            'metrics': metrics
        }
        
        checkpoint_path = os.path.join(
            self.checkpoint_dir,
            f"checkpoint_iteration_{iteration}.pt"
        )
        
        torch.save(checkpoint, checkpoint_path)
        
        # Also save the best model
        if metrics and 'avg_fitness' in metrics:
            best_path = os.path.join(self.checkpoint_dir, "best_model.pt")
            if not os.path.exists(best_path):
                torch.save(checkpoint, best_path)
                print(f"💾 Saved first checkpoint as best: {metrics['avg_fitness']:.1f}")
            else:
                best_checkpoint = torch.load(best_path)
                if best_checkpoint['metrics']['avg_fitness'] < metrics['avg_fitness']:
                    torch.save(checkpoint, best_path)
                    print(f"🏆 New best model saved: {metrics['avg_fitness']:.1f}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        self.network.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.training_history = checkpoint['training_history']
        
        print(f"✅ Loaded checkpoint from iteration {checkpoint['iteration']}")
        return checkpoint['iteration']
    
    def plot_training_history(self):
        """Plot training metrics"""
        if not self.training_history['total_loss']:
            print("No training history to plot")
            return
        
        try:
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            
            # Loss curves
            axes[0, 0].plot(self.training_history['policy_loss'], label='Policy Loss')
            axes[0, 0].plot(self.training_history['value_loss'], label='Value Loss')
            axes[0, 0].plot(self.training_history['total_loss'], label='Total Loss')
            axes[0, 0].set_title('Training Losses')
            axes[0, 0].legend()
            axes[0, 0].grid(True)
            
            # Fitness scores
            if self.training_history['fitness_scores']:
                axes[0, 1].plot(self.training_history['fitness_scores'])
                axes[0, 1].set_title('Average Fitness Score')
                axes[0, 1].grid(True)
            
            # Completion rates
            if self.training_history['completion_rates']:
                axes[1, 0].plot(self.training_history['completion_rates'])
                axes[1, 0].set_title('Completion Rate')
                axes[1, 0].grid(True)
            
            # Learning rate
            axes[1, 1].plot(self.training_history['learning_rate'])
            axes[1, 1].set_title('Learning Rate')
            axes[1, 1].grid(True)
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.logs_dir, 'training_history.png'), dpi=300)
            plt.show()
            
        except Exception as e:
            print(f"Error plotting training history: {e}")
    
    def run_training(self):
        """Main training loop"""
        print("🚀 Starting Training Pipeline")
        print("=" * 70)
        
        start_time = time.time()
        
        for iteration in range(self.config['num_training_iterations']):
            iteration_start = time.time()
            
            print(f"\n🔄 Training Iteration {iteration + 1}/{self.config['num_training_iterations']}")
            print("-" * 50)
            
            # Generate self-play data
            try:
                new_examples = self.generate_self_play_data(self.config['self_play_games'])
                self.training_buffer.extend(new_examples)
            except Exception as e:
                print(f"Error generating self-play data: {e}")
                continue
            
            # Train only if we have enough data
            if len(self.training_buffer) >= self.config['min_buffer_size']:
                try:
                    # Sample training data from buffer
                    training_examples = list(self.training_buffer)
                    
                    # Train network
                    loss_metrics = self.train_network(training_examples)
                    
                    # Update training history
                    self.training_history['policy_loss'].append(loss_metrics['policy_loss'])
                    self.training_history['value_loss'].append(loss_metrics['value_loss'])
                    self.training_history['total_loss'].append(loss_metrics['total_loss'])
                    self.training_history['learning_rate'].append(loss_metrics['learning_rate'])
                    
                except Exception as e:
                    print(f"Error during training: {e}")
                    continue
            
            # Evaluate model periodically
            eval_metrics = None
            if (iteration + 1) % self.config['evaluation_interval'] == 0:
                try:
                    eval_metrics = self.evaluate_model()
                    self.training_history['fitness_scores'].append(eval_metrics['avg_fitness'])
                    self.training_history['completion_rates'].append(eval_metrics['completion_rate'])
                    
                    # Log to wandb if enabled
                    if self.config['use_wandb']:
                        try:
                            import wandb
                            wandb.log({
                                'iteration': iteration + 1,
                                'avg_fitness': eval_metrics['avg_fitness'],
                                'completion_rate': eval_metrics['completion_rate'],
                                'policy_loss': self.training_history['policy_loss'][-1] if self.training_history['policy_loss'] else 0,
                                'value_loss': self.training_history['value_loss'][-1] if self.training_history['value_loss'] else 0,
                                'total_loss': self.training_history['total_loss'][-1] if self.training_history['total_loss'] else 0
                            })
                        except ImportError:
                            pass
                        
                except Exception as e:
                    print(f"Error during evaluation: {e}")
            
            # Save checkpoint periodically
            if (iteration + 1) % self.config['save_interval'] == 0:
                try:
                    self.save_checkpoint(iteration + 1, eval_metrics)
                except Exception as e:
                    print(f"Error saving checkpoint: {e}")
            
            iteration_time = time.time() - iteration_start
            self.training_history['training_time'].append(iteration_time)
            
            print(f"⏱ Iteration {iteration + 1} completed in {iteration_time:.1f}s")
        
        total_time = time.time() - start_time
        print(f"\n🎉 Training completed in {total_time:.1f}s!")
        
        # Final evaluation and save
        try:
            final_metrics = self.evaluate_model(num_games=10)
            self.save_checkpoint(self.config['num_training_iterations'], final_metrics)
        except Exception as e:
            print(f"Error in final evaluation: {e}")
            final_metrics = {'avg_fitness': 0.0, 'completion_rate': 0.0}
        
        # Plot training history
        self.plot_training_history()
        
        # Save final training data
        try:
            training_data_path = os.path.join(self.logs_dir, 'training_history.json')
            with open(training_data_path, 'w') as f:
                json.dump(self.training_history, f, indent=2)
            print(f"📊 Training data saved to {training_data_path}")
        except Exception as e:
            print(f"Error saving training data: {e}")
        
        if self.config['use_wandb']:
            try:
                import wandb
                wandb.finish()
            except ImportError:
                pass
        
        return final_metrics

# ============================================================================
# CONFIGURATION AND USAGE EXAMPLE
# ============================================================================

def create_training_config():
    """Create training configuration"""
    return {
        'learning_rate': 1e-3,
        'weight_decay': 1e-4,
        'batch_size': 16,  # Smaller batch size for graph data
        'epochs_per_iteration': 10,
        'num_training_iterations': 30,
        'mcts_simulations': 150,
        'self_play_games': 5,
        'evaluation_games': 3,
        'data_buffer_size': 5000,
        'min_buffer_size': 500,
        'temperature': 1.0,
        'value_loss_weight': 1.0,
        'policy_loss_weight': 1.0,
        'save_interval': 5,
        'evaluation_interval': 2,
        'use_wandb': False  # Set to True if you want to use Weights & Biases
    }

def main_training():
    """Main training function"""
    print("🎓 TIMETABLE GENERATOR TRAINING PIPELINE")
    print("=" * 60)
    
    try:
        # Create sample data
        teachers, classrooms, subjects, class_groups = create_enhanced_sample_data()
        
        # Create generator
        generator = HybridTimetableGenerator(teachers, classrooms, subjects, class_groups)
        
        # Create training pipeline
        config = create_training_config()
        pipeline = TimetableTrainingPipeline(generator, config)
        
        # Run training
        final_metrics = pipeline.run_training()
        
        print("\n🏆 Training Results:")
        print(f"  Final Average Fitness: {final_metrics['avg_fitness']:.1f}")
        print(f"  Final Completion Rate: {final_metrics['completion_rate']:.2%}")
        
        return pipeline
        
    except Exception as e:
        print(f"❌ Error in main training: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    # Install required packages if needed
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("⚠ matplotlib not installed. Install with: pip install matplotlib")
    
    # Run training
    pipeline = main_training()