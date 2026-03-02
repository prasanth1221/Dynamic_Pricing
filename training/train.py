"""
Enhanced Training Pipeline for Multi-Route, Multi-Class Airline RL
Supports training across multiple routes and joint Economy+Business pricing
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import json
import torch
from datetime import datetime
from pathlib import Path
import pickle
import os
import sys

# Add project root to Python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

# Import custom modules
from environment.airline_env import AirlineRevenueEnv
from agents.model import DQNAgent
from config.config import AGENT_CONFIG, TRAINING_CONFIG


class AirlineRLTrainer:
    """
    Enhanced training pipeline for Multi-Route, Multi-Class Airline RL

    Features:
    - Multi-route training (agent learns across different routes)
    - Multi-class pricing (Economy + Business joint optimization)
    - Comprehensive evaluation and visualization
    - Route-specific and overall performance tracking
    """

    def __init__(self, env, agent, save_dir='models/trained_models/',
                 results_dir='results/', log_dir='logs/'):
        self.env = env
        self.agent = agent
        self.save_dir = save_dir
        self.results_dir = results_dir
        self.log_dir = log_dir

        # Create directories
        for directory in [save_dir, results_dir, log_dir]:
            Path(directory).mkdir(parents=True, exist_ok=True)

        # Training statistics - OVERALL
        self.episode_rewards = []
        self.episode_revenues = []
        self.episode_load_factors = []
        self.episode_econ_load = []
        self.episode_bus_load = []
        self.episode_lengths = []
        self.losses = []
        self.episode_routes = []  # Track which route was used

        # Per-route statistics
        self.route_performance = {}

        print(f"✓ Trainer initialized")
        print(f"  Save dir: {save_dir}")
        print(f"  Results dir: {results_dir}")

    # ─────────────────────────────────────────────────────────────────────────
    # CURRICULUM HELPER  (proper class method — NOT nested inside train())
    # ─────────────────────────────────────────────────────────────────────────
    def _get_curriculum_routes(self, episode):
        """
        Return the list of routes to sample from based on current episode.

        Phases (from TRAINING_CONFIG['curriculum_phases']):
          - Phase 1 (ep 0-1499):   3 busiest routes  → agent learns basics fast
          - Phase 2 (ep 1500-3499): 10 routes        → broader generalisation
          - Phase 3 (ep 3500+):    all 30 routes     → full multi-route mastery

        If curriculum_learning is False (or phases not defined), returns all routes.
        """
        curriculum_phases = TRAINING_CONFIG.get('curriculum_phases', [])

        if not curriculum_phases or not TRAINING_CONFIG.get('curriculum_learning', False):
            return self.env.routes  # curriculum disabled → use all routes

        # Sort routes: highest avg economy price first (proxy for busy/easy routes)
        try:
            sorted_routes = sorted(
                self.env.routes,
                key=lambda r: self.env.all_route_stats[r].get('econ_avg_price', 0),
                reverse=True
            )
        except Exception:
            sorted_routes = list(self.env.routes)  # fallback: original order

        # Find which phase we are currently in
        for phase in curriculum_phases:
            if episode < phase['end_episode']:
                n = phase['num_routes']
                if n is None:
                    return sorted_routes          # None means "all routes"
                return sorted_routes[:n]

        return sorted_routes  # past all phases → all routes

    # ─────────────────────────────────────────────────────────────────────────
    # MAIN TRAINING LOOP
    # ─────────────────────────────────────────────────────────────────────────
    def train(self, num_episodes=1000, target_update_freq=10,
              save_freq=100, eval_freq=50, verbose=True):
        """
        Train the DQN agent across multiple routes and classes.

        Args:
            num_episodes:       Total training episodes
            target_update_freq: Update target network every N episodes
            save_freq:          Save checkpoint every N episodes
            eval_freq:          Evaluate agent every N episodes
            verbose:            Print detailed progress
        """

        print("\n" + "=" * 80)
        print(f"  🚀 STARTING MULTI-ROUTE MULTI-CLASS TRAINING")
        print("=" * 80)
        print(f"Episodes:        {num_episodes}")
        print(f"Device:          {self.agent.device}")
        print(f"Available routes:{len(self.env.routes)}")
        print(f"Action space:    {self.env.action_space.n} (joint Economy+Business pricing)")
        print(f"State space:     {self.env.observation_space.shape[0]} features")
        curriculum_on = TRAINING_CONFIG.get('curriculum_learning', False)
        print(f"Curriculum:      {'ON' if curriculum_on else 'OFF'}")
        print("=" * 80)

        best_avg_reward = -np.inf
        best_episode = 0

        # ── Training loop ────────────────────────────────────────────────────
        for episode in tqdm(range(num_episodes), desc="Training"):

            # Curriculum: choose which route to train on this episode
            curriculum_routes = self._get_curriculum_routes(episode)
            sampled_route = np.random.choice(curriculum_routes)
            self.env.fixed_route = sampled_route

            state, info = self.env.reset()
            episode_reward = 0
            episode_loss = []
            done = False
            step = 0

            # Track route for this episode
            current_route = self.env.route

            while not done:
                # Select action (epsilon-greedy)
                action = self.agent.select_action(state, training=True)

                # Take action in environment
                next_state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated

                # Store transition in replay buffer
                self.agent.store_transition(state, action, reward, next_state, done)

                # Train agent (if enough samples)
                loss = self.agent.train_step()
                if loss is not None:
                    episode_loss.append(loss)

                episode_reward += reward
                state = next_state
                step += 1

            # Update target network periodically
            if episode % target_update_freq == 0:
                self.agent.update_target_network()

            # Decay epsilon
            self.agent.update_epsilon()

            # Update episode count in agent
            self.agent.episode_count += 1

            # Store episode statistics
            total_load = (info['econ_sold'] + info['bus_sold']) / self.env.total_seats
            econ_load  = info['econ_sold'] / self.env.econ_seats_total
            bus_load   = info['bus_sold']  / self.env.bus_seats_total

            self.episode_rewards.append(episode_reward)
            self.episode_revenues.append(info['revenue'])
            self.episode_load_factors.append(total_load)
            self.episode_econ_load.append(econ_load)
            self.episode_bus_load.append(bus_load)
            self.episode_lengths.append(step)
            self.episode_routes.append(current_route)

            if episode_loss:
                self.losses.append(np.mean(episode_loss))

            # Update per-route statistics
            if current_route not in self.route_performance:
                self.route_performance[current_route] = {
                    'rewards': [], 'revenues': [],
                    'load_factors': [], 'econ_load': [], 'bus_load': []
                }

            self.route_performance[current_route]['rewards'].append(episode_reward)
            self.route_performance[current_route]['revenues'].append(info['revenue'])
            self.route_performance[current_route]['load_factors'].append(total_load)
            self.route_performance[current_route]['econ_load'].append(econ_load)
            self.route_performance[current_route]['bus_load'].append(bus_load)

            # ── Verbose logging every 50 episodes ───────────────────────────
            if verbose and episode % 50 == 0 and episode > 0:
                avg_reward  = np.mean(self.episode_rewards[-50:])
                avg_revenue = np.mean(self.episode_revenues[-50:])
                avg_load    = np.mean(self.episode_load_factors[-50:])
                avg_econ    = np.mean(self.episode_econ_load[-50:])
                avg_bus     = np.mean(self.episode_bus_load[-50:])

                # Curriculum phase info
                active_routes = self._get_curriculum_routes(episode)
                phase_info = (f"{len(active_routes)}/{len(self.env.routes)} routes active"
                              if curriculum_on else "all routes")

                print(f"\n{'=' * 80}")
                print(f"Episode {episode}/{num_episodes}  |  📚 Curriculum: {phase_info}")
                print(f"{'=' * 80}")
                print(f"  Avg Reward  (last 50):  {avg_reward:>10.2f}")
                print(f"  Avg Revenue:            ₹{avg_revenue:>10,.0f}")
                print(f"  Avg Load Factor:        {avg_load * 100:>9.1f}%")
                print(f"    ├─ Economy Load:      {avg_econ * 100:>9.1f}%")
                print(f"    └─ Business Load:     {avg_bus  * 100:>9.1f}%")
                print(f"  Epsilon:                {self.agent.epsilon:>10.3f}")
                if self.losses:
                    print(f"  Avg Loss (last 50):     {np.mean(self.losses[-50:]):>10.4f}")
                print(f"  Current Route:          {current_route}")
                print("=" * 80)

                # Track best model
                if avg_reward > best_avg_reward:
                    best_avg_reward = avg_reward
                    best_episode = episode
                    self.save_best_model()

            # ── Periodic evaluation ──────────────────────────────────────────
            if eval_freq > 0 and episode % eval_freq == 0 and episode > 0:
                print(f"\n🎯 Running evaluation at episode {episode}...")
                eval_results = self.evaluate(num_episodes=5, render=False)
                self.log_evaluation(episode, eval_results)

            # ── Save checkpoint ──────────────────────────────────────────────
            if episode % save_freq == 0 and episode > 0:
                self.save_checkpoint(episode)

        print("\n" + "=" * 80)
        print("  ✅ TRAINING COMPLETED!")
        print("=" * 80)
        print(f"Best average reward: {best_avg_reward:.2f} (episode {best_episode})")

        # Save final model and statistics
        self.save_final_model()
        self.save_training_stats()

        # Create visualizations
        print("\n📊 Generating training visualizations...")
        self.plot_training_progress()
        self.plot_route_performance()
        self.plot_class_performance()

        return self.get_training_summary()

    # ─────────────────────────────────────────────────────────────────────────
    # EVALUATION
    # ─────────────────────────────────────────────────────────────────────────
    def evaluate(self, num_episodes=5, render=False):
        """Evaluate agent performance (greedy policy)."""

        print(f"\n{'=' * 80}")
        print(f"  📊 EVALUATION ({num_episodes} episodes)")
        print(f"{'=' * 80}")

        eval_rewards = []
        eval_revenues = []
        eval_load_factors = []
        eval_routes = []

        # Save current epsilon and go greedy
        original_epsilon = self.agent.epsilon
        self.agent.epsilon = 0.0

        for ep in range(num_episodes):
            state, info = self.env.reset()
            episode_reward = 0
            done = False

            while not done:
                action = self.agent.select_action(state, training=False)
                next_state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                state = next_state
                episode_reward += reward

                if render:
                    self.env.render()

            summary = self.env.get_episode_summary()
            eval_rewards.append(episode_reward)
            eval_revenues.append(summary['total_revenue'])
            eval_load_factors.append(summary['load_factor'])
            eval_routes.append(summary['route'])

        # Restore epsilon
        self.agent.epsilon = original_epsilon

        print(f"\n  Results:")
        print(f"    Avg Reward:      {np.mean(eval_rewards):>10.2f}")
        print(f"    Avg Revenue:     ₹{np.mean(eval_revenues):>10,.0f}")
        print(f"    Avg Load Factor: {np.mean(eval_load_factors) * 100:>10.1f}%")
        print(f"    Routes tested:   {len(set(eval_routes))}")
        print(f"{'=' * 80}\n")

        return {
            'rewards': eval_rewards,
            'revenues': eval_revenues,
            'load_factors': eval_load_factors,
            'routes': eval_routes
        }

    def log_evaluation(self, episode, eval_results):
        log_file = Path(self.results_dir) / "evaluation_log.txt"

        revenues     = eval_results.get("revenues", [])
        rewards      = eval_results.get("rewards", [])
        load_factors = eval_results.get("load_factors", [])

        revenues_mean = sum(revenues) / len(revenues)
        revenues_std  = (sum((x - revenues_mean) ** 2 for x in revenues) / len(revenues)) ** 0.5
        rewards_mean  = sum(rewards) / len(rewards)
        load_mean     = sum(load_factors) / len(load_factors)

        with open(log_file, "a", encoding="utf-8") as f:
            f.write("\n" + "=" * 60 + "\n")
            f.write(f"Episode {episode}\n")
            f.write(f"Avg Reward:        {rewards_mean:.2f}\n")
            f.write(f"Avg Revenue:       ₹{revenues_mean:,.0f} ± {revenues_std:,.0f}\n")
            f.write(f"Avg Load Factor:   {load_mean * 100:.1f}%\n")
            f.write("=" * 60 + "\n")

    # ─────────────────────────────────────────────────────────────────────────
    # SAVE / LOAD
    # ─────────────────────────────────────────────────────────────────────────
    def save_checkpoint(self, episode):
        filepath = os.path.join(self.save_dir, f'checkpoint_ep{episode}.pth')
        self.agent.save_model(filepath)
        print(f"\n💾 Checkpoint saved: {filepath}")

    def save_best_model(self):
        filepath = os.path.join(self.save_dir, 'best_model.pth')
        self.agent.save_model(filepath)

    def save_final_model(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(self.save_dir, f'final_model_{timestamp}.pth')
        self.agent.save_model(filepath)
        print(f"\n💾 Final model saved: {filepath}")

    def save_training_stats(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        stats = {
            'timestamp': timestamp,
            'num_episodes': len(self.episode_rewards),
            'episode_rewards': self.episode_rewards,
            'episode_revenues': self.episode_revenues,
            'episode_load_factors': self.episode_load_factors,
            'episode_econ_load': self.episode_econ_load,
            'episode_bus_load': self.episode_bus_load,
            'episode_lengths': self.episode_lengths,
            'episode_routes': self.episode_routes,
            'losses': self.losses,
            'route_performance': {
                k: {
                    'rewards': v['rewards'],
                    'revenues': v['revenues'],
                    'load_factors': v['load_factors'],
                    'econ_load': v['econ_load'],
                    'bus_load': v['bus_load']
                }
                for k, v in self.route_performance.items()
            },
            'final_epsilon': self.agent.epsilon
        }

        filepath = os.path.join(self.results_dir, f'training_stats_{timestamp}.json')
        with open(filepath, 'w') as f:
            json.dump(stats, f, indent=2)

        print(f"📊 Training stats saved: {filepath}")

    def get_training_summary(self):
        summary = {
            'total_episodes': len(self.episode_rewards),
            'final_avg_reward': np.mean(self.episode_rewards[-100:]),
            'final_avg_revenue': np.mean(self.episode_revenues[-100:]),
            'final_avg_load': np.mean(self.episode_load_factors[-100:]),
            'final_econ_load': np.mean(self.episode_econ_load[-100:]),
            'final_bus_load': np.mean(self.episode_bus_load[-100:]),
            'best_reward': np.max(self.episode_rewards),
            'best_revenue': np.max(self.episode_revenues),
            'routes_trained': list(self.route_performance.keys()),
            'num_routes': len(self.route_performance)
        }
        return summary

    # ─────────────────────────────────────────────────────────────────────────
    # VISUALISATIONS
    # ─────────────────────────────────────────────────────────────────────────
    def plot_training_progress(self):
        fig, axes = plt.subplots(3, 2, figsize=(18, 14))
        fig.suptitle('Multi-Route Multi-Class Training Progress',
                     fontsize=16, fontweight='bold')

        axes[0, 0].plot(self.episode_rewards, alpha=0.3, label='Episode Reward', color='skyblue')
        axes[0, 0].plot(self._moving_average(self.episode_rewards, 50),
                        label='MA(50)', linewidth=2, color='blue')
        axes[0, 0].set_xlabel('Episode'); axes[0, 0].set_ylabel('Reward')
        axes[0, 0].set_title('Training Rewards'); axes[0, 0].legend(); axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].plot(self.episode_revenues, alpha=0.3, label='Episode Revenue', color='lightgreen')
        axes[0, 1].plot(self._moving_average(self.episode_revenues, 50),
                        label='MA(50)', linewidth=2, color='green')
        axes[0, 1].set_xlabel('Episode'); axes[0, 1].set_ylabel('Revenue (₹)')
        axes[0, 1].set_title('Total Revenue per Episode'); axes[0, 1].legend(); axes[0, 1].grid(True, alpha=0.3)

        load_factors_pct = [lf * 100 for lf in self.episode_load_factors]
        axes[1, 0].plot(load_factors_pct, alpha=0.3, label='Load Factor', color='coral')
        axes[1, 0].plot(self._moving_average(load_factors_pct, 50),
                        label='MA(50)', linewidth=2, color='red')
        axes[1, 0].axhline(y=80, color='green', linestyle='--', alpha=0.5, label='Target (80%)')
        axes[1, 0].set_xlabel('Episode'); axes[1, 0].set_ylabel('Load Factor (%)')
        axes[1, 0].set_title('Overall Load Factor'); axes[1, 0].legend(); axes[1, 0].grid(True, alpha=0.3)

        econ_pct = [lf * 100 for lf in self.episode_econ_load]
        bus_pct  = [lf * 100 for lf in self.episode_bus_load]
        axes[1, 1].plot(self._moving_average(econ_pct, 50),
                        label='Economy MA(50)', linewidth=2, color='#3b82f6')
        axes[1, 1].plot(self._moving_average(bus_pct, 50),
                        label='Business MA(50)', linewidth=2, color='#8b5cf6')
        axes[1, 1].set_xlabel('Episode'); axes[1, 1].set_ylabel('Load Factor (%)')
        axes[1, 1].set_title('Economy vs Business Load Factors')
        axes[1, 1].legend(); axes[1, 1].grid(True, alpha=0.3)

        if self.losses:
            axes[2, 0].plot(self.losses, alpha=0.4, color='orange')
            axes[2, 0].plot(self._moving_average(self.losses, 50),
                            linewidth=2, label='MA(50)', color='darkorange')
            axes[2, 0].set_xlabel('Episode'); axes[2, 0].set_ylabel('Loss')
            axes[2, 0].set_title('Training Loss'); axes[2, 0].set_yscale('log')
            axes[2, 0].legend(); axes[2, 0].grid(True, alpha=0.3)

        axes[2, 1].plot(self.episode_lengths, alpha=0.3, color='purple')
        axes[2, 1].plot(self._moving_average(self.episode_lengths, 50),
                        linewidth=2, label='MA(50)', color='darkviolet')
        axes[2, 1].set_xlabel('Episode'); axes[2, 1].set_ylabel('Steps')
        axes[2, 1].set_title('Episode Length'); axes[2, 1].legend(); axes[2, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = os.path.join(self.results_dir, 'training_progress.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Training progress plot saved: {save_path}")
        plt.close()

    def plot_route_performance(self):
        if not self.route_performance:
            return

        num_routes = len(self.route_performance)
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle('Performance by Route', fontsize=16, fontweight='bold')

        routes = list(self.route_performance.keys())
        colors = plt.cm.tab10(np.linspace(0, 1, min(num_routes, 10)))

        avg_rewards  = [np.mean(self.route_performance[r]['rewards'])  for r in routes]
        std_rewards  = [np.std(self.route_performance[r]['rewards'])   for r in routes]
        axes[0, 0].barh(routes, avg_rewards, xerr=std_rewards, color=colors, alpha=0.7)
        axes[0, 0].set_xlabel('Average Reward'); axes[0, 0].set_title('Average Reward by Route')
        axes[0, 0].grid(True, alpha=0.3, axis='x')

        avg_revenues = [np.mean(self.route_performance[r]['revenues']) for r in routes]
        std_revenues = [np.std(self.route_performance[r]['revenues'])  for r in routes]
        bars = axes[0, 1].barh(routes, avg_revenues, xerr=std_revenues, color=colors, alpha=0.7)
        axes[0, 1].set_xlabel('Average Revenue (₹)'); axes[0, 1].set_title('Average Revenue by Route')
        axes[0, 1].grid(True, alpha=0.3, axis='x')
        for bar, val in zip(bars, avg_revenues):
            axes[0, 1].text(bar.get_width(), bar.get_y() + bar.get_height() / 2,
                            f' ₹{val:,.0f}', va='center', fontsize=8)

        avg_loads = [np.mean(self.route_performance[r]['load_factors']) * 100 for r in routes]
        axes[1, 0].barh(routes, avg_loads, color=colors, alpha=0.7)
        axes[1, 0].axvline(x=80, color='green', linestyle='--', alpha=0.5, label='Target')
        axes[1, 0].set_xlabel('Average Load Factor (%)'); axes[1, 0].set_title('Average Load Factor by Route')
        axes[1, 0].legend(); axes[1, 0].grid(True, alpha=0.3, axis='x')

        x = np.arange(len(routes)); width = 0.35
        econ_loads = [np.mean(self.route_performance[r]['econ_load']) * 100 for r in routes]
        bus_loads  = [np.mean(self.route_performance[r]['bus_load'])  * 100 for r in routes]
        axes[1, 1].bar(x - width / 2, econ_loads, width, label='Economy', color='#3b82f6', alpha=0.8)
        axes[1, 1].bar(x + width / 2, bus_loads,  width, label='Business', color='#8b5cf6', alpha=0.8)
        axes[1, 1].set_ylabel('Average Load Factor (%)'); axes[1, 1].set_title('Economy vs Business Load by Route')
        axes[1, 1].set_xticks(x); axes[1, 1].set_xticklabels(routes, rotation=45, ha='right')
        axes[1, 1].legend(); axes[1, 1].grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        save_path = os.path.join(self.results_dir, 'route_performance.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Route performance plot saved: {save_path}")
        plt.close()

    def plot_class_performance(self):
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle('Economy vs Business Class Performance',
                     fontsize=16, fontweight='bold')

        econ_pct = [lf * 100 for lf in self.episode_econ_load]
        bus_pct  = [lf * 100 for lf in self.episode_bus_load]

        axes[0, 0].plot(self._moving_average(econ_pct, 50),
                        label='Economy MA(50)', linewidth=2, color='#3b82f6')
        axes[0, 0].plot(self._moving_average(bus_pct, 50),
                        label='Business MA(50)', linewidth=2, color='#8b5cf6')
        axes[0, 0].set_xlabel('Episode'); axes[0, 0].set_ylabel('Load Factor (%)')
        axes[0, 0].set_title('Load Factor Trend'); axes[0, 0].legend(); axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].plot(self._moving_average(self.episode_revenues, 50),
                        label='Total Revenue MA(50)', linewidth=2, color='green')
        axes[0, 1].set_xlabel('Episode'); axes[0, 1].set_ylabel('Revenue (₹)')
        axes[0, 1].set_title('Revenue Trend'); axes[0, 1].legend(); axes[0, 1].grid(True, alpha=0.3)

        axes[1, 0].scatter(econ_pct, bus_pct, alpha=0.3)
        axes[1, 0].set_xlabel('Economy Load (%)'); axes[1, 0].set_ylabel('Business Load (%)')
        axes[1, 0].set_title('Economy vs Business Load Relationship'); axes[1, 0].grid(True, alpha=0.3)

        route_counts = pd.Series(self.episode_routes).value_counts()
        route_counts.plot(kind='bar', ax=axes[1, 1])
        axes[1, 1].set_title('Training Distribution Across Routes')
        axes[1, 1].set_xlabel('Route'); axes[1, 1].set_ylabel('Episodes'); axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = os.path.join(self.results_dir, 'class_performance.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Class performance plot saved: {save_path}")
        plt.close()

    def _moving_average(self, data, window):
        """Calculate moving average."""
        if len(data) < window:
            return data
        return np.convolve(data, np.ones(window) / window, mode='valid')


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("\n" + "=" * 80)
    print("  MULTI-ROUTE MULTI-CLASS AIRLINE RL TRAINING")
    print("=" * 80)

    route_stats_path = "data/route_stats.pkl"
    if not os.path.exists(route_stats_path):
        print(f"\n❌ Error: {route_stats_path} not found!")
        print("Please run data calibration first to generate route_stats.pkl")
        return

    with open(route_stats_path, "rb") as f:
        route_stats = pickle.load(f)

    env = AirlineRevenueEnv(
        route_stats_path=route_stats_path,
        fixed_route=None  # multi-route training
    )

    state_size  = env.observation_space.shape[0]
    action_size = env.action_space.n

    print(f"\n📐 Environment Configuration:")
    print(f"  State size:  {state_size}")
    print(f"  Action size: {action_size}")
    print(f"  Routes:      {len(env.routes)}")

    accepted_params = [
        'learning_rate', 'gamma', 'epsilon', 'epsilon_decay', 'epsilon_min',
        'batch_size', 'hidden_size', 'use_prioritized_replay', 'device'
    ]

    agent_params = {k: v for k, v in AGENT_CONFIG.items() if k in accepted_params}

    print(f"\n🤖 Agent Configuration:")
    for key, value in agent_params.items():
        print(f"  {key}: {value}")

    # agent = DQNAgent(
    #     state_size=state_size,
    #     action_size=action_size,
    #     **agent_params
    # )

    # trainer = AirlineRLTrainer(env=env, agent=agent)

    # print("\n🚀 Starting training...")
    # summary = trainer.train(
    #     num_episodes=TRAINING_CONFIG.get('num_episodes', 1000),
    #     target_update_freq=TRAINING_CONFIG.get('target_update_freq', 10),
    #     save_freq=TRAINING_CONFIG.get('save_freq', 100),
    #     eval_freq=TRAINING_CONFIG.get('eval_freq', 50),
    #     verbose=True
    # )
    agent = DQNAgent(
        state_size=state_size,
        action_size=action_size,
        **agent_params
    )

    # ── Load existing 6000-episode model and continue from it ──
    existing_model = 'models/trained_models/final_model_20260301_171317.pth'
    if os.path.exists(existing_model):
        agent.load_model(existing_model, load_optimizer=False)
        agent.epsilon = 0.05   # small exploration — not from scratch
        print(f"\n✅ Loaded existing model: {existing_model}")
        print(f"   Continuing from episode 6000 with epsilon=0.05")
    else:
        print(f"\n⚠️  No existing model found — training from scratch")

    trainer = AirlineRLTrainer(env=env, agent=agent)

    print("\n🚀 Starting fine-tune training (2000 episodes)...")
    summary = trainer.train(
        num_episodes=2000,     # only 2000 more — not 6000 again
        target_update_freq=TRAINING_CONFIG.get('target_update_freq', 10),
        save_freq=TRAINING_CONFIG.get('save_freq', 100),
        eval_freq=TRAINING_CONFIG.get('eval_freq', 50),
        verbose=True
    )

    print("\n" + "=" * 80)
    print("  📊 TRAINING SUMMARY")
    print("=" * 80)
    print(f"Total Episodes:    {summary['total_episodes']}")
    print(f"Final Avg Reward:  {summary['final_avg_reward']:.2f}")
    print(f"Final Avg Revenue: ₹{summary['final_avg_revenue']:,.0f}")
    print(f"Final Avg Load:    {summary['final_avg_load'] * 100:.1f}%")
    print(f"  ├─ Economy:      {summary['final_econ_load'] * 100:.1f}%")
    print(f"  └─ Business:     {summary['final_bus_load']  * 100:.1f}%")
    print(f"Best Reward:       {summary['best_reward']:.2f}")
    print(f"Best Revenue:      ₹{summary['best_revenue']:,.0f}")
    print(f"Routes Trained:    {summary['num_routes']}")
    print("=" * 80)


if __name__ == "__main__":
    main()