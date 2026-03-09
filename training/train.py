"""
Enhanced Training Pipeline for Multi-Route, Multi-Class Airline RL
File: training/train.py

FIXES applied:
  1. _get_curriculum_routes: fixed key path (was 'econ_avg_price', doesn't exist)
  2. save best model by avg_revenue (not avg_reward which is inflated by bonuses)
  3. periodic epsilon reset when revenue is declining — escape local optima
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

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from environment.airline_env import AirlineRevenueEnv
from agents.model import DQNAgent
from config.config import AGENT_CONFIG, TRAINING_CONFIG


class AirlineRLTrainer:
    """
    Enhanced training pipeline for Multi-Route, Multi-Class Airline RL
    """

    def __init__(self, env, agent, save_dir='models/trained_models/',
                 results_dir='results/', log_dir='logs/'):
        self.env        = env
        self.agent      = agent
        self.save_dir   = save_dir
        self.results_dir = results_dir
        self.log_dir    = log_dir

        for directory in [save_dir, results_dir, log_dir]:
            Path(directory).mkdir(parents=True, exist_ok=True)

        # Overall training statistics
        self.episode_rewards      = []
        self.episode_revenues     = []
        self.episode_load_factors = []
        self.episode_econ_load    = []
        self.episode_bus_load     = []
        self.episode_lengths      = []
        self.losses               = []
        self.episode_routes       = []

        # Per-route statistics
        self.route_performance = {}

        print(f"✓ Trainer initialized")
        print(f"  Save dir: {save_dir}")
        print(f"  Results dir: {results_dir}")

    # ─────────────────────────────────────────────────────────────────────────
    # FIX 1: CURRICULUM — correct key path into route_stats nested dict
    # ─────────────────────────────────────────────────────────────────────────
    def _get_curriculum_routes(self, episode):
        """
        Return list of routes for this episode based on curriculum phase.

        FIX: old code used self.env.all_route_stats[r].get('econ_avg_price', 0)
             that key does NOT exist in route_stats. Correct path is:
             route_stats[r]['Economy']['price_stats']['mean']
        """
        curriculum_phases = TRAINING_CONFIG.get('curriculum_phases', [])

        if not curriculum_phases or not TRAINING_CONFIG.get('curriculum_learning', False):
            return self.env.routes

        # FIX: correct nested key path
        try:
            sorted_routes = sorted(
                self.env.routes,
                key=lambda r: (
                    self.env.all_route_stats[r]
                    .get('Economy', {})
                    .get('price_stats', {})
                    .get('mean', 0)
                ),
                reverse=True
            )
        except Exception:
            sorted_routes = list(self.env.routes)

        for phase in curriculum_phases:
            if episode < phase['end_episode']:
                n = phase['num_routes']
                if n is None:
                    return sorted_routes
                return sorted_routes[:n]

        return sorted_routes

    # ─────────────────────────────────────────────────────────────────────────
    # MAIN TRAINING LOOP
    # ─────────────────────────────────────────────────────────────────────────
    def train(self, num_episodes=1000, target_update_freq=10,
              save_freq=100, eval_freq=50, verbose=True):

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
        print(f"Epsilon decay:   {self.agent.epsilon_decay}  (min={self.agent.epsilon_min})")
        print("=" * 80)

        # FIX 2: track best by REVENUE not reward
        best_avg_revenue = -np.inf
        best_avg_reward  = -np.inf
        best_episode     = 0

        for episode in tqdm(range(num_episodes), desc="Training"):

            curriculum_routes = self._get_curriculum_routes(episode)
            sampled_route = np.random.choice(curriculum_routes)
            self.env.fixed_route = sampled_route

            state, info = self.env.reset()
            episode_reward = 0
            episode_loss   = []
            done           = False
            step           = 0

            current_route = self.env.route

            while not done:
                action = self.agent.select_action(state, training=True)
                next_state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated

                self.agent.store_transition(state, action, reward, next_state, done)

                loss = self.agent.train_step()
                if loss is not None:
                    episode_loss.append(loss)

                episode_reward += reward
                state = next_state
                step += 1

            # Update target network
            if episode % target_update_freq == 0:
                self.agent.update_target_network()

            # Decay epsilon
            self.agent.update_epsilon()

            # FIX 3: Periodic epsilon reset if revenue is not improving
            if episode > 0 and episode % 1000 == 0:
                recent = self.episode_revenues[-100:] if len(self.episode_revenues) >= 100 else self.episode_revenues
                if len(recent) >= 50:
                    trend = np.mean(recent[-50:]) - np.mean(recent[:50])
                    if trend < 0:  # revenue declining → stuck in bad policy
                        kick = max(self.agent.epsilon, 0.20)
                        self.agent.epsilon = kick
                        print(f"\n  🔄 Epsilon reset to {kick:.2f} at ep {episode} — revenue declining (trend={trend:+,.0f})")

            self.agent.episode_count += 1

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

            # Verbose logging every 50 episodes
            if verbose and episode % 50 == 0 and episode > 0:
                avg_reward  = np.mean(self.episode_rewards[-50:])
                avg_revenue = np.mean(self.episode_revenues[-50:])
                avg_load    = np.mean(self.episode_load_factors[-50:])
                avg_econ    = np.mean(self.episode_econ_load[-50:])
                avg_bus     = np.mean(self.episode_bus_load[-50:])

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
                print(f"  Epsilon:                {self.agent.epsilon:>10.4f}")
                if self.losses:
                    print(f"  Avg Loss (last 50):     {np.mean(self.losses[-50:]):>10.4f}")
                print(f"  Current Route:          {current_route}")
                print("=" * 80)

                # FIX 2: Save best model by REVENUE (not reward)
                if avg_revenue > best_avg_revenue:
                    best_avg_revenue = avg_revenue
                    best_episode     = episode
                    self.save_best_model()
                    print(f"  💾 New best model saved! Revenue: ₹{avg_revenue:,.0f}")

                # Still track best reward for logging
                if avg_reward > best_avg_reward:
                    best_avg_reward = avg_reward

            # Periodic evaluation
            if eval_freq > 0 and episode % eval_freq == 0 and episode > 0:
                print(f"\n🎯 Running evaluation at episode {episode}...")
                eval_results = self.evaluate(num_episodes=5, render=False)
                self.log_evaluation(episode, eval_results)

            # Save checkpoint
            if episode % save_freq == 0 and episode > 0:
                self.save_checkpoint(episode)

        print("\n" + "=" * 80)
        print("  ✅ TRAINING COMPLETED!")
        print("=" * 80)
        print(f"Best avg revenue: ₹{best_avg_revenue:,.0f} (episode {best_episode})")
        print(f"Best avg reward:  {best_avg_reward:.2f}")

        self.save_final_model()
        self.save_training_stats()

        print("\n📊 Generating training visualizations...")
        self.plot_training_progress()
        self.plot_route_performance()
        self.plot_class_performance()

        return self.get_training_summary()

    # ─────────────────────────────────────────────────────────────────────────
    # EVALUATION
    # ─────────────────────────────────────────────────────────────────────────
    def evaluate(self, num_episodes=5, render=False):
        print(f"\n{'=' * 80}")
        print(f"  📊 EVALUATION ({num_episodes} episodes)")
        print(f"{'=' * 80}")

        eval_rewards      = []
        eval_revenues     = []
        eval_load_factors = []
        eval_routes       = []

        original_epsilon  = self.agent.epsilon
        self.agent.epsilon = 0.0

        for ep in range(num_episodes):
            state, info = self.env.reset()
            episode_reward = 0
            done = False

            while not done:
                action = self.agent.select_action(state, training=False)
                next_state, reward, terminated, truncated, info = self.env.step(action)
                done  = terminated or truncated
                state = next_state
                episode_reward += reward

                if render:
                    self.env.render()

            summary = self.env.get_episode_summary()
            eval_rewards.append(episode_reward)
            eval_revenues.append(summary['total_revenue'])
            eval_load_factors.append(summary['load_factor'])
            eval_routes.append(summary['route'])

        self.agent.epsilon = original_epsilon

        print(f"\n  Results:")
        print(f"    Avg Reward:      {np.mean(eval_rewards):>10.2f}")
        print(f"    Avg Revenue:     ₹{np.mean(eval_revenues):>10,.0f}")
        print(f"    Avg Load Factor: {np.mean(eval_load_factors) * 100:>10.1f}%")
        print(f"    Routes tested:   {len(set(eval_routes))}")
        print(f"{'=' * 80}\n")

        return {
            'rewards':      eval_rewards,
            'revenues':     eval_revenues,
            'load_factors': eval_load_factors,
            'routes':       eval_routes,
        }

    def log_evaluation(self, episode, eval_results):
        log_file = Path(self.results_dir) / "evaluation_log.txt"

        revenues     = eval_results.get("revenues", [])
        rewards      = eval_results.get("rewards", [])
        load_factors = eval_results.get("load_factors", [])

        revenues_mean = float(np.mean(revenues))
        revenues_std  = float(np.std(revenues))
        rewards_mean  = float(np.mean(rewards))
        load_mean     = float(np.mean(load_factors))

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
        filepath  = os.path.join(self.save_dir, f'final_model_{timestamp}.pth')
        self.agent.save_model(filepath)
        print(f"\n💾 Final model saved: {filepath}")

    def save_training_stats(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        stats = {
            'timestamp':            timestamp,
            'num_episodes':         len(self.episode_rewards),
            'episode_rewards':      self.episode_rewards,
            'episode_revenues':     self.episode_revenues,
            'episode_load_factors': self.episode_load_factors,
            'episode_econ_load':    self.episode_econ_load,
            'episode_bus_load':     self.episode_bus_load,
            'episode_lengths':      self.episode_lengths,
            'episode_routes':       self.episode_routes,
            'losses':               self.losses,
            'route_performance': {
                k: {
                    'rewards':      v['rewards'],
                    'revenues':     v['revenues'],
                    'load_factors': v['load_factors'],
                    'econ_load':    v['econ_load'],
                    'bus_load':     v['bus_load'],
                }
                for k, v in self.route_performance.items()
            },
            'final_epsilon': self.agent.epsilon,
        }

        filepath = os.path.join(self.results_dir, f'training_stats_{timestamp}.json')
        with open(filepath, 'w') as f:
            json.dump(stats, f, indent=2)

        print(f"📊 Training stats saved: {filepath}")

    def get_training_summary(self):
        return {
            'total_episodes':    len(self.episode_rewards),
            'final_avg_reward':  float(np.mean(self.episode_rewards[-100:])),
            'final_avg_revenue': float(np.mean(self.episode_revenues[-100:])),
            'final_avg_load':    float(np.mean(self.episode_load_factors[-100:])),
            'final_econ_load':   float(np.mean(self.episode_econ_load[-100:])),
            'final_bus_load':    float(np.mean(self.episode_bus_load[-100:])),
            'best_reward':       float(np.max(self.episode_rewards)),
            'best_revenue':      float(np.max(self.episode_revenues)),
            'routes_trained':    list(self.route_performance.keys()),
            'num_routes':        len(self.route_performance),
        }

    # ─────────────────────────────────────────────────────────────────────────
    # VISUALISATIONS
    # ─────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _moving_average(data, window):
        if len(data) < window:
            return data
        return np.convolve(data, np.ones(window) / window, mode='valid').tolist()

    def plot_training_progress(self):
        fig, axes = plt.subplots(3, 2, figsize=(18, 14))
        fig.suptitle('Multi-Route Multi-Class Training Progress', fontsize=16, fontweight='bold')

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

        lf_pct = [lf * 100 for lf in self.episode_load_factors]
        axes[1, 0].plot(lf_pct, alpha=0.3, label='Load Factor', color='coral')
        axes[1, 0].plot(self._moving_average(lf_pct, 50), label='MA(50)', linewidth=2, color='red')
        axes[1, 0].axhline(y=80, color='green', linestyle='--', alpha=0.5, label='Target (80%)')
        axes[1, 0].set_xlabel('Episode'); axes[1, 0].set_ylabel('Load Factor (%)')
        axes[1, 0].set_title('Overall Load Factor'); axes[1, 0].legend(); axes[1, 0].grid(True, alpha=0.3)

        econ_pct = [lf * 100 for lf in self.episode_econ_load]
        bus_pct  = [lf * 100 for lf in self.episode_bus_load]
        axes[1, 1].plot(self._moving_average(econ_pct, 50), label='Economy MA(50)', linewidth=2, color='#3b82f6')
        axes[1, 1].plot(self._moving_average(bus_pct, 50),  label='Business MA(50)', linewidth=2, color='#8b5cf6')
        axes[1, 1].set_xlabel('Episode'); axes[1, 1].set_ylabel('Load Factor (%)')
        axes[1, 1].set_title('Economy vs Business Load Factors')
        axes[1, 1].legend(); axes[1, 1].grid(True, alpha=0.3)

        if self.losses:
            axes[2, 0].plot(self.losses, alpha=0.4, color='orange')
            axes[2, 0].plot(self._moving_average(self.losses, 50), linewidth=2, label='MA(50)', color='darkorange')
            axes[2, 0].set_xlabel('Episode'); axes[2, 0].set_ylabel('Loss')
            axes[2, 0].set_title('Training Loss'); axes[2, 0].set_yscale('log')
            axes[2, 0].legend(); axes[2, 0].grid(True, alpha=0.3)

        axes[2, 1].plot(self.episode_lengths, alpha=0.3, color='purple')
        axes[2, 1].plot(self._moving_average(self.episode_lengths, 50), linewidth=2, label='MA(50)', color='darkviolet')
        axes[2, 1].set_xlabel('Episode'); axes[2, 1].set_ylabel('Steps')
        axes[2, 1].set_title('Episode Length'); axes[2, 1].legend(); axes[2, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = os.path.join(self.results_dir, 'training_progress.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Training progress saved: {save_path}")
        plt.close()

    def plot_route_performance(self):
        if not self.route_performance:
            return

        routes       = list(self.route_performance.keys())
        avg_revenues = [np.mean(self.route_performance[r]['revenues']) for r in routes]
        avg_rewards  = [np.mean(self.route_performance[r]['rewards'])  for r in routes]
        avg_loads    = [np.mean(self.route_performance[r]['load_factors']) * 100 for r in routes]
        avg_econ     = [np.mean(self.route_performance[r]['econ_load']) * 100 for r in routes]
        avg_bus      = [np.mean(self.route_performance[r]['bus_load'])  * 100 for r in routes]
        episode_counts = [len(self.route_performance[r]['rewards']) for r in routes]

        sorted_idx = np.argsort(avg_rewards)[::-1]
        routes_s   = [routes[i] for i in sorted_idx]
        revenues_s = [avg_revenues[i] for i in sorted_idx]
        rewards_s  = [avg_rewards[i]  for i in sorted_idx]
        loads_s    = [avg_loads[i]    for i in sorted_idx]
        econ_s     = [avg_econ[i]     for i in sorted_idx]
        bus_s      = [avg_bus[i]      for i in sorted_idx]
        counts_s   = [episode_counts[i] for i in sorted_idx]

        colors = plt.cm.tab20(np.linspace(0, 1, len(routes_s)))

        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        fig.suptitle('Performance by Route', fontsize=16, fontweight='bold')

        # Reward
        bars = axes[0, 0].barh(routes_s, rewards_s, color=colors)
        for bar, val in zip(bars, rewards_s):
            axes[0, 0].text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                            f'{val:.1f}', va='center', fontsize=7)
        axes[0, 0].set_xlabel('Average Reward'); axes[0, 0].set_title('Average Reward by Route')
        axes[0, 0].grid(True, alpha=0.3)

        # Revenue
        bars = axes[0, 1].barh(routes_s, revenues_s, color=colors)
        for bar, val in zip(bars, revenues_s):
            axes[0, 1].text(bar.get_width() * 1.01, bar.get_y() + bar.get_height() / 2,
                            f'₹{val:,.0f}', va='center', fontsize=7)
        axes[0, 1].set_xlabel('Average Revenue (₹)'); axes[0, 1].set_title('Average Revenue by Route')
        axes[0, 1].grid(True, alpha=0.3)

        # Load factor
        bars = axes[1, 0].barh(routes_s, loads_s, color=colors)
        axes[1, 0].axvline(x=80, color='green', linestyle='--', alpha=0.7, label='Target')
        axes[1, 0].set_xlabel('Average Load Factor (%)'); axes[1, 0].set_title('Average Load Factor by Route')
        axes[1, 0].legend(); axes[1, 0].grid(True, alpha=0.3)

        # Economy vs Business
        x      = np.arange(len(routes_s))
        width  = 0.35
        axes[1, 1].bar(x - width / 2, econ_s, width, label='Economy', color='#3b82f6', alpha=0.8)
        axes[1, 1].bar(x + width / 2, bus_s,  width, label='Business', color='#8b5cf6', alpha=0.8)
        axes[1, 1].set_xlabel('Route'); axes[1, 1].set_ylabel('Average Load Factor (%)')
        axes[1, 1].set_title('Economy vs Business Load by Route')
        axes[1, 1].set_xticks(x); axes[1, 1].set_xticklabels(routes_s, rotation=45, ha='right', fontsize=7)
        axes[1, 1].legend(); axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = os.path.join(self.results_dir, 'route_performance.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Route performance saved: {save_path}")
        plt.close()

    def plot_class_performance(self):
        if not self.episode_econ_load:
            return

        econ_pct = [lf * 100 for lf in self.episode_econ_load]
        bus_pct  = [lf * 100 for lf in self.episode_bus_load]

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Economy vs Business Class Performance', fontsize=16, fontweight='bold')

        # Load factor trend
        axes[0, 0].plot(self._moving_average(econ_pct, 50), label='Economy MA(50)', color='#3b82f6')
        axes[0, 0].plot(self._moving_average(bus_pct,  50), label='Business MA(50)', color='#8b5cf6')
        axes[0, 0].set_xlabel('Episode'); axes[0, 0].set_ylabel('Load Factor (%)')
        axes[0, 0].set_title('Load Factor Trend'); axes[0, 0].legend(); axes[0, 0].grid(True, alpha=0.3)

        # Revenue trend
        axes[0, 1].plot(self._moving_average(self.episode_revenues, 50),
                        label='Total Revenue MA(50)', color='green', linewidth=2)
        axes[0, 1].set_xlabel('Episode'); axes[0, 1].set_ylabel('Revenue (₹)')
        axes[0, 1].set_title('Revenue Trend'); axes[0, 1].legend(); axes[0, 1].grid(True, alpha=0.3)

        # Economy vs Business scatter
        axes[1, 0].scatter(econ_pct, bus_pct, alpha=0.3, s=10, color='#3b82f6')
        axes[1, 0].set_xlabel('Economy Load (%)'); axes[1, 0].set_ylabel('Business Load (%)')
        axes[1, 0].set_title('Economy vs Business Load Relationship')
        axes[1, 0].grid(True, alpha=0.3)

        # Training distribution
        if self.episode_routes:
            route_counts = {}
            for r in self.episode_routes:
                route_counts[r] = route_counts.get(r, 0) + 1
            sorted_rc = sorted(route_counts.items(), key=lambda x: x[1], reverse=True)
            rnames, rcounts = zip(*sorted_rc)
            axes[1, 1].bar(range(len(rnames)), rcounts, color='#3b82f6', alpha=0.8)
            axes[1, 1].set_xticks(range(len(rnames)))
            axes[1, 1].set_xticklabels(rnames, rotation=45, ha='right', fontsize=7)
            axes[1, 1].set_xlabel('Route'); axes[1, 1].set_ylabel('Episodes')
            axes[1, 1].set_title('Training Distribution Across Routes')
            axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = os.path.join(self.results_dir, 'class_performance.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Class performance saved: {save_path}")
        plt.close()


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("=" * 80)
    print("  AIRLINE RL — MULTI-ROUTE MULTI-CLASS TRAINING  (v4)")
    print("=" * 80)

    from config.config import compute_state_size, ROUTE_STATS_PATH

    # Load environment
    env = AirlineRevenueEnv(route_stats_path=str(ROUTE_STATS_PATH))

    # Compute state size dynamically
    state_size = compute_state_size(env.num_routes)
    print(f"\n✓ State size: {state_size} (7 base + {env.num_routes} routes + 5 extra)")

    accepted_params = [
        'learning_rate', 'gamma', 'epsilon', 'epsilon_decay', 'epsilon_min',
        'batch_size', 'hidden_size', 'use_prioritized_replay',
        'priority_alpha', 'priority_beta', 'priority_beta_increment',
        'gradient_clip', 'learning_rate_decay', 'lr_decay_step', 'device',
        'replay_buffer_size',
    ]
    agent_params = {k: v for k, v in AGENT_CONFIG.items() if k in accepted_params}

    print(f"\n🤖 Agent Configuration:")
    for key, value in agent_params.items():
        print(f"  {key}: {value}")

    agent = DQNAgent(
        state_size  = state_size,
        action_size = AGENT_CONFIG['action_size'],
        **agent_params
    )

    trainer = AirlineRLTrainer(
        env         = env,
        agent       = agent,
        save_dir    = 'models/trained_models/',
        results_dir = 'results/',
        log_dir     = 'logs/',
    )

    summary = trainer.train(
        num_episodes      = TRAINING_CONFIG['num_episodes'],
        target_update_freq = TRAINING_CONFIG['target_update_freq'],
        save_freq         = TRAINING_CONFIG['save_freq'],
        eval_freq         = TRAINING_CONFIG['eval_freq'],
        verbose           = TRAINING_CONFIG['verbose'],
    )

    # ── Optional: load existing model and fine-tune (uncomment to use) ──────
    # existing_model = 'models/trained_models/final_model_YYYYMMDD_HHMMSS.pth'
    # if os.path.exists(existing_model):
    #     agent.load_model(existing_model, load_optimizer=False)
    #     agent.epsilon = 0.05   # small exploration — not from scratch
    #     print(f"\n✅ Loaded existing model: {existing_model}")
    # trainer = AirlineRLTrainer(env=env, agent=agent)
    # summary = trainer.train(
    #     num_episodes=2000,     # only 2000 more — not 6000 again
    #     target_update_freq=TRAINING_CONFIG.get('target_update_freq', 10),
    #     save_freq=TRAINING_CONFIG.get('save_freq', 100),
    #     eval_freq=TRAINING_CONFIG.get('eval_freq', 50),
    #     verbose=True
    # )

    print("\n" + "=" * 80)
    print("  📊 TRAINING SUMMARY")
    print("=" * 80)
    print(f"Total Episodes:    {summary['total_episodes']}")
    print(f"Final Avg Reward:  {summary['final_avg_reward']:.2f}")
    print(f"Final Avg Revenue: ₹{summary['final_avg_revenue']:,.0f}")
    print(f"Final Avg Load:    {summary['final_avg_load'] * 100:.1f}%")
    print(f"  ├─ Economy:      {summary['final_econ_load'] * 100:.1f}%")
    print(f"  └─ Business:     {summary['final_bus_load']  * 100:.1f}%")
    print(f"Best Revenue:      ₹{summary['best_revenue']:,.0f}")
    print(f"Routes Trained:    {summary['num_routes']}")
    print("=" * 80)


if __name__ == "__main__":
    main()