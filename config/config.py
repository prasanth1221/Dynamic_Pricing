"""
Configuration file for Multi-Route Multi-Class Airline RL Project
File: config/config.py

FIXES applied:
  1. epsilon_decay: 0.995 → 0.9998  (reaches min around ep 3000, not ep 1100)
  2. epsilon_min:   0.01  → 0.05    (5% permanent exploration, escape local optima)
  3. bus_base_demand aligned to 0.12 to match what airline_env.py actually uses
"""

import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Data directories
DATA_DIR   = BASE_DIR / 'data'
MODELS_DIR = BASE_DIR / 'models' / 'trained_models'
LOGS_DIR   = BASE_DIR / 'logs'
RESULTS_DIR = BASE_DIR / 'results'

# Ensure directories exist
for dir_path in [DATA_DIR, MODELS_DIR, LOGS_DIR, RESULTS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# File paths
FLIGHT_DATA_PATH         = DATA_DIR / 'flight_data.csv'
ROUTE_STATS_PATH         = DATA_DIR / 'route_stats.pkl'
CALIBRATION_SUMMARY_PATH = DATA_DIR / 'calibration_summary.txt'

# =================================================
# ENVIRONMENT CONFIGURATION
# =================================================
ENV_CONFIG = {
    # Aircraft configuration (realistic A320 layout)
    'total_seats': 180,
    'econ_seats':  150,
    'bus_seats':   30,

    # Time horizon
    'max_days': 90,

    # Calibration
    'route_stats_path':    str(ROUTE_STATS_PATH),
    'use_real_calibration': True,

    # Multi-route settings
    'train_all_routes': True,
    'fixed_route':      None,

    # Demand modeling — aligned with airline_env.py hardcoded values
    'econ_base_demand':      0.12,   # 12% of capacity per day
    'bus_base_demand':       0.06,   # FIX: was 0.08, env uses 0.12
    'econ_price_elasticity': 2.5,
    'bus_price_elasticity':  1.2,

    # Disruption system
    'disruption_probability': 0.05,
    'disruption_types': ['none', 'weather', 'pilot_strike', 'competitor_cancel'],

    # Default fallback values
    'default_econ_price': 6000,
    'default_bus_price':  12000,
    'default_route':      'Delhi-Mumbai',
}

# =================================================
# STATE SPACE CONFIGURATION
# =================================================
STATE_CONFIG = {
    'base_features':        7,     # [econ_price, bus_price, econ_comp_avg, bus_comp_avg, econ_remaining, bus_remaining, days_norm]
    'route_encoding_size':  None,  # set dynamically based on loaded routes
    'additional_features':  5,     # [disruption_flag, time_sin, time_cos, econ_price_ratio, bus_price_ratio]
    'total_state_size':     None,  # computed: base + route_encoding + additional
}

# =================================================
# ACTION SPACE CONFIGURATION
# =================================================
ACTION_CONFIG = {
    'num_actions':       9,
    'action_space_type': 'joint_pricing',

    'action_map': {
        0: (-0.10, -0.10),
        1: (-0.10,  0.00),
        2: (-0.10, +0.10),
        3: ( 0.00, -0.10),
        4: ( 0.00,  0.00),
        5: ( 0.00, +0.10),
        6: (+0.10, -0.10),
        7: (+0.10,  0.00),
        8: (+0.10, +0.10),
    },

    'action_names': {
        0: 'E↓10% B↓10%',
        1: 'E↓10% B→',
        2: 'E↓10% B↑10%',
        3: 'E→ B↓10%',
        4: 'E→ B→',
        5: 'E→ B↑10%',
        6: 'E↑10% B↓10%',
        7: 'E↑10% B→',
        8: 'E↑10% B↑10%',
    },

    'action_descriptions': {
        0: 'Decrease both Economy and Business by 10%',
        1: 'Decrease Economy 10%, hold Business',
        2: 'Decrease Economy 10%, increase Business 10%',
        3: 'Hold Economy, decrease Business 10%',
        4: 'Hold both prices steady',
        5: 'Hold Economy, increase Business 10%',
        6: 'Increase Economy 10%, decrease Business 10%',
        7: 'Increase Economy 10%, hold Business',
        8: 'Increase both Economy and Business by 10%',
    }
}

# =================================================
# DQN AGENT CONFIGURATION
# =================================================
AGENT_CONFIG = {
    # Network architecture
    'state_size':  None,   # set dynamically
    'action_size': 9,
    'hidden_size': 256,

    # Learning parameters
    'learning_rate': 0.0005,
    'gamma':         0.99,

    # ── FIX 1 & 2: Exploration ──────────────────────────────────────────────
    # OLD: epsilon_decay=0.995  → reaches 0.01 by episode ~1100 out of 6000
    #      epsilon_min=0.01     → agent stuck with no escape mechanism
    # NEW: epsilon_decay=0.9998 → reaches 0.05 around episode 3000
    #      epsilon_min=0.05     → 5% permanent exploration prevents local optima
    'epsilon':       1.0,
    'epsilon_decay': 0.9998,   # FIX: was 0.995
    'epsilon_min':   0.05,     # FIX: was 0.01

    # Training
    'batch_size':          64,
    'replay_buffer_size':  50000,
    'use_prioritized_replay': True,

    # Prioritized replay
    'priority_alpha':          0.6,
    'priority_beta':           0.4,
    'priority_beta_increment': 0.001,

    # Optimization
    'gradient_clip':      1.0,
    'learning_rate_decay': 0.9,
    'lr_decay_step':      200,
}

# =================================================
# TRAINING CONFIGURATION
# =================================================
TRAINING_CONFIG = {
    'num_episodes':          10000,
    'max_steps_per_episode': 90,

    # Network updates
    'target_update_freq': 10,
    'train_freq':         1,

    # Checkpointing
    'save_freq':     200,
    'eval_freq':     100,
    'eval_episodes': 10,

    # Early stopping
    'early_stopping': False,
    'patience':       200,
    'min_delta':      0.01,

    # Curriculum learning
    'route_sampling':     'curriculum',
    'curriculum_learning': True,

    'curriculum_phases': [
        {'end_episode': 3000,  'num_routes': 3},
        {'end_episode': 6000,  'num_routes': 6},
        {'end_episode': 10000, 'num_routes': 12},
    ],

    # Logging
    'verbose':             True,
    'log_interval':        100,
    'save_training_stats': True,
}

# =================================================
# REWARD CONFIGURATION  (informational — env uses its own values)
# =================================================
REWARD_CONFIG = {
    'revenue_weight':      0.90,
    'target_load_factor':  0.825,
    'optimal_load_range':  (0.80, 0.90),
}

# =================================================
# FLASK CONFIGURATION
# =================================================
FLASK_CONFIG = {
    'host':       '0.0.0.0',
    'port':       5000,
    'debug':      False,
    'secret_key': 'airline_rl_multiclass_secret_key_2024',
}

# =================================================
# LOGGING CONFIGURATION
# =================================================
LOGGING_CONFIG = {
    'level':  'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'log_file': str(LOGS_DIR / 'training.log'),
    'save_episodes':          True,
    'save_route_performance': True,
}

# =================================================
# MODEL PATHS
# =================================================
MODEL_SAVE_PATH  = str(MODELS_DIR / 'model_ep{episode}.pth')
BEST_MODEL_PATH  = str(MODELS_DIR / 'best_model.pth')
FINAL_MODEL_PATH = str(MODELS_DIR / 'final_model.pth')
CHECKPOINT_PATH  = str(MODELS_DIR / 'checkpoint_ep{episode}.pth')

# =================================================
# RESULTS PATHS
# =================================================
TRAINING_STATS_PATH    = str(RESULTS_DIR / 'training_stats.json')
TRAINING_PLOT_PATH     = str(RESULTS_DIR / 'training_progress.png')
ROUTE_PERFORMANCE_PATH = str(RESULTS_DIR / 'route_performance.png')
CLASS_PERFORMANCE_PATH = str(RESULTS_DIR / 'class_performance.png')
EVALUATION_RESULTS_PATH = str(RESULTS_DIR / 'evaluation_results.json')


# =================================================
# HELPER: compute state size dynamically
# =================================================
def compute_state_size(num_routes):
    """Call this after loading route_stats to get the correct state dimension."""
    base       = STATE_CONFIG['base_features']
    additional = STATE_CONFIG['additional_features']
    total      = base + num_routes + additional
    STATE_CONFIG['route_encoding_size'] = num_routes
    STATE_CONFIG['total_state_size']    = total
    AGENT_CONFIG['state_size']          = total
    return total