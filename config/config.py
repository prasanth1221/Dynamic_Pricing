"""
Configuration file for Multi-Route Multi-Class Airline RL Project
File: config/config.py

ENHANCEMENTS:
- Multi-route support with dynamic state sizing
- 9-action space for joint Economy+Business pricing
- Enhanced training parameters for complex multi-class scenarios
- Route-specific calibration settings
"""

import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Data directories
DATA_DIR = BASE_DIR / 'data'
MODELS_DIR = BASE_DIR / 'models' / 'trained_models'
LOGS_DIR = BASE_DIR / 'logs'
RESULTS_DIR = BASE_DIR / 'results'

# Ensure directories exist
for dir_path in [DATA_DIR, MODELS_DIR, LOGS_DIR, RESULTS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# File paths
FLIGHT_DATA_PATH = DATA_DIR / 'flight_data.csv'
ROUTE_STATS_PATH = DATA_DIR / 'route_stats.pkl'
CALIBRATION_SUMMARY_PATH = DATA_DIR / 'calibration_summary.txt'

# =================================================
# ENVIRONMENT CONFIGURATION
# =================================================
ENV_CONFIG = {
    # Aircraft configuration (realistic A320 layout)
    'total_seats': 180,
    'econ_seats': 150,
    'bus_seats': 30,
    
    # Time horizon
    'max_days': 90,
    
    # Calibration
    'route_stats_path': str(ROUTE_STATS_PATH),
    'use_real_calibration': True,
    
    # Multi-route settings
    'train_all_routes': True,  # Train on all available routes
    'fixed_route': None,  # Set to specific route name to train on single route
    
    # Demand modeling
    'econ_base_demand': 0.12,  # 12% of capacity per day
    'bus_base_demand': 0.08,   # 8% of capacity per day
    'econ_price_elasticity': 2.5,  # High elasticity
    'bus_price_elasticity': 1.2,   # Lower elasticity (business travelers)
    
    # Disruption system
    'disruption_probability': 0.05,  # 5% chance per step
    'disruption_types': ['none', 'weather', 'pilot_strike', 'competitor_cancel'],
    
    # Default fallback values (if calibration fails)
    'default_econ_price': 6000,
    'default_bus_price': 12000,
    'default_route': 'Delhi-Mumbai',
}

# =================================================
# STATE SPACE CONFIGURATION
# =================================================
# State components:
# - Base features: 7 (prices, inventory, time)
# - Route one-hot: variable (depends on number of routes)
# - Additional features: 5 (disruption, cyclic time, price ratios)
# Total = 7 + num_routes + 5 = 12 + num_routes

STATE_CONFIG = {
    'base_features': 7,  # [econ_price, bus_price, econ_comp_avg, bus_comp_avg, econ_remaining, bus_remaining, days_norm]
    'route_encoding_size': None,  # Will be set dynamically based on loaded routes
    'additional_features': 5,  # [disruption_flag, time_sin, time_cos, econ_price_ratio, bus_price_ratio]
    'total_state_size': None,  # Will be computed: base + route_encoding + additional
}

# =================================================
# ACTION SPACE CONFIGURATION
# =================================================
# 9 actions for joint Economy + Business pricing
# 3 Economy adjustments × 3 Business adjustments
ACTION_CONFIG = {
    'num_actions': 9,
    'action_space_type': 'joint_pricing',  # Economy + Business simultaneous
    
    # Action mapping: (economy_adjustment, business_adjustment)
    'action_map': {
        0: (-0.10, -0.10),  # Both decrease
        1: (-0.10,  0.00),  # Econ down, Bus hold
        2: (-0.10, +0.10),  # Econ down, Bus up
        3: ( 0.00, -0.10),  # Econ hold, Bus down
        4: ( 0.00,  0.00),  # Both hold
        5: ( 0.00, +0.10),  # Econ hold, Bus up
        6: (+0.10, -0.10),  # Econ up, Bus down
        7: (+0.10,  0.00),  # Econ up, Bus hold
        8: (+0.10, +0.10),  # Both increase
    },
    
    # Action names for display
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
    
    # Detailed action descriptions
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
    'state_size': None,  # Will be set based on STATE_CONFIG
    'action_size': 9,  # Joint pricing actions
    'hidden_size': 256,  # Larger for multi-class complexity
    
    # Learning parameters
    'learning_rate': 0.0005,  # Lower for stability
    'gamma': 0.99,  # Discount factor
    
    # Exploration
    'epsilon': 1.0,  # Initial exploration
    'epsilon_decay': 0.995,
    'epsilon_min': 0.01,
    
    # Training
    'batch_size': 64,
    'replay_buffer_size': 50000,  # Larger for diverse experiences
    'use_prioritized_replay': True,  # Better for multi-class scenarios
    
    # Prioritized replay parameters
    'priority_alpha': 0.6,
    'priority_beta': 0.4,
    'priority_beta_increment': 0.001,
    
    # Optimization
    'gradient_clip': 1.0,
    'learning_rate_decay': 0.9,
    'lr_decay_step': 200,
}

# =================================================
# TRAINING CONFIGURATION
# =================================================
TRAINING_CONFIG = {
    # Training episodes — increase significantly
    'num_episodes': 6000,          # was 2000 — 3x more training
    'max_steps_per_episode': 90,
    
    # Network updates
    'target_update_freq': 10,
    'train_freq': 1,
    
    # Checkpointing
    'save_freq': 200,
    'eval_freq': 100,
    'eval_episodes': 10,
    
    # Early stopping
    'early_stopping': False,
    'patience': 200,
    'min_delta': 0.01,
    
    # ─── CURRICULUM LEARNING ─────────────────────────────────
    'route_sampling': 'curriculum',    # was 'uniform'
    'curriculum_learning': True,       # was False
    
    # Phase 1: Learn on 3 easiest/highest-traffic routes (episodes 0-1500)
    # Phase 2: Add 7 more medium routes (episodes 1500-3500)
    # Phase 3: All routes (episodes 3500-6000)
    'curriculum_phases': [
        {'end_episode': 1500, 'num_routes': 3},
        {'end_episode': 3500, 'num_routes': 10},
        {'end_episode': 6000, 'num_routes': None},   # None = all routes
    ],
    
    # Logging
    'verbose': True,
    'log_interval': 100,
    'save_training_stats': True,
}

# =================================================
# REWARD CONFIGURATION
# =================================================
REWARD_CONFIG = {
    # Revenue scaling
    'revenue_scale': 1000.0,  # Divide by this for reward
    
    # Load factor targets
    'target_load_factor': 0.825,  # 82.5% target
    'optimal_load_range': (0.80, 0.85),
    
    # Bonuses and penalties
    'perfect_load_bonus': 10.0,
    'good_load_bonus': 5.0,
    'acceptable_load_bonus': 3.0,
    'low_load_penalty_multiplier': 0.8,
    'moderate_load_penalty': 5.0,
    'early_sellout_penalty': 15.0,
    'business_booking_bonus': 0.5,  # Per booking
    
    # Disruption penalties
    'pilot_strike_penalty': 8.0,
    'weather_penalty': 4.0,
    'competitor_cancel_bonus': 5.0,  # If we capture demand
    
    # Time-based penalties
    'late_low_load_threshold': 7,  # Days before departure
    'late_low_load_min': 0.6,
}

# =================================================
# FLASK CONFIGURATION
# =================================================
FLASK_CONFIG = {
    'host': '0.0.0.0',
    'port': 5000,
    'debug': True,
    'secret_key': 'airline_rl_multiclass_secret_key_2024',
}

# =================================================
# LOGGING CONFIGURATION
# =================================================
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'log_file': str(LOGS_DIR / 'training.log'),
    'save_episodes': True,
    'save_route_performance': True,
}

# =================================================
# MODEL PATHS
# =================================================
MODEL_SAVE_PATH = str(MODELS_DIR / 'model_ep{episode}.pth')
BEST_MODEL_PATH = str(MODELS_DIR / 'best_model.pth')
FINAL_MODEL_PATH = str(MODELS_DIR / 'final_model.pth')
CHECKPOINT_PATH = str(MODELS_DIR / 'checkpoint_ep{episode}.pth')

# =================================================
# RESULTS PATHS
# =================================================
TRAINING_STATS_PATH = str(RESULTS_DIR / 'training_stats.json')
TRAINING_PLOT_PATH = str(RESULTS_DIR / 'training_progress.png')
ROUTE_PERFORMANCE_PATH = str(RESULTS_DIR / 'route_performance.png')
CLASS_PERFORMANCE_PATH = str(RESULTS_DIR / 'class_performance.png')
EVALUATION_RESULTS_PATH = str(RESULTS_DIR / 'evaluation_results.json')

# =================================================
# VISUALIZATION CONFIGURATION
# =================================================
VIZ_CONFIG = {
    'style': 'darkgrid',
    'figure_dpi': 300,
    'figure_format': 'png',
    'color_palette': 'tab10',
    
    # Class colors
    'economy_color': '#3b82f6',
    'business_color': '#8b5cf6',
    
    # Chart settings
    'moving_average_window': 50,
    'plot_alpha': 0.3,
    'line_width': 2,
}

# =================================================
# UTILITY FUNCTIONS
# =================================================
def compute_state_size(num_routes):
    """
    Compute total state size based on number of routes
    
    Args:
        num_routes: Number of routes in the dataset
    
    Returns:
        total_state_size: Complete state dimension
    """
    base = STATE_CONFIG['base_features']
    route_encoding = num_routes
    additional = STATE_CONFIG['additional_features']
    total = base + route_encoding + additional
    
    STATE_CONFIG['route_encoding_size'] = route_encoding
    STATE_CONFIG['total_state_size'] = total
    AGENT_CONFIG['state_size'] = total
    
    return total


def get_config():
    """Get all configuration as a dictionary"""
    return {
        'env': ENV_CONFIG,
        'state': STATE_CONFIG,
        'action': ACTION_CONFIG,
        'agent': AGENT_CONFIG,
        'training': TRAINING_CONFIG,
        'reward': REWARD_CONFIG,
        'flask': FLASK_CONFIG,
        'logging': LOGGING_CONFIG,
        'visualization': VIZ_CONFIG,
    }


def update_config_for_routes(route_stats):
    """
    Update configuration based on loaded route statistics
    
    Args:
        route_stats: Dictionary of route statistics from calibration
    """
    num_routes = len(route_stats)
    
    print(f"\n Updating config for {num_routes} routes...")
    
    # Update state size
    state_size = compute_state_size(num_routes)
    
    print(f"  State size: {state_size}")
    print(f"    - Base features: {STATE_CONFIG['base_features']}")
    print(f"    - Route encoding: {STATE_CONFIG['route_encoding_size']}")
    print(f"    - Additional features: {STATE_CONFIG['additional_features']}")
    
    return state_size


def print_config():
    """Print current configuration"""
    print("="*80)
    print("  MULTI-ROUTE MULTI-CLASS AIRLINE RL - CONFIGURATION")
    print("="*80)
    
    print("\n Directories:")
    print(f"   Base: {BASE_DIR}")
    print(f"   Data: {DATA_DIR}")
    print(f"   Models: {MODELS_DIR}")
    print(f"   Logs: {LOGS_DIR}")
    print(f"   Results: {RESULTS_DIR}")
    
    print("\n Environment:")
    print(f"   Total Seats: {ENV_CONFIG['total_seats']}")
    print(f"     € Economy: {ENV_CONFIG['econ_seats']}")
    print(f"     € Business: {ENV_CONFIG['bus_seats']}")
    print(f"   Max Days: {ENV_CONFIG['max_days']}")
    print(f"   Disruption Probability: {ENV_CONFIG['disruption_probability']}")
    
    print("\n Action Space:")
    print(f"   Actions: {ACTION_CONFIG['num_actions']}")
    print(f"   Type: {ACTION_CONFIG['action_space_type']}")
    print("   Action Map:")
    for action_id, (e_adj, b_adj) in ACTION_CONFIG['action_map'].items():
        name = ACTION_CONFIG['action_names'][action_id]
        print(f"      {action_id}: {name:>15} → E:{e_adj:+.0%}, B:{b_adj:+.0%}")
    
    print("\nðŸ§  State Space:")
    if STATE_CONFIG['total_state_size']:
        print(f"   Total Size: {STATE_CONFIG['total_state_size']}")
        print(f"     € Base: {STATE_CONFIG['base_features']}")
        print(f"     € Route Encoding: {STATE_CONFIG['route_encoding_size']}")
        print(f"     € Additional: {STATE_CONFIG['additional_features']}")
    else:
        print(f"   Size: Not yet configured (depends on routes)")
    
    print("\n Agent:")
    print(f"   Hidden Size: {AGENT_CONFIG['hidden_size']}")
    print(f"   Learning Rate: {AGENT_CONFIG['learning_rate']}")
    print(f"   Gamma: {AGENT_CONFIG['gamma']}")
    print(f"   Epsilon: {AGENT_CONFIG['epsilon']} → {AGENT_CONFIG['epsilon_min']}")
    print(f"   Batch Size: {AGENT_CONFIG['batch_size']}")
    print(f"   Replay Buffer: {AGENT_CONFIG['replay_buffer_size']}")
    print(f"   Prioritized Replay: {AGENT_CONFIG['use_prioritized_replay']}")
    
    print("\n Training:")
    print(f"   Episodes: {TRAINING_CONFIG['num_episodes']}")
    print(f"   Target Update Freq: {TRAINING_CONFIG['target_update_freq']}")
    print(f"   Save Freq: {TRAINING_CONFIG['save_freq']}")
    print(f"   Eval Freq: {TRAINING_CONFIG['eval_freq']}")
    print(f"   Route Sampling: {TRAINING_CONFIG['route_sampling']}")
    
    print("\n Reward:")
    print(f"   Revenue Scale: {REWARD_CONFIG['revenue_scale']}")
    print(f"   Target Load Factor: {REWARD_CONFIG['target_load_factor']*100:.1f}%")
    print(f"   Optimal Range: {REWARD_CONFIG['optimal_load_range'][0]*100:.0f}%-{REWARD_CONFIG['optimal_load_range'][1]*100:.0f}%")
    print(f"   Perfect Load Bonus: {REWARD_CONFIG['perfect_load_bonus']}")
    print(f"   Business Booking Bonus: {REWARD_CONFIG['business_booking_bonus']}")
    
    print("\n" + "="*80)


def validate_config():
    """Validate configuration consistency"""
    issues = []
    
    # Check action size consistency
    if len(ACTION_CONFIG['action_map']) != ACTION_CONFIG['num_actions']:
        issues.append(f"Action map size ({len(ACTION_CONFIG['action_map'])}) != num_actions ({ACTION_CONFIG['num_actions']})")
    
    # Check seat configuration
    if ENV_CONFIG['econ_seats'] + ENV_CONFIG['bus_seats'] != ENV_CONFIG['total_seats']:
        issues.append(f"Seat configuration inconsistent: {ENV_CONFIG['econ_seats']} + {ENV_CONFIG['bus_seats']} != {ENV_CONFIG['total_seats']}")
    
    # Check paths exist
    if not DATA_DIR.exists():
        issues.append(f"Data directory does not exist: {DATA_DIR}")
    
    # Check agent state size
    if AGENT_CONFIG['state_size'] is not None and AGENT_CONFIG['state_size'] != STATE_CONFIG['total_state_size']:
        issues.append(f"Agent state size mismatch: {AGENT_CONFIG['state_size']} != {STATE_CONFIG['total_state_size']}")
    
    if issues:
        print("\n¸  Configuration Issues:")
        for issue in issues:
            print(f"   - {issue}")
        return False
    else:
        print("\n Configuration validated successfully")
        return True


# =================================================
# INITIALIZATION
# =================================================
if __name__ == "__main__":
    print_config()
    validate_config()
    
    # Example: Update config for specific number of routes
    print("\n" + "="*80)
    print("  EXAMPLE: Configuring for 5 routes")
    print("="*80)
    
    example_routes = ['Delhi-Mumbai', 'Delhi-Bangalore', 'Mumbai-Chennai', 
                      'Mumbai-Kolkata', 'Delhi-Hyderabad']
    
    example_route_stats = {route: {} for route in example_routes}
    state_size = update_config_for_routes(example_route_stats)
    
    print(f"\n Configuration updated")
    print(f"  State size: {state_size}")
    print(f"  Ready for training on {len(example_routes)} routes")