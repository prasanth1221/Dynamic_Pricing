"""
Traditional Pricing Strategies for Airline Revenue Management
File: baselines/traditional_pricing.py

These are NON-LEARNING baseline strategies used for comparison with RL agent.
"""

import numpy as np


class TraditionalPricingStrategies:
    """
    Collection of traditional (non-RL) pricing strategies
    All strategies return action indices (0-8) for the multi-class environment
    """
    
    @staticmethod
    def static_pricing(env):
        """
        Strategy 1: Static Pricing (Baseline-1)
        
        - Never changes prices
        - Uses market average
        - Action 4 = Hold both Economy and Business
        
        This is the WORST strategy but shows what happens with no dynamic pricing.
        """
        return 4  # E→ B→ (hold both)
    
    @staticmethod
    def rule_based_pricing(env):
        """
        Strategy 2: Rule-Based Pricing (Baseline-2) ⭐ BEST TRADITIONAL
        
        Uses human-designed rules based on:
        - Load factor
        - Days to departure
        - Class-specific demand
        
        This mimics real airline revenue management systems.
        """
        # Calculate load factors
        total_load = (env.econ_sold + env.bus_sold) / env.total_seats
        econ_load = env.econ_sold / env.econ_seats_total
        bus_load = env.bus_sold / env.bus_seats_total
        
        days = env.days_to_departure
        
        # Rule 1: Very early (>60 days) - aggressive economy, hold business
        if days > 60:
            if econ_load < 0.5:
                return 1  # E↓10% B→
            else:
                return 4  # Hold both
        
        # Rule 2: Early period (30-60 days) - standard pricing
        elif days > 30:
            if total_load < 0.6:
                return 0  # E↓10% B↓10% (both down)
            elif total_load < 0.75:
                return 4  # Hold
            else:
                return 7  # E↑10% B→ (economy up)
        
        # Rule 3: Mid period (14-30 days) - optimize by class
        elif days > 14:
            if econ_load < 0.6 and bus_load < 0.5:
                return 0  # Both down
            elif econ_load < 0.7:
                return 1  # E↓10% B→
            elif bus_load < 0.6:
                return 3  # E→ B↓10%
            elif total_load > 0.85:
                return 8  # Both up
            else:
                return 4  # Hold
        
        # Rule 4: Late period (7-14 days) - maximize revenue
        elif days > 7:
            if total_load < 0.7:
                # Last minute deals
                return 2  # E↓10% B↑10%
            elif total_load > 0.9:
                # Nearly full - increase both
                return 8  # E↑10% B↑10%
            elif bus_load < 0.7:
                # Business not full - lower it
                return 3  # E→ B↓10%
            else:
                return 7  # E↑10% B→
        
        # Rule 5: Very late (<7 days) - aggressive pricing
        else:
            if total_load < 0.6:
                # Desperate - drop both
                return 0  # E↓10% B↓10%
            elif total_load > 0.95:
                # Nearly sold out - increase both
                return 8  # E↑10% B↑10%
            elif econ_load < 0.75:
                return 1  # E↓10% B→
            else:
                return 7  # E↑10% B→
    
    @staticmethod
    def time_based_pricing(env):
        """
        Strategy 3: Time-Based Pricing (Baseline-3)
        
        Classic airline rule:
        - Early booking → cheaper prices
        - Late booking → expensive prices
        
        Simple time-dependent strategy without considering load.
        """
        days = env.days_to_departure
        
        # Phase 1: Far out (>60 days) - low prices
        if days > 60:
            return 0  # E↓10% B↓10%
        
        # Phase 2: Early (45-60 days) - slightly lower
        elif days > 45:
            return 1  # E↓10% B→
        
        # Phase 3: Mid-early (30-45 days) - hold
        elif days > 30:
            return 4  # E→ B→
        
        # Phase 4: Mid (15-30 days) - start increasing
        elif days > 15:
            return 5  # E→ B↑10%
        
        # Phase 5: Late (7-15 days) - increase economy
        elif days > 7:
            return 7  # E↑10% B→
        
        # Phase 6: Very late (<7 days) - maximize both
        else:
            return 8  # E↑10% B↑10%
    
    @staticmethod
    def competitor_following_pricing(env):
        """
        Strategy 4: Competitor-Following Pricing (Baseline-4)
        
        Adjusts prices to stay competitive with market average
        - If above market → decrease
        - If below market → increase
        """
        # Calculate average competitor prices
        econ_comp_avg = np.mean(list(env.econ_competitors.values()))
        bus_comp_avg = np.mean(list(env.bus_competitors.values()))
        
        # Calculate our position relative to market
        econ_ratio = env.econ_price / econ_comp_avg if econ_comp_avg > 0 else 1.0
        bus_ratio = env.bus_price / bus_comp_avg if bus_comp_avg > 0 else 1.0
        
        # Decision matrix based on price ratios
        # We want to be slightly below market (0.95-1.05 range)
        
        # Economy too high
        if econ_ratio > 1.10:
            if bus_ratio > 1.10:
                return 0  # Both too high - decrease both
            elif bus_ratio < 0.95:
                return 2  # E↓10% B↑10%
            else:
                return 1  # E↓10% B→
        
        # Economy too low
        elif econ_ratio < 0.90:
            if bus_ratio < 0.90:
                return 8  # Both too low - increase both
            elif bus_ratio > 1.05:
                return 6  # E↑10% B↓10%
            else:
                return 7  # E↑10% B→
        
        # Economy about right
        else:
            if bus_ratio > 1.10:
                return 3  # E→ B↓10%
            elif bus_ratio < 0.90:
                return 5  # E→ B↑10%
            else:
                return 4  # Both about right - hold
    
    @staticmethod
    def load_factor_optimizer(env):
        """
        Strategy 5: Load Factor Optimizer (Baseline-5)
        
        Focuses purely on achieving target load factor (80-85%)
        Ignores revenue - just tries to fill the plane optimally
        """
        total_load = (env.econ_sold + env.bus_sold) / env.total_seats
        econ_load = env.econ_sold / env.econ_seats_total
        bus_load = env.bus_sold / env.bus_seats_total
        
        days = env.days_to_departure
        
        # Target load factor
        target_load = 0.825
        
        # Early booking period - be aggressive to hit target
        if days > 30:
            if total_load < 0.6:
                return 0  # E↓10% B↓10% (fill seats)
            elif total_load < target_load:
                return 1  # E↓10% B→
            else:
                return 4  # On track - hold
        
        # Mid period - optimize by class
        elif days > 14:
            if total_load < 0.7:
                # Behind target - lower prices
                if econ_load < bus_load:
                    return 1  # E↓10% B→
                else:
                    return 3  # E→ B↓10%
            elif total_load > 0.9:
                # Ahead of target - slow down
                return 8  # E↑10% B↑10%
            else:
                return 4  # Hold
        
        # Late period - final push
        else:
            if total_load < 0.75:
                # Emergency - drop both
                return 0  # E↓10% B↓10%
            elif total_load > 0.95:
                # Too full too early - maximize revenue
                return 8  # E↑10% B↑10%
            elif econ_load < 0.8:
                return 1  # E↓10% B→
            elif bus_load < 0.7:
                return 3  # E→ B↓10%
            else:
                return 4  # Hold


# Convenience dictionary for easy access
TRADITIONAL_STRATEGIES = {
    'static': TraditionalPricingStrategies.static_pricing,
    'rule_based': TraditionalPricingStrategies.rule_based_pricing,
    'time_based': TraditionalPricingStrategies.time_based_pricing,
    'competitor_following': TraditionalPricingStrategies.competitor_following_pricing,
    'load_factor': TraditionalPricingStrategies.load_factor_optimizer
}


def evaluate_traditional_strategy(env, strategy_fn, num_episodes=10):
    """
    Evaluate a traditional pricing strategy
    
    Args:
        env: AirlineRevenueEnv instance
        strategy_fn: Function that takes env and returns action
        num_episodes: Number of episodes to evaluate
    
    Returns:
        dict: Performance metrics
    """
    revenues = []
    load_factors = []
    econ_loads = []
    bus_loads = []
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False
        
        while not done:
            action = strategy_fn(env)
            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        
        summary = env.get_episode_summary()
        revenues.append(summary['total_revenue'])
        load_factors.append(summary['load_factor'])
        econ_loads.append(summary['econ_load_factor'])
        bus_loads.append(summary['bus_load_factor'])
    
    return {
        'avg_revenue': np.mean(revenues),
        'std_revenue': np.std(revenues),
        'avg_load_factor': np.mean(load_factors),
        'avg_econ_load': np.mean(econ_loads),
        'avg_bus_load': np.mean(bus_loads),
        'revenues': revenues,
        'load_factors': load_factors
    }


def compare_all_strategies(env, rl_agent=None, num_episodes=10):
    """
    Compare all traditional strategies (and optionally RL agent)
    
    Args:
        env: AirlineRevenueEnv instance
        rl_agent: Optional DQN agent for comparison
        num_episodes: Episodes per strategy
    
    Returns:
        dict: Comparison results
    """
    results = {}
    
    print("\n" + "="*80)
    print("  📊 COMPARING TRADITIONAL PRICING STRATEGIES")
    print("="*80)
    
    # Evaluate each traditional strategy
    for name, strategy_fn in TRADITIONAL_STRATEGIES.items():
        print(f"\n🔄 Evaluating: {name.replace('_', ' ').title()}...")
        results[name] = evaluate_traditional_strategy(env, strategy_fn, num_episodes)
        
        print(f"   Avg Revenue: ₹{results[name]['avg_revenue']:,.0f}")
        print(f"   Avg Load Factor: {results[name]['avg_load_factor']*100:.1f}%")
    
    # Evaluate RL agent if provided
    if rl_agent is not None:
        print(f"\n🤖 Evaluating: RL Agent...")
        
        revenues = []
        load_factors = []
        econ_loads = []
        bus_loads = []
        
        original_epsilon = rl_agent.epsilon
        rl_agent.epsilon = 0.0  # Greedy
        
        for episode in range(num_episodes):
            state, _ = env.reset()
            done = False
            
            while not done:
                action = rl_agent.select_action(state, training=False)
                state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
            
            summary = env.get_episode_summary()
            revenues.append(summary['total_revenue'])
            load_factors.append(summary['load_factor'])
            econ_loads.append(summary['econ_load_factor'])
            bus_loads.append(summary['bus_load_factor'])
        
        rl_agent.epsilon = original_epsilon
        
        results['rl_agent'] = {
            'avg_revenue': np.mean(revenues),
            'std_revenue': np.std(revenues),
            'avg_load_factor': np.mean(load_factors),
            'avg_econ_load': np.mean(econ_loads),
            'avg_bus_load': np.mean(bus_loads),
            'revenues': revenues,
            'load_factors': load_factors
        }
        
        print(f"   Avg Revenue: ₹{results['rl_agent']['avg_revenue']:,.0f}")
        print(f"   Avg Load Factor: {results['rl_agent']['avg_load_factor']*100:.1f}%")
    
    print("\n" + "="*80)
    
    return results


# Example usage
if __name__ == "__main__":
    print("="*80)
    print("  TRADITIONAL PRICING STRATEGIES - TESTING")
    print("="*80)
    
    print("\n📋 Available Strategies:")
    for i, (name, _) in enumerate(TRADITIONAL_STRATEGIES.items(), 1):
        print(f"   {i}. {name.replace('_', ' ').title()}")
    
    print("\n✅ All strategies loaded successfully!")
    print("\nTo use:")
    print("  from baselines.traditional_pricing import TRADITIONAL_STRATEGIES")
    print("  action = TRADITIONAL_STRATEGIES['rule_based'](env)")