# """
# Traditional Pricing Strategies for Airline Revenue Management
# File: baselines/traditional_pricing.py

# These are NON-LEARNING baseline strategies used for comparison with RL agent.
# """

# import numpy as np


# class TraditionalPricingStrategies:
#     """
#     Collection of traditional (non-RL) pricing strategies
#     All strategies return action indices (0-8) for the multi-class environment
#     """
    
#     @staticmethod
#     def static_pricing(env):
#         """
#         Strategy 1: Static Pricing (Baseline-1)
        
#         - Never changes prices
#         - Uses market average
#         - Action 4 = Hold both Economy and Business
        
#         This is the WORST strategy but shows what happens with no dynamic pricing.
#         """
#         return 4  # E→ B→ (hold both)
    
#     @staticmethod
#     def rule_based_pricing(env):
#         """
#         Strategy 2: Rule-Based Pricing (Baseline-2) ⭐ BEST TRADITIONAL
        
#         Uses human-designed rules based on:
#         - Load factor
#         - Days to departure
#         - Class-specific demand
        
#         This mimics real airline revenue management systems.
#         """
#         # Calculate load factors
#         total_load = (env.econ_sold + env.bus_sold) / env.total_seats
#         econ_load = env.econ_sold / env.econ_seats_total
#         bus_load = env.bus_sold / env.bus_seats_total
        
#         days = env.days_to_departure
        
#         # Rule 1: Very early (>60 days) - aggressive economy, hold business
#         if days > 60:
#             if econ_load < 0.5:
#                 return 1  # E↓10% B→
#             else:
#                 return 4  # Hold both
        
#         # Rule 2: Early period (30-60 days) - standard pricing
#         elif days > 30:
#             if total_load < 0.6:
#                 return 0  # E↓10% B↓10% (both down)
#             elif total_load < 0.75:
#                 return 4  # Hold
#             else:
#                 return 7  # E↑10% B→ (economy up)
        
#         # Rule 3: Mid period (14-30 days) - optimize by class
#         elif days > 14:
#             if econ_load < 0.6 and bus_load < 0.5:
#                 return 0  # Both down
#             elif econ_load < 0.7:
#                 return 1  # E↓10% B→
#             elif bus_load < 0.6:
#                 return 3  # E→ B↓10%
#             elif total_load > 0.85:
#                 return 8  # Both up
#             else:
#                 return 4  # Hold
        
#         # Rule 4: Late period (7-14 days) - maximize revenue
#         elif days > 7:
#             if total_load < 0.7:
#                 # Last minute deals
#                 return 2  # E↓10% B↑10%
#             elif total_load > 0.9:
#                 # Nearly full - increase both
#                 return 8  # E↑10% B↑10%
#             elif bus_load < 0.7:
#                 # Business not full - lower it
#                 return 3  # E→ B↓10%
#             else:
#                 return 7  # E↑10% B→
        
#         # Rule 5: Very late (<7 days) - aggressive pricing
#         else:
#             if total_load < 0.6:
#                 # Desperate - drop both
#                 return 0  # E↓10% B↓10%
#             elif total_load > 0.95:
#                 # Nearly sold out - increase both
#                 return 8  # E↑10% B↑10%
#             elif econ_load < 0.75:
#                 return 1  # E↓10% B→
#             else:
#                 return 7  # E↑10% B→
    
#     @staticmethod
#     def time_based_pricing(env):
#         """
#         Strategy 3: Time-Based Pricing (Baseline-3)
        
#         Classic airline rule:
#         - Early booking → cheaper prices
#         - Late booking → expensive prices
        
#         Simple time-dependent strategy without considering load.
#         """
#         days = env.days_to_departure
        
#         # Phase 1: Far out (>60 days) - low prices
#         if days > 60:
#             return 0  # E↓10% B↓10%
        
#         # Phase 2: Early (45-60 days) - slightly lower
#         elif days > 45:
#             return 1  # E↓10% B→
        
#         # Phase 3: Mid-early (30-45 days) - hold
#         elif days > 30:
#             return 4  # E→ B→
        
#         # Phase 4: Mid (15-30 days) - start increasing
#         elif days > 15:
#             return 5  # E→ B↑10%
        
#         # Phase 5: Late (7-15 days) - increase economy
#         elif days > 7:
#             return 7  # E↑10% B→
        
#         # Phase 6: Very late (<7 days) - maximize both
#         else:
#             return 8  # E↑10% B↑10%
    
#     @staticmethod
#     def competitor_following_pricing(env):
#         """
#         Strategy 4: Competitor-Following Pricing (Baseline-4)
        
#         Adjusts prices to stay competitive with market average
#         - If above market → decrease
#         - If below market → increase
#         """
#         # Calculate average competitor prices
#         econ_comp_avg = np.mean(list(env.econ_competitors.values()))
#         bus_comp_avg = np.mean(list(env.bus_competitors.values()))
        
#         # Calculate our position relative to market
#         econ_ratio = env.econ_price / econ_comp_avg if econ_comp_avg > 0 else 1.0
#         bus_ratio = env.bus_price / bus_comp_avg if bus_comp_avg > 0 else 1.0
        
#         # Decision matrix based on price ratios
#         # We want to be slightly below market (0.95-1.05 range)
        
#         # Economy too high
#         if econ_ratio > 1.10:
#             if bus_ratio > 1.10:
#                 return 0  # Both too high - decrease both
#             elif bus_ratio < 0.95:
#                 return 2  # E↓10% B↑10%
#             else:
#                 return 1  # E↓10% B→
        
#         # Economy too low
#         elif econ_ratio < 0.90:
#             if bus_ratio < 0.90:
#                 return 8  # Both too low - increase both
#             elif bus_ratio > 1.05:
#                 return 6  # E↑10% B↓10%
#             else:
#                 return 7  # E↑10% B→
        
#         # Economy about right
#         else:
#             if bus_ratio > 1.10:
#                 return 3  # E→ B↓10%
#             elif bus_ratio < 0.90:
#                 return 5  # E→ B↑10%
#             else:
#                 return 4  # Both about right - hold
    
#     @staticmethod
#     def load_factor_optimizer(env):
#         """
#         Strategy 5: Load Factor Optimizer (Baseline-5)
        
#         Focuses purely on achieving target load factor (80-85%)
#         Ignores revenue - just tries to fill the plane optimally
#         """
#         total_load = (env.econ_sold + env.bus_sold) / env.total_seats
#         econ_load = env.econ_sold / env.econ_seats_total
#         bus_load = env.bus_sold / env.bus_seats_total
        
#         days = env.days_to_departure
        
#         # Target load factor
#         target_load = 0.825
        
#         # Early booking period - be aggressive to hit target
#         if days > 30:
#             if total_load < 0.6:
#                 return 0  # E↓10% B↓10% (fill seats)
#             elif total_load < target_load:
#                 return 1  # E↓10% B→
#             else:
#                 return 4  # On track - hold
        
#         # Mid period - optimize by class
#         elif days > 14:
#             if total_load < 0.7:
#                 # Behind target - lower prices
#                 if econ_load < bus_load:
#                     return 1  # E↓10% B→
#                 else:
#                     return 3  # E→ B↓10%
#             elif total_load > 0.9:
#                 # Ahead of target - slow down
#                 return 8  # E↑10% B↑10%
#             else:
#                 return 4  # Hold
        
#         # Late period - final push
#         else:
#             if total_load < 0.75:
#                 # Emergency - drop both
#                 return 0  # E↓10% B↓10%
#             elif total_load > 0.95:
#                 # Too full too early - maximize revenue
#                 return 8  # E↑10% B↑10%
#             elif econ_load < 0.8:
#                 return 1  # E↓10% B→
#             elif bus_load < 0.7:
#                 return 3  # E→ B↓10%
#             else:
#                 return 4  # Hold


# # Convenience dictionary for easy access
# TRADITIONAL_STRATEGIES = {
#     'static': TraditionalPricingStrategies.static_pricing,
#     'rule_based': TraditionalPricingStrategies.rule_based_pricing,
#     'time_based': TraditionalPricingStrategies.time_based_pricing,
#     'competitor_following': TraditionalPricingStrategies.competitor_following_pricing,
#     'load_factor': TraditionalPricingStrategies.load_factor_optimizer
# }


# def evaluate_traditional_strategy(env, strategy_fn, num_episodes=10):
#     """
#     Evaluate a traditional pricing strategy
    
#     Args:
#         env: AirlineRevenueEnv instance
#         strategy_fn: Function that takes env and returns action
#         num_episodes: Number of episodes to evaluate
    
#     Returns:
#         dict: Performance metrics
#     """
#     revenues = []
#     load_factors = []
#     econ_loads = []
#     bus_loads = []
    
#     for episode in range(num_episodes):
#         state, _ = env.reset()
#         done = False
        
#         while not done:
#             action = strategy_fn(env)
#             state, reward, terminated, truncated, info = env.step(action)
#             done = terminated or truncated
        
#         summary = env.get_episode_summary()
#         revenues.append(summary['total_revenue'])
#         load_factors.append(summary['load_factor'])
#         econ_loads.append(summary['econ_load_factor'])
#         bus_loads.append(summary['bus_load_factor'])
    
#     return {
#         'avg_revenue': np.mean(revenues),
#         'std_revenue': np.std(revenues),
#         'avg_load_factor': np.mean(load_factors),
#         'avg_econ_load': np.mean(econ_loads),
#         'avg_bus_load': np.mean(bus_loads),
#         'revenues': revenues,
#         'load_factors': load_factors
#     }


# def compare_all_strategies(env, rl_agent=None, num_episodes=10):
#     """
#     Compare all traditional strategies (and optionally RL agent)
    
#     Args:
#         env: AirlineRevenueEnv instance
#         rl_agent: Optional DQN agent for comparison
#         num_episodes: Episodes per strategy
    
#     Returns:
#         dict: Comparison results
#     """
#     results = {}
    
#     print("\n" + "="*80)
#     print("  📊 COMPARING TRADITIONAL PRICING STRATEGIES")
#     print("="*80)
    
#     # Evaluate each traditional strategy
#     for name, strategy_fn in TRADITIONAL_STRATEGIES.items():
#         print(f"\n🔄 Evaluating: {name.replace('_', ' ').title()}...")
#         results[name] = evaluate_traditional_strategy(env, strategy_fn, num_episodes)
        
#         print(f"   Avg Revenue: ₹{results[name]['avg_revenue']:,.0f}")
#         print(f"   Avg Load Factor: {results[name]['avg_load_factor']*100:.1f}%")
    
#     # Evaluate RL agent if provided
#     if rl_agent is not None:
#         print(f"\n🤖 Evaluating: RL Agent...")
        
#         revenues = []
#         load_factors = []
#         econ_loads = []
#         bus_loads = []
        
#         original_epsilon = rl_agent.epsilon
#         rl_agent.epsilon = 0.0  # Greedy
        
#         for episode in range(num_episodes):
#             state, _ = env.reset()
#             done = False
            
#             while not done:
#                 action = rl_agent.select_action(state, training=False)
#                 state, reward, terminated, truncated, info = env.step(action)
#                 done = terminated or truncated
            
#             summary = env.get_episode_summary()
#             revenues.append(summary['total_revenue'])
#             load_factors.append(summary['load_factor'])
#             econ_loads.append(summary['econ_load_factor'])
#             bus_loads.append(summary['bus_load_factor'])
        
#         rl_agent.epsilon = original_epsilon
        
#         results['rl_agent'] = {
#             'avg_revenue': np.mean(revenues),
#             'std_revenue': np.std(revenues),
#             'avg_load_factor': np.mean(load_factors),
#             'avg_econ_load': np.mean(econ_loads),
#             'avg_bus_load': np.mean(bus_loads),
#             'revenues': revenues,
#             'load_factors': load_factors
#         }
        
#         print(f"   Avg Revenue: ₹{results['rl_agent']['avg_revenue']:,.0f}")
#         print(f"   Avg Load Factor: {results['rl_agent']['avg_load_factor']*100:.1f}%")
    
#     print("\n" + "="*80)
    
#     return results


# # Example usage
# if __name__ == "__main__":
#     print("="*80)
#     print("  TRADITIONAL PRICING STRATEGIES - TESTING")
#     print("="*80)
    
#     print("\n📋 Available Strategies:")
#     for i, (name, _) in enumerate(TRADITIONAL_STRATEGIES.items(), 1):
#         print(f"   {i}. {name.replace('_', ' ').title()}")
    
#     print("\n✅ All strategies loaded successfully!")
#     print("\nTo use:")
#     print("  from baselines.traditional_pricing import TRADITIONAL_STRATEGIES")
#     print("  action = TRADITIONAL_STRATEGIES['rule_based'](env)")



"""
Traditional Pricing Strategies for Airline Revenue Management
File: baselines/traditional_pricing.py

NON-LEARNING baseline strategies used for comparison with RL agent.

FIXES & ENHANCEMENTS (v3):
- SAFE env reconstruction via direct AirlineRevenueEnv() instead of fragile deepcopy
- Fixed_route preserved per episode so comparisons are route-consistent
- Each strategy evaluated on a guaranteed-fresh env (no state bleed)
- RL agent evaluated identically for fair comparison
- Built-in diagnostics to verify env freshness
- Summary table sorted by revenue with trophy marker
- All strategies return int actions (guard against accidental float returns)
"""

import numpy as np
import os
import pickle
import sys

# ── Make sure project root is on path so AirlineRevenueEnv is importable ──
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from environment.airline_env import AirlineRevenueEnv


# ─────────────────────────────────────────────────────────────────────────────
# INTERNAL HELPERS
# ─────────────────────────────────────────────────────────────────────────────

_ROUTE_STATS_PATH = "data/route_stats.pkl"   # default; overridden at runtime


def _make_fresh_env(template_env):
    """
    Build a brand-new AirlineRevenueEnv from scratch using the same
    calibration file and fixed_route as the template.

    WHY NOT deepcopy?
    -----------------
    deepcopy on a live Gym env (numpy RNG state, file handles, etc.)
    can silently produce a broken copy that still appears to work but
    shares hidden mutable state.  Direct construction is 100% safe.

    Returns
    -------
    AirlineRevenueEnv  — freshly reset, ready for an episode
    """
    route_stats_path = getattr(template_env, "_route_stats_path", _ROUTE_STATS_PATH)
    fixed_route      = getattr(template_env, "fixed_route", None)

    fresh = AirlineRevenueEnv(
        route_stats_path=route_stats_path,
        fixed_route=fixed_route,
    )
    fresh.reset()
    return fresh


def _assert_env_fresh(env, label="env"):
    """Quick sanity-check: sold seats should be 0 at episode start."""
    assert env.econ_sold == 0 and env.bus_sold == 0, (
        f"[traditional_pricing] {label} is NOT fresh! "
        f"econ_sold={env.econ_sold}, bus_sold={env.bus_sold}. "
        "State bleed detected — check _make_fresh_env()."
    )


# ─────────────────────────────────────────────────────────────────────────────
# STRATEGY IMPLEMENTATIONS
# ─────────────────────────────────────────────────────────────────────────────

class TraditionalPricingStrategies:
    """
    Collection of traditional (non-RL) pricing strategies.
    All return int action indices 0-8 for the 9-action joint pricing space:

        0: E↓10% B↓10%    1: E↓10% B→     2: E↓10% B↑10%
        3: E→   B↓10%     4: E→   B→      5: E→   B↑10%
        6: E↑10% B↓10%    7: E↑10% B→     8: E↑10% B↑10%
    """

    # ------------------------------------------------------------------ #
    @staticmethod
    def static_pricing(env) -> int:
        """
        Strategy 1 — Static Pricing  (Baseline-1, WORST)
        Never changes prices.  Shows the cost of zero dynamic pricing.
        """
        return 4   # Hold both

    # ------------------------------------------------------------------ #
    @staticmethod
    def rule_based_pricing(env) -> int:
        """
        Strategy 2 — Rule-Based Pricing  (Baseline-2, ⭐ BEST TRADITIONAL)
        Mimics real airline Revenue Management using human-designed rules:
        load factor × days-to-departure × class-specific demand.
        """
        total_load = (env.econ_sold + env.bus_sold) / env.total_seats
        econ_load  = env.econ_sold / env.econ_seats_total
        bus_load   = env.bus_sold  / env.bus_seats_total
        days       = env.days_to_departure

        # ── Phase 1: Very early (>60 days) ───────────────────────────────
        if days > 60:
            if econ_load < 0.3:
                return 0   # Both low demand — drop both
            elif econ_load < 0.5:
                return 1   # E↓10% B→
            else:
                return 4   # On track — hold

        # ── Phase 2: Early (30-60 days) ──────────────────────────────────
        elif days > 30:
            if total_load < 0.50:
                return 0   # Behind — drop both
            elif total_load < 0.65:
                return 1   # Slightly behind — ease economy
            elif total_load < 0.80:
                return 4   # On track — hold
            else:
                return 7   # Ahead — lift economy

        # ── Phase 3: Mid period (14-30 days) ─────────────────────────────
        elif days > 14:
            if econ_load < 0.55 and bus_load < 0.45:
                return 0   # Both lagging
            elif econ_load < 0.65:
                return 1   # Economy lagging
            elif bus_load < 0.55:
                return 3   # Business lagging
            elif total_load > 0.88:
                return 8   # Strong demand — raise both
            elif total_load > 0.75:
                return 7   # Good demand — lift economy
            else:
                return 4   # Hold

        # ── Phase 4: Late (7-14 days) ────────────────────────────────────
        elif days > 7:
            if total_load < 0.60:
                return 0   # Seriously behind — drop both
            elif total_load < 0.70:
                return 2   # E↓10% B↑10% — last-minute leisure/biz split
            elif total_load > 0.92:
                return 8   # Nearly full — maximise
            elif bus_load < 0.65:
                return 3   # Business still has room — lower it
            else:
                return 7   # Economy up

        # ── Phase 5: Very late (<7 days) ─────────────────────────────────
        else:
            if total_load < 0.55:
                return 0   # Desperate — drop both
            elif total_load < 0.70:
                return 1   # E↓10% to stimulate last-minute
            elif total_load > 0.95:
                return 8   # Sold out imminent — squeeze revenue
            elif econ_load < 0.80:
                return 1   # Economy still has seats
            else:
                return 7   # Raise economy only

    # ------------------------------------------------------------------ #
    @staticmethod
    def time_based_pricing(env) -> int:
        """
        Strategy 3 — Time-Based Pricing  (Baseline-3)
        Classic airline bucket system: early = cheap, late = expensive.
        """
        days = env.days_to_departure
        if   days > 60:  return 0   # Very early — cheap
        elif days > 45:  return 1   # E↓ B→
        elif days > 30:  return 4   # Hold
        elif days > 20:  return 5   # E→ B↑
        elif days > 10:  return 7   # E↑ B→
        else:            return 8   # Very late — maximise both

    # ------------------------------------------------------------------ #
    @staticmethod
    def competitor_following_pricing(env) -> int:
        """
        Strategy 4 — Competitor-Following  (Baseline-4)
        Stays within ±10% of market average.
        """
        econ_comp_avg = np.mean(list(env.econ_competitors.values())) if env.econ_competitors else env.econ_price
        bus_comp_avg  = np.mean(list(env.bus_competitors.values()))  if env.bus_competitors  else env.bus_price

        econ_ratio = env.econ_price / econ_comp_avg if econ_comp_avg > 0 else 1.0
        bus_ratio  = env.bus_price  / bus_comp_avg  if bus_comp_avg  > 0 else 1.0

        # Economy too high
        if econ_ratio > 1.10:
            if bus_ratio > 1.10:    return 0   # Both high — drop both
            elif bus_ratio < 0.92:  return 2   # E↓ B↑
            else:                   return 1   # E↓ B→

        # Economy too low
        elif econ_ratio < 0.90:
            if bus_ratio < 0.90:    return 8   # Both low — raise both
            elif bus_ratio > 1.08:  return 6   # E↑ B↓
            else:                   return 7   # E↑ B→

        # Economy about right
        else:
            if bus_ratio > 1.10:    return 3   # E→ B↓
            elif bus_ratio < 0.90:  return 5   # E→ B↑
            else:                   return 4   # Both fine — hold

    # ------------------------------------------------------------------ #
    @staticmethod
    def load_factor_optimizer(env) -> int:
        """
        Strategy 5 — Load Factor Optimizer  (Baseline-5)
        Ignores revenue; purely tries to fill the plane to ~82% target.
        """
        total_load = (env.econ_sold + env.bus_sold) / env.total_seats
        econ_load  = env.econ_sold / env.econ_seats_total
        bus_load   = env.bus_sold  / env.bus_seats_total
        days       = env.days_to_departure
        urgency    = 1.0 - (days / env.max_days)   # 0 at start → 1 at departure

        TARGET = 0.82

        if total_load < 0.45:
            # Severely behind — drop both aggressively
            return 0 if urgency > 0.4 else 1

        elif total_load < TARGET - 0.15:
            # Behind target
            if urgency > 0.5:
                return 0   # Urgent — drop both
            elif econ_load < bus_load:
                return 1   # Economy is the bottleneck
            else:
                return 3   # Business is the bottleneck

        elif total_load < TARGET:
            # Slightly behind — gentle push
            return 1 if urgency > 0.3 else 4

        elif total_load < TARGET + 0.10:
            # On target — hold or minor lift
            return 7 if urgency > 0.65 else 4

        else:
            # Ahead of target — maximise revenue
            return 8


# ─────────────────────────────────────────────────────────────────────────────
# STRATEGY REGISTRY
# ─────────────────────────────────────────────────────────────────────────────

TRADITIONAL_STRATEGIES = {
    "static":               TraditionalPricingStrategies.static_pricing,
    "rule_based":           TraditionalPricingStrategies.rule_based_pricing,
    "time_based":           TraditionalPricingStrategies.time_based_pricing,
    "competitor_following": TraditionalPricingStrategies.competitor_following_pricing,
    "load_factor":          TraditionalPricingStrategies.load_factor_optimizer,
}


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATION FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_traditional_strategy(env, strategy_fn, num_episodes: int = 10,
                                   verbose: bool = False) -> dict:
    """
    Evaluate a single traditional pricing strategy.

    Each episode uses a **fresh, independently constructed** env so there is
    zero state bleed between episodes.

    Parameters
    ----------
    env          : AirlineRevenueEnv — used only as a configuration template
    strategy_fn  : callable(env) -> int
    num_episodes : int
    verbose      : bool — print per-episode details if True

    Returns
    -------
    dict with keys: avg_revenue, std_revenue, min_revenue, max_revenue,
                    avg_load_factor, avg_econ_load, avg_bus_load,
                    revenues, load_factors
    """
    revenues     = []
    load_factors = []
    econ_loads   = []
    bus_loads    = []

    for ep in range(num_episodes):
        ep_env = _make_fresh_env(env)
        _assert_env_fresh(ep_env, label=f"ep{ep}")

        state, _ = ep_env.reset()
        done = False

        while not done:
            action = int(strategy_fn(ep_env))   # guard: ensure int
            state, reward, terminated, truncated, info = ep_env.step(action)
            done = terminated or truncated

        summary = ep_env.get_episode_summary()
        revenues.append(summary["total_revenue"])
        load_factors.append(summary["load_factor"])
        econ_loads.append(summary["econ_load_factor"])
        bus_loads.append(summary["bus_load_factor"])

        if verbose:
            print(f"    ep{ep+1:02d}: rev=₹{summary['total_revenue']:,.0f}  "
                  f"load={summary['load_factor']*100:.1f}%")

    return {
        "avg_revenue":     float(np.mean(revenues)),
        "std_revenue":     float(np.std(revenues)),
        "min_revenue":     float(np.min(revenues)),
        "max_revenue":     float(np.max(revenues)),
        "avg_load_factor": float(np.mean(load_factors)),
        "avg_econ_load":   float(np.mean(econ_loads)),
        "avg_bus_load":    float(np.mean(bus_loads)),
        "revenues":        revenues,
        "load_factors":    load_factors,
    }


def compare_all_strategies(env, rl_agent=None, num_episodes: int = 10,
                            verbose: bool = True) -> dict:
    """
    Compare all traditional strategies — and optionally the RL agent.

    FAIR COMPARISON GUARANTEES
    --------------------------
    • Every strategy (including RL) gets its own freshly constructed env.
    • The original `env` object is NEVER mutated (safe to pass live env).
    • RL agent runs with epsilon=0 (pure greedy) for fair evaluation.
    • fixed_route is preserved so all strategies run on the same route.

    Parameters
    ----------
    env          : AirlineRevenueEnv — configuration template (not mutated)
    rl_agent     : DQNAgent or None
    num_episodes : int
    verbose      : bool

    Returns
    -------
    dict  keyed by strategy name → metrics dict
    """
    results = {}

    if verbose:
        print("\n" + "=" * 80)
        print("  📊 COMPARING PRICING STRATEGIES  [v3 — safe fresh-env]")
        print("=" * 80)

    # ── Traditional strategies ────────────────────────────────────────────
    for name, strategy_fn in TRADITIONAL_STRATEGIES.items():
        if verbose:
            print(f"\n🔄  {name.replace('_', ' ').title()} ...")
        results[name] = evaluate_traditional_strategy(
            env, strategy_fn, num_episodes, verbose=verbose
        )
        if verbose:
            m = results[name]
            print(f"    Avg Revenue:     ₹{m['avg_revenue']:>12,.0f}  "
                  f"(±{m['std_revenue']:,.0f})")
            print(f"    Avg Load Factor: {m['avg_load_factor']*100:>6.1f}%  "
                  f"(econ {m['avg_econ_load']*100:.1f}%  "
                  f"bus {m['avg_bus_load']*100:.1f}%)")

    # ── RL agent (optional) ───────────────────────────────────────────────
    if rl_agent is not None:
        if verbose:
            print(f"\n🤖  RL Agent (greedy, ε=0) ...")

        revenues     = []
        load_factors = []
        econ_loads   = []
        bus_loads    = []

        original_epsilon = rl_agent.epsilon
        rl_agent.epsilon = 0.0   # pure greedy — no exploration noise

        for ep in range(num_episodes):
            ep_env = _make_fresh_env(env)
            _assert_env_fresh(ep_env, label=f"rl_ep{ep}")

            state, _ = ep_env.reset()
            done = False

            while not done:
                action = int(rl_agent.select_action(state, training=False))
                state, reward, terminated, truncated, info = ep_env.step(action)
                done = terminated or truncated

            summary = ep_env.get_episode_summary()
            revenues.append(summary["total_revenue"])
            load_factors.append(summary["load_factor"])
            econ_loads.append(summary["econ_load_factor"])
            bus_loads.append(summary["bus_load_factor"])

            if verbose:
                print(f"    ep{ep+1:02d}: rev=₹{summary['total_revenue']:,.0f}  "
                      f"load={summary['load_factor']*100:.1f}%")

        rl_agent.epsilon = original_epsilon   # always restore

        results["rl_agent"] = {
            "avg_revenue":     float(np.mean(revenues)),
            "std_revenue":     float(np.std(revenues)),
            "min_revenue":     float(np.min(revenues)),
            "max_revenue":     float(np.max(revenues)),
            "avg_load_factor": float(np.mean(load_factors)),
            "avg_econ_load":   float(np.mean(econ_loads)),
            "avg_bus_load":    float(np.mean(bus_loads)),
            "revenues":        revenues,
            "load_factors":    load_factors,
        }

        if verbose:
            m = results["rl_agent"]
            print(f"    Avg Revenue:     ₹{m['avg_revenue']:>12,.0f}  "
                  f"(±{m['std_revenue']:,.0f})")
            print(f"    Avg Load Factor: {m['avg_load_factor']*100:>6.1f}%")

    # ── Summary table ─────────────────────────────────────────────────────
    if verbose:
        print("\n" + "=" * 80)
        print(f"  {'Strategy':<25} {'Avg Revenue':>14}  {'±Std':>10}  "
              f"{'Load%':>7}  {'Econ%':>6}  {'Bus%':>5}")
        print("  " + "─" * 72)
        sorted_items = sorted(results.items(),
                              key=lambda x: x[1]["avg_revenue"],
                              reverse=True)
        for rank, (name, m) in enumerate(sorted_items):
            trophy = " ← 🏆" if rank == 0 else ""
            tag    = " [RL]" if name == "rl_agent" else "     "
            print(f"  {name.replace('_', ' ').title():<25}"
                  f"{tag}"
                  f"  ₹{m['avg_revenue']:>12,.0f}"
                  f"  ±{m['std_revenue']:>9,.0f}"
                  f"  {m['avg_load_factor']*100:>6.1f}%"
                  f"  {m['avg_econ_load']*100:>5.1f}%"
                  f"  {m['avg_bus_load']*100:>4.1f}%"
                  f"{trophy}")
        print("=" * 80)

        # ── RL vs best-traditional improvement ───────────────────────────
        if "rl_agent" in results:
            rl_rev   = results["rl_agent"]["avg_revenue"]
            trad_names = [k for k in results if k != "rl_agent"]
            best_name  = max(trad_names, key=lambda k: results[k]["avg_revenue"])
            best_rev   = results[best_name]["avg_revenue"]
            improvement = (rl_rev - best_rev) / best_rev * 100
            sign  = "📈" if improvement >= 0 else "📉"
            print(f"\n  {sign}  RL Agent vs Best Traditional "
                  f"({best_name.replace('_', ' ').title()}): "
                  f"{improvement:+.1f}%")
            print("=" * 80)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# QUICK SMOKE TEST
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 80)
    print("  TRADITIONAL PRICING STRATEGIES  v3 — smoke test")
    print("=" * 80)

    print("\n📋 Registered strategies:")
    for i, name in enumerate(TRADITIONAL_STRATEGIES, 1):
        print(f"   {i}. {name.replace('_', ' ').title()}")

    # ── Try a live comparison if calibration data is present ─────────────
    if os.path.exists(_ROUTE_STATS_PATH):
        print(f"\n✅ Calibration file found at '{_ROUTE_STATS_PATH}'")
        print("   Running quick 3-episode comparison…\n")

        test_env = AirlineRevenueEnv(
            route_stats_path=_ROUTE_STATS_PATH,
            fixed_route=None,
        )
        compare_all_strategies(test_env, rl_agent=None,
                               num_episodes=3, verbose=True)
    else:
        print(f"\n⚠️  '{_ROUTE_STATS_PATH}' not found — skipping live test.")
        print("   Run:  python analyze_data.py   to generate calibration data.")

    print("\n✅ Module loaded successfully — ready to use.")
    print("   from baselines.traditional_pricing import TRADITIONAL_STRATEGIES, compare_all_strategies")