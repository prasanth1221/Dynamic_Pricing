"""
Enhanced Multi-Route, Multi-Class Airline Revenue Management Environment
Calibrated using REAL flight data with proper disruptions and competitor dynamics
"""

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
import random
import pickle
import os


class AirlineRevenueEnv(gym.Env):
    """
    Single-agent RL environment with:
    - Multiple routes (sampled per episode)
    - Economy + Business class joint pricing
    - Realistic competitor dynamics
    - Disruption modeling
    - Proper demand curves and booking patterns
    """

    def __init__(self, route_stats_path="data/route_stats.pkl", 
                 fixed_route=None, seed=None):
        super(AirlineRevenueEnv, self).__init__()

        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        # Load calibrated route statistics
        if not os.path.exists(route_stats_path):
            raise FileNotFoundError(
                f"❌ {route_stats_path} not found. Run data calibration first."
            )

        with open(route_stats_path, "rb") as f:
            self.all_route_stats = pickle.load(f)

        self.routes = list(self.all_route_stats.keys())
        self.num_routes = len(self.routes)
        self.fixed_route = fixed_route

        print(f"✓ Loaded calibration for {self.num_routes} routes: {self.routes}")

        # Aircraft configuration (realistic A320 layout)
        self.econ_seats_total = 150
        self.bus_seats_total = 30
        self.total_seats = self.econ_seats_total + self.bus_seats_total
        self.max_days = 90

        # Action space: Joint pricing for Economy & Business
        self.ACTION_MAP = {
            0: (-0.10, -0.10),  # Both decrease
            1: (-0.10,  0.00),  # Econ down, Bus hold
            2: (-0.10, +0.10),  # Econ down, Bus up
            3: ( 0.00, -0.10),  # Econ hold, Bus down
            4: ( 0.00,  0.00),  # Both hold
            5: ( 0.00, +0.10),  # Econ hold, Bus up
            6: (+0.10, -0.10),  # Econ up, Bus down
            7: (+0.10,  0.00),  # Econ up, Bus hold
            8: (+0.10, +0.10),  # Both increase
        }

        self.action_space = spaces.Discrete(9)

        # State space
        state_dim = 7 + self.num_routes + 5
        
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(state_dim,),
            dtype=np.float32
        )

        # Demand parameters
        self.econ_base_demand = 0.12
        self.bus_base_demand = 0.12
        self.econ_price_elasticity = 2.5
        self.bus_price_elasticity = 1.2

        # Disruption system
        self.disruption_types = ["none", "weather", "pilot_strike", "competitor_cancel"]
        self.disruption_probability = 0.05
        self.prev_action = None

        print(f"✓ Environment initialized")
        print(f"  - Action space: {self.action_space.n} joint pricing actions")
        print(f"  - State space: {self.observation_space.shape[0]} features")
        print(f"  - Total seats: {self.total_seats} (Econ: {self.econ_seats_total}, Bus: {self.bus_seats_total})")

    def reset(self, seed=None, options=None):
        """Reset environment for new episode"""
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        # Sample route for this episode
        if self.fixed_route and self.fixed_route in self.routes:
            self.route = self.fixed_route
            self.route_index = self.routes.index(self.route)
        else:
            self.route_index = random.randint(0, self.num_routes - 1)
            self.route = self.routes[self.route_index]
        
        self.route_stats = self.all_route_stats[self.route]

        # Load class-specific calibration
        econ = self.route_stats.get("Economy", {})
        bus  = self.route_stats.get("Business", {})

        # Competitor prices
        self.econ_competitors = econ.get("competitor_prices", {})
        self.bus_competitors  = bus.get("competitor_prices", {})
        
        if not self.econ_competitors:
            self.econ_competitors = {"Competitor_A": 6000}
        if not self.bus_competitors:
            self.bus_competitors  = {"Competitor_A": 12000}

        # Initial prices — start at market average
        self.econ_price = np.mean(list(self.econ_competitors.values()))
        self.bus_price  = np.mean(list(self.bus_competitors.values()))

        # FIX: Wider price bounds — allow agent more room to explore pricing
        econ_stats = econ.get("price_stats", {})
        bus_stats  = bus.get("price_stats", {})
        
        self.econ_min = econ_stats.get("q25", self.econ_price * 0.7) * 0.85   # 15% below q25
        self.econ_max = econ_stats.get("q75", self.econ_price * 1.5) * 1.5    # higher ceiling
        self.bus_min  = bus_stats.get("q25",  self.bus_price  * 0.7) * 0.85
        self.bus_max  = bus_stats.get("q75",  self.bus_price  * 1.5) * 1.5

        # State variables
        self.econ_sold = 0
        self.bus_sold  = 0
        self.days_to_departure = self.max_days
        self.current_step = 0
        
        # Revenue tracking
        self.revenue_econ  = 0.0
        self.revenue_bus   = 0.0
        self.total_revenue = 0.0
        
        # Disruption state
        self.current_disruption  = "none"
        self.disruption_duration = 0

        # History
        self.episode_history = []
        self.prev_action = None

        return self._get_state(), {}

    def _get_state(self):
        """Comprehensive state representation"""
        
        econ_remaining_pct = (self.econ_seats_total - self.econ_sold) / self.econ_seats_total
        bus_remaining_pct  = (self.bus_seats_total  - self.bus_sold)  / self.bus_seats_total
        days_normalized    = self.days_to_departure / self.max_days
        
        econ_comp_avg = np.mean(list(self.econ_competitors.values()))
        bus_comp_avg  = np.mean(list(self.bus_competitors.values()))
        
        econ_price_ratio = self.econ_price / econ_comp_avg if econ_comp_avg > 0 else 1.0
        bus_price_ratio  = self.bus_price  / bus_comp_avg  if bus_comp_avg  > 0 else 1.0
        
        route_encoding = np.zeros(self.num_routes)
        route_encoding[self.route_index] = 1.0
        
        disruption_flag = 1.0 if self.current_disruption != "none" else 0.0
        time_sin = np.sin(2 * np.pi * (self.current_step / self.max_days))
        time_cos = np.cos(2 * np.pi * (self.current_step / self.max_days))
        
        state = np.concatenate([
            [self.econ_price, self.bus_price],
            [econ_comp_avg, bus_comp_avg],
            [econ_remaining_pct, bus_remaining_pct],
            [days_normalized],
            route_encoding,
            [disruption_flag],
            [time_sin, time_cos],
            [econ_price_ratio, bus_price_ratio]
        ])
        
        return state.astype(np.float32)

    def _calculate_econ_demand(self):
        """Economy demand model"""
        days_ratio  = self.days_to_departure / self.max_days
        time_factor = 0.4 + (1 - days_ratio) * 1.2
        
        econ_comp_avg = np.mean(list(self.econ_competitors.values()))
        price_ratio   = self.econ_price / econ_comp_avg if econ_comp_avg > 0 else 1.0
        price_factor  = np.exp(-self.econ_price_elasticity * (price_ratio - 1))
        
        disruption_factor = self._get_disruption_factor()
        demand = self.econ_base_demand * time_factor * price_factor * disruption_factor
        return np.clip(demand, 0, 1)

    def _calculate_bus_demand(self):
        """
        Business demand model.
        Corporate travelers book closer to departure.
        """
        days_ratio = self.days_to_departure / self.max_days

        if days_ratio > 0.7:
            time_factor = 0.4
        elif days_ratio > 0.4:
            time_factor = 0.7
        else:
            time_factor = 1.2   # last 36 days — peak business demand

        bus_comp_avg  = np.mean(list(self.bus_competitors.values()))
        price_ratio   = self.bus_price / bus_comp_avg if bus_comp_avg > 0 else 1.0
        price_factor  = np.exp(-self.bus_price_elasticity * (price_ratio - 1))
        
        disruption_factor = self._get_disruption_factor()
        demand = self.bus_base_demand * time_factor * price_factor * disruption_factor
        return np.clip(demand, 0, 1)

    def _get_disruption_factor(self):
        """Calculate demand impact of current disruption"""
        if self.current_disruption == "weather":
            return 0.6
        elif self.current_disruption == "pilot_strike":
            return 0.3
        elif self.current_disruption == "competitor_cancel":
            return 1.5
        return 1.0

    def _update_competitor_prices(self):
        """Simulate realistic competitor price changes"""
        for airline in self.econ_competitors:
            change_pct = np.random.normal(0, 0.02)
            self.econ_competitors[airline] *= (1 + change_pct)
            self.econ_competitors[airline] = np.clip(
                self.econ_competitors[airline],
                self.econ_min,
                self.econ_max
            )
        
        for airline in self.bus_competitors:
            change_pct = np.random.normal(0, 0.015)
            self.bus_competitors[airline] *= (1 + change_pct)
            self.bus_competitors[airline] = np.clip(
                self.bus_competitors[airline],
                self.bus_min,
                self.bus_max
            )

    def _trigger_disruption(self):
        """Randomly trigger disruptions"""
        if self.disruption_duration > 0:
            self.disruption_duration -= 1
            if self.disruption_duration == 0:
                self.current_disruption = "none"
        elif random.random() < self.disruption_probability:
            self.current_disruption = random.choice(self.disruption_types[1:])
            self.disruption_duration = random.randint(1, 3)

    def step(self, action):
        """Execute pricing action and simulate one day"""
        
        econ_adj, bus_adj = self.ACTION_MAP[action]

        # Update prices
        self.econ_price *= (1 + econ_adj)
        self.bus_price  *= (1 + bus_adj)

        # Enforce bounds
        self.econ_price = np.clip(self.econ_price, self.econ_min, self.econ_max)
        self.bus_price  = np.clip(self.bus_price,  self.bus_min,  self.bus_max)

        self._update_competitor_prices()
        self._trigger_disruption()

        econ_demand = self._calculate_econ_demand()
        bus_demand  = self._calculate_bus_demand()

        econ_available = self.econ_seats_total - self.econ_sold
        bus_available  = self.bus_seats_total  - self.bus_sold
        
        expected_econ = econ_demand * self.econ_seats_total * 0.15
        expected_bus  = bus_demand  * self.bus_seats_total  * 0.1
        
        econ_bookings = min(np.random.poisson(expected_econ), econ_available)
        bus_bookings  = min(np.random.poisson(expected_bus),  bus_available)

        self.econ_sold += econ_bookings
        self.bus_sold  += bus_bookings

        step_revenue_econ = econ_bookings * self.econ_price
        step_revenue_bus  = bus_bookings  * self.bus_price
        
        self.revenue_econ  += step_revenue_econ
        self.revenue_bus   += step_revenue_bus
        self.total_revenue  = self.revenue_econ + self.revenue_bus

        self.days_to_departure -= 1
        self.current_step      += 1

        reward = self._calculate_reward(
            econ_bookings, bus_bookings,
            step_revenue_econ, step_revenue_bus
        )
        
        # Mild penalty: only penalize holding when load very low AND late
        if self.prev_action is not None:
            if action == self.prev_action and action == 4:
                load_factor = (self.econ_sold + self.bus_sold) / self.total_seats
                if load_factor < 0.4 and self.days_to_departure < 30:
                    reward -= 1.0

        done = (
            self.days_to_departure <= 0 or
            (self.econ_sold >= self.econ_seats_total and
             self.bus_sold  >= self.bus_seats_total)
        )
        
        terminated = done
        truncated  = False

        info = {
            "route":        self.route,
            "day":          self.max_days - self.days_to_departure,
            "econ_price":   self.econ_price,
            "bus_price":    self.bus_price,
            "econ_sold":    self.econ_sold,
            "bus_sold":     self.bus_sold,
            "total_sold":   self.econ_sold + self.bus_sold,
            "econ_bookings": econ_bookings,
            "bus_bookings":  bus_bookings,
            "revenue":      self.total_revenue,
            "load_factor":  (self.econ_sold + self.bus_sold) / self.total_seats,
            "disruption":   self.current_disruption,
            "econ_demand":  econ_demand,
            "bus_demand":   bus_demand
        }

        self.episode_history.append(info.copy())
        self.prev_action = action

        return self._get_state(), reward, terminated, truncated, info

    def _calculate_reward(self, econ_bookings, bus_bookings,
                          step_revenue_econ, step_revenue_bus):
        """
        Redesigned reward function.
        Goals:
          1. Maximize total revenue
          2. Maximize load factor (especially business class)
          3. Context-aware pricing during disruptions
        """

        # 1. PRIMARY: Revenue reward
        revenue_reward = (step_revenue_econ + step_revenue_bus) / 1000.0
        reward = revenue_reward

        # 2. Overall load factor
        load_factor = (self.econ_sold + self.bus_sold) / self.total_seats
        if load_factor >= 0.95:
            reward += 8
        elif load_factor >= 0.90:
            reward += 5
        elif load_factor >= 0.80:
            reward += 3
        elif load_factor >= 0.70:
            reward += 1
        else:
            reward -= 3

        # 3. Business class fill — strongly incentivised
        bus_load = self.bus_sold / self.bus_seats_total
        if bus_load >= 0.85:
            reward += 8
        elif bus_load >= 0.75:
            reward += 5
        elif bus_load >= 0.60:
            reward += 2
        elif bus_load < 0.50:
            reward -= 5

        # 4. Competitor pricing — context-aware
        econ_comp_avg = np.mean(list(self.econ_competitors.values()))
        bus_comp_avg  = np.mean(list(self.bus_competitors.values()))

        if self.current_disruption == "none":
            # Normal: economy at or above market
            if self.econ_price >= econ_comp_avg:
                reward += 2
            else:
                reward -= 1

            # Business: sweet spot slightly below to at-market (fills seats)
            if 0.85 * bus_comp_avg <= self.bus_price <= 1.05 * bus_comp_avg:
                reward += 4   # sweet spot
            elif 1.05 * bus_comp_avg < self.bus_price <= 1.15 * bus_comp_avg:
                reward += 1   # slightly above — acceptable
            elif self.bus_price > 1.15 * bus_comp_avg:
                reward -= 3   # too expensive → empty seats

        elif self.current_disruption == "competitor_cancel":
            # FIX: Competitor gone → strongly reward raising prices above market
            if self.econ_price > econ_comp_avg:
                reward += 8   # was 6, now stronger signal
            if self.bus_price > bus_comp_avg:
                reward += 8
            # Also penalise staying low — missed opportunity
            if self.econ_price < econ_comp_avg:
                reward -= 4
            if self.bus_price < bus_comp_avg:
                reward -= 4

        elif self.current_disruption in ["weather", "pilot_strike"]:
            # FIX: Demand drops → strongly reward cutting prices
            if self.econ_price < econ_comp_avg:
                reward += 6   # was 5
            if self.bus_price < bus_comp_avg:
                reward += 6
            # Penalise staying high during disruptions
            if self.econ_price > econ_comp_avg:
                reward -= 4
            if self.bus_price > bus_comp_avg:
                reward -= 4

        # 5. Late-stage empty seat penalty (last 7 days)
        if self.days_to_departure < 7 and load_factor < 0.70:
            empty_seats = self.total_seats - (self.econ_sold + self.bus_sold)
            reward -= empty_seats * 0.5

        # 6. Late-stage business empty seat penalty (last 14 days)
        if self.days_to_departure < 14 and bus_load < 0.60:
            bus_empty = self.bus_seats_total - self.bus_sold
            reward -= bus_empty * 0.8

        return reward

    def render(self, mode="human"):
        """Display current state"""
        load_factor = (self.econ_sold + self.bus_sold) / self.total_seats
        
        print(f"\n{'='*60}")
        print(f"Route: {self.route} | Day {self.max_days - self.days_to_departure}/{self.max_days}")
        print(f"{'='*60}")
        print(f"Economy:  ₹{self.econ_price:,.0f} | Sold: {self.econ_sold}/{self.econ_seats_total} ({self.econ_sold/self.econ_seats_total*100:.1f}%)")
        print(f"Business: ₹{self.bus_price:,.0f} | Sold: {self.bus_sold}/{self.bus_seats_total} ({self.bus_sold/self.bus_seats_total*100:.1f}%)")
        print(f"Load Factor: {load_factor*100:.1f}%")
        print(f"Revenue: ₹{self.total_revenue:,.0f} (E: ₹{self.revenue_econ:,.0f}, B: ₹{self.revenue_bus:,.0f})")
        if self.current_disruption != "none":
            print(f"⚠️ Disruption: {self.current_disruption} ({self.disruption_duration} days left)")
        print(f"{'='*60}")

    def get_episode_summary(self):
        """Return summary statistics for completed episode"""
        return {
            "route":            self.route,
            "total_revenue":    self.total_revenue,
            "econ_revenue":     self.revenue_econ,
            "bus_revenue":      self.revenue_bus,
            "load_factor":      (self.econ_sold + self.bus_sold) / self.total_seats,
            "econ_load_factor": self.econ_sold / self.econ_seats_total,
            "bus_load_factor":  self.bus_sold  / self.bus_seats_total,
            "final_econ_price": self.econ_price,
            "final_bus_price":  self.bus_price,
            "history":          self.episode_history
        }


# =================================================
# TESTING
# =================================================
if __name__ == "__main__":
    print("="*70)
    print("  TESTING ENHANCED MULTI-ROUTE MULTI-CLASS ENVIRONMENT")
    print("="*70)
    
    try:
        env = AirlineRevenueEnv(route_stats_path="data/route_stats.pkl")
        
        print("\n🎮 Running test episode...")
        state, _ = env.reset()
        print(f"Initial state shape: {state.shape}")
        print(f"Selected route: {env.route}")
        
        total_reward = 0
        for step in range(10):
            action = env.action_space.sample()
            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            
            if step % 3 == 0:
                env.render()
            
            if done:
                print(f"\n✓ Episode finished at step {step+1}")
                break
        
        print(f"\nTotal reward: {total_reward:.2f}")
        summary = env.get_episode_summary()
        print(f"Final load factor: {summary['load_factor']*100:.1f}%")
        print(f"Total revenue: ₹{summary['total_revenue']:,.0f}")
        
    except FileNotFoundError as e:
        print(f"\n❌ {e}")
        print("Please run data calibration first to generate route_stats.pkl")
    
    print("\n" + "="*70)
    print("  ✓ TEST COMPLETE")
    print("="*70)