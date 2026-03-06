"""
Multi-Route, Multi-Class Airline Revenue Management Environment
File: environment/airline_env.py

REWARD FUNCTION — v3 (revenue-dominant)
========================================
ROOT CAUSE of negative RL improvement:
  Old reward: bonuses (load factor, business fill, competitor) = up to ±30 pts/step
              revenue signal                                   = only  ~6 pts/step
  → Agent learned to fill seats cheaply, not maximise revenue
  → rule_based beat RL because rule_based targets revenue directly

New design:
  Revenue = 90% of signal  (normalised per-route so ₹3k and ₹15k routes train identically)
  Bonuses = 10% max        (small nudges, not decision-drivers)
  Late-stage penalty only  (unsold seats at departure = real cost)

Expected result after retraining:
  +10-20% over rule_based after 1000 episodes
  +15-25% after 3000 episodes
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import random
import pickle
import os


class AirlineRevenueEnv(gym.Env):
    """
    Single-agent RL environment:
    - Multiple routes (sampled each episode)
    - Economy + Business joint pricing (9 actions)
    - Realistic demand curves with price elasticity
    - Competitor dynamics
    - Disruption modelling
    - Revenue-dominant reward function
    """

    def __init__(self, route_stats_path="data/route_stats.pkl",
                 fixed_route=None, seed=None):
        super().__init__()

        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        if not os.path.exists(route_stats_path):
            raise FileNotFoundError(
                f"❌ {route_stats_path} not found. Run: python analyze_data.py"
            )

        # Store path so external tools can rebuild fresh envs safely
        self._route_stats_path = route_stats_path

        with open(route_stats_path, "rb") as f:
            self.all_route_stats = pickle.load(f)

        self.routes      = list(self.all_route_stats.keys())
        self.num_routes  = len(self.routes)
        self.fixed_route = fixed_route

        print(f"✓ Loaded calibration for {self.num_routes} routes: {self.routes}")

        # ── Aircraft (A320 layout) ────────────────────────────────────────
        self.econ_seats_total = 150
        self.bus_seats_total  = 30
        self.total_seats      = self.econ_seats_total + self.bus_seats_total
        self.max_days         = 90

        # ── Action space: 9 joint pricing actions ────────────────────────
        self.ACTION_MAP = {
            0: (-0.10, -0.10),   # E↓10% B↓10%
            1: (-0.10,  0.00),   # E↓10% B→
            2: (-0.10, +0.10),   # E↓10% B↑10%
            3: ( 0.00, -0.10),   # E→   B↓10%
            4: ( 0.00,  0.00),   # E→   B→   (hold)
            5: ( 0.00, +0.10),   # E→   B↑10%
            6: (+0.10, -0.10),   # E↑10% B↓10%
            7: (+0.10,  0.00),   # E↑10% B→
            8: (+0.10, +0.10),   # E↑10% B↑10%
        }
        self.action_space = spaces.Discrete(9)

        # ── State space: 7 base + num_routes one-hot + 5 extra ───────────
        state_dim = 7 + self.num_routes + 5
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32
        )

        # ── Demand parameters ─────────────────────────────────────────────
        self.econ_base_demand      = 0.12   # 12% of capacity/day baseline
        self.bus_base_demand       = 0.12
        self.econ_price_elasticity = 2.5    # leisure: price sensitive
        self.bus_price_elasticity  = 1.2    # corporate: less sensitive

        # ── Disruption system ─────────────────────────────────────────────
        self.disruption_types       = ["none", "weather", "pilot_strike", "competitor_cancel"]
        self.disruption_probability = 0.05
        self.prev_action            = None

        print(f"✓ Environment initialised")
        print(f"  Action space:  {self.action_space.n} joint pricing actions")
        print(f"  State space:   {self.observation_space.shape[0]} features")
        print(f"  Seats:         {self.total_seats} (E:{self.econ_seats_total} B:{self.bus_seats_total})")

    # =========================================================================
    # RESET
    # =========================================================================
    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        # Select route
        if self.fixed_route and self.fixed_route in self.routes:
            self.route       = self.fixed_route
            self.route_index = self.routes.index(self.route)
        else:
            self.route_index = random.randint(0, self.num_routes - 1)
            self.route       = self.routes[self.route_index]

        self.route_stats = self.all_route_stats[self.route]

        econ = self.route_stats.get("Economy",  {})
        bus  = self.route_stats.get("Business", {})

        self.econ_competitors = dict(econ.get("competitor_prices", {}))
        self.bus_competitors  = dict(bus.get("competitor_prices",  {}))

        if not self.econ_competitors:
            self.econ_competitors = {"Competitor_A": 6000.0}
        if not self.bus_competitors:
            self.bus_competitors  = {"Competitor_A": 12000.0}

        # Start at market average
        self.econ_price = float(np.mean(list(self.econ_competitors.values())))
        self.bus_price  = float(np.mean(list(self.bus_competitors.values())))

        # Price bounds from calibration
        econ_stats = econ.get("price_stats", {})
        bus_stats  = bus.get("price_stats",  {})

        self.econ_min = float(econ_stats.get("q25", self.econ_price * 0.70) * 0.85)
        self.econ_max = float(econ_stats.get("q75", self.econ_price * 1.50) * 1.50)
        self.bus_min  = float(bus_stats.get( "q25", self.bus_price  * 0.70) * 0.85)
        self.bus_max  = float(bus_stats.get( "q75", self.bus_price  * 1.50) * 1.50)

        # ── Revenue normaliser (KEY for reward function) ──────────────────
        # "Typical good step" for THIS route's price level.
        # Makes reward scale identical across cheap and expensive routes.
        # Econ: ~10% of seats booked per step | Bus: ~5%
        self._revenue_norm = max(
            self.econ_seats_total * 0.10 * self.econ_price +
            self.bus_seats_total  * 0.05 * self.bus_price,
            5000.0
        )

        # State variables
        self.econ_sold         = 0
        self.bus_sold          = 0
        self.days_to_departure = self.max_days
        self.current_step      = 0

        self.revenue_econ  = 0.0
        self.revenue_bus   = 0.0
        self.total_revenue = 0.0

        self.current_disruption  = "none"
        self.disruption_duration = 0
        self.episode_history     = []
        self.prev_action         = None

        return self._get_state(), {}

    # =========================================================================
    # STEP
    # =========================================================================
    def step(self, action):
        econ_adj, bus_adj = self.ACTION_MAP[action]
        self.econ_price  *= (1 + econ_adj)
        self.bus_price   *= (1 + bus_adj)
        self.econ_price   = float(np.clip(self.econ_price, self.econ_min, self.econ_max))
        self.bus_price    = float(np.clip(self.bus_price,  self.bus_min,  self.bus_max))

        self._update_competitor_prices()
        self._trigger_disruption()

        econ_demand = self._calculate_econ_demand()
        bus_demand  = self._calculate_bus_demand()

        econ_available = self.econ_seats_total - self.econ_sold
        bus_available  = self.bus_seats_total  - self.bus_sold

        expected_econ = econ_demand * self.econ_seats_total * 0.15
        expected_bus  = bus_demand  * self.bus_seats_total  * 0.10

        econ_bookings = int(min(np.random.poisson(max(expected_econ, 0)), econ_available))
        bus_bookings  = int(min(np.random.poisson(max(expected_bus,  0)), bus_available))

        self.econ_sold += econ_bookings
        self.bus_sold  += bus_bookings

        step_rev_econ = econ_bookings * self.econ_price
        step_rev_bus  = bus_bookings  * self.bus_price

        self.revenue_econ  += step_rev_econ
        self.revenue_bus   += step_rev_bus
        self.total_revenue  = self.revenue_econ + self.revenue_bus

        self.days_to_departure -= 1
        self.current_step      += 1

        reward = self._calculate_reward(econ_bookings, bus_bookings,
                                        step_rev_econ, step_rev_bus)

        # Small inaction penalty: holding when very empty and late
        if (self.prev_action is not None and action == self.prev_action == 4):
            load = (self.econ_sold + self.bus_sold) / self.total_seats
            if load < 0.40 and self.days_to_departure < 30:
                reward -= 0.5

        done       = (self.days_to_departure <= 0 or
                      (self.econ_sold >= self.econ_seats_total and
                       self.bus_sold  >= self.bus_seats_total))
        terminated = done
        truncated  = False

        info = {
            "route":         self.route,
            "day":           self.max_days - self.days_to_departure,
            "econ_price":    self.econ_price,
            "bus_price":     self.bus_price,
            "econ_sold":     self.econ_sold,
            "bus_sold":      self.bus_sold,
            "total_sold":    self.econ_sold + self.bus_sold,
            "econ_bookings": econ_bookings,
            "bus_bookings":  bus_bookings,
            "revenue":       self.total_revenue,
            "load_factor":   (self.econ_sold + self.bus_sold) / self.total_seats,
            "disruption":    self.current_disruption,
            "econ_demand":   econ_demand,
            "bus_demand":    bus_demand,
        }

        self.episode_history.append(info.copy())
        self.prev_action = action

        return self._get_state(), float(reward), terminated, truncated, info

    # =========================================================================
    # REWARD FUNCTION  ← THE CORE FIX
    # =========================================================================
    def _calculate_reward(self, econ_bookings, bus_bookings,
                          step_rev_econ, step_rev_bus):
        """
        Revenue-dominant reward.

        Signal budget per step (with typical bookings):
        ─────────────────────────────────────────────
        Revenue signal:   ~10 pts   (90% of total)
        All bonuses max:  ~4.5 pts  (10% of total)
        Late penalties:   ~7 pts    (last 7 days only)

        Compare to OLD function:
        Revenue:   ~6 pts
        Bonuses:  ~24 pts  ← agent chased these instead
        """

        step_revenue = step_rev_econ + step_rev_bus

        # ── 1. PRIMARY: normalised revenue (always dominates) ─────────────
        reward = (step_revenue / self._revenue_norm) * 10.0

        load_factor = (self.econ_sold + self.bus_sold) / self.total_seats
        bus_load    = self.bus_sold  / self.bus_seats_total

        econ_comp = float(np.mean(list(self.econ_competitors.values()))) if self.econ_competitors else self.econ_price
        bus_comp  = float(np.mean(list(self.bus_competitors.values())))  if self.bus_competitors  else self.bus_price
        bus_ratio = self.bus_price / bus_comp if bus_comp > 0 else 1.0

        # ── 2. Bonus: strong occupancy + revenue together (max +2) ────────
        # Only fires when BOTH conditions are good — prevents filling cheaply
        if load_factor >= 0.85 and step_revenue > 0.8 * self._revenue_norm:
            reward += 2.0
        elif load_factor >= 0.70 and step_revenue > 0.5 * self._revenue_norm:
            reward += 1.0

        # ── 3. Bonus: business class health (max +1.5) ────────────────────
        # Reward filling business at a decent price only
        if bus_load >= 0.70 and bus_ratio >= 0.85:
            reward += 1.5
        elif bus_load >= 0.50 and bus_ratio >= 0.80:
            reward += 0.5

        # ── 4. Disruption nudge (max ±1 each class = ±2 total) ────────────
        if self.current_disruption == "competitor_cancel":
            if self.econ_price > econ_comp:              reward += 1.0
            if self.bus_price  > bus_comp:               reward += 1.0
            if self.econ_price < econ_comp * 0.95:       reward -= 1.0
            if self.bus_price  < bus_comp  * 0.95:       reward -= 1.0

        elif self.current_disruption in ["weather", "pilot_strike"]:
            if self.econ_price < econ_comp:              reward += 1.0
            if self.bus_price  < bus_comp:               reward += 1.0
            if self.econ_price > econ_comp * 1.05:       reward -= 1.0
            if self.bus_price  > bus_comp  * 1.05:       reward -= 1.0

        # ── 5. Late-stage empty-seat penalty (last 7 days only) ───────────
        # Unsold seat at departure = ₹0 forever. Create urgency.
        # Capped so it never overrides a good revenue step.
        if self.days_to_departure <= 7:
            if load_factor < 0.65:
                reward -= (1.0 - load_factor) * 4.0      # max -4 pts
            if bus_load < 0.50:
                reward -= (1.0 - bus_load)    * 3.0      # max -3 pts

        elif self.days_to_departure <= 14:
            if bus_load    < 0.35:  reward -= 1.5
            if load_factor < 0.50:  reward -= 1.0

        return float(reward)

    # =========================================================================
    # STATE
    # =========================================================================
    def _get_state(self):
        econ_remaining = (self.econ_seats_total - self.econ_sold) / self.econ_seats_total
        bus_remaining  = (self.bus_seats_total  - self.bus_sold)  / self.bus_seats_total
        days_norm      = self.days_to_departure / self.max_days

        econ_comp = float(np.mean(list(self.econ_competitors.values()))) if self.econ_competitors else self.econ_price
        bus_comp  = float(np.mean(list(self.bus_competitors.values())))  if self.bus_competitors  else self.bus_price

        econ_ratio = self.econ_price / econ_comp if econ_comp > 0 else 1.0
        bus_ratio  = self.bus_price  / bus_comp  if bus_comp  > 0 else 1.0

        route_enc = np.zeros(self.num_routes, dtype=np.float32)
        route_enc[self.route_index] = 1.0

        disruption = 1.0 if self.current_disruption != "none" else 0.0
        t_sin = float(np.sin(2 * np.pi * self.current_step / self.max_days))
        t_cos = float(np.cos(2 * np.pi * self.current_step / self.max_days))

        state = np.concatenate([
            [self.econ_price, self.bus_price],
            [econ_comp,       bus_comp],
            [econ_remaining,  bus_remaining],
            [days_norm],
            route_enc,
            [disruption, t_sin, t_cos],
            [econ_ratio, bus_ratio],
        ])
        return state.astype(np.float32)

    # =========================================================================
    # DEMAND MODELS
    # =========================================================================
    def _calculate_econ_demand(self):
        days_ratio   = self.days_to_departure / self.max_days
        time_factor  = 0.4 + (1.0 - days_ratio) * 1.2

        econ_comp    = float(np.mean(list(self.econ_competitors.values()))) if self.econ_competitors else self.econ_price
        price_ratio  = self.econ_price / econ_comp if econ_comp > 0 else 1.0
        price_factor = np.exp(-self.econ_price_elasticity * (price_ratio - 1.0))

        demand = self.econ_base_demand * time_factor * price_factor * self._get_disruption_factor()
        return float(np.clip(demand, 0.0, 1.0))

    def _calculate_bus_demand(self):
        days_ratio = self.days_to_departure / self.max_days

        if   days_ratio > 0.70:  time_factor = 0.40
        elif days_ratio > 0.40:  time_factor = 0.70
        else:                    time_factor = 1.20

        bus_comp     = float(np.mean(list(self.bus_competitors.values()))) if self.bus_competitors else self.bus_price
        price_ratio  = self.bus_price / bus_comp if bus_comp > 0 else 1.0
        price_factor = np.exp(-self.bus_price_elasticity * (price_ratio - 1.0))

        demand = self.bus_base_demand * time_factor * price_factor * self._get_disruption_factor()
        return float(np.clip(demand, 0.0, 1.0))

    def _get_disruption_factor(self):
        return {"weather": 0.60, "pilot_strike": 0.30, "competitor_cancel": 1.50}.get(
            self.current_disruption, 1.0
        )

    # =========================================================================
    # MARKET DYNAMICS
    # =========================================================================
    def _update_competitor_prices(self):
        for airline in list(self.econ_competitors):
            chg = np.random.normal(0, 0.02)
            self.econ_competitors[airline] = float(np.clip(
                self.econ_competitors[airline] * (1 + chg), self.econ_min, self.econ_max
            ))
        for airline in list(self.bus_competitors):
            chg = np.random.normal(0, 0.015)
            self.bus_competitors[airline] = float(np.clip(
                self.bus_competitors[airline] * (1 + chg), self.bus_min, self.bus_max
            ))

    def _trigger_disruption(self):
        if self.disruption_duration > 0:
            self.disruption_duration -= 1
            if self.disruption_duration == 0:
                self.current_disruption = "none"
        elif random.random() < self.disruption_probability:
            self.current_disruption  = random.choice(self.disruption_types[1:])
            self.disruption_duration = random.randint(1, 3)

    # =========================================================================
    # DISPLAY & SUMMARY
    # =========================================================================
    def render(self, mode="human"):
        load = (self.econ_sold + self.bus_sold) / self.total_seats
        print(f"\n{'='*62}")
        print(f"Route: {self.route}  |  Day {self.max_days - self.days_to_departure}/{self.max_days}")
        print(f"{'='*62}")
        print(f"Economy:  ₹{self.econ_price:>8,.0f}  |  {self.econ_sold}/{self.econ_seats_total} ({self.econ_sold/self.econ_seats_total*100:.1f}%)")
        print(f"Business: ₹{self.bus_price:>8,.0f}  |  {self.bus_sold}/{self.bus_seats_total}  ({self.bus_sold/self.bus_seats_total*100:.1f}%)")
        print(f"Load: {load*100:.1f}%   Revenue: ₹{self.total_revenue:,.0f}")
        if self.current_disruption != "none":
            print(f"⚠️  Disruption: {self.current_disruption} ({self.disruption_duration}d left)")
        print(f"{'='*62}")

    def get_episode_summary(self):
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
            "history":          self.episode_history,
        }


# =============================================================================
# QUICK TEST
# =============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("  TESTING airline_env.py — revenue-dominant reward (v3)")
    print("=" * 70)

    try:
        env = AirlineRevenueEnv(route_stats_path="data/route_stats.pkl")
        state, _ = env.reset()

        print(f"\nRoute:            {env.route}")
        print(f"Econ start price: ₹{env.econ_price:,.0f}")
        print(f"Bus  start price: ₹{env.bus_price:,.0f}")
        print(f"Revenue norm:     ₹{env._revenue_norm:,.0f}  ← typical good step")
        print(f"State shape:      {state.shape}")

        total_reward = 0.0
        done = False
        while not done:
            action = 7 if env.days_to_departure < 30 else 1
            state, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated

        summary = env.get_episode_summary()
        print(f"\nFull episode (simple rule):")
        print(f"  Total reward:  {total_reward:.1f}")
        print(f"  Revenue:       ₹{summary['total_revenue']:,.0f}")
        print(f"  Load:          {summary['load_factor']*100:.1f}%")
        print(f"  Economy load:  {summary['econ_load_factor']*100:.1f}%")
        print(f"  Business load: {summary['bus_load_factor']*100:.1f}%")
        print(f"\n✅ Env test passed")

    except FileNotFoundError as e:
        print(f"\n❌ {e}")