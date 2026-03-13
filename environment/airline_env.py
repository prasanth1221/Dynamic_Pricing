"""
Multi-Route, Multi-Class Airline Revenue Management Environment
File: environment/airline_env.py

REWARD FUNCTION — v4 (market-anchored, anti-underpricing)
==========================================================
FIXES applied:
  1. Price floor: econ_min = q25 (removed *0.85 that caused race to bottom)
  2. Revenue norm: anchored to COMPETITOR market average, not agent's own price
  3. Occupancy bonus: gated on price ratio >= 0.90 (can't bonus cheap fills)
  4. Underpricing penalty: explicit -reward when agent prices below 88% of market
  5. Price ceiling: q75 * 1.25 (was 1.50, symmetry with floor improves exploration)
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
    - Market-anchored revenue reward function (v4)
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
        self.econ_base_demand = 0.20
        self.bus_base_demand  = 0.10  # business books at lower volume, earlier
        self.econ_price_elasticity = 2.5    # leisure: price sensitive
        self.bus_price_elasticity  = 1.2    # corporate: less sensitive

        # ── Disruption system ─────────────────────────────────────────────
        self.disruption_types       = ["none", "weather", "pilot_strike", "competitor_cancel"]
        self.disruption_probability = 0.05
        self.prev_action            = None

        print(f"✓ Environment initialised (v4 — market-anchored reward)")
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

        # ── FIX 1: Price bounds — floor at q25 (no *0.85 discount) ──────
        econ_stats = econ.get("price_stats", {})
        bus_stats  = bus.get("price_stats",  {})

        self.econ_min = float(econ_stats.get("q25", self.econ_price * 0.80))
        self.econ_max = float(econ_stats.get("q75", self.econ_price * 1.25) * 1.25)
        self.bus_min  = float(bus_stats.get( "q25", self.bus_price  * 0.80))
        self.bus_max  = float(bus_stats.get( "q75", self.bus_price  * 1.25) * 1.25)

        # ── FIX 2: Revenue normaliser anchored to MARKET, not agent price ─
        # Use competitor average so norm stays stable even if agent drops prices.
        econ_market = float(np.mean(list(self.econ_competitors.values())))
        bus_market  = float(np.mean(list(self.bus_competitors.values())))
        self._revenue_norm = max(
            self.econ_seats_total * 0.10 * econ_market +
            self.bus_seats_total  * 0.05 * bus_market,
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

        expected_econ = econ_demand * self.econ_seats_total * 0.055
        expected_bus  = bus_demand  * self.bus_seats_total  * 0.065

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
    # REWARD FUNCTION v4 — market-anchored, anti-underpricing
    # =========================================================================
    def _calculate_reward(self, econ_bookings, bus_bookings,
                          step_rev_econ, step_rev_bus):
        """
        Revenue-dominant reward, market-anchored.

        KEY CHANGES vs v3:
          - Revenue norm uses market price (fixed), not agent's drifting price
          - Occupancy bonuses gated on price ratio >= 0.90
          - Explicit penalty when agent prices below 88% of market
          - Business bonus also gated on price ratio
        """

        step_revenue = step_rev_econ + step_rev_bus

        # ── 1. PRIMARY: normalised revenue (always dominates) ─────────────
        reward = (step_revenue / self._revenue_norm) * 10.0

        load_factor = (self.econ_sold + self.bus_sold) / self.total_seats
        bus_load    = self.bus_sold  / self.bus_seats_total

        econ_comp = float(np.mean(list(self.econ_competitors.values()))) if self.econ_competitors else self.econ_price
        bus_comp  = float(np.mean(list(self.bus_competitors.values())))  if self.bus_competitors  else self.bus_price

        econ_ratio = self.econ_price / econ_comp if econ_comp > 0 else 1.0
        bus_ratio  = self.bus_price  / bus_comp  if bus_comp  > 0 else 1.0

        # ── FIX 4: Underpricing penalty — active signal not to race to bottom
        # Economy underpricing
        if econ_ratio < 0.85:
            reward -= 3.0   # severely underpriced vs market
        elif econ_ratio < 0.92:
            reward -= 1.0   # moderately underpriced

        # Business underpricing
        if bus_ratio < 0.85:
            reward -= 2.0
        elif bus_ratio < 0.92:
            reward -= 0.8

        # ── FIX 3: Occupancy bonus gated on price ratio ────────────────────
        # ONLY bonus if BOTH load is good AND pricing is competitive (>= 90% of market)
        if load_factor >= 0.85 and step_revenue > 0.8 * self._revenue_norm and econ_ratio >= 0.90:
            reward += 2.0
        elif load_factor >= 0.70 and step_revenue > 0.5 * self._revenue_norm and econ_ratio >= 0.88:
            reward += 1.0

        # ── 3. Bonus: business class health — gated on price ratio ────────
        if bus_load >= 0.70 and bus_ratio >= 0.85:
            reward += 1.5
        elif bus_load >= 0.50 and bus_ratio >= 0.80:
            reward += 0.5

        # ── 4. Disruption nudge ───────────────────────────────────────────
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
        if self.days_to_departure <= 7:
            if load_factor < 0.65:
                reward -= (1.0 - load_factor) * 4.0
            if bus_load < 0.50:
                reward -= (1.0 - bus_load)    * 3.0

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
                [self.econ_price / self.econ_max, self.bus_price / self.bus_max],
                [econ_comp / self.econ_max,       bus_comp / self.bus_max],
                [econ_remaining, bus_remaining],
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
        econ_market = float(np.mean(list(self.econ_competitors.values()))) if self.econ_competitors else self.econ_price
        bus_market  = float(np.mean(list(self.bus_competitors.values())))  if self.bus_competitors  else self.bus_price

        for airline in list(self.econ_competitors):
            chg = np.random.normal(0, 0.02)
            # Competitors react if agent significantly undercuts market
            if self.econ_price < econ_market * 0.88:
                chg -= 0.01   # competitor also lowers price slightly
            elif self.econ_price > econ_market * 1.12:
                chg += 0.01   # competitor raises when agent raises
            self.econ_competitors[airline] = float(np.clip(
                self.econ_competitors[airline] * (1 + chg),
                self.econ_min * 0.8, self.econ_max * 1.2
            ))

        for airline in list(self.bus_competitors):
            chg = np.random.normal(0, 0.02)
            if self.bus_price < bus_market * 0.88:
                chg -= 0.01
            elif self.bus_price > bus_market * 1.12:
                chg += 0.01
            self.bus_competitors[airline] = float(np.clip(
                self.bus_competitors[airline] * (1 + chg),
                self.bus_min * 0.8, self.bus_max * 1.2
            ))

    def _trigger_disruption(self):
        if self.disruption_duration > 0:
            self.disruption_duration -= 1
            if self.disruption_duration == 0:
                self.current_disruption = "none"
            return

        if random.random() < self.disruption_probability:
            self.current_disruption  = random.choice(self.disruption_types[1:])
            self.disruption_duration = random.randint(1, 4)
        else:
            self.current_disruption = "none"

    # =========================================================================
    # UTILITIES
    # =========================================================================
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

    def render(self, mode="human"):
        load = (self.econ_sold + self.bus_sold) / self.total_seats
        econ_comp = float(np.mean(list(self.econ_competitors.values()))) if self.econ_competitors else self.econ_price
        bus_comp  = float(np.mean(list(self.bus_competitors.values())))  if self.bus_competitors  else self.bus_price
        print(f"Day {self.max_days - self.days_to_departure:3d} | "
              f"Route: {self.route} | "
              f"E: ₹{self.econ_price:,.0f} (mkt ₹{econ_comp:,.0f}) | "
              f"B: ₹{self.bus_price:,.0f} (mkt ₹{bus_comp:,.0f}) | "
              f"Load: {load*100:.1f}% | "
              f"Rev: ₹{self.total_revenue:,.0f}")


# =============================================================================
# QUICK TEST
# =============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("  TESTING airline_env.py — market-anchored reward (v4)")
    print("=" * 70)

    try:
        env = AirlineRevenueEnv(route_stats_path="data/route_stats.pkl")
        state, _ = env.reset()

        print(f"\nRoute:            {env.route}")
        print(f"Econ start price: ₹{env.econ_price:,.0f}")
        print(f"Bus  start price: ₹{env.bus_price:,.0f}")
        print(f"Econ floor/ceil:  ₹{env.econ_min:,.0f} / ₹{env.econ_max:,.0f}")
        print(f"Revenue norm:     ₹{env._revenue_norm:,.0f}  ← market-anchored")
        print(f"State shape:      {state.shape}")

        total_reward = 0.0
        done = False
        while not done:
            action = 7 if env.days_to_departure < 30 else 4
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
        print(f"\n✅ Env v4 test passed")

    except FileNotFoundError as e:
        print(f"\n❌ {e}")