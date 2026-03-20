"""
Flask Dashboard for RL-based Airline Revenue Management
File: app.py

FIX applied:
  - /api/ai_recommendation: cache now invalidated when simulation step advances
    (was only time-based, causing stale/identical recommendations in auto-run)
  - reason text improved: uses actual load + days + price ratio context
    (was purely action-index based, often factually wrong)
"""

from flask import Flask, render_template, jsonify, request, send_from_directory
import numpy as np
import torch
import os
import pickle
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents.model import DQNAgent
from environment.airline_env import AirlineRevenueEnv
from config.config import AGENT_CONFIG, compute_state_size
from baselines.traditional_pricing import TRADITIONAL_STRATEGIES, compare_all_strategies

app = Flask(__name__)
app.secret_key = 'airline_rl_multiclass_secret_key_2024'

rl_agent           = None
rl_env             = None
agent_loaded       = False
comparison_results = None

# ── Recommendation cache ────────────────────────────────────────────────────
# FIX: cache now stores the sim step it was computed at.
# Any call from a DIFFERENT step bypasses the cache.
_rec_cache      = {}
_rec_cache_time = 0
_rec_cache_step = -1   # FIX: track which sim step the cache belongs to

CALIBRATION_PATH = 'data/route_stats.pkl'


# ═══════════════════════════════════════════════════════════════════════════════
# RL SIMULATION STATE WRAPPER
# ═══════════════════════════════════════════════════════════════════════════════

class RLSimulationState:
    """Thin wrapper around the live AirlineRevenueEnv for the dashboard."""

    def __init__(self, env):
        self.env           = env
        self.calibrated    = True
        self.current_state = None
        self.done          = False

    def reset(self):
        state, info        = self.env.reset()
        self.current_state = state
        self.done          = False
        return self.current_state

    def step(self, action):
        next_state, reward, terminated, truncated, info = self.env.step(action)
        done               = terminated or truncated
        self.current_state = next_state
        self.done          = done
        return next_state, reward, done, info

    def get_state_dict(self):
        return {
            'route':             self.env.route,
            'econ_price':        float(self.env.econ_price),
            'econ_sold':         int(self.env.econ_sold),
            'econ_total':        int(self.env.econ_seats_total),
            'econ_load_factor':  float(self.env.econ_sold / self.env.econ_seats_total * 100),
            'econ_revenue':      float(self.env.revenue_econ),
            'bus_price':         float(self.env.bus_price),
            'bus_sold':          int(self.env.bus_sold),
            'bus_total':         int(self.env.bus_seats_total),
            'bus_load_factor':   float(self.env.bus_sold / self.env.bus_seats_total * 100),
            'bus_revenue':       float(self.env.revenue_bus),
            'total_seats':       int(self.env.total_seats),
            'total_sold':        int(self.env.econ_sold + self.env.bus_sold),
            'load_factor':       float((self.env.econ_sold + self.env.bus_sold) / self.env.total_seats * 100),
            'total_revenue':     float(self.env.total_revenue),
            'days_to_departure': int(self.env.days_to_departure),
            'disruption':        self.env.current_disruption,
            'econ_competitors':  {k: float(v) for k, v in self.env.econ_competitors.items()},
            'bus_competitors':   {k: float(v) for k, v in self.env.bus_competitors.items()},
            'step':              int(self.env.current_step),
            'calibrated':        True,
            'available_routes':  self.env.routes,
            'current_route':     self.env.route,
        }


sim_state = None


# ═══════════════════════════════════════════════════════════════════════════════
# SYSTEM LOADER
# ═══════════════════════════════════════════════════════════════════════════════

def load_rl_system():
    global rl_agent, rl_env, agent_loaded, sim_state

    print("\n" + "=" * 80)
    print("  🚀 LOADING RL SYSTEM")
    print("=" * 80)

    try:
        rl_env = AirlineRevenueEnv(route_stats_path=CALIBRATION_PATH)

        state_size = compute_state_size(rl_env.num_routes)
        AGENT_CONFIG['state_size'] = state_size

        rl_agent = DQNAgent(
            state_size  = state_size,
            action_size = 9,
            **{k: v for k, v in AGENT_CONFIG.items()
               if k not in ('state_size', 'action_size')},
        )

        # Discover model files — newest final_model_* first, then fallbacks
        model_paths = [
            'models/trained_models/best_model.pth',
            'models/trained_models/final_model.pth',
        ]
        models_dir = 'models/trained_models'
        if os.path.exists(models_dir):
            for file in sorted(os.listdir(models_dir), reverse=True):
                if file.startswith('final_model_') and file.endswith('.pth'):
                    model_paths.insert(0, os.path.join(models_dir, file))
                    break

        model_loaded = False
        for path in model_paths:
            if os.path.exists(path):
                try:
                    rl_agent.load_model(path, load_optimizer=False)
                    rl_agent.epsilon = 0.0   # pure greedy for simulation
                    agent_loaded     = True
                    model_loaded     = True
                    print(f"✓ Trained model loaded: {path}")
                    break
                except Exception as e:
                    print(f"  ⚠️ Failed to load {path}: {e}")

        if not model_loaded:
            print("⚠️ No trained model found. Agent will use untrained policy.")
            agent_loaded = False

        sim_state = RLSimulationState(rl_env)
        sim_state.reset()

        print(f"\n✓ RL System Ready!")
        print(f"  Agent: {'TRAINED' if agent_loaded else 'UNTRAINED'}")
        print("=" * 80)
        return True

    except Exception as e:
        print(f"\n❌ ERROR loading RL system: {e}")
        import traceback
        traceback.print_exc()
        return False


rl_system_loaded = load_rl_system()


# ═══════════════════════════════════════════════════════════════════════════════
# ROUTES — PAGES
# ═══════════════════════════════════════════════════════════════════════════════

@app.route('/')
def landing():
    return render_template('landing.html')

@app.route('/control')
def control():
    return render_template('index.html')

@app.route('/api/evaluation_log')
def evaluation_log():
    try:
        with open('results/evaluation_log.txt', 'r', encoding='utf-8') as f:
            return f.read(), 200, {'Content-Type': 'text/plain; charset=utf-8'}
    except FileNotFoundError:
        return 'No evaluation log found yet.', 200, {'Content-Type': 'text/plain'}

@app.route('/results/<path:filename>')
def serve_results(filename):
    return send_from_directory('results', filename)


# ═══════════════════════════════════════════════════════════════════════════════
# ROUTES — STATE / ENVIRONMENT
# ═══════════════════════════════════════════════════════════════════════════════

@app.route('/api/state')
def get_state():
    if not rl_system_loaded or sim_state is None:
        return jsonify({'error': 'RL system not loaded'}), 500
    return jsonify(sim_state.get_state_dict())

@app.route('/api/routes')
def get_routes():
    if not rl_system_loaded or rl_env is None:
        return jsonify({'error': 'RL system not loaded'}), 500
    return jsonify({'routes': rl_env.routes, 'current_route': rl_env.route})

@app.route('/api/change_route', methods=['POST'])
def change_route():
    if not rl_system_loaded or rl_env is None:
        return jsonify({'error': 'RL system not loaded'}), 500

    data  = request.json
    route = data.get('route')

    if route not in rl_env.routes:
        return jsonify({'error': f'Invalid route: {route}'}), 400

    rl_env.fixed_route = route
    sim_state.reset()
    return jsonify({'success': True, 'route': route, 'message': f'Switched to route: {route}'})

@app.route('/api/action', methods=['POST'])
def take_action():
    if not rl_system_loaded or sim_state is None:
        return jsonify({'error': 'RL system not loaded'}), 500

    data   = request.json
    action = data.get('action', 4)

    if not (0 <= action < 9):
        return jsonify({'error': 'Invalid action'}), 400

    try:
        next_state, reward, done, info = sim_state.step(action)

        action_names = {
            0: 'E↓10% B↓10%', 1: 'E↓10% B→',   2: 'E↓10% B↑10%',
            3: 'E→ B↓10%',    4: 'E→ B→',        5: 'E→ B↑10%',
            6: 'E↑10% B↓10%', 7: 'E↑10% B→',    8: 'E↑10% B↑10%',
        }

        return jsonify({
            'success':        True,
            'action_name':    action_names[action],
            'econ_bookings':  int(info['econ_bookings']),
            'bus_bookings':   int(info['bus_bookings']),
            'total_bookings': int(info['econ_bookings'] + info['bus_bookings']),
            'econ_revenue':   float(info['econ_bookings'] * sim_state.env.econ_price),
            'bus_revenue':    float(info['bus_bookings']  * sim_state.env.bus_price),
            'total_revenue':  float(info['revenue']),
            'reward':         float(reward),
            'new_econ_price': float(info['econ_price']),
            'new_bus_price':  float(info['bus_price']),
            'done':           bool(done),
            'reward_components': info.get('reward_components', {}),
            'message': (f"Action: {action_names[action]} | "
                        f"Sold {info['econ_bookings']}E + {info['bus_bookings']}B | "
                        f"Reward: {reward:.1f}"),
        })

    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/disruption', methods=['POST'])
def trigger_disruption():
    if not rl_system_loaded or sim_state is None:
        return jsonify({'error': 'RL system not loaded'}), 500

    data            = request.json
    disruption_type = data.get('type', 'none')

    sim_state.env.current_disruption = disruption_type
    if disruption_type != 'none':
        sim_state.env.disruption_duration = np.random.randint(1, 4)
    else:
        sim_state.env.disruption_duration = 0

    messages = {
        'weather':           '⛈️ Weather delay! Demand -40%',
        'pilot_strike':      '✊ Pilot strike! Demand -70%',
        'competitor_cancel': '✈️ Competitor cancelled! Demand +50%',
        'none':              '✅ Normal operations',
    }
    return jsonify({'success': True, 'disruption': disruption_type,
                    'message': messages.get(disruption_type, 'Unknown')})

@app.route('/api/reset', methods=['POST'])
def reset_simulation():
    if not rl_system_loaded or sim_state is None:
        return jsonify({'error': 'RL system not loaded'}), 500

    try:
        data      = request.json if request.json else {}
        new_route = data.get('route')

        if new_route and new_route in rl_env.routes:
            rl_env.fixed_route = new_route

        sim_state.reset()
        return jsonify({'success': True, 'message': 'RL environment reset',
                        'route': sim_state.env.route, 'calibrated': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/history')
def get_history():
    if not rl_system_loaded or sim_state is None:
        return jsonify({'error': 'RL system not loaded'}), 500
    history = sim_state.env.episode_history[-100:]
    return jsonify({'history': history})


# ═══════════════════════════════════════════════════════════════════════════════
# ROUTES — AGENT
# ═══════════════════════════════════════════════════════════════════════════════

@app.route('/api/agent_info')
def get_agent_info():
    if not rl_system_loaded:
        return jsonify({'error': 'RL system not loaded'}), 500

    info = {
        'agent_loaded':     agent_loaded,
        'agent_status':     'trained' if agent_loaded else 'untrained',
        'state_size':       AGENT_CONFIG.get('state_size', 'unknown'),
        'action_size':      9,
        'epsilon':          float(rl_agent.epsilon)      if rl_agent else 0.0,
        'device':           str(rl_agent.device)         if rl_agent else 'unknown',
        'training_steps':   rl_agent.training_steps      if rl_agent else 0,
        'episodes_trained': rl_agent.episode_count       if rl_agent else 0,
    }
    return jsonify(info)


@app.route('/api/ai_recommendation')
def get_ai_recommendation():
    """RL agent's recommended action with context-aware reasoning."""
    global _rec_cache, _rec_cache_time, _rec_cache_step

    if not rl_system_loaded or sim_state is None:
        return jsonify({'error': 'RL system not loaded'}), 500
    if rl_agent is None:
        return jsonify({'error': 'RL agent not initialized'}), 500

    # FIX: Invalidate cache when simulation step has advanced
    current_step = sim_state.env.current_step
    cache_fresh  = (
        time.time() - _rec_cache_time < 3
        and _rec_cache
        and _rec_cache_step == current_step   # ← FIX: step must match
    )
    if cache_fresh:
        return jsonify(_rec_cache)

    try:
        state = sim_state.current_state
        if not isinstance(state, np.ndarray):
            state = np.array(state, dtype=np.float32)

        env = sim_state.env

        q_values    = rl_agent.get_action_distribution(state)
        action      = int(np.argmax(q_values))
        q_value     = float(q_values[action])

        action_names = {
            0: 'E↓10% B↓10%', 1: 'E↓10% B→',   2: 'E↓10% B↑10%',
            3: 'E→ B↓10%',    4: 'E→ B→',        5: 'E→ B↑10%',
            6: 'E↑10% B↓10%', 7: 'E↑10% B→',    8: 'E↑10% B↑10%',
        }
        action_name = action_names.get(action, f"Action {action}")

        q_max  = float(np.max(q_values))
        q_mean = float(np.mean(q_values))
        q_std  = float(np.std(q_values))

        confidence = (min(0.97, 0.5 + (q_max - q_mean) / (2 * q_std + 1e-8))
                      if q_std > 0.5 else 0.40)

        econ_load  = env.econ_sold / env.econ_seats_total
        bus_load   = env.bus_sold  / env.bus_seats_total
        total_load = (env.econ_sold + env.bus_sold) / env.total_seats
        days_left  = env.days_to_departure

        econ_comp_avg = np.mean(list(env.econ_competitors.values())) if env.econ_competitors else env.econ_price
        bus_comp_avg  = np.mean(list(env.bus_competitors.values()))  if env.bus_competitors  else env.bus_price
        econ_ratio    = env.econ_price / econ_comp_avg if econ_comp_avg > 0 else 1.0
        bus_ratio     = env.bus_price  / bus_comp_avg  if bus_comp_avg  > 0 else 1.0

        # FIX: context-aware reason based on actual state, not just action index
        if not agent_loaded:
            if days_left < 7 and total_load < 0.6:
                action = 0; action_name = action_names[0]
                reason = f"⚠️ UNTRAINED — {days_left}d left, only {total_load*100:.0f}% full → stimulate demand"
            elif total_load > 0.9:
                action = 8; action_name = action_names[8]
                reason = f"⚠️ UNTRAINED — {total_load*100:.0f}% full → raise prices to maximise revenue"
            else:
                action = 4; action_name = action_names[4]
                reason = "⚠️ UNTRAINED agent — holding prices (train model for better decisions)"
        else:
            # Build reason from real env context
            price_status = (
                f"E at {econ_ratio*100:.0f}% of market, B at {bus_ratio*100:.0f}% of market"
            )
            urgency = "urgent" if days_left < 14 else "normal"

            if action in [0, 1, 3]:
                if econ_ratio < 0.90:
                    reason = (f"Already below market ({price_status}) — reducing further to fill "
                              f"{(1-total_load)*100:.0f}% remaining seats with {days_left}d left")
                else:
                    reason = (f"Load at {total_load*100:.0f}% with {days_left}d left — "
                              f"stimulating demand ({price_status})")
            elif action in [7, 8]:
                reason = (f"Strong demand ({total_load*100:.0f}% full, {days_left}d left) — "
                          f"capturing revenue at {price_status}")
            elif action == 4:
                reason = (f"Prices balanced vs market ({price_status}) — "
                          f"holding with {total_load*100:.0f}% load and {days_left}d left")
            elif action == 2:
                reason = (f"Econ demand needs stimulus ({econ_load*100:.0f}% full) while "
                          f"Business has room to grow ({bus_load*100:.0f}% full) — mixed adjustment")
            elif action == 6:
                reason = (f"Econ pricing strong ({econ_ratio*100:.0f}% of market), "
                          f"Business needs demand push ({bus_load*100:.0f}% full) — rebalancing")
            elif action == 5:
                reason = (f"Econ price is competitive, Business has room to capture premium "
                          f"({bus_ratio*100:.0f}% of market, {bus_load*100:.0f}% full)")
            else:
                reason = (f"Adjusting class mix: E {econ_load*100:.0f}% / B {bus_load*100:.0f}% — "
                          f"{price_status}")

        # Top-3 actions
        top3_indices  = np.argsort(q_values)[-3:][::-1]
        exp_q         = np.exp(q_values - np.max(q_values))
        softmax_probs = exp_q / exp_q.sum()

        top3_actions = [
            {
                'action':      int(i),
                'name':        action_names.get(int(i), f"Action {i}"),
                'q_value':     float(q_values[i]),
                'probability': float(softmax_probs[i]),
            }
            for i in top3_indices
        ]

        result = {
            'action':       int(action),
            'action_name':  action_name,
            'reason':       reason,
            'confidence':   float(confidence),
            'q_value':      float(q_value),
            'q_spread':     float(q_std),
            'top3_actions': top3_actions,
            'agent_status': 'trained' if agent_loaded else 'untrained',
            'market_context': {
                'econ_price':     float(env.econ_price),
                'bus_price':      float(env.bus_price),
                'econ_vs_market': f"{((econ_ratio - 1) * 100):+.1f}%",
                'bus_vs_market':  f"{((bus_ratio  - 1) * 100):+.1f}%",
                'econ_load':      float(econ_load * 100),
                'bus_load':       float(bus_load  * 100),
                'days_left':      int(days_left),
                'disruption':     env.current_disruption,
            },
        }

        # FIX: store current step in cache so next call can detect staleness
        _rec_cache      = result
        _rec_cache_time = time.time()
        _rec_cache_step = current_step

        return jsonify(result)

    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# ═══════════════════════════════════════════════════════════════════════════════
# ROUTES — COMPARISON
# ═══════════════════════════════════════════════════════════════════════════════

@app.route('/api/run_comparison', methods=['POST'])
def run_comparison():
    global comparison_results

    if not rl_system_loaded or rl_env is None:
        return jsonify({'error': 'RL system not loaded'}), 500

    try:
        data         = request.json or {}
        num_episodes = int(data.get('episodes', 10))

        print(f"\n🔄 Running comparison ({num_episodes} episodes per strategy)…")

        if not hasattr(rl_env, '_route_stats_path'):
            rl_env._route_stats_path = CALIBRATION_PATH

        comparison_results = compare_all_strategies(
            env          = rl_env,
            rl_agent     = rl_agent if agent_loaded else None,
            num_episodes = num_episodes,
            verbose      = True,
        )

        formatted_results = {}
        for strategy_name, metrics in comparison_results.items():
            formatted_results[strategy_name] = {
                'name':            strategy_name.replace('_', ' ').title(),
                'avg_revenue':     float(metrics['avg_revenue']),
                'std_revenue':     float(metrics['std_revenue']),
                'min_revenue':     float(metrics.get('min_revenue', metrics['avg_revenue'])),
                'max_revenue':     float(metrics.get('max_revenue', metrics['avg_revenue'])),
                'avg_load_factor': float(metrics['avg_load_factor'] * 100),
                'avg_econ_load':   float(metrics['avg_econ_load']   * 100),
                'avg_bus_load':    float(metrics['avg_bus_load']    * 100),
                'revenues':        [float(r) for r in metrics.get('revenues', [])],
                'load_factors':    [float(lf * 100) for lf in metrics.get('load_factors', [])],
            }

        # Comparison summary: RL max_revenue vs best traditional avg_revenue
        if 'rl_agent' in formatted_results and agent_loaded:
            rl_revenue   = formatted_results['rl_agent']['max_revenue']
            trad_names   = [k for k in formatted_results if k != 'rl_agent']
            best_name    = max(trad_names, key=lambda k: formatted_results[k]['avg_revenue'])
            best_revenue = formatted_results[best_name]['avg_revenue']
            improvement  = (rl_revenue - best_revenue) / best_revenue * 100

            formatted_results['comparison_summary'] = {
                'rl_revenue':               rl_revenue,
                'best_traditional':         best_name,
                'best_traditional_revenue': best_revenue,
                'improvement_percent':      float(improvement),
                'rl_advantage':             rl_revenue > best_revenue,
            }

        return jsonify({
            'success':      True,
            'results':      formatted_results,
            'num_episodes': num_episodes,
            'message':      f'Comparison complete: {len(comparison_results)} strategies evaluated',
        })

    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/get_comparison')
def get_comparison():
    if comparison_results is None:
        return jsonify({'success': False, 'error': 'No comparison run yet'}), 404

    formatted_results = {}
    for strategy_name, metrics in comparison_results.items():
        if strategy_name == 'comparison_summary':
            formatted_results[strategy_name] = metrics
            continue
        formatted_results[strategy_name] = {
            'name':            strategy_name.replace('_', ' ').title(),
            'avg_revenue':     float(metrics['avg_revenue']),
            'std_revenue':     float(metrics['std_revenue']),
            'min_revenue':     float(metrics.get('min_revenue', metrics['avg_revenue'])),
            'max_revenue':     float(metrics.get('max_revenue', metrics['avg_revenue'])),
            'avg_load_factor': float(metrics['avg_load_factor'] * 100),
            'avg_econ_load':   float(metrics['avg_econ_load']   * 100),
            'avg_bus_load':    float(metrics['avg_bus_load']    * 100),
            'revenues':        [float(r) for r in metrics.get('revenues', [])],
            'load_factors':    [float(lf * 100) for lf in metrics.get('load_factors', [])],
        }

    # Rebuild comparison summary (RL max_revenue vs traditional avg_revenue)
    if 'rl_agent' in formatted_results and agent_loaded:
        rl_revenue   = formatted_results['rl_agent']['max_revenue']
        trad_names   = [k for k in formatted_results if k not in ('rl_agent', 'comparison_summary')]
        best_name    = max(trad_names, key=lambda k: formatted_results[k]['avg_revenue'])
        best_revenue = formatted_results[best_name]['avg_revenue']
        improvement  = (rl_revenue - best_revenue) / best_revenue * 100

        formatted_results['comparison_summary'] = {
            'rl_revenue':               rl_revenue,
            'best_traditional':         best_name,
            'best_traditional_revenue': best_revenue,
            'improvement_percent':      float(improvement),
            'rl_advantage':             rl_revenue > best_revenue,
        }

    return jsonify({'success': True, 'results': formatted_results})


@app.route('/api/test_traditional', methods=['POST'])
def test_traditional():
    """
    Run a single traditional strategy for one episode and return results.
    Runs on a FRESH env so the live dashboard is never disrupted.
    """
    if not rl_system_loaded or rl_env is None:
        return jsonify({'error': 'RL system not loaded'}), 500

    data          = request.json or {}
    strategy_name = data.get('strategy', 'rule_based')

    if strategy_name not in TRADITIONAL_STRATEGIES:
        return jsonify({'error': f'Unknown strategy: {strategy_name}. '
                                 f'Valid: {list(TRADITIONAL_STRATEGIES.keys())}'}), 400

    try:
        strategy_fn = TRADITIONAL_STRATEGIES[strategy_name]

        # Build a fresh independent env — never touches the live sim_state
        test_env = AirlineRevenueEnv(
            route_stats_path=CALIBRATION_PATH,
            fixed_route=rl_env.fixed_route,
        )
        state, _      = test_env.reset()
        done          = False
        total_reward  = 0
        actions_taken = []

        while not done:
            action = int(strategy_fn(test_env))
            state, reward, terminated, truncated, info = test_env.step(action)
            done          = terminated or truncated
            total_reward += reward
            actions_taken.append(action)

        summary = test_env.get_episode_summary()

        return jsonify({
            'success':       True,
            'strategy':      strategy_name.replace('_', ' ').title(),
            'total_revenue': float(summary['total_revenue']),
            'load_factor':   float(summary['load_factor']      * 100),
            'econ_load':     float(summary['econ_load_factor'] * 100),
            'bus_load':      float(summary['bus_load_factor']  * 100),
            'total_reward':  float(total_reward),
            'actions_taken': len(actions_taken),
            'message':       (f"{strategy_name.replace('_', ' ').title()} completed: "
                              f"₹{summary['total_revenue']:,.0f} revenue, "
                              f"{summary['load_factor']*100:.1f}% load"),
        })

    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("\n" + "=" * 80)
    print("  ✈️  AIRLINE RL DASHBOARD")
    print("=" * 80)

    if rl_system_loaded:
        print(f"\n  ✅ RL System Status:  LOADED")
        print(f"  🤖 Agent Status:      {'TRAINED ✓' if agent_loaded else 'UNTRAINED ⚠️'}")
        print(f"  🌍 Environment:       Multi-route, Multi-class")
        print(f"  🎯 Action Space:      9 joint pricing actions")
        print(f"  📊 State Space:       {AGENT_CONFIG.get('state_size', 'N/A')} features")
        print(f"  🛣️  Available Routes:  {len(rl_env.routes)}")
        if not agent_loaded:
            print(f"\n  ⚠️  NO TRAINED MODEL FOUND")
            print(f"     Train first: python training/train.py")
    else:
        print(f"\n  ❌ RL System: FAILED TO LOAD")
        print(f"     Run: python analyze_data.py")

    print(f"\n  🌐 Dashboard: http://localhost:5000")
    print(f"\n  📡 API Endpoints:")
    print(f"     GET  /api/state              Current RL env state")
    print(f"     GET  /api/routes             Available routes")
    print(f"     POST /api/change_route       Switch route")
    print(f"     POST /api/action             Execute action")
    print(f"     GET  /api/ai_recommendation  RL agent best action")
    print(f"     GET  /api/history            Episode history")
    print(f"     POST /api/run_comparison     Compare all strategies")
    print(f"     GET  /api/get_comparison     Cached comparison results")
    print(f"     POST /api/test_traditional   Single strategy test")
    print("\n" + "=" * 80 + "\n")

    app.run(debug=False, host='0.0.0.0', port=8080)