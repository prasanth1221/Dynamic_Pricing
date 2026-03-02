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

rl_agent = None
rl_env = None
agent_loaded = False
comparison_results = None

# ── FIX 2: Cache for /api/ai_recommendation to stop excessive polling ──
_rec_cache = {}
_rec_cache_time = 0


class RLSimulationState:
    """Wrapper around the actual RL environment"""

    def __init__(self, env):
        self.env = env
        self.calibrated = True
        self.current_state = None
        self.done = False

    def reset(self):
        """Reset the RL environment"""
        state, info = self.env.reset()
        self.current_state = state
        self.done = False
        return self.current_state

    def step(self, action):
        """Execute action in RL environment"""
        next_state, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        self.current_state = next_state
        self.done = done
        return next_state, reward, done, info

    def get_state_dict(self):
        """Get current state as dictionary for API"""
        info = {
            'route': self.env.route,
            # Economy
            'econ_price': float(self.env.econ_price),
            'econ_sold': int(self.env.econ_sold),
            'econ_total': int(self.env.econ_seats_total),
            'econ_load_factor': float(self.env.econ_sold / self.env.econ_seats_total * 100),
            'econ_revenue': float(self.env.revenue_econ),
            # Business
            'bus_price': float(self.env.bus_price),
            'bus_sold': int(self.env.bus_sold),
            'bus_total': int(self.env.bus_seats_total),
            'bus_load_factor': float(self.env.bus_sold / self.env.bus_seats_total * 100),
            'bus_revenue': float(self.env.revenue_bus),
            # Overall
            'total_seats': int(self.env.total_seats),
            'total_sold': int(self.env.econ_sold + self.env.bus_sold),
            'load_factor': float((self.env.econ_sold + self.env.bus_sold) / self.env.total_seats * 100),
            'total_revenue': float(self.env.total_revenue),
            'days_to_departure': int(self.env.days_to_departure),
            'disruption': self.env.current_disruption,
            # Competitors
            'econ_competitors': {k: float(v) for k, v in self.env.econ_competitors.items()},
            'bus_competitors': {k: float(v) for k, v in self.env.bus_competitors.items()},
            'step': int(self.env.current_step),
            'calibrated': True,
            'available_routes': self.env.routes,
            'current_route': self.env.route
        }
        return info


sim_state = None


def load_rl_system():
    """Load RL environment and trained agent"""
    global rl_agent, rl_env, sim_state, agent_loaded

    print("\n" + "=" * 80)
    print("  🤖 LOADING RL SYSTEM")
    print("=" * 80)

    calibration_path = 'data/route_stats.pkl'

    if not os.path.exists(calibration_path):
        print(f"\n❌ ERROR: No calibration file at {calibration_path}")
        print(f"   Run: python analyze_data.py")
        return False

    try:
        with open(calibration_path, 'rb') as f:
            route_stats = pickle.load(f)

        print(f"✓ Loaded calibration for {len(route_stats)} routes")

        rl_env = AirlineRevenueEnv(
            route_stats_path=calibration_path,
            fixed_route=None
        )

        print(f"✓ Created RL environment")
        print(f"  State space: {rl_env.observation_space.shape[0]}")
        print(f"  Action space: {rl_env.action_space.n}")

        state_size = rl_env.observation_space.shape[0]
        AGENT_CONFIG['state_size'] = state_size

        rl_agent = DQNAgent(
            state_size=state_size,
            action_size=9,
            **{k: v for k, v in AGENT_CONFIG.items() if k not in ['state_size', 'action_size']}
        )

        print(f"✓ Created DQN agent")

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
        for model_path in model_paths:
            if os.path.exists(model_path):
                try:
                    rl_agent.load_model(model_path, load_optimizer=False)
                    rl_agent.epsilon = 0.0
                    print(f"✓ Loaded trained model: {model_path}")
                    model_loaded = True
                    agent_loaded = True
                    break
                except Exception as e:
                    print(f"⚠️  Failed to load {model_path}: {e}")
                    continue

        if not model_loaded:
            print(f"\n⚠️  WARNING: No trained model found!")
            print(f"   Agent will use UNTRAINED policy")
            agent_loaded = False

        sim_state = RLSimulationState(rl_env)
        sim_state.reset()

        print(f"\n✓ RL System Ready!")
        print(f"  Agent: {'TRAINED' if model_loaded else 'UNTRAINED'}")
        print("=" * 80)

        return True

    except Exception as e:
        print(f"\n❌ ERROR loading RL system: {e}")
        import traceback
        traceback.print_exc()
        return False


rl_system_loaded = load_rl_system()


@app.route('/')
def landing():
    return render_template('landing.html')


@app.route('/control')
def control():
    return render_template('index.html')


@app.route('/api/evaluation_log')
def evaluation_log():
    with open('results/evaluation_log.txt', 'r', encoding='utf-8') as f:
        return f.read(), 200, {'Content-Type': 'text/plain; charset=utf-8'}


@app.route('/results/<path:filename>')
def serve_results(filename):
    return send_from_directory('results', filename)


@app.route('/api/state')
def get_state():
    if not rl_system_loaded or sim_state is None:
        return jsonify({'error': 'RL system not loaded'}), 500
    return jsonify(sim_state.get_state_dict())


@app.route('/api/routes')
def get_routes():
    if not rl_system_loaded or rl_env is None:
        return jsonify({'error': 'RL system not loaded'}), 500
    return jsonify({
        'routes': rl_env.routes,
        'current_route': rl_env.route
    })


@app.route('/api/change_route', methods=['POST'])
def change_route():
    if not rl_system_loaded or rl_env is None:
        return jsonify({'error': 'RL system not loaded'}), 500

    data = request.json
    route = data.get('route')

    if route not in rl_env.routes:
        return jsonify({'error': f'Invalid route: {route}'}), 400

    rl_env.fixed_route = route
    sim_state.reset()

    return jsonify({
        'success': True,
        'route': route,
        'message': f'Switched to route: {route}'
    })


@app.route('/api/action', methods=['POST'])
def take_action():
    if not rl_system_loaded or sim_state is None:
        return jsonify({'error': 'RL system not loaded'}), 500

    data = request.json
    action = data.get('action', 4)

    if not (0 <= action < 9):
        return jsonify({'error': 'Invalid action'}), 400

    try:
        next_state, reward, done, info = sim_state.step(action)

        action_names = {
            0: 'E↓10% B↓10%', 1: 'E↓10% B→', 2: 'E↓10% B↑10%',
            3: 'E→ B↓10%', 4: 'E→ B→', 5: 'E→ B↑10%',
            6: 'E↑10% B↓10%', 7: 'E↑10% B→', 8: 'E↑10% B↑10%',
        }

        return jsonify({
            'success': True,
            'action_name': action_names[action],
            'econ_bookings': int(info['econ_bookings']),
            'bus_bookings': int(info['bus_bookings']),
            'total_bookings': int(info['econ_bookings'] + info['bus_bookings']),
            'econ_revenue': float(info['econ_bookings'] * sim_state.env.econ_price),
            'bus_revenue': float(info['bus_bookings'] * sim_state.env.bus_price),
            'total_revenue': float(info['revenue']),
            'reward': float(reward),
            'new_econ_price': float(info['econ_price']),
            'new_bus_price': float(info['bus_price']),
            'done': bool(done),
            'message': f"Action: {action_names[action]} | Sold {info['econ_bookings']}E + {info['bus_bookings']}B | Reward: {reward:.1f}"
        })

    except Exception as e:
        print(f"Error executing action: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/ai_recommendation')
def get_ai_recommendation():
    """Get RL agent's recommended action with dynamic, context-aware reasoning"""
    global _rec_cache, _rec_cache_time

    if not rl_system_loaded or sim_state is None:
        return jsonify({'error': 'RL system not loaded'}), 500

    if rl_agent is None:
        return jsonify({'error': 'RL agent not initialized'}), 500

    # ── FIX 2: Return cached result if less than 3 seconds old ──
    if time.time() - _rec_cache_time < 3 and _rec_cache:
        return jsonify(_rec_cache)

    try:
        state = sim_state.current_state
        if not isinstance(state, np.ndarray):
            state = np.array(state, dtype=np.float32)

        env = sim_state.env

        # ── Get Q-values ──────────────────────────────────────────
        q_values = rl_agent.get_action_distribution(state)
        action = int(np.argmax(q_values))
        q_value = float(q_values[action])

        action_names = {
            0: 'E↓10% B↓10%', 1: 'E↓10% B→',   2: 'E↓10% B↑10%',
            3: 'E→ B↓10%',    4: 'E→ B→',        5: 'E→ B↑10%',
            6: 'E↑10% B↓10%', 7: 'E↑10% B→',    8: 'E↑10% B↑10%',
        }
        action_name = action_names.get(action, f"Action {action}")

        # ── Confidence from Q-value spread ────────────────────────
        q_max  = float(np.max(q_values))
        q_mean = float(np.mean(q_values))
        q_std  = float(np.std(q_values))

        if q_std > 0.5:
            confidence = min(0.97, 0.5 + (q_max - q_mean) / (2 * q_std + 1e-8))
        else:
            confidence = 0.40

        # ── Context variables ──────────────────────────────────────
        econ_load  = env.econ_sold / env.econ_seats_total
        bus_load   = env.bus_sold  / env.bus_seats_total
        total_load = (env.econ_sold + env.bus_sold) / env.total_seats
        days_left  = env.days_to_departure

        econ_comp_avg = np.mean(list(env.econ_competitors.values()))
        bus_comp_avg  = np.mean(list(env.bus_competitors.values()))
        econ_ratio = env.econ_price / econ_comp_avg if econ_comp_avg > 0 else 1.0
        bus_ratio  = env.bus_price  / bus_comp_avg  if bus_comp_avg  > 0 else 1.0

        # ── Reason generation ──────────────────────────────────────
        if q_std < 0.5 and not agent_loaded:
            if days_left < 7 and total_load < 0.6:
                action = 0
                action_name = action_names[0]
                reason = f"⚠️ UNTRAINED — Rule override: {days_left}d left, only {total_load*100:.0f}% full → Drop prices!"
            elif econ_ratio > 1.15 and econ_load < 0.7:
                action = 1
                action_name = action_names[1]
                reason = f"⚠️ UNTRAINED — Rule override: Economy overpriced by {(econ_ratio-1)*100:.0f}% → Lower Economy"
            elif bus_ratio > 1.20 and bus_load < 0.6:
                action = 3
                action_name = action_names[3]
                reason = f"⚠️ UNTRAINED — Rule override: Business overpriced by {(bus_ratio-1)*100:.0f}% → Lower Business"
            elif econ_ratio < 0.90 and econ_load > 0.75:
                action = 8
                action_name = action_names[8]
                reason = f"⚠️ UNTRAINED — Rule override: Underpriced with high demand → Raise prices"
            elif env.current_disruption == 'competitor_cancel':
                action = 8
                action_name = action_names[8]
                reason = f"⚠️ UNTRAINED — Rule override: Competitor cancelled → Raise prices!"
            elif env.current_disruption in ['weather', 'pilot_strike']:
                action = 0
                action_name = action_names[0]
                reason = f"⚠️ UNTRAINED — Rule override: {env.current_disruption} disruption → Lower prices"
            else:
                action = 4
                action_name = action_names[4]
                reason = f"⚠️ UNTRAINED agent — Train model for better recommendations. Holding prices."
            confidence = 0.5

        elif agent_loaded:
            reasons = []
            if days_left < 7:
                reasons.append(f"⏰ Only {days_left}d to departure")
            if total_load < 0.5:
                reasons.append(f"🪑 Low fill: {total_load*100:.0f}%")
            if total_load > 0.85:
                reasons.append(f"🔥 High demand: {total_load*100:.0f}% full")
            if econ_ratio > 1.10:
                reasons.append(f"📉 Econ overpriced {(econ_ratio-1)*100:.0f}% vs market")
            if econ_ratio < 0.92:
                reasons.append(f"📈 Econ underpriced {(1-econ_ratio)*100:.0f}% vs market")
            if bus_ratio > 1.15:
                reasons.append(f"📉 Biz overpriced {(bus_ratio-1)*100:.0f}% vs market")
            if bus_ratio < 0.90:
                reasons.append(f"📈 Biz underpriced {(1-bus_ratio)*100:.0f}% vs market")
            if env.current_disruption != 'none':
                reasons.append(f"⚠️ Disruption: {env.current_disruption}")
            context_str = " | ".join(reasons) if reasons else "Stable market conditions"
            reason = f"🤖 RL Agent → {action_name} | {context_str} | Q-spread: {q_std:.2f}"
        else:
            reason = f"⚠️ Model loaded but uncertain (Q-spread: {q_std:.2f}) — Train more episodes"

        # ── FIX 1: Stable softmax (computed OUTSIDE the dict) ─────
        q_shifted     = q_values - np.max(q_values)
        exp_q         = np.exp(q_shifted)
        softmax_probs = exp_q / np.sum(exp_q)

        top3_indices = np.argsort(q_values)[::-1][:3]
        top3_actions = [
            {
                'action':      int(i),
                'name':        action_names[int(i)],
                'q_value':     float(q_values[i]),
                'probability': float(softmax_probs[i])
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
                'econ_price':      float(env.econ_price),
                'bus_price':       float(env.bus_price),
                'econ_vs_market':  f"{((econ_ratio - 1) * 100):+.1f}%",
                'bus_vs_market':   f"{((bus_ratio  - 1) * 100):+.1f}%",
                'econ_load':       float(econ_load * 100),
                'bus_load':        float(bus_load  * 100),
                'days_left':       int(days_left),
                'disruption':      env.current_disruption
            }
        }

        # ── FIX 2: Save to cache ───────────────────────────────────
        _rec_cache      = result
        _rec_cache_time = time.time()

        return jsonify(result)

    except Exception as e:
        print(f"Error getting AI recommendation: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/disruption', methods=['POST'])
def trigger_disruption():
    if not rl_system_loaded or sim_state is None:
        return jsonify({'error': 'RL system not loaded'}), 500

    data = request.json
    disruption_type = data.get('type', 'none')

    sim_state.env.current_disruption = disruption_type

    if disruption_type != 'none':
        sim_state.env.disruption_duration = np.random.randint(1, 4)
    else:
        sim_state.env.disruption_duration = 0

    messages = {
        'weather':            '⛈️ Weather delay! Demand -40%',
        'pilot_strike':       '✊ Pilot strike! Demand -70%',
        'competitor_cancel':  '✈️ Competitor cancelled! Demand +50%',
        'none':               '✅ Normal operations'
    }

    return jsonify({
        'success':   True,
        'disruption': disruption_type,
        'message':    messages.get(disruption_type, 'Unknown')
    })


@app.route('/api/reset', methods=['POST'])
def reset_simulation():
    if not rl_system_loaded or sim_state is None:
        return jsonify({'error': 'RL system not loaded'}), 500

    try:
        data = request.json if request.json else {}
        new_route = data.get('route')

        if new_route and new_route in rl_env.routes:
            rl_env.fixed_route = new_route

        sim_state.reset()

        return jsonify({
            'success':    True,
            'message':    'RL environment reset',
            'route':      sim_state.env.route,
            'calibrated': True
        })
    except Exception as e:
        print(f"Error resetting: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/history')
def get_history():
    if not rl_system_loaded or sim_state is None:
        return jsonify({'error': 'RL system not loaded'}), 500

    history = sim_state.env.episode_history[-100:]
    return jsonify({'history': history})


@app.route('/api/agent_info')
def get_agent_info():
    if not rl_system_loaded:
        return jsonify({'error': 'RL system not loaded'}), 500

    info = {
        'agent_loaded':    agent_loaded,
        'agent_status':    'trained' if agent_loaded else 'untrained',
        'state_size':      AGENT_CONFIG.get('state_size', 'unknown'),
        'action_size':     9,
        'epsilon':         float(rl_agent.epsilon) if rl_agent else 0.0,
        'device':          str(rl_agent.device) if rl_agent else 'unknown',
        'training_steps':  rl_agent.training_steps if rl_agent else 0,
        'episodes_trained': rl_agent.episode_count if rl_agent else 0,
    }

    return jsonify(info)


@app.route('/api/run_comparison', methods=['POST'])
def run_comparison():
    global comparison_results

    if not rl_system_loaded or rl_env is None:
        return jsonify({'error': 'RL system not loaded'}), 500

    try:
        data = request.json
        num_episodes = data.get('episodes', 50)

        print(f"\n🔄 Running comparison with {num_episodes} episodes per strategy...")

        comparison_results = compare_all_strategies(
            env=rl_env,
            rl_agent=rl_agent if agent_loaded else None,
            num_episodes=num_episodes
        )

        formatted_results = {}

        for strategy_name, metrics in comparison_results.items():
            formatted_results[strategy_name] = {
                'name':            strategy_name.replace('_', ' ').title(),
                'avg_revenue':     float(metrics['avg_revenue']),
                'std_revenue':     float(metrics['std_revenue']),
                'avg_load_factor': float(metrics['avg_load_factor'] * 100),
                'avg_econ_load':   float(metrics['avg_econ_load'] * 100),
                'avg_bus_load':    float(metrics['avg_bus_load'] * 100),
                'revenues':        [float(r) for r in metrics['revenues']],
                'load_factors':    [float(lf * 100) for lf in metrics['load_factors']]
            }

        if 'rl_agent' in formatted_results and agent_loaded:
            rl_revenue = formatted_results['rl_agent']['avg_revenue']
            traditional_strategies = [k for k in formatted_results.keys() if k != 'rl_agent']
            best_traditional = max(traditional_strategies,
                                   key=lambda k: formatted_results[k]['avg_revenue'])
            best_trad_revenue = formatted_results[best_traditional]['avg_revenue']
            improvement = ((rl_revenue - best_trad_revenue) / best_trad_revenue) * 100

            formatted_results['comparison_summary'] = {
                'rl_revenue':              rl_revenue,
                'best_traditional':        best_traditional,
                'best_traditional_revenue': best_trad_revenue,
                'improvement_percent':     float(improvement),
                'rl_advantage':            rl_revenue > best_trad_revenue
            }

        return jsonify({
            'success':      True,
            'results':      formatted_results,
            'num_episodes': num_episodes,
            'message':      f'Comparison complete: {len(formatted_results)} strategies evaluated'
        })

    except Exception as e:
        print(f"Error running comparison: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/get_comparison')
def get_comparison():
    if comparison_results is None:
        return jsonify({'error': 'No comparison results available. Run comparison first.'}), 404

    formatted_results = {}

    for strategy_name, metrics in comparison_results.items():
        formatted_results[strategy_name] = {
            'name':            strategy_name.replace('_', ' ').title(),
            'avg_revenue':     float(metrics['avg_revenue']),
            'std_revenue':     float(metrics['std_revenue']),
            'avg_load_factor': float(metrics['avg_load_factor'] * 100),
            'avg_econ_load':   float(metrics['avg_econ_load'] * 100),
            'avg_bus_load':    float(metrics['avg_bus_load'] * 100),
            'revenues':        [float(r) for r in metrics['revenues']],
            'load_factors':    [float(lf * 100) for lf in metrics['load_factors']]
        }

    if 'rl_agent' in formatted_results and agent_loaded:
        rl_revenue = formatted_results['rl_agent']['avg_revenue']
        traditional_strategies = [k for k in formatted_results.keys() if k != 'rl_agent']
        best_traditional = max(traditional_strategies,
                               key=lambda k: formatted_results[k]['avg_revenue'])
        best_trad_revenue = formatted_results[best_traditional]['avg_revenue']
        improvement = ((rl_revenue - best_trad_revenue) / best_trad_revenue) * 100

        formatted_results['comparison_summary'] = {
            'rl_revenue':              rl_revenue,
            'best_traditional':        best_traditional,
            'best_traditional_revenue': best_trad_revenue,
            'improvement_percent':     float(improvement),
            'rl_advantage':            rl_revenue > best_trad_revenue
        }

    return jsonify({'success': True, 'results': formatted_results})


@app.route('/api/test_traditional', methods=['POST'])
def test_traditional():
    if not rl_system_loaded or sim_state is None:
        return jsonify({'error': 'RL system not loaded'}), 500

    data = request.json
    strategy_name = data.get('strategy', 'rule_based')

    if strategy_name not in TRADITIONAL_STRATEGIES:
        return jsonify({'error': f'Unknown strategy: {strategy_name}'}), 400

    try:
        strategy_fn = TRADITIONAL_STRATEGIES[strategy_name]
        sim_state.reset()

        total_reward = 0
        actions_taken = []

        while not sim_state.done:
            action = strategy_fn(sim_state.env)
            next_state, reward, done, info = sim_state.step(action)
            total_reward += reward
            actions_taken.append(action)

        summary = sim_state.env.get_episode_summary()

        return jsonify({
            'success':       True,
            'strategy':      strategy_name.replace('_', ' ').title(),
            'total_revenue': float(summary['total_revenue']),
            'load_factor':   float(summary['load_factor'] * 100),
            'econ_load':     float(summary['econ_load_factor'] * 100),
            'bus_load':      float(summary['bus_load_factor'] * 100),
            'total_reward':  float(total_reward),
            'actions_taken': len(actions_taken),
            'message':       f'{strategy_name.replace("_", " ").title()} completed: ₹{summary["total_revenue"]:,.0f} revenue'
        })

    except Exception as e:
        print(f"Error testing traditional strategy: {e}")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("\n" + "=" * 80)
    print("  🚀 RL-INTEGRATED MULTI-CLASS AIRLINE DASHBOARD")
    print("=" * 80)

    if rl_system_loaded:
        print(f"\n  ✅ RL System Status: LOADED")
        print(f"  🤖 Agent Status: {'TRAINED ✓' if agent_loaded else 'UNTRAINED ⚠️'}")
        print(f"  🌍 Environment: Multi-route, Multi-class")
        print(f"  🎯 Action Space: 9 joint pricing actions")
        print(f"  📊 State Space: {AGENT_CONFIG.get('state_size', 'N/A')} features")
        print(f"  🛣️  Available Routes: {len(rl_env.routes)}")

        if not agent_loaded:
            print(f"\n  ⚠️  NO TRAINED MODEL FOUND")
            print(f"     The agent will use random/untrained policy")
            print(f"     Train a model first: python training/train.py")
    else:
        print(f"\n  ❌ RL System: FAILED TO LOAD")
        print(f"     Check calibration and dependencies")

    print(f"\n  🌐 Dashboard: http://localhost:5000")
    print(f"  📡 API Endpoints:")
    print(f"     GET  /api/state              - Current RL environment state")
    print(f"     GET  /api/routes             - Available routes")
    print(f"     POST /api/change_route       - Switch route")
    print(f"     POST /api/action             - Execute action")
    print(f"     GET  /api/ai_recommendation  - Get RL agent's best action")
    print(f"     GET  /api/history            - Episode history (daily revenue)")
    print("\n" + "=" * 80 + "\n")

    # ── FIX 3: debug=False to prevent double model loading ────────
    app.run(debug=False, host='0.0.0.0', port=5000)