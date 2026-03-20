# Airline Dynamic Pricing (RL-based)

A Reinforcement Learning based Dashboard and Environment for Dynamic Pricing in the Airline Industry. Evaluates pricing strategies for economy and business classes across multiple routes using a Deep Q-Network (DQN) agent and compares them to traditional rule-based strategies.

## Features
- **Custom RL Environment**: Simulates realistic demand, market competitiveness, and opportunity costs for airline seats.
- **Multi-class System**: Handles inventory management and dynamic pricing for both Economy and Business classes.
- **Live Disruption Simulation**: Inject real-world disruptions (weather delays, pilot strikes, competitor cancellations) and evaluate agent adaptability.
- **AI Recommendation Engine**: Explains its reasoning based on market context, load factor, and time to departure.
- **Baseline Comparisons**: Benchmarks the RL agent against traditional, random, and rule-based pricing strategies over customizable episode simulations.
- **High-Performance FastAPI Backend**: Dashboard logic completely implemented in completely async **FastAPI** to deliver rapid recommendations via local Uvicorn serving.

---

## 🚀 Quick Start
You can launch the entire project automatically via the provided setup script:

```bash
chmod +x run.sh
./run.sh
```
Follow the interactive prompt to run the system either via **Docker** or via a **Local Python Environment**.

---

## 🐳 Running with Docker (Recommended)
Running via Docker Compose is fully automated and creates persistent volumes for the trained agent and datasets.

```bash
docker compose up --build
```

**What this does during the build:**
1. Installs the blazing-fast `uv` package manager.
2. Synthesizes `requirements.txt` dependencies globally inside the container.
3. Bootstraps the environment using `setup.py` (generating initial folder structures/dummy flights).
4. Auto-trains the DQN agent if there are no existing model weights.
5. Boots the FastAPI app tightly managed under Uvicorn.

---

## 💻 Running Locally

If you prefer executing natively on your machine:

1. **Install uv (Faster pip replacement)**
```bash
pip install uv
```

2. **Setup virtual environment and sync dependencies**
```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

3. **Bootstrap setup & data calibration**
```bash
python3 setup.py
python3 analyze_data.py
```

4. **Train the RL Agent (Only strictly needed if no models exist)**
```bash
python3 training/train.py
```

5. **Start the FastAPI dashboard**
```bash
python3 app.py
```

---

## 🌐 Accessing the System
Once the server boot concludes (Local or Docker), open your browser to:

**[http://localhost:8080](http://localhost:8080)**

---

## 📁 Folder Structure

```text
├── agents/            # Reinforcement Learning agent networks (e.g. DQN model class)
├── baselines/         # Traditional and rule-based pricing logics for performance comparison
├── config/            # Project configurations, agent hyperparameters, and state mappers
├── data/              # Runtime datasets, parsed sample data, and route calibration output
├── environment/       # The core gymnasium-based RL environment (airline_env.py)
├── logs/              # Background training/assessment logs
├── models/            # Safetensor/Pickle model weights (.pth) dumped post-training
├── results/           # Performance charts, comparisons, and episode outcome logs
├── static/            # Frontend static files (CSS styles, specialized JS files, icons)
├── templates/         # Raw HTML files (landing.html, index.html) served by Jinja2
├── tests/             # PyTest integration points
├── training/          # Heavy-lifting agent training pipelines (train.py)
└── utils/             # Data preprocessing logic and data-load handlers
```

---

## 📡 Key API Endpoints
The dashboard performs background REST operations under FastAPI. You can interface with them manually:

- `GET /api/state` — View the live RL state (Days to departure, remaining supply, active competitor pricing).
- `GET /api/routes` — Look up supported dashboard flight routes.
- `POST /api/action` — Step the environment with 1 of 9 mapped joint pricing operations.
- `GET /api/ai_recommendation` — Request the RL Agent's best evaluated action + human-readable justification.
- `POST /api/disruption` — Apply a timeline disruption modifier against the network (weather/strike/competitor).
- `POST /api/run_comparison` — Fire an isolated N-episode benchmark against all traditional logic schemas.
- `POST /api/test_traditional` — Launch a background run for a singular traditional schema and return JSON metrics.
