#!/bin/bash

set -e  # Exit immediately if any command fails

# Airline RL Project Startup Script (Pip -> UV Integrated)

echo "=========================================="
echo "  🚀 Airline RL – Setup (run.sh) + Run (Docker)"
echo "=========================================="

# 0. Check for and install uv via pip if missing
if ! command -v uv &> /dev/null; then
    echo -e "\n[0/4] 'uv' not found. Installing uv via pip..."
    pip install uv
    
    if ! command -v uv &> /dev/null; then
        echo "⚠️  Note: 'uv' installed but not in PATH. Trying to run via 'python3 -m uv'..."
        UV_CMD="python3 -m uv"
    else
        UV_CMD="uv"
    fi
else
    UV_CMD="uv"
fi

echo "✓ Using command: $UV_CMD"

# 1. Setup structure
echo -e "\n[1/4] Checking project structure..."
python3 setup.py <<EOF
y
EOF

# 2. Create and prepare virtual environment
echo -e "\n[2/4] Setting up virtual environment..."
if [ ! -d ".venv" ]; then
    $UV_CMD venv
fi

# Activate virtual environment
source .venv/bin/activate

# 3. Install dependencies
echo -e "\n[3/5] Syncing dependencies..."
$UV_CMD pip install -r requirements.txt

# 4. Calibrate data
echo -e "\n[4/5] Calibrating environment from data..."
if [ -f "data/flight_data.csv" ] || [ -f "data/sample_data.csv" ]; then
    # Use sample data if real data isn't there yet
    if [ ! -f "data/flight_data.csv" ] && [ -f "data/sample_data.csv" ]; then
        cp data/sample_data.csv data/flight_data.csv
    fi
    python3 analyze_data.py
else
    echo "❌ Error: No data found in data/ folder."
    exit 1
fi

# 5. Train RL model
echo -e "\n[5/5] Training RL agent..."
MODEL_DIR="models/trained_models"
if [ -z "$(ls -A $MODEL_DIR 2>/dev/null)" ]; then
    echo "   🤖 No trained model found. Starting training (this may take a few minutes)..."
    python3 training/train.py
    echo "   ✓ Training complete"
else
    echo "   ✓ Trained model already exists — skipping training"
fi

echo ""
echo "=========================================="
echo "  ✅ Setup complete!"
echo "=========================================="
echo ""
echo "  Run the app with Docker Compose:"
echo "    docker compose up --build"
echo ""
echo "  Or in the background:  docker compose up --build -d"
echo ""
echo "  Then open: http://localhost:8080"
echo ""
echo "  Or run locally:  python3 app.py"
echo ""