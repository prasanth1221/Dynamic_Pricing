"""
Project Setup Script
Run this to initialize the project structure
"""

import os
from pathlib import Path


def create_project_structure():
    """Create all necessary directories and placeholder files"""
    
    print("="*70)
    print("  AIRLINE RL PROJECT SETUP")
    print("="*70)
    
    # Base directory
    base_dir = Path(__file__).parent
    
    # Directory structure
    directories = [
        'config',
        'data',
        'models/trained_models',
        'agents',
        'environment',
        'utils',
        'training',
        'templates',
        'static/css',
        'static/js',
        'static/images',
        'logs',
        'results',
        'tests',
    ]
    
    print("\n📁 Creating directory structure...")
    for directory in directories:
        dir_path = base_dir / directory
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"   ✓ {directory}")
        
        # Create .gitkeep for empty directories
        if directory in ['logs', 'results', 'models/trained_models', 'static/images']:
            gitkeep = dir_path / '.gitkeep'
            gitkeep.touch()
    
    # Create __init__.py files
    print("\n📝 Creating __init__.py files...")
    init_dirs = ['config', 'agents', 'environment', 'utils', 'training', 'tests']
    for directory in init_dirs:
        init_file = base_dir / directory / '__init__.py'
        if not init_file.exists():
            init_file.write_text(f'"""{directory.capitalize()} module"""\n')
            print(f"   ✓ {directory}/__init__.py")
    
    # Check for required files
    print("\n📄 Checking required files...")
    required_files = {
        'app.py': 'Flask application',
        'requirements.txt': 'Python dependencies',
        'README.md': 'Project documentation',
        '.gitignore': 'Git ignore rules',
        'config/config.py': 'Configuration settings',
        'agents/model.py': 'DQN Agent model',
        'environment/airline_env.py': 'RL Environment',
        'utils/preprocessing.py': 'Data preprocessing',
        'training/train.py': 'Training pipeline',
        'templates/index.html': 'Dashboard HTML',
        'static/css/style.css': 'Dashboard CSS',
        'static/js/dashboard.js': 'Dashboard JavaScript',
    }
    
    missing_files = []
    for file_path, description in required_files.items():
        full_path = base_dir / file_path
        if full_path.exists():
            print(f"   ✓ {file_path} - {description}")
        else:
            print(f"   ✗ {file_path} - {description} (MISSING)")
            missing_files.append(file_path)
    
    # Summary
    print("\n" + "="*70)
    if missing_files:
        print("  ⚠️  SETUP INCOMPLETE")
        print("="*70)
        print("\nMissing files:")
        for file_path in missing_files:
            print(f"   - {file_path}")
        print("\nPlease create these files to complete the setup.")
    else:
        print("  ✓ SETUP COMPLETE")
        print("="*70)
        print("\n🎉 Project structure ready!")
        print("\nNext steps:")
        print("   1. Install dependencies: pip install -r requirements.txt")
        print("   2. Add your flight data: data/flight_data.csv")
        print("   3. Run the app: python app.py")
        print("   4. Open browser: http://localhost:5000")
    
    print()


def create_sample_data():
    """Create sample flight data"""
    import pandas as pd
    import numpy as np
    
    print("\n📊 Creating sample flight data...")
    
    np.random.seed(42)
    n_samples = 1000
    
    airlines = ['SpiceJet', 'AirAsia', 'Vistara', 'IndiGo', 'GoAir']
    routes = [
        'Delhi-Mumbai',
        'Delhi-Bangalore',
        'Mumbai-Chennai',
        'Mumbai-Kolkata',
        'Delhi-Hyderabad'
    ]
    
    sample_data = {
        'airline': np.random.choice(airlines, n_samples),
        'from': [r.split('-')[0] for r in np.random.choice(routes, n_samples)],
        'to': [r.split('-')[1] for r in np.random.choice(routes, n_samples)],
        'route': np.random.choice(routes, n_samples),
        'price': np.random.normal(6000, 800, n_samples).clip(3000, 15000),
        'duration_in_min': np.random.normal(120, 30, n_samples).clip(60, 300),
        'stops': np.random.choice([0, 1, 2], n_samples, p=[0.6, 0.3, 0.1]),
        'class_category': np.random.choice(
            ['Economy', 'Business'], n_samples, p=[0.85, 0.15]
        ),
        'day': np.random.randint(1, 32, n_samples),
        'month': np.random.randint(1, 13, n_samples),
        'dep_hour': np.random.randint(0, 24, n_samples),
        'arr_hour': np.random.randint(0, 24, n_samples),
    }
    
    df = pd.DataFrame(sample_data)
    
    # Save sample data
    base_dir = Path(__file__).parent
    sample_path = base_dir / 'data' / 'sample_data.csv'
    df.to_csv(sample_path, index=False)
    
    print(f"   ✓ Created {len(df)} sample flight records")
    print(f"   ✓ Saved to: {sample_path}")


if __name__ == "__main__":
    create_project_structure()
    
    # Ask to create sample data
    response = input("\n📊 Create sample flight data? (y/n): ").lower()
    if response == 'y':
        try:
            create_sample_data()
        except ImportError:
            print("   ⚠️  pandas/numpy not installed. Run: pip install pandas numpy")
    
    print("\n✓ Setup complete!\n")
