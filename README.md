# Causal Simulation of Engineering Education Expansion, GDP Growth, and Talent Migration

## Overview
This project implements a comprehensive causal simulation model combining System Dynamics (SD), Structural Causal Models (SCM), and Agent-Based Modeling (ABM) to analyze the relationships between engineering education expansion, GDP growth, and talent migration in India.

## Requirements

### Python Version
- Python 3.8 or higher

### Dependencies
The project requires the following Python packages:
- `numpy` - Numerical computing
- `pandas` - Data manipulation and analysis
- `matplotlib` - Data visualization
- `seaborn` - Statistical data visualization
- `scipy` - Scientific computing (integration, special functions)

## Installation

### 1. Install Python
Download and install Python 3.8+ from [python.org](https://www.python.org/)

### 2. Install Required Packages

Run the following command in the terminal:
```bash
pip install numpy pandas matplotlib seaborn scipy
```

Or create a requirements file and install from it:
```bash
pip install -r requirements.txt
```

## Running the Main Script

### Method 1: Direct Execution
```bash
python main.py
```

### Method 2: Using Python IDE
1. Open the project folder in your IDE (VS Code, PyCharm, Jupyter, etc.)
2. Open `main.py`
3. Click the "Run" button or press `Ctrl+F5` (or equivalent in your IDE)

### Method 3: From Terminal/Command Prompt
```bash
cd c:\Users\chira\Sim
python main.py
```

## Project Structure

```
Sim/
├── main.py                                    # Main simulation script
├── 1.py                                       # Supporting script 1
├── act.py                                     # Supporting script for activity/actions
├── test_approved_intake.py                    # Test script
├── scenario_projections_2030_2040_summary.csv # Scenario projections data
├── data/
│   ├── raw/
│   │   └── aicte/                            # Raw AICTE data by year and state
│   ├── processed/
│   │   └── aicte_approved_institutes_master.csv
│   └── metadata/
│       └── aicte_states.csv
├── Excels/                                    # Excel files
├── PDFs/                                      # PDF documents
└── Zip/                                       # Compressed files
```

## What the Script Does

The `main.py` script performs:
1. **Historical Data Analysis** (2010-2024) - Education supply, labor market, migration, and macroeconomic indicators
2. **System Dynamics Modeling** - Causal relationships between education expansion, GDP growth, and employment
3. **Structural Causal Modeling** - Causal graphs and interventional analysis
4. **Agent-Based Modeling** - Simulation of individual graduate decisions and migrations
5. **Scenario Projections** - Future scenarios (2030-2040) under different policy interventions

## Output

The script generates:
- Console output with simulation results and statistics
- Visualization plots (if matplotlib backend is configured)
- Projection reports for different scenarios

## Troubleshooting

### Module Not Found Error
If you get an error like `ModuleNotFoundError: No module named 'pandas'`, install the missing package:
```bash
pip install [package_name]
```

### Data File Issues
Ensure all data files in the `data/` folder are present before running the main script.

## Notes
- The simulation uses historical data from AISHE, PLFS, MOSPI, and OECD sources
- Results are saved and can be analyzed for policy implications
- Warnings are suppressed to keep output clean

## Author
Policy Research Team  
Date: January 2026
