markdown

# Fractional Differential Inclusions in Economic Dynamics

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the complete **Python source code** for the numerical simulations and economic modelling presented in the paper:

> **"A Theoretical and Numerical Framework of Fractional Differential Inclusions Model in Economic Dynamics with Memory and Uncertainty"**  
> *Communications in Nonlinear Science and Numerical Simulation* (2026)  
> DOI: [10.1016/j.cnsns.2026.109893](https://doi.org/10.1016/j.cnsns.2026.109893)

The code implements a **predictor‑corrector L1‑scheme** for Caputo fractional derivatives, a **composite fractional operator**, and a **set‑valued inclusion** that models investment uncertainty under three economic scenarios: pessimistic, baseline, and optimistic. All figures and tables from the paper can be reproduced using the scripts provided here.

---

## 📁 Repository Structure

fractional-economic-inclusion/
│
├── src/ # Core numerical modules
│ ├── l1_derivative.py # L1 approximation of Caputo derivative
│ ├── composite_operator.py # D^σ( (1/p(t)) D^ω + λ ) x
│ ├── economic_model.py # Uncertainty functions & scenario selection
│ └── utils.py # Error norms, plotting helpers
│
├── benchmark/ # Validation against analytical solution (Section 4.2)
│ └── run_benchmark.py # Solves benchmark example & produces Table 1
│
├── economic_scenarios/ # Main economic simulations (Section 4.3)
│ └── run_economic_analysis.py # Generates all 6 figures + convergence table
│
├── convergence/ # Grid refinement study (Table 2 & Figure 9)
│ └── convergence_study.py
│
├── figures/ # Output directory for generated plots (auto‑created)
├── requirements.txt # Python dependencies
├── LICENSE # MIT License
└── README.md # This file
text


---

## 🚀 Getting Started

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/ghaderimehr-boop/fractional-economic-inclusion.git
   cd fractional-economic-inclusion

    Install dependencies
    bash

    pip install -r requirements.txt

    requirements.txt contains:
    text

    numpy>=1.21.0
    scipy>=1.7.0
    matplotlib>=3.4.0

📊 Reproducing the Paper's Results
1. Benchmark Example (Section 4.2 – validation of the L1 scheme)
bash

python benchmark/run_benchmark.py

This will:

    Compute the numerical solution for the test problem with exact solution x(t)=t+21t².

    Display the absolute error distribution.

    Print the convergence table (errors in L∞ and L₂ norms for N = 100, 200, 400, 800).

2. Economic Scenarios (Section 4.3 – main model)
bash

python economic_scenarios/run_economic_analysis.py

The script automatically:

    Builds the fractional composite operator matrices.

    Runs fixed‑point iterations for the three selection scenarios.

    Produces the following six figures (saved in the figures/ folder and shown on screen):

Figure in paper	Description
Fig. 7	Time trajectories of national income deviation x(t)x(t) under pessimistic, baseline, optimistic scenarios.
Fig. 8	Absolute deviations from the baseline scenario.
Fig. 9	Convergence of L∞L∞​ and L2L2​ errors with grid refinement (N = 100–800).
Fig. 10	First‑order derivative x′(t)x′(t) vs. fractional derivative D0.5x(t)D0.5x(t).
Fig. 11	Sensitivity analysis to fractional order σ = 0.5, 0.7, 0.9.
Fig. 12	Time evolution of state‑dependent uncertainty radius δ(t).

    Prints the convergence table (Table 2 in the paper) with observed rates.

3. Convergence Study Only (optional)
bash

python convergence/convergence_study.py

Runs the grid refinement study independently and saves the error data.
📝 Dependencies & Environment

All required libraries are listed in requirements.txt. It is recommended to use a virtual environment:
bash

python -m venv venv
source venv/bin/activate      # Linux/macOS
venv\Scripts\activate         # Windows
pip install -r requirements.txt

🧠 Core Numerical Methods

    Caputo fractional derivative approximated by the L1 scheme (for orders 0<α<1 and 1<α<2).

    Composite operator Dσ ⁣(1p(t)Dω+λ)x(t)Dσ(p(t)1​Dω+λ)x(t) implemented via matrix assembly.

    State‑dependent uncertainty band δ(t)=0.02(1+∣x′(t)∣+∣D0.5x(t)∣)δ(t)=0.02(1+∣x′(t)∣+∣D0.5x(t)∣).

    Fixed‑point iteration to satisfy the nonlocal boundary condition x′(0)=∫00.5x(s) dsx′(0)=∫00.5​x(s)ds.

    Three deterministic selections of the inclusion F(t)F(t):

        fpess=g(t)−δ(t)fpess​=g(t)−δ(t)

        fbase=g(t)fbase​=g(t)

        fopt=g(t)+δ(t)fopt​=g(t)+δ(t)

📖 Citation

If you use this code or the associated model in your research, please cite the original article:
bibtex

@article{ghaderimehr2026fractional,
  title   = {A Theoretical and Numerical Framework of Fractional Differential Inclusions Model in Economic Dynamics with Memory and Uncertainty},
  author  = {Ghaderi Mehr, B. and [other authors]},
  journal = {Communications in Nonlinear Science and Numerical Simulation},
  year    = {2026},
  doi     = {10.1016/j.cnsns.2026.109893},
  note    = {Code available at https://github.com/ghaderimehr-boop/fractional-economic-inclusion}
}

Additionally, please cite the Zenodo archive of this specific version ([10.5281/zenodo.19772111](https://doi.org/10.5281/zenodo.19772111)).
📜 License

This project is distributed under the MIT License. See the LICENSE file for details.
🤝 Contributing

Issues and pull requests are welcome. For major changes, please open an issue first to discuss what you would like to improve.
📧 Contact

For questions about the code or the paper, please open an issue on GitHub or contact the corresponding author at the email address given in the article.

Last updated: April 2026
Repository: https://github.com/ghaderimehr-boop/fractional-economic-inclusion
