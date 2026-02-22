# Scientific Computing
**By:** Tadhg Jones, David Kraakman, Storm Koper <br>
**Team:** 5

## Repository Structure

```text
├── pyproject.toml           # Project configuration and dependencies
├── README.md                # Project documentation and commands
└── set_1/                   # Code for Assignment Set 1
    ├── scripts/             # Executable scripts to generate plots and run experiments
    │   ├── benchmarks.py      # Benchmarking tool for base vs JIT performance
    │   ├── run_tide.py        # Scripts for Time-Independent Diffusion (Questions H-L)
    │   ├── run_wave1D.py      # Scripts for 1D Wave Equation (Questions B-C)
    │   └── run_wave2D.py      # Scripts for Time dependent 2D Diffusion Equation (Questions E-G)
    ├── tests/               # Pytest unit tests
    │   ├── test_misc.py       # Tests for helper functions
    │   ├── test_tide.py       # Tests for TIDE solvers (Jacobi, Gauss-Seidel, SOR)
    │   └── test_wave.py       # Tests for Wave solvers (1D, 2D, Leapfrog)
    └── utils/               # Core numerical solvers and helper modules
        ├── config.py          # Configuring global styles
        ├── misc.py            # Analytical solutions and image loading utilities
        ├── optimized.py       # Numba JIT-compiled optimized numerical schemes
        ├── TIDE.py            # Time-Independent Diffusion Equation solvers
        └── wave.py            # Wave Equation solvers
└── set_2/                   # Code for Assignment Set 2 (NOT YET IMPLEMENTED)
└── set_3/                   # Code for Assignment Set 3 (NOT YET IMPLEMENTED) 
```

## Dependency Management
For this project we use [uv](https://github.com/astral-sh/uv) for package and virtual environment management. If you have `uv` installed, you can set up the environment and install dependencies by running:
```bash
uv sync
```
However, if you use another package and virtual environt manager, then you can use the provided pyproject.toml. For example, using standard pip:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
pip install .
```

## Set 1
Below are the CLI commands to obtain the figures used in the report for set 1. Note that if you are not using `uv` you can replace `uv run` with `python` after activating your virtual environment.
### 1D Wave Equation
* **Question B** (Time development of 1D wave):
  ```bash
  uv run -m set_1.scripts.run_wave1D -b_type i -steps 1000 -save_every 10
  uv run -m set_1.scripts.run_wave1D -b_type ii -steps 1000 -save_every 10
  uv run -m set_1.scripts.run_wave1D -b_type iii -steps 1000 -save_every 10
  ```
* **Question C** (Animated Wave):
  ```bash
  uv run -m set_1.scripts.run_wave1D -b_type i -steps 1000 -save_every 10 --animate
  uv run -m set_1.scripts.run_wave1D -b_type ii -steps 1000 -save_every 10 --animate
  uv run -m set_1.scripts.run_wave1D -b_type iii -steps 1000 -save_every 10 --animate
  ```
### 2D Diffusion
* **Question E** (2D Diffusion vs Analytical Solution)
  ```bash
  uv run -m set_1.scripts.run_wave2D -question E
  ```
* **Question F** (2D domain at various time steps)
  ```bash
  uv run -m set_1.scripts.run_wave2D -question F
  ```
* **Question G** (Animation of 2D domain)
  ```bash
  uv run -m set_1.scripts.run_wave2D -question G
  ```
### Time-Independent Diffusion Equation (TIDE)
* **Question H** (3 iteration methods vs analytical solution)
  ```bash
  uv run -m set_1.scripts.run_tide -question H
  ```
* **Question I** (2D domain at various time steps)
  ```bash
  uv run -m set_1.scripts.run_tide -question I
  ```
* **Question J** (Omega sweep + golden section search for empty grid)
  ```bash
  uv run -m set_1.scripts.run_tide -question J
  ```
* **Question K** (Time evolution of diffusion with sinks + optimal omega)
  ```bash
  uv run -m set_1.scripts.run_tide -question K
  ```
* **Question L** (Time evolution of diffusion with insulators + optimal omega)
  ```bash
  uv run -m set_1.scripts.run_tide -question L
  ```
### Misc
* Run tests with:
  ```bash
  uv run pytest set_1
  ```
* Run benchmarks with:
  ```bash
  uv run -m set_1.scripts.benchmarks -method SOR -iterations 1000 -N 500 -repeats 5 --warmup_jit
  ```
