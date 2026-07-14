# CausalMan

A Causal simulator for manufacturing process systems. CausalMan generates synthetic observational and interventional datasets from complex production lines with known ground-truth causal graphs, enabling benchmarking of causal discovery algorithms.

## Overview

CausalMan implements a Functional Causal Model (FCM) framework over a hierarchical manufacturing production line. Parts flow through machines and parallel sections, and each path through the line has a distinct causal structure parameterized by symbolic structural equations (SymPy). The simulator returns:

- **Observational data** — sensor readings from the partially observable production process
- **Interventional data** — data under hard do-calculus interventions on selected variables
- **Path data** — per-part routing through parallel sections
- **Causal graph** — the ground-truth DAG (and optionally its ADMG/MAG projections after marginalizing latent variables)

## Project Structure

```
causal_simulator_to_release/
├── src/                    # Installable Python package
│   ├── causalman.py              # Main CausalMan simulator class
│   ├── fcm.py                    # FCM core: DAG construction, sampling, interventions. 
│   ├── node.py                   # Node model definitions
│   ├── sample_batch.py           # Per-batch sampling logic
│   ├── graph_plotter.py          # Interactive graph visualization (Pyvis)
│   ├── graph_projections.py      # ADMG/MAG latent variable projections
│   ├── marginalization.py        # Marginalization via R dagitty (optional)
│   ├── col_masking.py            # Filter datasets to observable columns
│   ├── FCM_Definitions/          # FCM equation templates for parallel sections
│   ├── line_structure/           # Production line hierarchy (line, section, machine)
│   ├── dataset_objects/          # Pre-computed simulation configurations (~7.6 GB)
│   ├── utils/
│   │   ├── sampling.py           # Sequential and parallel batch runners
│   │   ├── graph.py              # Graph manipulation and I/O utilities
│   │   ├── data.py               # DataFrame processing utilities
│   │   └── equation.py           # Sympy equation construction helpers
│   ├── output/                     # Generated simulation results
│   ├── example_observational.ipynb
│   └── interventions_example.ipynb
│   ├── causal_inference_data_generation.ipynb  # Generate CI benchmark datasets (notebook)
│   ├── rca_data_generation.ipynb               # Generate RCA benchmark datasets (notebook)
│   └── generate_causal_inference_data.py       # Generate CI benchmark datasets (CLI)
├── pyproject.toml
└── requirements.txt
```

## Installation

**Requirements**: Python 3.9+

Clone the repository and install the package in one step:

```bash
git clone <repository-url>
cd causalman
pip install .
```

This installs the `causalman` package along with all required dependencies. The precomputed dataset objects (~7.6 GB of pickle files) are bundled inside the package and installed automatically — no separate download needed.

> **Note**: Installation copies the full `dataset_objects/` directory into your Python environment. Make sure you have at least 8 GB of free disk space before installing.

**Optional extras** — install only what you need:

```bash
pip install ".[graph-layout]"   # Graphviz-based graph layouts (pygraphviz)
pip install ".[png-export]"     # Export graphs to PNG via headless browser (playwright)
pip install ".[r-dagitty]"      # R dagitty marginalization bridge (rpy2)
pip install ".[all]"            # Everything above
```

After installing `playwright`, run:

```bash
playwright install chromium
```

After installing `rpy2`, ensure R is installed locally with the `dagitty` package.

Set the `R_HOME` environment variable to your R installation directory **before** importing `causalman.marginalization`:

```bash
# Linux / macOS
export R_HOME=/usr/lib/R          # typical system R
export R_HOME=/usr/local/lib/R    # Homebrew R on macOS

# Windows (PowerShell)
$env:R_HOME = "C:\Program Files\R\R-4.4.2"

# Windows (Command Prompt)
set R_HOME=C:\Program Files\R\R-4.4.2
```

You can also set it in Python before the import:

```python
import os
os.environ["R_HOME"] = "/path/to/R"   # must be set before importing rpy2

from causalman.marginalization import marginalize_to_mag
```

To find your R home directory, run `R.home()` inside an R session.

## Quick Start

```python
from causalman import CausalMan

simulator = CausalMan(
    name="causalman_small",   # dataset variant
    seed=42,                  # random seed
    batch_multiplier=1,       # controls number of samples per batch
)

obs_df, int_table, path_df, causal_dag = simulator.sample()
```

### Returned values

| Variable | Type | Description |
|---|---|---|
| `obs_df` | `pd.DataFrame` | Observational sensor readings (partially observable) |
| `int_table` | `pd.DataFrame` | Binary indicator table of which variables were intervened on |
| `path_df` | `pd.DataFrame` | Path routing of each part through parallel sections |
| `causal_dag` | `nx.DiGraph` | Ground-truth causal DAG (level 2) |

## Dataset Variants

| Name | Description |
|---|---|
| `causalman_micro` | Minimal single-simulation variant for quick testing |
| `causalman_small` | Two product simulations on a small production line |
| `causalman_medium` | Two product simulations on a medium production line |
| `causalman_large` | Three product simulations on a large production line |

Each variant corresponds to a different product running on the same production line structure but with distinct SCM parameterizations (noise levels, failure modes, material properties).

## Configuration Options

```python
CausalMan(
    name="causalman_small",   # one of: micro, small, medium, large
    seed=42,                  # reproducibility seed
    batch_multiplier=1,       # scale factor for number of samples
    parallelize=False,        # enable multi-process batch sampling
    max_workers=5,            # worker count when parallelize=True
    debug_mode=False,         # write debug outputs to disk
    save_path="output/run1",  # directory for CSV and graph outputs
)
```

## Interventional Sampling

Specify a dictionary of hard interventions (do-calculus) before calling `sample()`:

```python
simulator = CausalMan(name="causalman_small", seed=42)
simulator.intervention_dict = {"PF_M1_T1_sgrad": 18500}
obs_df, int_table, path_df, dag = simulator.sample()
```

The simulator mutates the causal graph and resamples downstream variables according to the intervention.

## Working with Causal Graphs

The returned `causal_dag` is a NetworkX `DiGraph`. Observable and latent nodes are tracked as node attributes.

```python
import networkx as nx

# List observable nodes
observable = [n for n, d in dag.nodes(data=True) if d.get("observable")]

# Export to GraphML
nx.write_graphml(dag, "causal_graph.graphml")
```

### Latent Variable Projections

To compute the ADMG (Acyclic Mixed Graph) and MAG (Maximal Ancestral Graph) after marginalizing latent variables:

```python
from causalman.graph_projections import get_latent_projection_single, admg2mag

admg = get_latent_projection_single(dag)
mag  = admg2mag(admg)
```

### Graph Visualization

```python
from causalman.graph_plotter import GraphPlotter

plotter = GraphPlotter(dag)
plotter.plot()  # opens interactive HTML in browser
```

### Masking to Observable Columns

```bash
python -m causalman.col_masking \
    --graph output/run1/batch_data/batch_0/batch_graph.pkl \
    --csv   output/run1/merged.csv \
    --output_dir output/run1/observable/
```

## Output Directory Layout

When `save_path` is provided:

```
output/run1/
├── batch_data/
│   └── batch_0/
│       ├── batch_graph.pkl          # Ground-truth causal DAG (pickled)
│       ├── batch_graph.graphml      # Graph in GraphML format
│       └── observed_nodes_list.txt  # List of observable node names
└── DEBUG/                           # Debug outputs (if debug_mode=True)
```

## Parallel Processing

For large-scale sampling, enable multi-process execution:

```python
simulator = CausalMan(
    name="causalman_medium",
    parallelize=True,
    max_workers=8,
)
obs_df, int_table, path_df, dag = simulator.sample()
```

## Examples

Two tutorial notebooks are provided in `causalman/`:

- **`example_observational.ipynb`** — basic FCM construction and observational sampling
- **`interventions_example.ipynb`** — interventional sampling and comparison with observational distributions

### Benchmark Data Generation

Two notebooks in `src/` generate ready-to-use benchmark CSV datasets:

- **`src/causal_inference_data_generation.ipynb`** — generates causal inference benchmark datasets. Set `SCALE`, `SEEDS`, and `N_SAMPLES` at the top of the notebook and run all cells. Produces for each (scale, seed) combination:
  - `observational.csv` — training data with no interventions
  - `task1_force_ltl_control.csv` / `task1_force_ltl_treatment.csv` — do(PF_M1_T1_Force_LTL = 15000/18000)
  - `task2_force_control.csv` / `task2_force_treatment.csv` — do(PF_M1_T1_Force = 16000/30000)

- **`src/rca_data_generation.ipynb`** — generates root-cause analysis benchmark datasets across 4 tasks and multiple scales. Set `SCALES`, `SEED`, and `N_SAMPLES` at the top and run all cells.

A CLI equivalent of the causal inference notebook is also available:

```bash
# Basic usage (variant required)
python src/generate_causal_inference_data.py --variant small

# Full benchmark: 5 seeds, 10 000 rows per dataset
python src/generate_causal_inference_data.py --variant medium --seeds 4 6 42 66 90

# Custom output directory and sample count
python src/generate_causal_inference_data.py --variant large --samples 5000 --output my_output/
```

| Argument | Default | Description |
|---|---|---|
| `--variant` | *(required)* | Scale variant: `micro`, `small`, `medium`, `large` |
| `--seeds` | `42` | One or more random seeds |
| `--samples` | `10000` | Rows to write per dataset |
| `--output` | `output/causalman_causal_inference` | Root output directory |

## Citation 

If you use CausalMan in your research, please cite the associated work. 

```bibtex
@misc{tagliapietra2025causalman,
      title={CausalMan: A physics-based simulator for large-scale causality}, 
      author={Nicholas Tagliapietra and Juergen Luettin and Lavdim Halilaj and Moritz Willig and Tim Pychynski and Kristian Kersting},
      year={2025},
      eprint={2502.12707},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2502.12707},
      doi={10.48550/arXiv.2502.12707}
}
```

**arXiv:** [https://arxiv.org/abs/2502.12707](https://arxiv.org/abs/2502.12707)

## License

CausalMan is open-sourced under the AGPL-3.0 license. See the
[LICENSE](LICENSE) file for details.

For a list of other open source components included in PROJECT-NAME, see the
file [3rd-party-licenses.txt](3rd-party-licenses.txt).