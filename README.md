# Hydrogen truck and refuelling infrastructure rollout model

Monte Carlo model of heavy-duty fuel-cell truck adoption and the hydrogen
refuelling and production capacity built to serve it, with the cost, policy and
social-welfare accounting used in the accompanying journal article.

This repository is the code release for that article. It contains the model, the
input configuration, and every script that produces a figure or table in the
paper. Running the scripts listed below reproduces the published results.

## Requirements

Julia 1.9 or later. Two separate environments are used so that the heavy plotting
stack does not have to be installed to run the model:

| Environment | Purpose | Key packages |
|---|---|---|
| `Project.toml` (root) | the simulation model | Distributions, JSON |
| `figures/Project.toml` | figure rendering | CairoMakie, LaTeXStrings |

Both ship a `Manifest.toml`, so the exact package versions used for the published
results are pinned.

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=figures -e 'using Pkg; Pkg.instantiate()'
```

The `fig_model_schematic` figure is built by `figures/make_schematic.py`, which
needs Python 3 and `python-pptx`.

## Repository layout

```
hydrogen truck deployment.jl   the model: deployment, capacity investment,
                               hydrogen price build-up, TCO, policy credits
config/                        all inputs, as JSON
  model_defaults.json          horizon, Monte Carlo settings, discount rate
  stations_config.json         the named station pipeline and its status
  trucks_config.json           truck economics and the deployment scenarios
  tco_config.json              diesel comparator, mileage, fuel prices
  lcfs_config.json             pathway carbon intensities, credit prices
figures/                       one script per figure or table
  config_defaults.jl           build_config(): the shared baseline, with only
                               per-figure overrides given at each call site
  pub_theme.jl                 colours, sizes and save_pub(); see FIGURE_STYLE.md
  fig_routing.jl               figure -> manuscript/appendix ordering and numbering
  health_cost.jl               the single avoided-air-quality-damage basis
  out/                         output lands here (not version-controlled)
```

Inputs live only in `config/`. Some `description` fields in those JSON files
mention values being set "by the UI at runtime" — that refers to an interactive
front end that is not part of this release; for the results in the paper every
value comes from the JSON defaults and the per-figure overrides in
`figures/config_defaults.jl`.

## Running

The model on its own, with the defaults in `config/model_defaults.json`
(1000 runs, 2026-2045, seed 42):

```bash
julia --project=. "hydrogen truck deployment.jl"
```

Any figure, from the repository root:

```bash
julia --project=figures figures/fig_price_breakdown.jl
```

Each script sets its own seed, so figures are reproducible individually and do
not depend on the order they are run in. Output is written to
`figures/out/manuscript/`, `figures/out/appendix/` or `figures/out/old/`
according to `figures/fig_routing.jl`, as both PDF and 300 dpi PNG. Scripts that
report numbers also write a plain-text dump of the plotted values next to the
figure.

## Which script makes which figure

Manuscript figures are numbered by their position in `FIG_MANUSCRIPT` in
`figures/fig_routing.jl`, and that number is spliced into the output filename
(`fig_5_price_breakdown.pdf`).

| # | Figure | Script |
|---|---|---|
| 1 | Model schematic | `figures/make_schematic.py` |
| 2 | Technology learning curves | `fig_learning_curves.jl` |
| 3 | Station cost curves | `fig_station_costs.jl` |
| 4 | Truck deployment scenarios | `fig_scenario_inputs_split.jl` |
| 5 | Hydrogen fuel cost build-up | `fig_price_breakdown.jl` |
| 6 | Utilisation effect on cost | `fig_utilization_impact.jl` |
| 7 | Scenario matrix | `fig_scenario_matrix_overlay.jl` |
| 8 | Truck TCO comparison | `fig_tco_comparison.jl` |
| 9 | Capital vintage effect | `fig_vintage_effect.jl` |
| 10 | Sensitivity grid | `fig_sensitivity.jl` |
| 11 | Cost premium and health benefit | `fig_societal_cost_benefit.jl` |
| 12 | Net societal cost | `fig_societal_cost_benefit.jl` |
| 13 | Net abatement cost | `fig_abatement_cost.jl` |

Appendix figures and the policy table:

| Output | Script |
|---|---|
| `fig_background_demand` | `fig_background_demand.jl` |
| `fig_scc_fit` | `fig_scc_fit.jl` |
| `fig_capacity_vs_demand_combined` | `fig_capacity_vs_demand.jl`, `fig_capacity_vs_demand_high.jl`, then `stack_capacity_figs.jl` |
| `fig_tco_premium` | `fig_tco_premium.jl` |
| `fig_societal_benefit` | `fig_societal_benefit.jl` |
| `fig_policy_expenditure` | `fig_policy_expenditure.jl` |
| `fig_sensitivity_grid_high_dep` | `fig_sensitivity.jl high_dep` |
| `tab_2030_policy_gap.txt` | `tab_2030_policy_gap.jl` |

Scripts in `figures/` that are not listed above produce exploratory or superseded
views; they are kept because they share the same configuration path and are
useful for probing the model, and their output is routed to `figures/out/old/`.

## Two things worth knowing before changing inputs

**Deployment scenarios.** `limited_dep` and `high_dep` in
`config/trucks_config.json` are the two truck-deployment trajectories carried
through the paper; `nolow_dep` is a no-growth reference. Scripts take the
scenario either as a command-line argument or through an environment variable —
see the header comment of each script.

**Air-quality valuation.** `figures/health_cost.jl` is the single definition of
avoided health damage per displaced diesel mile, shared by every figure that
nets a health benefit against cost. It defaults to an EPA COBRA basis; the
alternative per-mile factors are selected with `HEALTH_BASIS=envcost`. The two
bases differ by roughly a factor of five and must never be mixed within a figure
suite.

## Validation case

`figures/val_2026_baseline.jl` runs a single year (2026) with the station list
frozen at what is operational today and no new build, so the result is
deterministic and can be held against observed conditions. It reports the price
build-up twice: with the existing stations carrying full annualised new-build
capital, and with that capital treated as sunk.

```bash
julia --project=figures figures/val_2026_baseline.jl
```

## Notes on the outputs

`figures/out/` is not version-controlled, with one exception:
`figures/out/cobra_input.csv` holds the avoided NOx and PM2.5 tonnages that were
submitted to the EPA COBRA model to obtain the monetised health benefits. COBRA
is run outside this repository, so this file is kept as the record of exactly
what was passed to it.

## Style

`figures/FIGURE_STYLE.md` documents the colour and typography conventions and the
reasoning behind them; `figures/pub_theme.jl` is the machine-readable version.

## License

MIT — see [`LICENSE`](LICENSE). Copyright (c) 2026 Joel Löfving.

You are free to use, modify and redistribute this code, including commercially,
provided the copyright notice is retained. If you use it in published work,
please cite the accompanying article.
