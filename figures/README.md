# Publication figures

Scripts in this folder produce the journal figures and tables directly from the
simulation model. Output goes to `figures/out/` as both `.pdf` (vector) and
`.png` (300 dpi), routed into `manuscript/`, `appendix/` or `old/` by
[`fig_routing.jl`](fig_routing.jl). The figure-to-script map is in the
[repository README](../README.md).

## First-time setup

```bash
julia --project=figures -e 'using Pkg; Pkg.instantiate()'
```

This installs CairoMakie into an environment separate from the model's, so the
model can be run without the plotting stack.

## Running a figure

Always from the repository root, not from inside this folder — the scripts
resolve `config/` relative to the root:

```bash
julia --project=figures figures/fig_price_breakdown.jl
julia --project=figures figures/fig_sensitivity.jl high_dep
```

## Shared pieces

| File | Role |
|---|---|
| `config_defaults.jl` | `build_config()` — the baseline configuration. Scripts pass only what differs from it, so a change to a shared assumption lands everywhere at once. |
| `pub_theme.jl` | Colours, font sizes, column widths and `save_pub()`. See [`FIGURE_STYLE.md`](FIGURE_STYLE.md) for the reasoning; keep the two in sync. |
| `fig_routing.jl` | Single source of figure ordering, numbering and output destination. Moving a name between `FIG_MANUSCRIPT` and `FIG_APPENDIX` is the only edit needed to promote or retire a figure. |
| `health_cost.jl` | The one definition of avoided air-quality damage per displaced diesel mile, shared by every figure that nets a health benefit against cost. |

## Adding a figure

1. Copy an existing script.
2. Call `build_config(; <overrides>)` for each scenario — specify only the
   parameters that differ from the defaults.
3. Call `run_monte_carlo(config, N_RUNS)` and destructure the returned channels.
4. Lay out with CairoMakie using the constants from `pub_theme.jl` rather than
   hard-coded colours or sizes.
5. Export with `save_pub(name, fig)`, and add `name` to `fig_routing.jl` if the
   figure is cited in the paper.
