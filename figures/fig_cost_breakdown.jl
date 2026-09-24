# =============================================================================
# FIGURE: STACKED COST BREAKDOWN BY YEAR AND SCENARIO
# =============================================================================
# Produces a 1×3 panel figure where each panel shows the mean H2 price
# decomposed into:
#   - Base production cost ($/kg)
#   - Infrastructure CAPEX (stations)
#   - Infrastructure O&M
#   - LCFS credits (negative — shown as reduction)
#
# Output: figures/out/fig_cost_breakdown.pdf
#
# Run from project root:
#   julia --project figures/fig_cost_breakdown.jl
# =============================================================================

using Statistics
using CairoMakie
using Random

cd(joinpath(@__DIR__, ".."))

include("../hydrogen truck deployment.jl")
include("config_defaults.jl")

const N_RUNS = 1000
const SEED   = 42
include(joinpath(@__DIR__, "fig_routing.jl"))   # figure → manuscript/appendix/old routing
const OUT_DIR = joinpath(@__DIR__, "out")
mkpath(OUT_DIR)

# ── Scenarios ────────────────────────────────────────────────────────────────
scenarios = [
    (
        label  = "SMR (natural gas)",
        config = build_config(
            h2_pathway_id               = "natural_gas_smr_liquid",
            expansion_pathway_id        = "natural_gas_smr_liquid",
            use_utilization_pricing     = true,
            electrolysis_pricing_enabled = false,
        ),
    ),
    (
        label  = "Electrolysis — grid",
        config = build_config(
            h2_pathway_id               = "electrolysis",
            expansion_pathway_id        = "electrolysis",
            use_utilization_pricing     = true,
            electrolysis_pricing_enabled = true,
            electrolysis_electricity_source = "grid",
            electrolyzer_capex_per_kw   = 1000.0,
            electricity_cost_per_kwh    = 0.05,
        ),
    ),
    (
        label  = "Electrolysis — solar",
        config = build_config(
            h2_pathway_id               = "electrolysis",
            expansion_pathway_id        = "electrolysis",
            use_utilization_pricing     = true,
            electrolysis_pricing_enabled = true,
            electrolysis_electricity_source = "solar",
            electrolyzer_capex_per_kw   = 1000.0,
            solar_capex_per_kw          = 800.0,
            solar_capacity_factor       = 0.25,
        ),
    ),
]

# ── Run ──────────────────────────────────────────────────────────────────────
println("Running simulations…")
results = map(scenarios) do s
    Random.seed!(SEED)
    price_r, _, _, _, base_r, infra_r, infra_gov_r, infra_om_r, lcfs_r, _, _, _, _, _, _, _, _, _, _ =
        run_monte_carlo(s.config, N_RUNS)
    years = collect(s.config.start_year : s.config.end_year)
    (
        years          = years,
        mean_base      = [mean(base_r[:, i])     for i in eachindex(years)],
        mean_infra     = [mean(infra_r[:, i])     for i in eachindex(years)],
        mean_infra_om  = [mean(infra_om_r[:, i])  for i in eachindex(years)],
        mean_lcfs      = [mean(lcfs_r[:, i])      for i in eachindex(years)],  # negative = credit
    )
end
println("Done.")

# ── Colour palette (components) ──────────────────────────────────────────────
c_base    = colorant"#4472C4"
c_infra   = colorant"#ED7D31"
c_om      = colorant"#FFC000"
c_lcfs    = colorant"#70AD47"   # shown going downward

# ── Figure ───────────────────────────────────────────────────────────────────
MM_TO_PT = 1 / 0.352778
fig = Figure(
    size     = (183 * MM_TO_PT, 80 * MM_TO_PT),
    fontsize = 8,
    fonts    = (; regular = "Arial", bold = "Arial Bold"),
)

axes = map(enumerate(zip(scenarios, results))) do (j, (s, r))
    ax = Axis(
        fig[1, j];
        xlabel             = "Year",
        ylabel             = j == 1 ? "H₂ delivery price (USD kg⁻¹)" : "",
        title              = s.label,
        titlesize          = 8,
        xlabelsize         = 7,
        ylabelsize         = 7,
        xticklabelsize     = 7,
        yticklabelsize     = 7,
        xticks             = r.years,
        xticklabelrotation = π / 4,
        yticklabelsvisible = j == 1,
    )

    w = 0.6   # bar width
    xs = Float64.(r.years)

    for (k, yr) in enumerate(r.years)
        base   = r.mean_base[k]
        infra  = r.mean_infra[k]
        om     = r.mean_infra_om[k]
        lcfs   = r.mean_lcfs[k]   # already negative (credit)

        # Positive stacks: base → infra → O&M
        y0 = 0.0
        for (val, col) in [(base, c_base), (infra, c_infra), (om, c_om)]
            barplot!(ax, [xs[k]], [val]; offset = y0, width = w, color = col, gap = 0)
            y0 += val
        end

        # LCFS credit: stacked below zero (downward from baseline)
        if lcfs < 0
            barplot!(ax, [xs[k]], [lcfs]; offset = 0.0, width = w, color = c_lcfs, gap = 0)
        end
    end

    ax
end

linkyaxes!(axes...)
colgap!(fig.layout, 8)

# Legend
Legend(
    fig[2, 1:3],
    [
        PolyElement(color = c_base),
        PolyElement(color = c_infra),
        PolyElement(color = c_om),
        PolyElement(color = c_lcfs),
    ],
    ["Production cost", "Infrastructure CAPEX", "Infrastructure O&M", "LCFS credits"];
    orientation  = :horizontal,
    tellwidth    = false,
    framevisible = false,
    labelsize    = 7,
)

rowgap!(fig.layout, 4)

out_path = fig_path(OUT_DIR, "fig_cost_breakdown.pdf")
save(out_path, fig; pt_per_unit = 1)
println("Saved → $out_path")
save(fig_path(OUT_DIR, "fig_cost_breakdown.png"), fig; px_per_unit = 300 / 72)
println("Saved → $(fig_path(OUT_DIR, "fig_cost_breakdown.png"))")
