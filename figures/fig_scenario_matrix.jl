# =============================================================================
# FIGURE: H2 PRICE — 2 × 3 SCENARIO MATRIX
# =============================================================================
# 6-panel figure (2 rows × 3 columns):
#   Rows    — 2 LCFS price scenarios
#   Columns — 3 truck deployment scenarios
#   Panels  — 3 production pathways overlaid in each panel
#             (SMR, electrolysis–grid, electrolysis–solar)
#
# Output: figures/out/fig_scenario_matrix.pdf  (vector, journal-ready)
#
# Run from the rollout model root directory:
#   julia --project figures/fig_scenario_matrix.jl
# =============================================================================

using Statistics
using CairoMakie
using Random

cd(joinpath(@__DIR__, ".."))
include("../hydrogen truck deployment.jl")
include("config_defaults.jl")

const N_RUNS  = 1000
const SEED    = 42
include(joinpath(@__DIR__, "fig_routing.jl"))   # figure → manuscript/appendix/old routing
const OUT_DIR = joinpath(@__DIR__, "out")
mkpath(OUT_DIR)

# ── Dimension definitions ─────────────────────────────────────────────────────

# 3 production pathways — plotted together in every panel
pathways = [
    (label = "SMR (2026 mix)",        color = colorant"#4472C4", kind = :smr),
    (label = "Electrolysis — grid",   color = colorant"#ED7D31", kind = :grid),
    (label = "Electrolysis — solar",  color = colorant"#70AD47", kind = :solar),
]

# 3 truck deployment scenarios — columns
truck_specs = [
    (key = "nolow_dep", label = "No/Low deployment"),
    (key = "limited_dep",   label = "Limited deployment"),
    (key = "high_dep",    label = "High deployment"),
]

# 2 LCFS price scenarios — rows
lcfs_specs = [
    (key = "no_change",      label = "LCFS flat"),
    (key = "high_inc", label = "LCFS high increase"),
]

# ── Config builder ────────────────────────────────────────────────────────────
function make_config(kind, truck_sched, lcfs_prices)
    base = (
        h2_pathway_id                 = "current_mix",
        use_utilization_pricing       = true,
        utilization_transport_cost    = 1.0,
        use_lcfs                      = true,
        enable_45v                    = true,
        bus_demand_scenario           = "growing",
        end_year                      = END_YEAR,
        use_truck_deployment_schedule = true,
        truck_deployment_schedule     = truck_sched,
        lcfs_price_schedule_dict      = lcfs_prices,
    )

    if kind == :smr
        build_config(;
            base...,
            expansion_pathway_id         = "current_mix",
            electrolysis_pricing_enabled = false,
        )
    elseif kind == :grid
        build_config(;
            base...,
            expansion_pathway_id            = "electrolysis",
            electrolysis_pricing_enabled    = true,
            electrolysis_electricity_source = "grid",
            electrolyzer_capex_per_kw       = 3000.0,
            electricity_cost_per_kwh        = 0.2,
            electrolyzer_learning_rate      = 0.233,
            electrolyzer_stack_fraction     = 0.60,
            bop_learning_rate               = 0.04,
        )
    else  # :solar
        build_config(;
            base...,
            expansion_pathway_id            = "electrolysis",
            electrolysis_pricing_enabled    = true,
            electrolysis_electricity_source = "solar",
            electrolyzer_capex_per_kw       = 3000.0,
            solar_capex_per_kw              = 1600.0,
            solar_capacity_factor           = 0.25,
            solar_lifetime                  = 25,
            electrolyzer_learning_rate      = 0.233,
            electrolyzer_stack_fraction     = 0.60,
            bop_learning_rate               = 0.04,
            solar_panel_learning_rate       = 0.267,
            solar_panel_fraction            = 0.80,
            solar_bop_learning_rate         = 0.04,
        )
    end
end

# ── Run all simulations ───────────────────────────────────────────────────────
# cell_results[i, j] = Vector of 3 pathway results for (lcfs_i, truck_j)
n_total = length(lcfs_specs) * length(truck_specs) * length(pathways)
println("Running $n_total simulations ($(length(lcfs_specs)) LCFS × $(length(truck_specs)) truck × $(length(pathways)) pathway)…")

run_count = Ref(0)

cell_results = [
    begin
        truck_sched = load_truck_scenario(truck_specs[j].key)
        lcfs_prices = load_lcfs_price_scenario(lcfs_specs[i].key)
        map(pathways) do pw
            run_count[] += 1
            cfg = make_config(pw.kind, truck_sched, lcfs_prices)
            Random.seed!(SEED)
            price_r = run_monte_carlo(cfg, N_RUNS)[1]
            years = collect(cfg.start_year : cfg.end_year)
            print("  [$(run_count[])/$(n_total)] $(lcfs_specs[i].key) | $(truck_specs[j].key) | $(pw.label)…")
            r = (
                years    = years,
                mean_p   = [mean(price_r[:, k])             for k in eachindex(years)],
                median_p = [median(price_r[:, k])           for k in eachindex(years)],
                p10      = [quantile(price_r[:, k], 0.10)   for k in eachindex(years)],
                p25      = [quantile(price_r[:, k], 0.25)   for k in eachindex(years)],
                p75      = [quantile(price_r[:, k], 0.75)   for k in eachindex(years)],
                p90      = [quantile(price_r[:, k], 0.90)   for k in eachindex(years)],
            )
            println(" done (median $(END_YEAR): $(round(r.median_p[end], digits=2)) \$/kg)")
            r
        end
    end
    for i in 1:length(lcfs_specs), j in 1:length(truck_specs)
]

println("All simulations complete.")

# ── Shared y-axis limits ──────────────────────────────────────────────────────
x_lo = 2026
x_hi = END_YEAR
y_lo = 0.0
y_hi = 30.0

# Year ticks / interior gridlines that adapt to the horizon (2040 or 2050)
xtick_years     = filter(y -> x_lo <= y <= x_hi, [2026, 2030, 2035, 2040, 2045, 2050])
xinterior_years = filter(y -> x_lo < y < x_hi,  [2030, 2035, 2040, 2045])

# ── Figure layout ─────────────────────────────────────────────────────────────
MM_TO_PT = 1 / 0.352778
fig = Figure(
    size     = (183 * MM_TO_PT, 155 * MM_TO_PT),
    fontsize = 8,
)

# Panel labels (a)–(f), row-major order
panel_label = ["(a)" "(b)" "(c)"; "(d)" "(e)" "(f)"]

# Column headers — truck scenarios (row 1 of grid; panels start at row 2)
for (j, ts) in enumerate(truck_specs)
    Label(fig[1, j];
          text      = ts.label,
          fontsize  = 10,
          font      = :bold,
          tellwidth = false,
          padding   = (0, 0, 0, 2),
    )
end

# ── Axes ──────────────────────────────────────────────────────────────────────
axes_all = Axis[]

for (i, ls) in enumerate(lcfs_specs)
    for (j, ts) in enumerate(truck_specs)
        cell  = cell_results[i, j]
        years = cell[1].years

        # Y-axis label: embed LCFS scenario name on leftmost column
        yl = if j == 1
            rich(rich("$(ls.label)", font = :bold), "\nH\u2082 delivery price (USD kg\u207b\u00b9)")
        else
            ""
        end

        ax = Axis(
            fig[i + 1, j];
            xlabel             = i == length(lcfs_specs) ? "Year" : "",
            ylabel             = yl,
            xlabelsize         = 10,
            ylabelsize         = 10,
            xticklabelsize     = 10,
            yticklabelsize     = 10,
            xticks             = (xtick_years, string.(xtick_years)),
            xticklabelrotation = π / 4,
            xticklabelsvisible = i == length(lcfs_specs),
            xticksize          = 0,
            xminorticks        = xtick_years,
            xminorticksize     = i == length(lcfs_specs) ? 4 : 0,
            xminorticksvisible = true,
            xminortickcolor    = :black,
            yticklabelsvisible = j == 1,
            yticksize          = j == 1 ? 4 : 0,
            yticks = (collect(y_lo:5:y_hi), [i % 10 == 0 ? string(Int(i)) : "" for i in y_lo:5:y_hi]),
            limits             = ((x_lo,x_hi), (y_lo, y_hi)),
            xgridvisible       = false,
        )

        # Grid lines at interior ticks only (not at the edge years)
        vlines!(ax, xinterior_years; color = (:black, 0.12), linewidth = 1)

        # Panel letter in top-left corner
        text!(ax, 0.03, 0.97;
              text      = panel_label[i, j],
              space     = :relative,
              align     = (:left, :top),
              fontsize  = 8,
              font      = :bold,
              color     = :black,
        )

        # Plot pathways
        for (pw, r) in zip(pathways, cell)
            band!(ax, r.years, r.p25, r.p75; color = (pw.color, 0.30))
            lines!(ax, r.years, r.median_p;  color = pw.color, linewidth = 1.5)
        end

        push!(axes_all, ax)
    end
end

linkyaxes!(axes_all...)
colgap!(fig.layout, 6)
rowgap!(fig.layout, 4)

# ── Legend (inside panel f) ───────────────────────────────────────────────────
Legend(
    fig[length(lcfs_specs) + 1, length(truck_specs)],
    [
        [LineElement(color = pw.color, linewidth = 1.5) for pw in pathways]...,
        PolyElement(color = (:grey40, 0.30)),
    ],
    [
        [pw.label for pw in pathways]...,
        "P25\u2013P75",
    ];
    rowgap       = 1,
    framevisible = true,
    labelsize    = 7,
    patchsize    = (12, 8),
    padding      = (4, 4, 2, 2),
    halign       = 0.95,
    valign       = 0.97,
    tellwidth    = false,
    tellheight   = false,
)

# ── Save ──────────────────────────────────────────────────────────────────────
out_pdf = fig_path(OUT_DIR, "fig_scenario_matrix.pdf")
out_png = fig_path(OUT_DIR, "fig_scenario_matrix.png")
save(out_pdf, fig; pt_per_unit = 1)
println("Saved → $out_pdf")
save(out_png, fig; px_per_unit = 300 / 72)
println("Saved → $out_png")
