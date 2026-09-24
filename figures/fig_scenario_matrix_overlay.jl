# =============================================================================
# FIGURE: H2 PRICE — LCFS SCENARIO PANELS (truck deployments overlaid)
# =============================================================================
# Second version of fig_scenario_matrix.jl.
#
# Instead of a 2×3 matrix, this draws a single panel for the flat-LCFS price
# scenario. Within the panel ALL three truck deployment scenarios are
# overlaid on top of each other:
#   • Production pathway → line COLOUR  (SMR / electrolysis-grid / -solar)
#   • Truck deployment   → line STYLE   (solid / dotted / dashed)
#
# Median H₂ delivery price only (no P25–P75 bands — 9 overlaid lines per panel
# would make bands unreadable).
#
# Output: figures/out/fig_scenario_matrix_overlay.{pdf,png}
#
# Run from the rollout model root directory:
#   julia --project figures/fig_scenario_matrix_overlay.jl
# =============================================================================

using Statistics
using CairoMakie
using Random
using JSON

cd(joinpath(@__DIR__, ".."))
include("../hydrogen truck deployment.jl")
include("config_defaults.jl")
include("pub_theme.jl")

const N_RUNS  = 1000
const SEED    = 42
const OUT_DIR = joinpath(@__DIR__, "out")
mkpath(OUT_DIR)

# ── Dimension definitions ─────────────────────────────────────────────────────

# 3 production pathways — distinguished by colour
pathways = [
    (label = "SMR (2026 mix)",        color = C_SMR,   kind = :smr),
    (label = "Electrolysis — grid",   color = C_GRID,  kind = :grid),
    (label = "Electrolysis — solar",  color = C_SOLAR, kind = :solar),
]

# 3 truck deployment scenarios — distinguished by line style
truck_specs = [
    (key = "nolow_dep", label = "No/Low deployment",          style = LS_NOLOW),
    (key = "limited_dep", label = "Limited deployment", style = LS_LIM),
    (key = "high_dep",  label = "High deployment",            style = LS_HIGH),
]

# LCFS price scenario — single panel (the increasing-LCFS sensitivity now lives
# in fig_sensitivity_grid.jl instead).
lcfs_specs = [
    (key = "no_change", label = "LCFS flat"),
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

# ── Diesel-equivalent H₂ delivery price ($/kg) ─────────────────────────────────
# Mirrors the two diesel lines in fig_tco_comparison panel (a): the $4.80 and
# $5.80/gal pump-price scenarios, both LCFS-adjusted. The diesel fuel cost
# ($/mile) is converted to the equivalent H₂ delivery price ($/kg) — the price
# at which H₂ fuel cost PER MILE equals diesel — so the lines are a COST-parity
# reference on this price axis:
#   $/kg-equiv = (diesel $/gal ÷ mpg) × miles_per_kg_H₂
# This is cost parity, not energy parity: it credits the fuel cell's efficiency
# advantage (9 mi/kg vs 7.43 mi/gal is ~30% more miles per MJ), so the line sits
# ABOVE the pump price it is named after — $4.80/gal plots at ≈$5.81/kg. Hence
# the "parity" wording in the legend; do not read it as a $/kg price of diesel.
tco_raw         = JSON.parsefile(joinpath(@__DIR__, "..", "config", "tco_config.json"))
miles_per_kg_h2 = Float64(tco_raw["hydrogen_truck"]["miles_per_kg_h2"])

# Current California diesel blend (66% RD, 6% BD, 28% fossil) and its carbon
# intensity, used to value the LCFS credit/deficit embedded in the fuel cost.
rd_frac = 0.66;  rd_ci = 43.74
bd_frac = 0.06;  bd_ci = 38.49
fd_frac = 1.0 - rd_frac - bd_frac;  fd_ci = 90.0
blend_ci = rd_frac * rd_ci + bd_frac * bd_ci + fd_frac * fd_ci   # gCO₂e/MJ

lcfs_raw     = JSON.parsefile(joinpath(@__DIR__, "..", "config", "lcfs_config.json"))
benchmark_ci = Dict{Int,Float64}(
    d["year"] => Float64(d["ci"])
    for d in lcfs_raw["diesel_ci_schedule"]["schedule"]
)

const DIESEL_MJ_PER_GAL = 128.45   # LHV energy content, MJ/gallon
const DIESEL_MPG        = TCO_DIESEL_MPG   # miles/gallon, from config/tco_config.json
const DIESEL_P_LO       = TCO_DIESEL_P_LO  # $/gal, from config/tco_config.json
const DIESEL_P_HI       = TCO_DIESEL_P_HI  # $/gal, from config/tco_config.json

# Diesel fuel cost at the low pump-price scenario ($/mile). The reference is a
# California pump price, not ATRI's national-average `fuel` field.
d_fuel = DIESEL_P_LO / DIESEL_MPG

# LCFS credit/deficit value per mile driven on diesel ($/mile).
function diesel_lcfs_per_mile(year::Int, price_schedule::Dict{Int,Float64})
    ci_std  = get(benchmark_ci, year, benchmark_ci[maximum(filter(y -> y <= year, keys(benchmark_ci)))])
    lcfs_p  = get(price_schedule, year, price_schedule[maximum(filter(y -> y <= year, keys(price_schedule)))])
    credits = (ci_std - blend_ci) / 1e6 * DIESEL_MJ_PER_GAL   # tCO₂e/gallon
    return credits * lcfs_p / DIESEL_MPG                       # $/mile
end

# Diesel-equivalent H₂ price trajectories ($/kg) for one LCFS price schedule.
# The ATRI fuel cost already embeds the 2026 LCFS credit; future years adjust by
# the change in credit/deficit value (same convention as fig_tco_comparison).
function diesel_equiv_kg(years, lcfs_prices::Dict{Int,Float64})
    lcfs_2026 = diesel_lcfs_per_mile(2026, lcfs_prices)
    d_fuel_hi = DIESEL_P_HI / DIESEL_MPG
    lo   = [(d_fuel    + (lcfs_2026 - diesel_lcfs_per_mile(yr, lcfs_prices))) * miles_per_kg_h2 for yr in years]
    hi   = [(d_fuel_hi + (lcfs_2026 - diesel_lcfs_per_mile(yr, lcfs_prices))) * miles_per_kg_h2 for yr in years]
    return (lo = lo, hi = hi)
end

# ── Shared y-axis limits ──────────────────────────────────────────────────────
x_lo = 2026
x_hi = END_YEAR
y_lo = 0.0
y_hi = 30.0

# Year ticks / interior gridlines that adapt to the horizon (2040 or 2050)
xtick_years     = filter(y -> x_lo <= y <= x_hi, [2026, 2030, 2035, 2040, 2045, 2050])
xinterior_years = filter(y -> x_lo < y < x_hi,  [2030, 2035, 2040, 2045])

# ── Figure layout (single panel: flat-LCFS price scenario) ─────────────────────
fig = Figure(size = (W_ONEHALF, 90 * MM_TO_PT))

panel_label = ["(a)", "(b)"]
axes_all = Axis[]

for (i, ls) in enumerate(lcfs_specs)
    ax = Axis(
        fig[1, i];
        xlabel             = "Year",
        ylabel             = i == 1 ? rich("H₂ fuel cost (USD kg⁻¹)") : "",
        xticks             = (xtick_years, string.(xtick_years)),
        xticklabelrotation = π / 4,
        yticks             = (collect(y_lo:5:y_hi),
                              [v % 10 == 0 ? string(Int(v)) : "" for v in y_lo:5:y_hi]),
        yticklabelsvisible = i == 1,
        yticksize          = i == 1 ? 4 : 0,
        limits             = ((x_lo, x_hi), (y_lo, y_hi)),
        xgridvisible       = false,
    )

    # Grid lines at interior years only
    vlines!(ax, xinterior_years; color = (:black, 0.12), linewidth = 1)

    # No panel letter — the figure is a single panel, so "(a)" would be orphaned.
    # Restore the text! call with panel_label[i] if a second LCFS panel is added back.

    # Overlay all truck deployment scenarios (line style) × pathways (colour)
    for (j, ts) in enumerate(truck_specs)
        cell = cell_results[i, j]
        for (pw, r) in zip(pathways, cell)
            lines!(ax, r.years, r.median_p;
                   color     = pw.color,
                   linewidth = 1.5,
                   linestyle = ts.style)
        end
    end

    # Diesel-equivalent break-even references (grey), matching fig_tco_comparison
    years_ref = cell_results[i, 1][1].years
    deq = diesel_equiv_kg(years_ref, load_lcfs_price_scenario(ls.key))
    lines!(ax, Float64.(years_ref), deq.lo;
           color = C_DIESEL, linewidth = 1.5, linestyle = :dashdot)
    lines!(ax, Float64.(years_ref), deq.hi;
           color = C_DIESEL, linewidth = 1.5, linestyle = :dot)

    push!(axes_all, ax)
end

linkyaxes!(axes_all...)
colgap!(fig.layout, 14)

# ── Legend (single box, embedded top-right) ───────────────────────────────────
# Colour entries = pathway; line-style entries = truck deployment scenario.
# horizontal + nbanks=3 ⇒ the 3 pathway colours fill the first column and the
# 3 deployment styles the second. No group headlines (entries are self-evident).
Legend(
    fig[1, 1],          # same cell as the axis
    [
        [LineElement(color = pw.color, linewidth = 1.5) for pw in pathways];
        [LineElement(color = :black, linewidth = 1.5, linestyle = ts.style)
         for ts in truck_specs];
        LineElement(color = C_DIESEL, linewidth = 1.5, linestyle = :dashdot);
        LineElement(color = C_DIESEL, linewidth = 1.5, linestyle = :dot)
    ],
    [
        [pw.label for pw in pathways];
        [ts.label for ts in truck_specs];
        "Diesel parity ($(usd_per_gal(DIESEL_P_LO)))";
        "Diesel parity ($(usd_per_gal(DIESEL_P_HI)))"
    ];
    orientation  = :horizontal,
    nbanks       = 3,
    tellwidth    = false,
    tellheight   = false,
    halign       = :right,
    valign       = :top,
    framevisible = true,
    labelsize    = FS_LEGEND,
    rowgap       = 2,
    colgap       = 6,
    patchsize    = (15, 8),
    padding      = (4, 4, 3, 3),
    margin       = (4, 8, 4, 8),  # (left, right, bottom, top) in points
)

# ── Save ──────────────────────────────────────────────────────────────────────
save_pub("fig_scenario_matrix_overlay", fig)

# ── Monte Carlo spread (P25–P75) ──────────────────────────────────────────────
# The figure draws the median delivered hydrogen fuel cost per scenario; the
# shaded bands are P10–P90. This dump reports the interquartile range for the
# same runs, so the spread can be quoted in the text without re-reading it off
# the bands.
let path = joinpath(OUT_DIR, "fig_scenario_matrix_overlay_spread.txt")
    open(path, "w") do io
        println(io, "="^92)
        println(io, " MONTE CARLO SPREAD — fig_scenario_matrix_overlay (manuscript fig $(FIG_NUMBER["fig_scenario_matrix_overlay"]))")
        println(io, " Delivered hydrogen fuel cost, USD/kg, net of LCFS + HRI + 45V")
        println(io, "="^92)
        @printf(io, " %d Monte Carlo runs per scenario, seed %d.\n", N_RUNS, SEED)
        println(io, " Plotted line = median; plotted band = P10–P90; IQR below = P25–P75.")
        println(io)
        println(io, " TWO THINGS THIS TABLE SHOWS THAT THE FIGURE DOES NOT. The panel's y-axis")
        @printf(io, " stops at \$%.0f/kg, so any row above that is drawn off-scale and clipped;\n", y_hi)
        println(io, " such rows are flagged with '>' in the last column. They occur only in the")
        println(io, " No/Low deployment scenario, where the fleet shrinks faster than it is")
        println(io, " replaced and the fixed station cost is spread over a collapsing volume.")
        println(io, " Once that fleet reaches zero the model cannot form a price at all and")
        println(io, " substitutes a \$40.00/kg sentinel to avoid dividing by zero — a placeholder,")
        println(io, " not a modelled cost. Those rows are the ones with an IQR of exactly 0.")
        println(io, "="^92)

        for i in eachindex(lcfs_specs), j in eachindex(truck_specs)
            for (pw, r) in zip(pathways, cell_results[i, j])
                println(io)
                println(io, "$(truck_specs[j].label) | $(pw.label) | $(lcfs_specs[i].label)")
                println(io, "-"^92)
                @printf(io, " %-6s %9s %9s %9s %9s %9s %9s %9s  %s\n",
                        "Year", "P10", "P25", "median", "P75", "P90", "IQR", "IQR/med", "")
                println(io, " " * "-"^92)
                for (k, y) in enumerate(r.years)
                    (y < 2026 || y > END_YEAR) && continue
                    iqr  = r.p75[k] - r.p25[k]
                    flag = r.median_p[k] > y_hi ? ">" : " "
                    @printf(io, " %-6d %9.3f %9.3f %9.3f %9.3f %9.3f %9.3f %8.1f %%  %s\n",
                            y, r.p10[k], r.p25[k], r.median_p[k], r.p75[k], r.p90[k],
                            iqr, r.median_p[k] != 0 ? 100 * iqr / r.median_p[k] : NaN, flag)
                end
            end
        end
        println(io)
        println(io, "="^92)
    end
    println("Monte Carlo spread → $path")
end
