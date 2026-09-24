# =============================================================================
# FIGURE: H2 PRICE BREAKDOWN — SCENARIO: LCFS FLAT × SUGG. DEMO DEPLOYMENT
# =============================================================================
# Stacked area chart decomposing the delivered H2 price into:
#
#   Costs (stacked above zero, bottom → top):
#     · H2 production cost   (gross, before 45V credit)
#     · Transportation cost  (constant)
#     · Station O&M
#     · Station CAPEX        (company-funded portion)
#
#   Credits (stacked below zero, top → bottom):
#     · LCFS regular credits
#     · HRI bonus credits
#     · 45V production tax credit
#
#   Total delivered price — mean line + P25–P75 shaded band
#
# Corresponds to cell (1,2) of fig_scenario_matrix.jl:
#   LCFS scenario  : no_change (flat)
#   Truck scenario : limited_dep (limited deployment)
#   H2 pathway     : Electrolysis — solar
#
# Run from project root:
#   julia --project=figures figures/fig_price_breakdown.jl
# =============================================================================

using Statistics
using CairoMakie
using LaTeXStrings
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

# ── Simulation ────────────────────────────────────────────────────────────────
println("Running Monte Carlo simulation ($N_RUNS runs)…")
Random.seed!(SEED)

cfg = build_config(
    h2_pathway_id                   = "current_mix",
    expansion_pathway_id            = "electrolysis",
    use_utilization_pricing         = true,
    utilization_transport_cost      = 1.0,
    use_lcfs                        = true,
    enable_45v                      = true,
    bus_demand_scenario             = "growing",
    end_year                        = END_YEAR,
    use_truck_deployment_schedule   = true,
    truck_deployment_schedule       = load_truck_scenario("limited_dep"),
    lcfs_price_schedule_dict        = load_lcfs_price_scenario("no_change"),
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

mc = run_monte_carlo(cfg, N_RUNS)
println("Simulation complete.\n")

# ── Unpack result matrices (n_runs × n_years) ─────────────────────────────────
# Index reference for run_monte_carlo return tuple:
#   [1]  price_results            — total delivered H2 price ($/kg)
#   [5]  base_price_results       — production + transport (45V already embedded)
#   [6]  infrastructure_results   — station CAPEX, company-funded ($/kg)
#   [8]  infrastructure_om_results— station O&M ($/kg)
#   [9]  lcfs_results             — LCFS credits ($/kg, positive = credit)
#   [17] hri_credits_results      — HRI bonus credits ($/kg)
#   [19] credits_45v_results      — 45V tax credits ($/kg)
price_r  = mc[1]
base_r   = mc[5]
capex_r  = mc[6]
om_r     = mc[8]
lcfs_r   = mc[9]
hri_r    = mc[17]
v45_r    = mc[19]

transport_cost = cfg.utilization_transport_cost   # constant $/kg

years = collect(cfg.start_year:cfg.end_year)
yr_f  = Float64.(years)
n_y   = length(years)

# ── Summary statistics ────────────────────────────────────────────────────────
colmean(m) = [mean(m[:, k]) for k in 1:size(m, 2)]
colq(m, p) = [quantile(m[:, k], p) for k in 1:size(m, 2)]

v45_mean   = colmean(v45_r)
# Gross production cost: add back 45V so it can be shown as a separate credit layer
# Identity: gross_prod + transport + O&M + CAPEX - LCFS - HRI - 45V = total price
prod_mean  = colmean(base_r) .- transport_cost .+ v45_mean
trans_mean = fill(transport_cost, n_y)
om_mean    = colmean(om_r)
capex_mean = colmean(capex_r)
lcfs_mean  = colmean(lcfs_r)   # positive = credit (reduces price)
hri_mean   = colmean(hri_r)    # positive = credit
price_mean = colmean(price_r)
price_p25  = colq(price_r, 0.25)
price_p75  = colq(price_r, 0.75)

# ── Print summary ─────────────────────────────────────────────────────────────
println("H2 PRICE BREAKDOWN SUMMARY (means)")
println("="^72)
@printf("%-6s  %8s  %8s  %8s  %8s  %8s  %8s  %8s\n",
        "Year", "Prod", "Trans", "O&M", "CAPEX", "LCFS", "HRI", "Total")
println("-"^72)
for (i, yr) in enumerate(years)
    @printf("%-6d  %8.3f  %8.3f  %8.3f  %8.3f  %8.3f  %8.3f  %8.3f\n",
            yr,
            prod_mean[i], trans_mean[i], om_mean[i], capex_mean[i],
            lcfs_mean[i], hri_mean[i],
            price_mean[i])
end
println("="^72)

# ── Stacking boundaries ───────────────────────────────────────────────────────
# Positive (cost) layers — stacked bottom to top
z  = zeros(n_y)
y1 = prod_mean                  # top of production layer
y2 = y1 .+ trans_mean           # top of transport layer
y3 = y2 .+ om_mean              # top of O&M layer
y4 = y3 .+ capex_mean           # top of CAPEX layer  (= total gross cost)

# Negative (credit) layers — stacked top to bottom
yn1 = .-lcfs_mean               # bottom of LCFS credit band  (= -LCFS)
yn2 = yn1 .- hri_mean           # bottom of HRI band
yn3 = yn2 .- v45_mean           # bottom of 45V band

# ── Colours (delivered-price cost/credit component palette; see FIGURE_STYLE.md) ─
c_prod  = colorant"#4472C4"   # blue        — production
c_trans = colorant"#9C27B0"   # purple      — transport
c_om    = colorant"#CD853F"   # tan         — O&M
c_capex = colorant"#FF9800"   # orange      — station CAPEX
c_lcfs  = colorant"#4CAF50"   # green       — LCFS credits
c_hri   = colorant"#1B5E20"   # dark green  — HRI credits
c_45v   = colorant"#FFD700"   # gold        — 45V credit
c_total = colorant"#111111"   # near-black  — total price

# ── Figure layout ─────────────────────────────────────────────────────────────
fig = Figure(size = (W_ONEHALF, 105 * MM_TO_PT))

ax = Axis(fig[1, 1];
    xlabel         = "Year",
    ylabel         = L"H$_2$ fuel cost (USD kg$^{-1}$)",
    xticks         = [2026, 2030, 2035, 2040, 2045], xticklabelrotation = π/4,
    xautolimitmargin = (0f0, 0f0),   # ← removes left/right padding
)

# ── Cost layers (positive stack, bottom → top) ────────────────────────────────
band!(ax, yr_f, z,  y1; color = (c_prod,  0.40))
band!(ax, yr_f, y1, y2; color = (c_trans, 0.40))
band!(ax, yr_f, y2, y3; color = (c_om,    0.40))
band!(ax, yr_f, y3, y4; color = (c_capex, 0.40))

# Thin boundary lines between cost layers
lines!(ax, yr_f, y1; color = (c_prod,  0.70), linewidth = 0.7, linestyle = :dash)
lines!(ax, yr_f, y2; color = (c_trans, 0.70), linewidth = 0.7, linestyle = :dash)
lines!(ax, yr_f, y3; color = (c_om,    0.70), linewidth = 0.7, linestyle = :dash)
lines!(ax, yr_f, y4; color = (c_capex, 0.70), linewidth = 0.7, linestyle = :dash)

# ── Credit layers (negative stack, zero → bottom) ────────────────────────────
band!(ax, yr_f, yn1, z;   color = (c_lcfs, 0.40))   # 0 down to –LCFS
band!(ax, yr_f, yn2, yn1; color = (c_hri,  0.40))   # –LCFS down to –LCFS–HRI
band!(ax, yr_f, yn3, yn2; color = (c_45v,  0.40))   # further down for 45V

lines!(ax, yr_f, yn1; color = (c_lcfs, 0.70), linewidth = 0.7, linestyle = :dash)
lines!(ax, yr_f, yn2; color = (c_hri,  0.70), linewidth = 0.7, linestyle = :dash)
lines!(ax, yr_f, yn3; color = (c_45v,  0.70), linewidth = 0.7, linestyle = :dash)

# Zero reference line
hlines!(ax, [0.0]; color = (:black, 0.35), linewidth = 0.6)

# ── Total price: P25–P75 uncertainty band + mean line ────────────────────────
band!(ax, yr_f, price_p25, price_p75; color = (c_total, 0.12))
lines!(ax, yr_f, price_mean; color = c_total, linewidth = 2.0)

# ── Legend ────────────────────────────────────────────────────────────────────
Legend(
    fig[1, 1],          # ← same cell as your axis
    [
        PolyElement(color = (c_prod,  0.5)),
        PolyElement(color = (c_trans, 0.5)),
        PolyElement(color = (c_om,    0.5)),
        PolyElement(color = (c_capex, 0.5)),
        PolyElement(color = (c_lcfs,  0.5)),
        PolyElement(color = (c_hri,   0.5)),
        PolyElement(color = (c_45v,   0.5)),
        LineElement( color = c_total, linewidth = 2.0),
        PolyElement(color = (c_total, 0.12)),
    ],
    [
        "H₂ production (gross)",
        "Transportation",
        "Station O&M",
        "Station CAPEX",
        "LCFS credits",
        "HRI credits",
        "45V tax credit",
        "Total price (mean)",
        "P25–P75",
    ];
    orientation  = :horizontal,
    tellwidth    = false,
    tellheight   = false,
    halign       = :right,
    valign       = :top,
    framevisible = true,
    labelsize    = FS_LEGEND,
    rowgap       = 3.5,
    nbanks       = 3,
    margin       = (4, 8, 4, 8),  # (left, right, bottom, top) in points
)

# ── Save ───────────────────────────────────────────────────────────────────────
save_pub("fig_price_breakdown", fig)
