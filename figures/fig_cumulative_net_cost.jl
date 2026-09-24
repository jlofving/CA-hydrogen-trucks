# =============================================================================
# FIGURE: CUMULATIVE NET SOCIETAL COST OVER TIME  (M USD)
# =============================================================================
# Timeline visualisation of the "cumulative net societal cost" table from
# fig_cost_premium_envcost_signed.jl. For each year t the running integral
#   ∫_2026^t (TCO premium − total societal benefit) dt'
# is plotted, so each curve's value at 2045 equals the corresponding table cell.
#
#   Panels      : deployment   (limited_dep, high_dep)
#   Colour      : SCC discount rate (1.5 / 2.0 / 2.5%)  — green→amber→red
#   Line style  : diesel price       ($4.80 solid, $5.80 dashed), central = PV 2%
#   Shaded band : appraisal PV-rate range (PV 0%–3%)
#
# Solar electrolysis pathway, diesel side LCFS-adjusted (anchored 2026), to match
# fig_cost_premium_envcost_signed.jl exactly (same seed, configs, machinery).
#
# Run from project root:
#   julia --project figures/fig_cumulative_net_cost.jl
# =============================================================================

using CairoMakie
using Statistics
using JSON
using Random
using LinearAlgebra
using Printf

cd(joinpath(@__DIR__, ".."))
include("../hydrogen truck deployment.jl")
include("config_defaults.jl")

const N_RUNS  = 1000
const SEED    = 42
include(joinpath(@__DIR__, "fig_routing.jl"))   # figure → manuscript/appendix/old routing
const OUT_DIR = joinpath(@__DIR__, "out")
mkpath(OUT_DIR)

# ─────────────────────────────────────────────────────────────────────────────
# TCO parameters (mirrors fig_cost_premium_envcost_signed.jl)
# ─────────────────────────────────────────────────────────────────────────────
tco_raw = JSON.parsefile(joinpath(@__DIR__, "..", "config", "tco_config.json"))
diesel  = tco_raw["diesel_tco_usd_per_mile"];  ht = tco_raw["hydrogen_truck"];  ident = tco_raw["identical_cost_categories"]

d_capital = Float64(diesel["truck_lease_or_purchase"])
d_rm      = Float64(diesel["repair_and_maintenance"])
d_tires   = Float64(diesel["tires"])
d_common  = Float64(ident["driver_and_other"])

truck_purchase_cost = Float64(ht["purchase_cost_usd"])
truck_platform_cost = Float64(ht["platform_cost_usd"])
truck_fuelcell_cost = truck_purchase_cost - truck_platform_cost
miles_per_kg_h2     = Float64(ht["miles_per_kg_h2"])
rm_mult             = Float64(ht["repair_and_maintenance_multiplier"])
h2_tires_val        = Float64(ht["tires_multiplier"]) * d_tires
learning_rate       = Float64(ht["learning"]["learning_rate"])
ref_year_truck      = Int(ht["learning"]["reference_year"])

_sub_sched_prem = Dict{Int,Float64}(d["year"] => Float64(d["subsidy_usd"]) for d in ht["subsidy_schedule"])
truck_subsidy_yr_prem(year::Int) = _sub_sched_prem[maximum(filter(y -> y <= year, keys(_sub_sched_prem)))]

let
    obs = ht["learning"]["observed_fleet_stock"]
    ys  = Float64.([Int(d["year"]) for d in obs]);  vs = Float64.([Int(d["trucks"]) for d in obs])
    ts  = ys .- 2019.0
    θ   = hcat(ts.^3, ts.^2, ts, ones(length(ts))) \ vs
    global _prem_a, _prem_b, _prem_c, _prem_d = θ[1], θ[2], θ[3], θ[4]
end
fleet_stock_prem(yr::Int) = max(1.0, _prem_a*(yr-2019)^3 + _prem_b*(yr-2019)^2 + _prem_c*(yr-2019) + _prem_d)
function truck_net_cost_prem(year::Int)
    α  = log(1 / (1 - learning_rate)) / log(2)
    fc = truck_fuelcell_cost * (fleet_stock_prem(year) / fleet_stock_prem(ref_year_truck))^(-α)
    return truck_platform_cost + fc - truck_subsidy_yr_prem(year)
end
h2_rm_prem(yr::Int) = d_rm * (rm_mult + (1.0 - rm_mult) * clamp((yr - 2026) / 9.0, 0.0, 1.0))
function h2_capital_prem(year::Int)
    ann = calculate_annuity_factor(7, DISCOUNT_RATE)
    return truck_net_cost_prem(year) * ann /
           (Float64(H2_PER_TRUCK_PER_DAY) * Float64(OPERATING_DAYS_PER_YEAR) * miles_per_kg_h2)
end
h2_tco_prem(lcoh::Float64, yr::Int) =
    lcoh / miles_per_kg_h2 + h2_capital_prem(yr) + h2_rm_prem(yr) + h2_tires_val + d_common

# ─────────────────────────────────────────────────────────────────────────────
# Diesel + emission constants
# ─────────────────────────────────────────────────────────────────────────────
const DIESEL_MPG  = TCO_DIESEL_MPG   # miles/gallon, from config/tco_config.json
const DIESEL_P_LO = TCO_DIESEL_P_LO  # $/gal, from config/tco_config.json
const DIESEL_P_HI = TCO_DIESEL_P_HI  # $/gal, from config/tco_config.json
const DIESEL_MJ_PER_GAL = 128.45
const CA_BLEND_CI    = 0.66 * 43.74 + 0.06 * 38.49 + (1.0 - 0.66 - 0.06) * 90.0
const CO2_G_PER_MILE = CA_BLEND_CI * DIESEL_MJ_PER_GAL / DIESEL_MPG
const MJ_H2_PER_KG   = 120.0
const SOLAR_H2_CI    = 0.0

# ── Diesel LCFS adjustment (anchored at 2026; mirrors fig_tco_comparison.jl) ──
const _benchmark_ci = Dict{Int,Float64}(
    Int(d["year"]) => Float64(d["ci"])
    for d in JSON.parsefile(joinpath(@__DIR__, "..", "config", "lcfs_config.json"))["diesel_ci_schedule"]["schedule"]
)
function diesel_lcfs_per_mile(year::Int, ps::Dict{Int,Float64})
    ci_std  = get(_benchmark_ci, year, _benchmark_ci[maximum(filter(y -> y <= year, keys(_benchmark_ci)))])
    lcfs_p  = get(ps, year, ps[maximum(filter(y -> y <= year, keys(ps)))])
    credits = (ci_std - CA_BLEND_CI) / 1e6 * DIESEL_MJ_PER_GAL
    return credits * lcfs_p / DIESEL_MPG
end
const _lcfs_prices_diesel = load_lcfs_price_scenario("no_change")
const _lcfs_2026_per_mile = diesel_lcfs_per_mile(2026, _lcfs_prices_diesel)

diesel_tco_mile(p_gal, yr::Int) =
    p_gal / DIESEL_MPG + (_lcfs_2026_per_mile - diesel_lcfs_per_mile(yr, _lcfs_prices_diesel)) +
    d_capital + d_rm + d_tires + d_common

const ENV_COST_DELTA_USD_PER_MILE = Dict{Int,Float64}(
    2015=>0.0480283801,2016=>0.0470866472,2017=>0.0461633796,2018=>0.0452582153,
    2019=>0.0443707993,2020=>0.0435007836,2021=>0.0426478271,2022=>0.0418115952,
    2023=>0.0409917600,2024=>0.0401880000,2025=>0.0394000000,2026=>0.0386274510,
    2027=>0.0378700500,2028=>0.0371275000,2029=>0.0363995098,2030=>0.0356857939,
    2031=>0.0349860725,2032=>0.0343000710,2033=>0.0336275206,2034=>0.0329681575,
    2035=>0.0323217230,2036=>0.0316879637,2037=>0.0310666311,2038=>0.0304574815,
    2039=>0.0298602760,2040=>0.0292747804,2041=>0.0287007651,2042=>0.0281380050,
    2043=>0.0275862794,2044=>0.0270453719,2045=>0.0265150705,2046=>0.0259951672,
    2047=>0.0254854580,2048=>0.0249857432,2049=>0.0244958266,2050=>0.0240155163,
)

# ─────────────────────────────────────────────────────────────────────────────
# SCC — piecewise-linear EPA SC-CO₂ points (2020 USD → 2024 USD)
# ─────────────────────────────────────────────────────────────────────────────
const SCC_DEFLATOR_2020_TO_2024 = 1.18
const SCC_YEARS_2020USD  = [2020, 2030, 2040, 2050, 2060, 2070, 2080]
const SCC_POINTS_2020USD = Dict(
    "1.5%" => [340.0, 380.0, 430.0, 480.0, 530.0, 570.0, 600.0],
    "2.0%" => [190.0, 230.0, 270.0, 310.0, 350.0, 380.0, 410.0],
    "2.5%" => [120.0, 140.0, 170.0, 200.0, 230.0, 260.0, 280.0],
)
function scc_per_ton(year; rate = "2.0%")
    xs = SCC_YEARS_2020USD;  ys = SCC_POINTS_2020USD[rate]
    yr = clamp(year, xs[1], xs[end])
    i  = min(searchsortedlast(xs, yr), length(xs) - 1)
    t  = (yr - xs[i]) / (xs[i+1] - xs[i])
    return (ys[i] + t * (ys[i+1] - ys[i])) * SCC_DEFLATOR_2020_TO_2024
end

# ─────────────────────────────────────────────────────────────────────────────
# Run solar MC for a deployment scenario → median premium / health / CO₂ flows
# (identical to solar_flows in fig_cost_premium_envcost_signed.jl)
# ─────────────────────────────────────────────────────────────────────────────
solar_cfg(sched) = build_config(;
    h2_pathway_id = "current_mix", use_utilization_pricing = true, utilization_transport_cost = 1.0,
    use_lcfs = true, enable_45v = true, bus_demand_scenario = "growing", end_year = END_YEAR,
    use_truck_deployment_schedule = true, truck_deployment_schedule = sched,
    lcfs_price_schedule_dict = _lcfs_prices_diesel,
    expansion_pathway_id = "electrolysis", electrolysis_pricing_enabled = true,
    electrolysis_electricity_source = "solar", electrolyzer_capex_per_kw = 3000.0,
    solar_capex_per_kw = 1600.0, solar_capacity_factor = 0.25, solar_lifetime = 25,
    electrolyzer_learning_rate = 0.233, electrolyzer_stack_fraction = 0.60, bop_learning_rate = 0.04,
    solar_panel_learning_rate = 0.267, solar_panel_fraction = 0.80, solar_bop_learning_rate = 0.04,
)
function solar_flows(scenario_key)
    cfg = solar_cfg(load_truck_scenario(scenario_key))
    Random.seed!(SEED)
    raw = run_monte_carlo(cfg, N_RUNS)
    price_r = raw[1];  truck_r = Float64.(raw[3])
    yrs = collect(cfg.start_year : cfg.end_year);  m = length(yrs)
    h2_yr = truck_r .* Float64(H2_PER_TRUCK_PER_DAY) .* Float64(OPERATING_DAYS_PER_YEAR)
    mi_yr = h2_yr .* miles_per_kg_h2
    tlo = [(h2_tco_prem(price_r[r,k], yrs[k]) - diesel_tco_mile(DIESEL_P_LO, yrs[k])) * mi_yr[r,k] / 1e6 for r in 1:N_RUNS, k in 1:m]
    thi = [(h2_tco_prem(price_r[r,k], yrs[k]) - diesel_tco_mile(DIESEL_P_HI, yrs[k])) * mi_yr[r,k] / 1e6 for r in 1:N_RUNS, k in 1:m]
    med_miles = [median(mi_yr[:, k]) for k in 1:m]
    return (years  = yrs,
            tcp_lo = [median(tlo[:, k]) for k in 1:m],
            tcp_hi = [median(thi[:, k]) for k in 1:m],
            co2_t  = med_miles .* CO2_G_PER_MILE ./ 1e6,
            health = [ENV_COST_DELTA_USD_PER_MILE[y] * med_miles[k] / 1e6 for (k, y) in enumerate(yrs)])
end

# Net societal cost stream (M USD/yr) = TCO premium − air-quality health − avoided SCC.
net_stream(F, dfield, rate) = getfield(F, dfield) .- F.health .-
    [scc_per_ton(F.years[k]; rate = rate) * F.co2_t[k] / 1e6 for k in eachindex(F.years)]
# Appraisal discounting of a real flow to base year t0.
discount_series(years, vals, r, t0) = [v / (1 + r)^(y - t0) for (y, v) in zip(years, vals)]
# Running cumulative trapezoidal integral; cum[1] = 0.
function cumulative_integral(years, vals)
    cum = zeros(length(vals));  s = 0.0
    for i in 2:length(vals)
        s += 0.5 * (vals[i-1] + vals[i]) * (years[i] - years[i-1])
        cum[i] = s
    end
    return cum
end

# ─────────────────────────────────────────────────────────────────────────────
# Compute cumulative trajectories
# ─────────────────────────────────────────────────────────────────────────────
const INTEG_Y0, INTEG_Y1, DBASE = 2026, 2045, 2026
println("Running solar MC for limited_dep and high_dep ($(N_RUNS) runs each)…")
F_demo = solar_flows("limited_dep");  println("  limited_dep done")
F_high = solar_flows("high_dep");   println("  high_dep done")
deploy_panels = [("Limited deployment", F_demo), ("High deployment", F_high)]

SCC_RATES = ["1.5%", "2.0%", "2.5%"]
PV_RATES  = (0.0, 0.02, 0.03)
diesels   = [(DIESEL_P_LO, :tcp_lo, :solid), (DIESEL_P_HI, :tcp_hi, :dash)]

# Cumulative cost (M USD) for one (F, diesel field, SCC rate, PV rate) over 2026–2045.
function cum_trajectory(F, dfield, scc_rate, pv)
    idx = findall(y -> INTEG_Y0 <= y <= INTEG_Y1, F.years)
    yrs = F.years[idx]
    net = net_stream(F, dfield, scc_rate)[idx]
    dv  = discount_series(yrs, net, pv, DBASE)
    return Float64.(yrs), cumulative_integral(yrs, dv)
end

# ─────────────────────────────────────────────────────────────────────────────
# Figure
# ─────────────────────────────────────────────────────────────────────────────
MM_TO_PT = 1 / 0.352778
scc_colors = Dict("1.5%" => colorant"#2E7D32",   # green  — high SCC, most benefit
                  "2.0%" => colorant"#F9A825",   # amber
                  "2.5%" => colorant"#C62828")   # red    — low SCC, most net cost

# Shared y-limits across panels for honest cross-deployment comparison.
all_vals = Float64[]
for (_, F) in deploy_panels, (_, dfield, _) in diesels, rate in SCC_RATES, pv in PV_RATES
    _, c = cum_trajectory(F, dfield, rate, pv);  append!(all_vals, c)
end
ypad = 0.05 * (maximum(all_vals) - minimum(all_vals))
ylim = (minimum(all_vals) - ypad, maximum(all_vals) + ypad)

# Shared manual legend elements (scale-independent): SCC colour, diesel style, PV band.
leg_elems = [
    LineElement(color = scc_colors["1.5%"], linewidth = 2),
    LineElement(color = scc_colors["2.0%"], linewidth = 2),
    LineElement(color = scc_colors["2.5%"], linewidth = 2),
    LineElement(color = :gray30, linewidth = 2, linestyle = :solid),
    LineElement(color = :gray30, linewidth = 2, linestyle = :dash),
    PolyElement(color = (:gray30, 0.18)),
]
leg_labels = [
    "SCC 1.5%", "SCC 2.0%", "SCC 2.5%",
    @sprintf("Diesel \$%.2f/gal", DIESEL_P_LO), @sprintf("Diesel \$%.2f/gal", DIESEL_P_HI),
    "PV 0–3% range\n(line = PV 2%)",
]

# Build the 2-panel figure under an arbitrary y-scale and save to a suffixed name.
function make_fig(; yscale = identity, suffix = "", yticks = Makie.automatic, note = "")
    fig = Figure(size = (200 * MM_TO_PT, 100 * MM_TO_PT), fontsize = 8)
    labels = ['a', 'b']
    for (j, (name, F)) in enumerate(deploy_panels)
        ax = Axis(fig[1, j];
            title              = "($(labels[j]))  $name",
            titlesize          = 8, titlefont = :bold,
            xlabel             = "Year",
            ylabel             = j == 1 ? "Cumulative net societal cost (M USD)" : "",
            xlabelsize         = 7, ylabelsize = 7, xticklabelsize = 7, yticklabelsize = 7,
            xticklabelrotation = π/4, xticks = [2026, 2030, 2035, 2040, 2045],
            yscale             = yscale, yticks = yticks,
            yticklabelsvisible = j == 1,
            limits             = ((INTEG_Y0, INTEG_Y1), ylim))
        hlines!(ax, 0; color = (:black, 0.45), linewidth = 0.8, linestyle = :dot)
        for rate in SCC_RATES, (dp, dfield, ls) in diesels
            c = scc_colors[rate]
            yrs, c0 = cum_trajectory(F, dfield, rate, PV_RATES[1])   # PV 0%
            _,   c2 = cum_trajectory(F, dfield, rate, PV_RATES[2])   # PV 2% (central)
            _,   c3 = cum_trajectory(F, dfield, rate, PV_RATES[3])   # PV 3%
            band!(ax, yrs, min.(c0, c3), max.(c0, c3); color = (c, 0.12))
            lines!(ax, yrs, c2; color = c, linewidth = ls == :solid ? 1.6 : 1.2, linestyle = ls)
        end
        isempty(note) || j != 1 || text!(ax, INTEG_Y0 + 0.4, ylim[1];
            text = note, align = (:left, :bottom), fontsize = 5.5, color = :gray40)
    end
    Legend(fig[1, 3], leg_elems, leg_labels;
           framevisible = true, labelsize = 6.5, rowgap = 3, patchsize = (16, 8),
           titlefont = :bold, tellheight = false, tellwidth = true)
    colgap!(fig.layout, 6)
    resize_to_layout!(fig)
    out_pdf = fig_path(OUT_DIR, "fig_cumulative_net_cost$(suffix).pdf")
    out_png = fig_path(OUT_DIR, "fig_cumulative_net_cost$(suffix).png")
    save(out_pdf, fig; pt_per_unit = 1);  println("Saved → $out_pdf")
    save(out_png, fig; px_per_unit = 300/72);  println("Saved → $out_png")
end

# Linear y-scale (original).
make_fig()

# Symmetric-log y-scale: log compression of large magnitudes with a linear window
# through zero (|value| < SYMLOG_THRESH M USD), so both signs and the zero-
# crossings remain visible. Ticks placed at decade-ish symlog positions.
const SYMLOG_THRESH = 5.0
symlog_ticks = ([-5000.0, -1000.0, -100.0, -10.0, 0.0, 10.0, 100.0, 1000.0, 5000.0],
                ["−5000", "−1000", "−100", "−10", "0", "10", "100", "1000", "5000"])
make_fig(yscale = Makie.Symlog10(SYMLOG_THRESH), suffix = "_log", yticks = symlog_ticks,
         note = "symlog: linear within ±$(Int(SYMLOG_THRESH)) M USD")

# ─────────────────────────────────────────────────────────────────────────────
# Console check — 2045 cumulative should equal the summary-table cells
# ─────────────────────────────────────────────────────────────────────────────
println("\n2045 cumulative net cost (M USD) — should match summary table")
println("="^64)
for (name, F) in deploy_panels, (dp, dfield, _) in diesels
    @printf("[ %-15s | diesel \$%.2f/gal ]\n", name, dp)
    @printf("    %-7s %11s %11s %11s\n", "SCC\\PV", "PV 0%", "PV 2%", "PV 3%")
    for rate in SCC_RATES
        vals = [cum_trajectory(F, dfield, rate, pv)[2][end] for pv in PV_RATES]
        @printf("    %-7s %+11.1f %+11.1f %+11.1f\n", rate, vals[1], vals[2], vals[3])
    end
end
