# =============================================================================
# FIGURE: ANNUAL POLICY EXPENDITURE — SOLAR-H2 FLEET, BY FUNDER
#         vs. the externality it buys (avoided SCC + air-quality health)
# =============================================================================
# The TCO premium in the companion figures is reported net of policy support.
# That support is a *transfer*, not a resource cost, so it does not belong in a
# social cost-benefit comparison. Here we make it explicit:
#
#   Row 1 — total annual support, stacked by instrument and grouped by who pays:
#       Taxpayer-funded   : 45V production tax credit ($/kg) + HVIP voucher ($/truck)
#       Fuel-market-funded: LCFS credit ($/kg) + HRI bonus credit ($/kg)
#   The other side of the LCFS instrument — the LCFS deficit the displaced diesel
#   would otherwise pay — is reported to the console (cumulative avoided), not
#   plotted, to keep the support stack clean.
#   Row 2 — that same total support against the avoided societal damage
#       (SCC + air-quality health) the fleet delivers, on a SHARED scale. The gap
#       is the answer to "should support be netted out of the benefit?": the area
#       you would subtract is small next to the damages avoided.
#
# All flows come from the SAME solar Monte Carlo used by fig_cost_premium_3row /
# fig_societal_benefit (same seed/configs), so the numbers reconcile.
#
# Run from project root:
#   julia --project figures/fig_policy_expenditure.jl
# =============================================================================

using CairoMakie
using Statistics
using JSON
using Random
using Printf

cd(joinpath(@__DIR__, ".."))
include("../hydrogen truck deployment.jl")
include("config_defaults.jl")
include("health_cost.jl")   # avoided air-quality damage — COBRA by default
include("pub_theme.jl")

# 1000 is the publication setting. Override only to exercise the drawing code —
# the medians are meaningless at a handful of runs:
#   PE_N_RUNS=5 julia --project=figures figures/fig_policy_expenditure.jl
const N_RUNS  = parse(Int, get(ENV, "PE_N_RUNS", "1000"))
const SEED    = 42
const OUT_DIR = joinpath(@__DIR__, "out")
mkpath(OUT_DIR)

# ── TCO / fleet parameters ───────────────────────────────────────────────────
tco_raw = JSON.parsefile(joinpath(@__DIR__, "..", "config", "tco_config.json"))
ht      = tco_raw["hydrogen_truck"]
miles_per_kg_h2 = Float64(ht["miles_per_kg_h2"])

# HVIP truck voucher schedule ($/truck, steps down to 0 by 2035)
_hvip_sched = Dict{Int,Float64}(d["year"] => Float64(d["subsidy_usd"]) for d in ht["subsidy_schedule"])
hvip_per_truck(year::Int) = _hvip_sched[maximum(filter(y -> y <= year, keys(_hvip_sched)))]

# ── Diesel-side lifecycle CO₂ + air-quality health (mirror fig_societal_benefit) ─
const DIESEL_MPG        = TCO_DIESEL_MPG   # miles/gallon, from config/tco_config.json
const DIESEL_MJ_PER_GAL = 128.45
const CA_BLEND_CI       = 0.66 * 43.74 + 0.06 * 38.49 + (1.0 - 0.66 - 0.06) * 90.0
const CO2_G_PER_MILE    = CA_BLEND_CI * DIESEL_MJ_PER_GAL / DIESEL_MPG

# Avoided air-quality damage per displaced diesel mile comes from health_cost.jl
# (EPA COBRA by default; HEALTH_BASIS=envcost for the older per-mile factors).
# It is the smaller half of the row-2 benefit line — SCC dominates — so the basis
# barely moves the support-vs-damages gap this figure is built to show.

# SCC — piecewise-linear EPA SC-CO₂ points (2020 USD → 2024 USD); same as 3row
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

# ── Solar MC config (identical to fig_cost_premium_3row / fig_societal_benefit) ─
lcfs_prices = load_lcfs_price_scenario("no_change")

# Displaced-diesel LCFS position ($/mile). The CA diesel blend (CI = CA_BLEND_CI)
# sits BELOW the LCFS benchmark early (so it earns credits) and ABOVE it later (so
# it pays deficits), as the benchmark CI declines per lcfs_config.json. Sign
# convention: + = credit (diesel earns), − = deficit (diesel pays).
const _benchmark_ci = Dict{Int,Float64}(
    Int(d["year"]) => Float64(d["ci"])
    for d in JSON.parsefile(joinpath(@__DIR__, "..", "config", "lcfs_config.json"))["diesel_ci_schedule"]["schedule"]
)
function diesel_lcfs_per_mile(year::Int)
    ci_std = get(_benchmark_ci, year, _benchmark_ci[maximum(filter(y -> y <= year, keys(_benchmark_ci)))])
    lcfs_p = get(lcfs_prices, year, lcfs_prices[maximum(filter(y -> y <= year, keys(lcfs_prices)))])
    return (ci_std - CA_BLEND_CI) / 1e6 * DIESEL_MJ_PER_GAL * lcfs_p / DIESEL_MPG   # $/mile
end
solar_cfg(sched) = build_config(;
    h2_pathway_id = "current_mix", use_utilization_pricing = true, utilization_transport_cost = 1.0,
    use_lcfs = true, enable_45v = true, bus_demand_scenario = "growing", end_year = END_YEAR,
    use_truck_deployment_schedule = true, truck_deployment_schedule = sched,
    lcfs_price_schedule_dict = lcfs_prices,
    expansion_pathway_id = "electrolysis", electrolysis_pricing_enabled = true,
    electrolysis_electricity_source = "solar", electrolyzer_capex_per_kw = 3000.0,
    solar_capex_per_kw = 1600.0, solar_capacity_factor = 0.25, solar_lifetime = 25,
    electrolyzer_learning_rate = 0.233, electrolyzer_stack_fraction = 0.60, bop_learning_rate = 0.04,
    solar_panel_learning_rate = 0.267, solar_panel_fraction = 0.80, solar_bop_learning_rate = 0.04,
)

# Run MC → per-year median expenditure (M USD/yr) by instrument + societal benefit.
# MC tuple indices:  [3] trucks  [9] LCFS $/kg  [17] HRI $/kg  [19] 45V $/kg
function flows(scenario_key)
    cfg = solar_cfg(load_truck_scenario(scenario_key))
    Random.seed!(SEED)
    raw = run_monte_carlo(cfg, N_RUNS)
    truck_r = Float64.(raw[3]);  lcfs_r = raw[9];  hri_r = raw[17];  v45_r = raw[19]
    yrs = collect(cfg.start_year : cfg.end_year);  m = length(yrs)

    kg_yr = truck_r .* Float64(H2_PER_TRUCK_PER_DAY) .* Float64(OPERATING_DAYS_PER_YEAR)   # kg H₂/yr
    # New trucks bought each year (col 1 = initial fleet purchase); HVIP is one-off per truck.
    new_tr = similar(truck_r);  new_tr[:, 1] = truck_r[:, 1]
    for k in 2:m;  new_tr[:, k] = max.(0.0, truck_r[:, k] .- truck_r[:, k-1]);  end

    # Per-instrument expenditure (M USD/yr), per run → median
    med(M) = [median(M[:, k]) for k in 1:m]
    lcfs = med(lcfs_r .* kg_yr ./ 1e6)
    hri  = med(hri_r  .* kg_yr ./ 1e6)
    v45  = med(v45_r  .* kg_yr ./ 1e6)
    hvip = [median(hvip_per_truck(yrs[k]) .* new_tr[:, k]) for k in 1:m] ./ 1e6

    # Societal benefit (avoided SCC + health), median flows
    mi_yr   = kg_yr .* miles_per_kg_h2
    md_mi   = med(mi_yr)
    co2_t   = md_mi .* CO2_G_PER_MILE ./ 1e6
    health  = [health_cost_per_mile(y) * md_mi[k] / 1e6 for (k, y) in enumerate(yrs)]
    benefit(rate) = health .+ [scc_per_ton(yrs[k]; rate = rate) * co2_t[k] / 1e6 for k in 1:m]

    # Displaced-diesel LCFS position (M USD/yr) over the fleet's served miles:
    # + = diesel earns credits (early), − = diesel pays deficits (late).
    dlcfs = [diesel_lcfs_per_mile(yrs[k]) * md_mi[k] / 1e6 for k in 1:m]

    return (years = yrs, lcfs = lcfs, hri = hri, v45 = v45, hvip = hvip,
            support = lcfs .+ hri .+ v45 .+ hvip, dlcfs = dlcfs,
            ben_md = benefit("2.0%"), ben_lo = benefit("2.5%"), ben_hi = benefit("1.5%"))
end

# Drift check lives in fig_societal_cost_benefit.jl, which runs the same fleet.
health_basis_banner()

println("Running solar MC for limited_dep and high_dep ($(N_RUNS) runs each)…")
F_demo = flows("limited_dep");  println("  limited_dep done")
F_high = flows("high_dep");   println("  high_dep done")
panels = [("Limited deployment", F_demo), ("High deployment", F_high)]

# ── Colours: instrument = colour; funder = colour FAMILY (gold=taxpayer, green=
#    fuel-market). LCFS/HRI/45V reuse the fig_price_breakdown instrument colours. ─
c_lcfs = colorant"#4CAF50"   # green       — LCFS credit      (fuel-market)
c_hri  = colorant"#1B5E20"   # dark green  — HRI bonus credit (fuel-market)
c_45v  = colorant"#FFD700"   # gold        — 45V tax credit   (taxpayer)
c_hvip = colorant"#B8860B"   # dark gold   — HVIP voucher     (taxpayer)
c_ben  = C_TOTAL_BENEFIT     # indigo      — avoided SCC + health
c_supp = colorant"#666666"   # grey        — total support (row 2)

# Stack order (bottom→top): fuel-market group, then taxpayer group.
stack = [(:lcfs, c_lcfs), (:hri, c_hri), (:v45, c_45v), (:hvip, c_hvip)]

# Shared y-limits: row 1 = support composition; row 2 = support vs benefit.
r1_max = maximum(maximum(F.support) for (_, F) in panels) * 1.08
r2_max = maximum(maximum(F.ben_hi)  for (_, F) in panels) * 1.05

# Displaced-diesel LCFS deficit avoided by switching to H₂ (reported, not plotted).
# dlcfs is the counterfactual diesel's LCFS position (+ credit / − deficit); the
# deficit avoided is the negative part, the credit foregone is the positive part.
println("\nDisplaced-diesel LCFS position avoided by switching to H₂, 2026–$(END_YEAR) (undiscounted M USD):")
for (name, F) in panels
    idx = findall(y -> 2026 <= y <= END_YEAR, F.years)
    deficit_avoided = -sum(min.(0.0, F.dlcfs[idx]))   # deficits the diesel would have paid
    credit_foregone =  sum(max.(0.0, F.dlcfs[idx]))   # credits the diesel would have earned
    @printf("  %-16s deficit avoided = %7.1f | credit foregone = %6.1f | net avoided = %7.1f\n",
            name, deficit_avoided, credit_foregone, deficit_avoided - credit_foregone)
end

# ── Support composition by year (what row 1 stacks), for the caption ─────────
# "Excluding LCFS" is reported two ways because HRI is itself an LCFS-program
# bonus credit: ex-LCFS drops only the base credit, ex-LCFS-programme drops the
# HRI bonus too and so leaves the taxpayer-funded instruments alone.
println("\nPOLICY SUPPORT BY INSTRUMENT (median of $N_RUNS runs, M USD/yr)")
for (name, F) in panels
    println("\n  $name")
    println("  " * "-"^86)
    @printf("  %-6s %9s %9s %9s %9s %9s %11s %13s\n",
            "Year", "LCFS", "HRI", "45V", "HVIP", "TOTAL", "ex-LCFS", "ex-LCFS-prog")
    println("  " * "-"^86)
    idx = findall(y -> 2026 <= y <= END_YEAR, F.years)
    for k in idx
        @printf("  %-6d %9.1f %9.1f %9.1f %9.1f %9.1f %11.1f %13.1f\n",
                F.years[k], F.lcfs[k], F.hri[k], F.v45[k], F.hvip[k], F.support[k],
                F.support[k] - F.lcfs[k], F.v45[k] + F.hvip[k])
    end
    s(v) = sum(v[idx])
    println("  " * "-"^86)
    @printf("  %-6s %9.1f %9.1f %9.1f %9.1f %9.1f %11.1f %13.1f\n",
            "TOTAL", s(F.lcfs), s(F.hri), s(F.v45), s(F.hvip), s(F.support),
            s(F.support) - s(F.lcfs), s(F.v45) + s(F.hvip))
end

# ── Cumulative support 2026–END_YEAR, by funder ──────────────────────────────
# Undiscounted first (the sum of the annual bars above), then present-valued to
# 2026 at the same 2%/yr the societal figures use for their cumulative panels,
# so support and avoided damages are discounted on a common basis.
const PV_RATE = 0.02
println("\nTOTAL POLICY SUPPORT 2026–$(END_YEAR)  (M USD, median of $N_RUNS runs)")
println("-"^92)
@printf("  %-17s %10s %10s %12s %12s %12s %12s\n",
        "Deployment", "LCFS", "HRI", "45V", "HVIP", "TOTAL", "PV@2%")
println("-"^92)
for (name, F) in panels
    idx = findall(y -> 2026 <= y <= END_YEAR, F.years)
    s(v) = sum(v[idx])
    pv   = sum(F.support[k] / (1 + PV_RATE)^(F.years[k] - 2026) for k in idx)
    @printf("  %-17s %10.1f %10.1f %12.1f %12.1f %12.1f %12.1f\n",
            name, s(F.lcfs), s(F.hri), s(F.v45), s(F.hvip), s(F.support), pv)
end
println("-"^92)
for (name, F) in panels
    idx = findall(y -> 2026 <= y <= END_YEAR, F.years)
    s(v) = sum(v[idx])
    tax, fuel = s(F.v45) + s(F.hvip), s(F.lcfs) + s(F.hri)
    tot = tax + fuel
    @printf("  %-17s taxpayer (45V+HVIP) %7.1f (%4.1f%%) | fuel-market (LCFS+HRI) %7.1f (%4.1f%%)\n",
            name, tax, 100tax / tot, fuel, 100fuel / tot)
end

# ── Legends ──────────────────────────────────────────────────────────────────
fuel_elems  = [PolyElement(color = (c_lcfs, 0.85)), PolyElement(color = (c_hri, 0.85))]
fuel_labels = ["LCFS credit", "HRI bonus credit"]
tax_elems   = [PolyElement(color = (c_45v, 0.85)), PolyElement(color = (c_hvip, 0.85))]
tax_labels  = ["45V tax credit", "HVIP truck voucher"]

# ── Figure — 2 rows × 2 deployment columns ───────────────────────────────────
fig = Figure(size = (W_DOUBLE, 150 * MM_TO_PT))
L = ['a', 'b', 'c', 'd']

# Row 1 — stacked policy support by funder (own scale).
for (j, (name, F)) in enumerate(panels)
    ax = Axis(fig[1, j];
        title = "($(L[j]))  $name — policy support by funder", titlesize = 8,
        ylabel = j == 1 ? L"Policy support (M USD yr$^{-1}$)" : "",
        xticks = year_ticks(2026, END_YEAR), xticklabelsvisible = false,
        yticklabelsvisible = j == 1,
        limits = ((2026, END_YEAR), (0, r1_max)))
    y = F.years;  base = zeros(length(y))
    for (sym, col) in stack
        top = base .+ getfield(F, sym)
        band!(ax, y, base, top; color = (col, 0.85))
        lines!(ax, y, top; color = (col, 0.9), linewidth = 0.5)
        base = top
    end
    lines!(ax, y, base; color = (:black, 0.55), linewidth = 1.0)
    j == 1 && axislegend(ax, [fuel_elems, tax_elems], [fuel_labels, tax_labels],
        ["Fuel-market-funded", "Taxpayer-funded"];
        position = :lt, rowgap = 0, labelsize = 6, titlesize = 6.5,
        framevisible = true, titlegap = 1, groupgap = 5, patchsize = (10, 7))
end

# Row 2 — total support vs avoided societal damage (shared scale).
for (j, (name, F)) in enumerate(panels)
    ax = Axis(fig[2, j];
        title = "($(L[j+2]))  $name — support vs. avoided damage", titlesize = 8,
        xlabel = "Year", ylabel = j == 1 ? L"M USD yr$^{-1}$" : "",
        xticklabelrotation = π/4, xticks = year_ticks(2026, END_YEAR),
        yticklabelsvisible = j == 1,
        limits = ((2026, END_YEAR), (0, r2_max)))
    y = F.years
    band!(ax, y, zeros(length(y)), F.support; color = (c_supp, 0.45))
    lines!(ax, y, F.support; color = c_supp, linewidth = 1.3, label = "Total policy support")
    band!(ax, y, F.ben_lo, F.ben_hi; color = (c_ben, 0.12))
    lines!(ax, y, F.ben_md; color = c_ben, linewidth = 1.8, label = "Avoided SCC + health (2%)")
    j == 1 && axislegend(ax; position = :lt, rowgap = 1, framevisible = true, labelsize = 6.5)
end

rowgap!(fig.layout, 6); colgap!(fig.layout, 6)
resize_to_layout!(fig)
save_pub("fig_policy_expenditure", fig)
