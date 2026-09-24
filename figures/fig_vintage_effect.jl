# =============================================================================
# FIGURE 9: WHAT DROPPING CAPEX VINTAGING WOULD SAVE
#   (a) change in the dispensed H₂ price      (% of the vintaged price)
#   (b) change in H₂ truck TCO                (% of the vintaged TCO)
# =============================================================================
# The baseline model VINTAGES production capital: every electrolysis facility
# keeps the electrolyser and solar CAPEX of the year it opened, so the
# capacity-weighted CAPEX behind the price in year t is a mix of vintages that
# lags the learning curve. This script re-runs the model with that turned off
# (`capex_vintage = false`), which values ALL standing capacity at the CURRENT
# year's technology cost, and plots what that no-vintage accounting saves:
#
#     Δ = current-year costing − vintaged     (NEGATIVE = no-vintage is cheaper)
#
# so a curve at −6% reads "costing all capacity at this year's technology cost
# comes out 6% below the vintaged model". The vintaged model is the baseline,
# hence also the denominator.
#
# Both figures plot Δ RELATIVE to that vintaged cost for the same pathway in the
# same year — a percentage, not $/kg or $/mile, since the two denominators fall
# steeply over the horizon and an absolute difference alone understates how much
# of the late-horizon cost is at stake. The absolute $/kg and $/mile magnitudes
# are tabulated alongside the percentages in
# out/diagnostics/fig_vintage_effect_values.txt.
#
# The two instances are run from the same seed. Nothing in the deployment logic
# reads the hydrogen price — the investment trigger keys off demand versus
# capacity, and the truck fleet follows a fixed schedule — so both instances
# build exactly the same facilities in the same years, and the difference is a
# clean PAIRED difference per Monte Carlo run: same fleet, same plants, same
# stations, only the CAPEX vintage year differs. Lines are the median of the
# per-run differences (not the difference of the medians, which would mix runs).
# Their P25–P75 spread across runs — deployment timing varies, so the vintage mix
# does too — is computed but reported only in the values text file; on the panels
# it was ≈ ±0.3 pp even on the widest series, too narrow to be worth shading.
#
# Scope of the effect: BOTH learning-curve CAPEX items are re-valued — the
# electrolyser and, on the solar pathway, the PV plant that powers it (the two
# `capex_year(facility)` call sites in calculate_blended_production_cost). Solar
# PV is the larger of the two: switching each learning curve off in turn splits
# the solar pathway's 2045 saving of $0.46/kg into ≈$0.27 from PV and ≈$0.19 from
# the electrolyser, PV leading because its curve is steeper (45% vs 37% decline
# from a 2030 vintage to 2045 technology) and a larger share of its CAPEX is
# fast-learning (80% panel vs 60% stack).
#
# Nothing else in the cost stack is vintaged, so only the two electrolysis
# pathways can differ at all: SMR facility CAPEX is a fixed cost, and station
# CAPEX comes from the capacity-cost curve with no year term, so their
# differences are identically zero — the SMR line is plotted as a check on
# exactly that. The H₂ truck's own fuel-cell learning already uses
# the purchase year in the TCO, so it is not vintaged either: the vintage effect
# reaches the TCO only through the fuel term, which the script asserts.
#
# Manuscript figure 9, sitting behind the fig 8 TCO comparison it qualifies. The
# number comes from the position of "fig_vintage_effect" in FIG_MANUSCRIPT
# (fig_routing.jl) — move it there, not here, to renumber.
#
# Run from project root:
#   julia --project=figures figures/fig_vintage_effect.jl
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
include("pub_theme.jl")

const N_RUNS  = 1000
const SEED    = 42
const OUT_DIR = joinpath(@__DIR__, "out")

# ─────────────────────────────────────────────────────────────────────────────
# TCO parameters — same definition as fig_tco_comparison.jl (manuscript fig 8)
# ─────────────────────────────────────────────────────────────────────────────
# Only the fuel term can move with the CAPEX vintage, but the full TCO is built
# here anyway so the difference can be reported against the TCO level it sits
# on, and so the fuel-only claim above is testable rather than asserted.
tco_raw = JSON.parsefile(joinpath(@__DIR__, "..", "config", "tco_config.json"))
diesel  = tco_raw["diesel_tco_usd_per_mile"];  ht = tco_raw["hydrogen_truck"]
ident   = tco_raw["identical_cost_categories"]

d_rm     = Float64(diesel["repair_and_maintenance"])
d_tires  = Float64(diesel["tires"])
d_common = Float64(ident["driver_and_other"])

truck_purchase_cost = Float64(ht["purchase_cost_usd"])
truck_platform_cost = Float64(ht["platform_cost_usd"])
truck_fuelcell_cost = truck_purchase_cost - truck_platform_cost
miles_per_kg_h2     = Float64(ht["miles_per_kg_h2"])
rm_mult             = Float64(ht["repair_and_maintenance_multiplier"])
h2_tires_val        = Float64(ht["tires_multiplier"]) * d_tires
learning_rate       = Float64(ht["learning"]["learning_rate"])
ref_year_truck      = Int(ht["learning"]["reference_year"])

_subsidy_sched = Dict{Int,Float64}(d["year"] => Float64(d["subsidy_usd"])
                                   for d in ht["subsidy_schedule"])
truck_subsidy_yr(year::Int) = _subsidy_sched[maximum(filter(y -> y <= year, keys(_subsidy_sched)))]

# Global H₂ fleet stock (cubic OLS fit to observed stock) drives Wright's Law
let raw = ht["learning"]["observed_fleet_stock"]
    ts = Float64.([Int(d["year"]) for d in raw]) .- 2019.0
    ys = Float64.([Int(d["trucks"]) for d in raw])
    θ  = hcat(ts.^3, ts.^2, ts, ones(length(ts))) \ ys
    global _lr_a, _lr_b, _lr_c, _lr_d = θ[1], θ[2], θ[3], θ[4]
end
fleet_stock(year::Int) = max(1.0, _lr_a*(year-2019)^3 + _lr_b*(year-2019)^2 +
                                  _lr_c*(year-2019) + _lr_d)

# Net truck price in `year`: learning on the gross fuel-cell cost, then HVIP
function truck_net_cost_yr(year::Int)
    α = log(1 / (1 - learning_rate)) / log(2)
    truck_platform_cost +
        truck_fuelcell_cost * (fleet_stock(year) / fleet_stock(ref_year_truck))^(-α) -
        truck_subsidy_yr(year)
end

annual_miles_per_truck() = H2_PER_TRUCK_PER_DAY * OPERATING_DAYS_PER_YEAR * miles_per_kg_h2
h2_capital_per_mile(year::Int) =
    truck_net_cost_yr(year) * calculate_annuity_factor(7, DISCOUNT_RATE) / annual_miles_per_truck()
h2_rm_per_mile(year::Int) = d_rm * (rm_mult + (1.0 - rm_mult) * clamp((year - 2026) / 9.0, 0.0, 1.0))
h2_tco(lcoh::Float64, year::Int) = lcoh / miles_per_kg_h2 + h2_capital_per_mile(year) +
                                   h2_rm_per_mile(year) + h2_tires_val + d_common

# ─────────────────────────────────────────────────────────────────────────────
# Scenarios — same grid as manuscript fig 8, run twice (vintaged / current-year)
# ─────────────────────────────────────────────────────────────────────────────
pathways = [
    (label = "SMR (current mix)",    short = "SMR",         color = C_SMR,   kind = :smr),
    (label = "Electrolysis — grid",  short = "Grid elec.",  color = C_GRID,  kind = :grid),
    (label = "Electrolysis — solar", short = "Solar elec.", color = C_SOLAR, kind = :solar),
]

lcfs_prices = load_lcfs_price_scenario("no_change")

function make_config(kind, sched, vintage::Bool)
    base = (
        h2_pathway_id                 = "current_mix",
        use_utilization_pricing       = true,
        utilization_transport_cost    = 1.0,
        use_lcfs                      = true,
        enable_45v                    = true,
        bus_demand_scenario           = "growing",
        end_year                      = END_YEAR,
        use_truck_deployment_schedule = true,
        truck_deployment_schedule     = sched,
        lcfs_price_schedule_dict      = lcfs_prices,
        capex_vintage                 = vintage,
    )
    if kind == :smr
        build_config(; base..., expansion_pathway_id = "current_mix",
                       electrolysis_pricing_enabled = false)
    elseif kind == :grid
        build_config(; base..., expansion_pathway_id = "electrolysis",
            electrolysis_pricing_enabled = true, electrolysis_electricity_source = "grid",
            electrolyzer_capex_per_kw = 3000.0, electricity_cost_per_kwh = 0.2,
            electrolyzer_learning_rate = 0.233, electrolyzer_stack_fraction = 0.60,
            bop_learning_rate = 0.04)
    else  # :solar
        build_config(; base..., expansion_pathway_id = "electrolysis",
            electrolysis_pricing_enabled = true, electrolysis_electricity_source = "solar",
            electrolyzer_capex_per_kw = 3000.0, solar_capex_per_kw = 1600.0,
            solar_capacity_factor = 0.25, solar_lifetime = 25,
            electrolyzer_learning_rate = 0.233, electrolyzer_stack_fraction = 0.60,
            bop_learning_rate = 0.04, solar_panel_learning_rate = 0.267,
            solar_panel_fraction = 0.80, solar_bop_learning_rate = 0.04)
    end
end

# Paired run of the two model instances for one (pathway, deployment) cell.
# The seed is reset before each so the two share every stochastic draw.
function paired_run(kind, sched)
    Random.seed!(SEED);  p_vint = run_monte_carlo(make_config(kind, sched, true),  N_RUNS)[1]
    Random.seed!(SEED);  p_flat = run_monte_carlo(make_config(kind, sched, false), N_RUNS)[1]
    years = collect(START_YEAR:END_YEAR)

    t_vint = [h2_tco(p_vint[r, k], years[k]) for r in 1:N_RUNS, k in eachindex(years)]
    t_flat = [h2_tco(p_flat[r, k], years[k]) for r in 1:N_RUNS, k in eachindex(years)]

    # Per-run differences, current-year costing − vintaged (negative = cheaper)
    dl = p_flat .- p_vint        # USD/kg
    dt = t_flat .- t_vint        # USD/mile
    # …and as a share of the vintaged (baseline) cost in the SAME run and year,
    # which is what the two figures plot. Formed per run before any quantile is
    # taken, so the percentage is never a ratio of two different runs' medians.
    dl_pct = 100 .* dl ./ p_vint
    dt_pct = 100 .* dt ./ t_vint

    stat(M, f) = [f(M[:, k]) for k in eachindex(years)]
    q25(x) = quantile(x, 0.25);  q75(x) = quantile(x, 0.75)
    (years = years,
     lcoh_vint = stat(p_vint, median), lcoh_flat = stat(p_flat, median),
     tco_vint  = stat(t_vint,  median), tco_flat = stat(t_flat,  median),
     dl_med = stat(dl, median), dt_med = stat(dt, median),
     dlp_med = stat(dl_pct, median), dlp_p25 = stat(dl_pct, q25), dlp_p75 = stat(dl_pct, q75),
     dtp_med = stat(dt_pct, median), dtp_p25 = stat(dt_pct, q25), dtp_p75 = stat(dt_pct, q75),
     # Largest deviation from ΔTCO = ΔLCOH / miles_per_kg — should be ~0
     fuel_only_resid = maximum(abs.(dt .- dl ./ miles_per_kg_h2)))
end

deployments = [(key = "limited_dep", label = "Limited deployment",  ls = :solid, lw = 1.6),
               (key = "high_dep",  label = "High deployment", ls = :dash,  lw = 1.2)]

println("Running $(length(pathways)) pathways × $(length(deployments)) deployments × 2 " *
        "vintage settings ($(N_RUNS) runs each)…")
R = Dict{Tuple{Symbol,String},Any}()
for d in deployments
    sched = load_truck_scenario(d.key)
    for pw in pathways
        print("  [$(d.key)] $(pw.label)…")
        R[(pw.kind, d.key)] = paired_run(pw.kind, sched)
        r = R[(pw.kind, d.key)]
        @printf(" done (%d saving: %+.3f \$/kg = %+.1f%% of price, %+.4f \$/mile = %+.1f%% of TCO)\n",
                END_YEAR, r.dl_med[end], r.dlp_med[end], r.dt_med[end], r.dtp_med[end])
    end
end
println("All simulations complete.\n")

YEARS = R[(:smr, "limited_dep")].years

# The vintage effect must reach the TCO through the fuel term only — every other
# component is a function of the year, not of when capacity was built.
let resid = maximum(r.fuel_only_resid for r in values(R))
    @printf("Check: max |ΔTCO − ΔLCOH/miles_per_kg| = %.2e USD/mile (fuel-only pass-through)\n", resid)
    resid < 1e-9 || @warn "ΔTCO is not purely the fuel term — a non-fuel component moved with the vintage flag"
end
let mx = maximum(abs, R[(:smr, "limited_dep")].dl_med)
    @printf("Check: SMR pathway max |ΔLCOH| = %.2e USD/kg (no learning-curve CAPEX)\n", mx)
end

# ─────────────────────────────────────────────────────────────────────────────
# Figure — (a) H₂ price, (b) TCO, one shared legend
# ─────────────────────────────────────────────────────────────────────────────
# The pathway/deployment key is identical in both panels, so per FIGURE_STYLE.md
# §4b it is a single figure-level grouped legend in its own layout row rather
# than an axislegend per panel: it cannot overlap the data and is not repeated.
fig = Figure(size = (W_DOUBLE, 85 * MM_TO_PT))

# Axis for one panel. Every series is ≤ 0, so the y-limits run from the deepest
# median down-padded up to a sliver above zero, keeping the zero line visible.
function panel_axis(col, panel_title, ylabel, med)
    ylo = min(-eps(), minimum(minimum(med(R[(pw.kind, d.key)]))
                              for pw in pathways for d in deployments))
    Axis(fig[1, col];
        title              = panel_title,
        xlabel             = "Year",
        ylabel             = ylabel,
        xticks             = year_ticks(2026, END_YEAR),
        xticklabelrotation = π/4,
        limits             = ((2026, END_YEAR), (ylo * 1.06, -ylo * 0.05)),
    )
end

# Median of the per-run differences only. The P25–P75 spread is in the values
# text file rather than on the panel: it is narrow enough (≈ ±0.3 pp on the
# widest series) that plotting it added shading without adding information.
function panel!(ax, med)
    hlines!(ax, 0; color = C_ZERO_LINE, linewidth = 0.8, linestyle = :dot)
    for d in deployments, pw in pathways
        lines!(ax, YEARS, med(R[(pw.kind, d.key)]);
               color = pw.color, linewidth = d.lw, linestyle = d.ls)
    end
end

ax_a = panel_axis(1, "(a)", "H₂ fuel cost change (% of vintaged cost)", r -> r.dlp_med)
panel!(ax_a, r -> r.dlp_med)

ax_b = panel_axis(2, "(b)", "TCO change (% of vintaged TCO)", r -> r.dtp_med)
panel!(ax_b, r -> r.dtp_med)

# Two orthogonal encodings (colour = pathway, line style = deployment) → two
# titled sections instead of every colour × style combination.
leg_groups = Vector{Vector}([
    [LineElement(color = pw.color, linewidth = 1.6) for pw in pathways],
    [LineElement(color = :black, linestyle = d.ls, linewidth = d.lw) for d in deployments],
])
leg_labels = Vector{Vector{String}}([
    [pw.label for pw in pathways],
    [d.label  for d in deployments],
])
Legend(fig[2, 1:2], leg_groups, leg_labels,
    ["H₂ pathway (colour)", "Deployment (line style)"];
    orientation = :horizontal, nbanks = 2, titlefont = :bold,
    tellheight = true, tellwidth = false, colgap = 10, framevisible = true)

colgap!(fig.layout, 12)
rowgap!(fig.layout, 4)
resize_to_layout!(fig)
save_pub("fig_vintage_effect", fig)

# ─────────────────────────────────────────────────────────────────────────────
# Values → text file
# ─────────────────────────────────────────────────────────────────────────────
report_years = filter(y -> y in YEARS, [2030, 2035, 2040, END_YEAR])
out_txt = joinpath(OUT_DIR, "fig_vintage_effect_values.txt")
open(out_txt, "w") do io
    println(io, "="^120)
    println(io, " SAVING FROM DROPPING CAPEX VINTAGING — current-year-cost model minus vintaged model")
    @printf(io, " N_RUNS = %d, SEED = %d | LCFS = no_change | 45V on | median of PAIRED per-run differences\n",
            N_RUNS, SEED)
    println(io, " NEGATIVE Δ = costing all standing capacity at the current year's technology cost")
    println(io, " comes out BELOW the vintaged baseline, in which each facility keeps its opening-year CAPEX.")
    println(io, " ΔTCO is the fuel term only: Δ(USD/mile) = Δ(USD/kg) / " *
                @sprintf("%.3f miles per kg", miles_per_kg_h2))
    println(io, " Δ% is the PLOTTED series: median over runs of Δ / vintaged cost in the same run and")
    println(io, " year — not the ratio of the two medians beside it. P25–P75 is that ratio's spread")
    println(io, " across runs (deployment timing varies, so the vintage mix does); the panels no")
    println(io, " longer shade it.")
    println(io, "="^120)
    println(io)
    for d in deployments
        @printf(io, "  Deployment = %s\n", d.key)
        @printf(io, "    %-14s %5s %10s %10s %9s %8s %16s %10s %10s %8s %16s\n",
                "Pathway", "Year", "LCOH vint", "LCOH now", "Δ\$/kg", "Δ%", "Δ% P25–P75",
                "TCO vint", "Δ\$/mile", "Δ%", "Δ% P25–P75")
        for pw in pathways
            r = R[(pw.kind, d.key)]
            for y in report_years
                k = findfirst(==(y), r.years)
                iqr(lo, hi) = @sprintf("[%.2f, %.2f]", lo[k], hi[k])
                @printf(io, "    %-14s %5d %10.3f %10.3f %9.3f %7.2f%% %16s %10.3f %10.4f %7.2f%% %16s\n",
                        pw.short, y, r.lcoh_vint[k], r.lcoh_flat[k],
                        r.dl_med[k], r.dlp_med[k], iqr(r.dlp_p25, r.dlp_p75),
                        r.tco_vint[k], r.dt_med[k], r.dtp_med[k], iqr(r.dtp_p25, r.dtp_p75))
            end
        end
        println(io)
    end
    println(io, "="^120)
end
println("Saved → $out_txt")
