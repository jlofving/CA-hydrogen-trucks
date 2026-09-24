# =============================================================================
# FIGURE: CO2 ABATEMENT COST — SOLAR ELECTROLYSIS H2 vs. DIESEL  ($/tCO2e)
#         Both deployment scenarios overlaid on one panel; SCC shown as a band.
# =============================================================================
# Net marginal abatement cost (MAC) per year for solar-electrolysis H₂ trucks:
#   net MAC = (resource-cost premium − air-quality co-benefit) / tCO₂e   [$/tCO₂e]
# The premium is UNSUBSIDIZED (per-kg LCFS+HRI+45V credits added back, HVIP voucher
# removed): subsidies are transfers, not resource costs, and carbon is counted once
# via the SCC. The diesel reference is held flat in real terms (LCFS-on-diesel
# excluded for the same reason). Stripped-out support: fig_policy_expenditure.jl.
# for limited_dep and high_dep deployments × $4.80/$5.80 diesel references,
# overlaid so the deployment difference is directly visible (colour = scenario,
# linestyle = diesel). Gross MAC (premium only, ≈ net + ~30 $/tCO₂) is in the
# console table, omitted from the plot for legibility.
#
# The social cost of carbon is a BAND spanning EPA's 1.5%–2.5% near-term Ramsey
# discount rates (2% centre). Where a MAC line sits below the band, abating
# carbon this way costs less than the social value of the carbon avoided.
#
# Run from project root:
#   julia --project figures/fig_abatement_cost.jl
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
include("health_cost.jl")   # avoided air-quality damage — COBRA by default
include("pub_theme.jl")

# 1000 is the publication setting. Override only to exercise the drawing code —
# the medians are meaningless at a handful of runs:
#   MAC_N_RUNS=5 julia --project=figures figures/fig_abatement_cost.jl
const N_RUNS  = parse(Int, get(ENV, "MAC_N_RUNS", "1000"))
const SEED    = 42
const OUT_DIR = joinpath(@__DIR__, "out")
mkpath(OUT_DIR)

# ─────────────────────────────────────────────────────────────────────────────
# TCO parameters (mirrors fig_cost_premium_envcost.jl)
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

# Unsubsidized (resource-cost) variant for the social abatement-cost comparison:
# gross truck capital (no HVIP voucher) + an LCOH with the per-kg policy credits
# (LCFS + HRI + 45V) added back. Subsidies are transfers, not resource costs, and
# the carbon externality is counted once — here against the explicit SCC. The
# stripped-out support is shown by funder in fig_policy_expenditure.jl.
function h2_capital_gross(year::Int)
    α  = log(1 / (1 - learning_rate)) / log(2)
    fc = truck_fuelcell_cost * (fleet_stock_prem(year) / fleet_stock_prem(ref_year_truck))^(-α)
    truck_gross = truck_platform_cost + fc            # no HVIP subsidy subtracted
    ann = calculate_annuity_factor(7, DISCOUNT_RATE)
    return truck_gross * ann /
           (Float64(H2_PER_TRUCK_PER_DAY) * Float64(OPERATING_DAYS_PER_YEAR) * miles_per_kg_h2)
end
h2_tco_unsub(lcoh_gross::Float64, yr::Int) =
    lcoh_gross / miles_per_kg_h2 + h2_capital_gross(yr) + h2_rm_prem(yr) + h2_tires_val + d_common

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

# Literature benchmark for the abatement cost of hydrogen trucking, shown as a
# subtle grey reference line. Reported as 778 ± 383; only the central estimate is
# drawn, to keep the panel's only shaded range the SCC band.
const LIT_MAC_MID    = 778.0    # 2024 USD / tCO₂e
const LIT_MAC_SPREAD = 383.0    # ± reported uncertainty (kept for reference; see below)
const LIT_MAC_LABEL  = "Shafiee & Schrag (2024)"

# Second literature benchmark, year-dependent rather than flat: the declining
# abatement-cost trajectory reported by Martin et al. (2023), read off at five-
# year intervals and reported in 2020 €/tCO₂.
const MARTIN_LABEL   = "Martin et al. (2023)"
const MARTIN_COLOR   = colorant"#595959"   # a shade darker than C_GREY, to separate the two references
const MARTIN_EUR_PER_T = [(2020, 1022.0), (2025, 600.0), (2030, 265.0),
                          (2035,  138.0), (2040,  85.0), (2045,  55.0)]
# 2020 € → 2024 USD: 2020 average EUR/USD (1.142) × US GDP implicit price
# deflator 2020→2024 (1.19). Set to 1.0 to plot the reported euro figures
# unconverted.
const MARTIN_EUR_TO_USD24 = 1.36
# The reported points step down by roughly a constant factor every five years,
# so interpolate in log space; linear interpolation leaves visible kinks at the
# knots on a curve that nearly halves per step.
function martin_mac(year)
    xs = first.(MARTIN_EUR_PER_T);  ys = last.(MARTIN_EUR_PER_T)
    yr = clamp(year, xs[1], xs[end])
    i  = min(searchsortedlast(xs, yr), length(xs) - 1)
    t  = (yr - xs[i]) / (xs[i+1] - xs[i])
    return exp(log(ys[i]) + t * (log(ys[i+1]) - log(ys[i]))) * MARTIN_EUR_TO_USD24
end

# Diesel reference held FLAT in real terms for the social comparison. The
# LCFS-on-diesel adjustment (diesel cost rising as the benchmark CI declines) is a
# transfer / carbon-price proxy; carbon is already valued explicitly via the SCC,
# so including it would double-count carbon on the diesel side. Excluded here to
# stay symmetric with stripping LCFS from the H₂ side.
diesel_tco_mile(p_gal, yr::Int) =
    p_gal / DIESEL_MPG + d_capital + d_rm + d_tires + d_common

# Avoided air-quality damage per displaced diesel mile comes from health_cost.jl
# (EPA COBRA by default; HEALTH_BASIS=envcost for the older per-mile factors).
# It is what separates the gross and net MAC curves, so the basis moves the net
# curve directly: on COBRA the net MAC sits above where the older factors put it,
# by ~31 USD/tCO₂e in 2026 narrowing to ~19 by 2045 (the older factors decline
# with year, the COBRA rate does not). The gross curves are unaffected.

# ─────────────────────────────────────────────────────────────────────────────
# SCC — piecewise-linear interpolation of EPA SC-CO₂ points (2020 USD → 2024 USD)
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
# Run solar MC for a deployment scenario → MAC quantities
# ─────────────────────────────────────────────────────────────────────────────
lcfs_prices = load_lcfs_price_scenario("no_change")
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
function solar_mac(scenario_key)
    cfg = solar_cfg(load_truck_scenario(scenario_key))
    Random.seed!(SEED)
    raw = run_monte_carlo(cfg, N_RUNS)
    price_r = raw[1];  truck_r = Float64.(raw[3])
    # Per-kg policy credits to strip out: [9] LCFS, [17] HRI, [19] 45V (price is net of all three).
    price_unsub = price_r .+ raw[9] .+ raw[17] .+ raw[19]
    years = collect(cfg.start_year : cfg.end_year);  n = length(years)
    h2_yr = truck_r .* Float64(H2_PER_TRUCK_PER_DAY) .* Float64(OPERATING_DAYS_PER_YEAR)
    mi_yr = h2_yr .* miles_per_kg_h2
    tcp_lo = [(h2_tco_unsub(price_unsub[r,k], years[k]) - diesel_tco_mile(DIESEL_P_LO, years[k])) * mi_yr[r,k] / 1e6 for r in 1:N_RUNS, k in 1:n]
    tcp_hi = [(h2_tco_unsub(price_unsub[r,k], years[k]) - diesel_tco_mile(DIESEL_P_HI, years[k])) * mi_yr[r,k] / 1e6 for r in 1:N_RUNS, k in 1:n]
    med_miles  = [median(mi_yr[:, k]) for k in 1:n]
    tcp_med_lo = [median(tcp_lo[:, k]) for k in 1:n]
    tcp_med_hi = [median(tcp_hi[:, k]) for k in 1:n]
    co2_t      = med_miles .* (CO2_G_PER_MILE - SOLAR_H2_CI * MJ_H2_PER_KG / miles_per_kg_h2) ./ 1e6
    health     = [health_cost_per_mile(yr) * med_miles[k] / 1e6 for (k, yr) in enumerate(years)]

    # Per-run net MAC, for the Monte Carlo spread reported next to the figure.
    # The fleet cancels out of the ratio — cost premium and abated CO2e are both
    # proportional to the same per-run mileage — and under the scheduled
    # deployment used here the fleet is identical in every run anyway. So the
    # quantiles below sit on the plotted median rather than near it; the text
    # dump prints the per-run median against the plotted value so that stays
    # checked rather than assumed.
    co2_r    = mi_yr .* (CO2_G_PER_MILE - SOLAR_H2_CI * MJ_H2_PER_KG / miles_per_kg_h2) ./ 1e6
    hlth_r   = [health_cost_per_mile(years[k]) * mi_yr[r, k] / 1e6 for r in 1:N_RUNS, k in 1:n]
    net_r_lo = (tcp_lo .- hlth_r) .* 1e6 ./ co2_r
    net_r_hi = (tcp_hi .- hlth_r) .* 1e6 ./ co2_r
    qt(M, p) = [quantile(M[:, k], p) for k in 1:n]

    return (years = years,
            gross_lo = tcp_med_lo .* 1e6 ./ co2_t,
            gross_hi = tcp_med_hi .* 1e6 ./ co2_t,
            net_lo   = (tcp_med_lo .- health) .* 1e6 ./ co2_t,
            net_hi   = (tcp_med_hi .- health) .* 1e6 ./ co2_t,
            lo_p25 = qt(net_r_lo, 0.25), lo_med = qt(net_r_lo, 0.50), lo_p75 = qt(net_r_lo, 0.75),
            hi_p25 = qt(net_r_hi, 0.25), hi_med = qt(net_r_hi, 0.50), hi_p75 = qt(net_r_hi, 0.75))
end

# Drift check lives in fig_societal_cost_benefit.jl, which runs the same fleet.
health_basis_banner()

println("Running solar MC for limited_dep and high_dep ($(N_RUNS) runs each)…")
F_demo = solar_mac("limited_dep");  println("  limited_dep done")
F_high = solar_mac("high_dep");   println("  high_dep done")

for (title, F) in (("Limited deployment", F_demo), ("High deployment", F_high))
    println("\nCO₂ ABATEMENT COST — solar, $title (2024 USD / tCO₂e), median")
    println("="^78)
    @printf("%-6s %11s %11s %11s %11s %9s %9s\n",
            "Year","gross4.80","gross5.80","net4.80","net5.80","SCC1.5%","SCC2.5%")
    println("-"^78)
    for (k, y) in enumerate(F.years)
        (y < 2026 || y > 2045) && continue
        @printf("%-6d %11.1f %11.1f %11.1f %11.1f %9.1f %9.1f\n",
                y, F.gross_lo[k], F.gross_hi[k], F.net_lo[k], F.net_hi[k],
                scc_per_ton(y; rate="1.5%"), scc_per_ton(y; rate="2.5%"))
    end
end
println()

println("LITERATURE BENCHMARKS (2024 USD / tCO₂e)")
println("-"^78)
@printf("  %-24s %s\n", LIT_MAC_LABEL, "flat $(LIT_MAC_MID) (± $(LIT_MAC_SPREAD), band not drawn)")
@printf("  %-24s %s\n", MARTIN_LABEL,
        join([@sprintf("%d: %.0f", y, martin_mac(y)) for y in 2026:5:2045], "   "))
@printf("  %-24s €→2024 USD × %.2f, log-linear between reported points\n", "", MARTIN_EUR_TO_USD24)
println()

# ─────────────────────────────────────────────────────────────────────────────
# Monte Carlo spread (P25–P75) — written next to the figure
# ─────────────────────────────────────────────────────────────────────────────
# The plotted curves are the net MAC at each year's median. The interquartile
# range below comes from the per-run net MAC: the same calculation applied to
# each Monte Carlo draw separately, then quantiled. Only the delivered hydrogen
# price varies across runs — the truck fleet follows a fixed schedule, so the
# abated tonnage is identical in every run and drops out of the ratio. That is
# why the per-run median reproduces the plotted value exactly; the check column
# would expose it if a future change made the fleet stochastic.
let path = joinpath(OUT_DIR, "fig_abatement_cost_spread.txt")
    open(path, "w") do io
        println(io, "="^96)
        println(io, " MONTE CARLO SPREAD — fig_abatement_cost (manuscript fig $(FIG_NUMBER["fig_abatement_cost"]))")
        println(io, " Net CO2 abatement cost of solar-electrolysis hydrogen trucking, 2024 USD/tCO2e")
        println(io, "="^96)
        @printf(io, " %d Monte Carlo runs, seed %d. Net = unsubsidised resource-cost premium\n", N_RUNS, SEED)
        println(io, " minus avoided air-quality damage, per tonne of CO2e abated.")
        println(io, " Health basis: $HEALTH_BASIS.  Diesel bracket: \$$(DIESEL_P_LO) and \$$(DIESEL_P_HI)/gal.")
        println(io)
        println(io, " 'plotted' is the curve in the figure (net MAC evaluated at the median).")
        println(io, " 'run-med' is the median of the per-run net MAC. The two agree to within")
        println(io, " rounding because the truck schedule is deterministic, so the abated")
        println(io, " tonnage cancels from the ratio; the column is kept as a standing check.")
        println(io, "="^96)

        for (title, F) in (("LIMITED DEPLOYMENT", F_demo), ("HIGH DEPLOYMENT", F_high))
            for (dp, plotted, p25, rmed, p75) in
                    ((DIESEL_P_LO, F.net_lo, F.lo_p25, F.lo_med, F.lo_p75),
                     (DIESEL_P_HI, F.net_hi, F.hi_p25, F.hi_med, F.hi_p75))
                println(io)
                println(io, "$title — diesel \$$(dp)/gal")
                println(io, "-"^96)
                @printf(io, " %-6s %10s %10s %10s %10s %10s %9s\n",
                        "Year", "P25", "plotted", "P75", "IQR", "run-med", "IQR/med")
                println(io, " " * "-"^94)
                for (k, y) in enumerate(F.years)
                    (y < 2026 || y > END_YEAR) && continue
                    iqr = p75[k] - p25[k]
                    @printf(io, " %-6d %10.1f %10.1f %10.1f %10.1f %10.1f %8.1f %%\n",
                            y, p25[k], plotted[k], p75[k], iqr, rmed[k],
                            plotted[k] != 0 ? 100 * iqr / abs(plotted[k]) : NaN)
                end
            end
        end
        println(io)
        println(io, "="^96)
        maxdev = maximum(abs.(vcat(F_demo.lo_med .- F_demo.net_lo, F_demo.hi_med .- F_demo.net_hi,
                                   F_high.lo_med .- F_high.net_lo, F_high.hi_med .- F_high.net_hi)))
        @printf(io, " CHECK  max |run-med − plotted| over all years and both scenarios: %.4f USD/tCO2e\n", maxdev)
        println(io, maxdev < 1e-6 ?
            " Exact, as expected for a deterministic fleet." :
            " *** NON-ZERO: the fleet has become stochastic and the plotted curve is now a" *
            " ratio of medians rather than a median ratio. Revisit before citing the IQR. ***")
        println(io, "="^96)
    end
    println("Monte Carlo spread → $path")
end

# ─────────────────────────────────────────────────────────────────────────────
# Figure — single panel, both deployments overlaid (net MAC), SCC band
# ─────────────────────────────────────────────────────────────────────────────
c_demo = C_DEP_LIM   # violet — limited deployment
c_high = C_DEP_HIGH   # teal   — high deployment
c_scc  = C_SCC        # pink   — SCC band / benchmark

YEARS  = F_demo.years
scc_lo = scc_per_ton.(YEARS; rate = "2.5%")     # lowest band edge
scc_hi = scc_per_ton.(YEARS; rate = "1.5%")     # highest band edge
scc_md = scc_per_ton.(YEARS; rate = "2.0%")
ymax   = maximum(vcat(F_demo.net_lo, F_high.net_lo)) * 1.05

fig = Figure(size = (W_ONEHALF, 100 * MM_TO_PT))
ax = Axis(fig[1, 1];
    title              = "Solar electrolysis — net CO₂ abatement cost vs. SCC",
    titlesize          = 8,
    xlabel             = "Year",
    ylabel             = L"Net abatement cost (2024 USD tCO$_2$e$^{-1}$)",
    xticklabelrotation = π/4,
    xticks             = [2026, 2030, 2035, 2040, 2045],
    limits             = ((2026, END_YEAR), (0, ymax)),
)
# Literature benchmark: central estimate only, as a thin grey dashed reference
# line drawn first so every model series sits on top of it.
hlines!(ax, LIT_MAC_MID; color = (C_GREY, 0.75), linewidth = 0.7, linestyle = :dash)
# To show the ±LIT_MAC_SPREAD range as a hatched band instead, uncomment:
#   NB Rect2{Float64}, not Rect2f — at year-magnitude x the Float32 UV mapping
#   loses precision and the pattern silently renders as nothing.
# poly!(ax, CairoMakie.Rect2{Float64}(2026.0, LIT_MAC_MID - LIT_MAC_SPREAD,
#                                     Float64(END_YEAR) - 2026.0, 2 * LIT_MAC_SPREAD);
#       color = CairoMakie.Makie.LinePattern(
#                   direction = CairoMakie.Vec2f(1, 1), width = 1.5, tilesize = (8, 8),
#                   linecolor = (C_GREY, 0.55), backgroundcolor = :transparent),
#       strokewidth = 0)
text!(ax, END_YEAR - 0.4, LIT_MAC_MID + 25; text = LIT_MAC_LABEL,
      align = (:right, :bottom), fontsize = FS_ANNOT, color = C_GREY)

# Martin et al. — same grey reference treatment a shade darker, dash-dot to
# separate it from the flat benchmark above. Labelled at the left end, sitting on
# top of its own curve: the curve is steep here, so it drops away under the text
# rather than running through it, and the label clears the flat benchmark above.
lines!(ax, YEARS, martin_mac.(YEARS);
       color = (MARTIN_COLOR, 0.85), linewidth = 0.8, linestyle = :dashdot)
text!(ax, 2027.5, martin_mac(2027.5) + 12; text = MARTIN_LABEL,
      align = (:left, :bottom), fontsize = FS_ANNOT, color = MARTIN_COLOR)

band!(ax, YEARS, scc_lo, scc_hi; color = (c_scc, 0.13))
lines!(ax, YEARS, scc_md; color = c_scc, linewidth = 2, label = "SCC, 2% (band 1.5–2.5%)")
hlines!(ax, 0; color = (:black, 0.4), linewidth = 0.8, linestyle = :dot)
for (F, c, name) in ((F_demo, c_demo, "Limited deployment"), (F_high, c_high, "High deployment"))
    lines!(ax, YEARS, F.net_lo; color = c, linewidth = 1.6,
           label = "$name (diesel \$$(DIESEL_P_LO)/gal)")
    lines!(ax, YEARS, F.net_hi; color = c, linewidth = 1.1, linestyle = :dash,
           label = "$name (diesel \$$(DIESEL_P_HI)/gal)")
end
axislegend(ax; position = :rt, rowgap = 1, framevisible = true)

resize_to_layout!(fig)
save_pub("fig_abatement_cost", fig)
