# =============================================================================
# TABLE: 2030 TCO GAP AND WHAT IT WOULD TAKE TO CLOSE IT
# =============================================================================
# One snapshot year (2030), two deployment scenarios, for the policy-support
# discussion. Columns, all per displaced diesel mile unless stated:
#
#   baseline gap      policy-inclusive H2 TCO − diesel TCO at $4.80/gal
#   HVIP extension    effect of keeping the voucher at its 2026 level ($240k)
#                     instead of the scheduled 2030 step-down to $120k
#   LCFS at 250       effect of the `high_inc` credit-price path, which reaches
#                     $250/tonne in 2030, instead of the flat $65 baseline
#   remaining gap     baseline − both measures
#   required subsidy  remaining gap converted to $/kg H2 (x miles_per_kg_h2)
#   total cost        that subsidy over the fleet's 2030 hydrogen throughput
#
# All costs are ANNUAL 2030 flows (M USD/yr), not cumulative, matching the
# single-year framing of the gap columns.
#
# BOTH SIDES SEE THE CREDIT PRICE. Raising LCFS to $250/tonne does not only
# cheapen hydrogen: the CA diesel blend (CI 56.4) still sits BELOW the benchmark
# in 2030, so it is still earning credits, and a higher price makes diesel
# cheaper too — handing back part of the hydrogen-side gain. The headline
# columns are therefore net of that diesel response. The hydrogen-side-only
# figures are printed underneath for reconciliation, since they are what a
# one-sided hand calculation produces. Same distinction as the daggered rows in
# fig_sensitivity_grid.
#
# The baseline gap already carries the anchored LCFS-on-diesel adjustment (the
# observed pump price embeds 2026 conditions; only the change since 2026 is
# applied), so the diesel reference is consistent across both columns.
#
# Run from project root:
#   julia --project=figures figures/tab_2030_policy_gap.jl
# =============================================================================

using Statistics
using JSON
using Random
using Printf

cd(joinpath(@__DIR__, ".."))
include("../hydrogen truck deployment.jl")
include("config_defaults.jl")

const N_RUNS    = parse(Int, get(ENV, "TAB_N_RUNS", "1000"))
const SEED      = 42
const YEAR      = 2030
const OUT_DIR   = joinpath(@__DIR__, "out")
mkpath(OUT_DIR)

# ── TCO parameters (mirrors fig_cost_premium.jl — POLICY-INCLUSIVE framing) ──
tco_raw = JSON.parsefile(joinpath(@__DIR__, "..", "config", "tco_config.json"))
diesel  = tco_raw["diesel_tco_usd_per_mile"];  ht = tco_raw["hydrogen_truck"]
ident   = tco_raw["identical_cost_categories"]

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

_sub_sched = Dict{Int,Float64}(d["year"] => Float64(d["subsidy_usd"]) for d in ht["subsidy_schedule"])
hvip_per_truck(y::Int) = _sub_sched[maximum(filter(k -> k <= y, keys(_sub_sched)))]

let obs = ht["learning"]["observed_fleet_stock"]
    ys = Float64.([Int(d["year"]) for d in obs]);  vs = Float64.([Int(d["trucks"]) for d in obs])
    ts = ys .- 2019.0
    θ  = hcat(ts.^3, ts.^2, ts, ones(length(ts))) \ vs
    global _a, _b, _c, _d = θ[1], θ[2], θ[3], θ[4]
end
fleet_stock(yr::Int) = max(1.0, _a*(yr-2019)^3 + _b*(yr-2019)^2 + _c*(yr-2019) + _d)

truck_net_cost(yr::Int) = let α = log(1 / (1 - learning_rate)) / log(2)
    truck_platform_cost +
    truck_fuelcell_cost * (fleet_stock(yr) / fleet_stock(ref_year_truck))^(-α) -
    hvip_per_truck(yr)
end
h2_rm(yr::Int) = d_rm * (rm_mult + (1.0 - rm_mult) * clamp((yr - 2026) / 9.0, 0.0, 1.0))

const MILES_PER_TRUCK_YR = Float64(H2_PER_TRUCK_PER_DAY) * Float64(OPERATING_DAYS_PER_YEAR) * miles_per_kg_h2
const ANNUITY_7YR        = calculate_annuity_factor(7, DISCOUNT_RATE)

h2_capital(yr::Int) = truck_net_cost(yr) * ANNUITY_7YR / MILES_PER_TRUCK_YR
h2_tco(lcoh, yr::Int) = lcoh / miles_per_kg_h2 + h2_capital(yr) + h2_rm(yr) + h2_tires_val + d_common

# ── Diesel side ─────────────────────────────────────────────────────────────
const DIESEL_MPG        = TCO_DIESEL_MPG
const DIESEL_P_LO       = TCO_DIESEL_P_LO
const DIESEL_MJ_PER_GAL = 128.45
const CA_BLEND_CI       = 0.66 * 43.74 + 0.06 * 38.49 + (1.0 - 0.66 - 0.06) * 90.0
const _benchmark_ci = Dict{Int,Float64}(
    Int(d["year"]) => Float64(d["ci"])
    for d in JSON.parsefile(joinpath(@__DIR__, "..", "config", "lcfs_config.json"))["diesel_ci_schedule"]["schedule"])

# Diesel's own LCFS position ($/mile, + = diesel earns credits ⇒ cheaper).
function diesel_lcfs_per_mile(yr::Int, p::AbstractDict)
    ci_std = get(_benchmark_ci, yr, _benchmark_ci[maximum(filter(y -> y <= yr, keys(_benchmark_ci)))])
    lcfs_p = get(p, yr, p[maximum(filter(y -> y <= yr, keys(p)))])
    (ci_std - CA_BLEND_CI) / 1e6 * DIESEL_MJ_PER_GAL * lcfs_p / DIESEL_MPG
end

const P_BASE = load_lcfs_price_scenario("no_change")   # flat $65
const P_HIGH = load_lcfs_price_scenario("high_inc")    # $250 from 2030

# Anchored convention (fig 8): the observed pump price already embeds 2026 LCFS
# conditions, so only the CHANGE since 2026 adjusts the later-year diesel cost.
diesel_tco_mile(p_gal, yr::Int, p::AbstractDict) =
    p_gal / DIESEL_MPG + d_capital + d_rm + d_tires + d_common +
    (diesel_lcfs_per_mile(2026, P_BASE) - diesel_lcfs_per_mile(yr, p))

# ── Monte Carlo ─────────────────────────────────────────────────────────────
solar_cfg(sched, p) = build_config(;
    h2_pathway_id = "current_mix", use_utilization_pricing = true, utilization_transport_cost = 1.0,
    use_lcfs = true, enable_45v = true, bus_demand_scenario = "growing", end_year = END_YEAR,
    use_truck_deployment_schedule = true, truck_deployment_schedule = sched,
    lcfs_price_schedule_dict = p,
    expansion_pathway_id = "electrolysis", electrolysis_pricing_enabled = true,
    electrolysis_electricity_source = "solar", electrolyzer_capex_per_kw = 3000.0,
    solar_capex_per_kw = 1600.0, solar_capacity_factor = 0.25, solar_lifetime = 25,
    electrolyzer_learning_rate = 0.233, electrolyzer_stack_fraction = 0.60, bop_learning_rate = 0.04,
    solar_panel_learning_rate = 0.267, solar_panel_fraction = 0.80, solar_bop_learning_rate = 0.04,
)

# Median LCOH, per-instrument credits and fleet size in YEAR, for one price path.
function snapshot(scenario_key, p)
    cfg = solar_cfg(load_truck_scenario(scenario_key), p)
    Random.seed!(SEED)
    raw = run_monte_carlo(cfg, N_RUNS)
    yrs = collect(cfg.start_year : cfg.end_year)
    k   = findfirst(==(YEAR), yrs)
    trucks   = Float64.(raw[3])
    new_tr   = k == 1 ? trucks[:, 1] : max.(0.0, trucks[:, k] .- trucks[:, k-1])
    kg_yr    = trucks[:, k] .* Float64(H2_PER_TRUCK_PER_DAY) .* Float64(OPERATING_DAYS_PER_YEAR)
    (lcoh    = median(raw[1][:, k]),      # $/kg, net of LCFS + HRI + 45V
     lcfs    = median(raw[9][:, k]),      # $/kg
     hri     = median(raw[17][:, k]),     # $/kg
     v45     = median(raw[19][:, k]),     # $/kg
     kg      = median(kg_yr),             # kg/yr
     trucks  = median(trucks[:, k]),
     new_tr  = median(new_tr))
end

scenarios = [("Limited deployment", "limited_dep"), ("High deployment", "high_dep")]
println("Running solar MC at $N_RUNS runs — 2 deployments × 2 LCFS price paths…")
results = map(scenarios) do (name, key)
    b = snapshot(key, P_BASE);  println("  $name  baseline done")
    h = snapshot(key, P_HIGH);  println("  $name  LCFS-250 done")
    (name = name, base = b, high = h)
end

# ── Derived quantities ──────────────────────────────────────────────────────
# HVIP extension: restore the 2026 voucher level, i.e. an extra
# (240k − 120k) per truck, annuitised over 7 years across one truck's annual miles.
const HVIP_EXTRA_PER_TRUCK = hvip_per_truck(2026) - hvip_per_truck(YEAR)
const HVIP_EXT_PER_MILE    = HVIP_EXTRA_PER_TRUCK * ANNUITY_7YR / MILES_PER_TRUCK_YR

const DIESEL_BASE = diesel_tco_mile(DIESEL_P_LO, YEAR, P_BASE)
const DIESEL_HIGH = diesel_tco_mile(DIESEL_P_LO, YEAR, P_HIGH)
const DIESEL_SHIFT = DIESEL_HIGH - DIESEL_BASE   # + = diesel more expensive at $250

rows = map(results) do r
    gap_base  = h2_tco(r.base.lcoh, YEAR) - DIESEL_BASE
    lcfs_h2   = (r.high.lcoh - r.base.lcoh) / miles_per_kg_h2   # − = H₂ cheaper
    lcfs_net  = lcfs_h2 - DIESEL_SHIFT
    rem_h2    = gap_base - HVIP_EXT_PER_MILE + lcfs_h2
    rem_net   = gap_base - HVIP_EXT_PER_MILE + lcfs_net
    (name = r.name, gap = gap_base,
     hvip_pm = -HVIP_EXT_PER_MILE, lcfs_pm = lcfs_h2, lcfs_pm_net = lcfs_net,
     rem = rem_h2, rem_net = rem_net,
     sub_kg = rem_h2 * miles_per_kg_h2, sub_kg_net = rem_net * miles_per_kg_h2,
     cost_sub = rem_h2 * miles_per_kg_h2 * r.base.kg / 1e6,
     cost_sub_net = rem_net * miles_per_kg_h2 * r.base.kg / 1e6,
     cost_hvip = HVIP_EXTRA_PER_TRUCK * r.base.new_tr / 1e6,
     cost_lcfs = (r.high.lcfs - r.base.lcfs) * r.base.kg / 1e6,
     support = (r.base.lcfs + r.base.hri + r.base.v45) * r.base.kg / 1e6 +
               hvip_per_truck(YEAR) * r.base.new_tr / 1e6,
     kg = r.base.kg, trucks = r.base.trucks, new_tr = r.base.new_tr)
end

function report(io)
    println(io, "2030 TCO GAP AND POLICY MEASURES TO CLOSE IT")
    println(io, "Solar electrolysis, median of $N_RUNS runs, vs diesel at \$$(DIESEL_P_LO)/gal.")
    println(io, "Policy-inclusive H₂ TCO (LCFS + HRI + 45V in the price, HVIP in the capital).")
    println(io, "="^112)
    @printf(io, "HVIP extension = +\$%s/truck (restore %s → %s), annuitised 7 yr over %.0f mi/yr = \$%.4f/mi\n",
            Int(HVIP_EXTRA_PER_TRUCK), Int(hvip_per_truck(YEAR)), Int(hvip_per_truck(2026)),
            MILES_PER_TRUCK_YR, HVIP_EXT_PER_MILE)
    @printf(io, "LCFS at 250    = high_inc path (\$250/t in %d) vs flat \$65/t baseline\n", YEAR)
    @printf(io, "                 diesel side %+.4f \$/mi (− = diesel gets cheaper), the same in both\n",
            DIESEL_SHIFT)
    println(io, "                 scenarios; the CA blend (CI 56.4) is still under the benchmark in 2030")
    println(io, "                 and so still earns credits. The columns below are net of that.")
    println(io, "  'cost LCFS' is the extra credit revenue to the H₂ fleet only. The diesel-side")
    println(io, "  effect moves the REFERENCE, not a real flow — that diesel is never driven.")
    println(io, "="^112)

    println(io, "\nMAIN TABLE — both fuels see the credit price")
    println(io, "-"^112)
    @printf(io, "  %-17s %9s %10s %10s %11s %11s %11s %11s %11s\n",
            "Deployment", "gap", "HVIP ext", "LCFS@250", "remaining",
            "subsidy", "cost sub", "cost HVIP", "cost LCFS")
    @printf(io, "  %-17s %9s %10s %10s %11s %11s %11s %11s %11s\n",
            "", "\$/mi", "\$/mi", "\$/mi", "\$/mi", "\$/kg H₂", "MUSD", "MUSD", "MUSD")
    println(io, "-"^112)
    for r in rows
        @printf(io, "  %-17s %9.2f %10.2f %10.2f %11.2f %11.2f %11.1f %11.1f %11.1f\n",
                r.name, r.gap, r.hvip_pm, r.lcfs_pm_net, r.rem_net, r.sub_kg_net,
                r.cost_sub_net, r.cost_hvip, r.cost_lcfs)
    end

    println(io, "\n  Reconciliation — hydrogen side only, ignoring the diesel response:")
    @printf(io, "  %-17s %10s %11s %11s %11s\n",
            "", "LCFS@250", "remaining", "subsidy", "cost sub")
    for r in rows
        @printf(io, "  %-17s %10.2f %11.2f %11.2f %11.1f\n",
                r.name, r.lcfs_pm, r.rem, r.sub_kg, r.cost_sub)
    end
    @printf(io, "  A one-sided calculation understates the remaining gap by \$%.4f/mi in %d,\n",
            -DIESEL_SHIFT, YEAR)
    println(io, "  because it credits hydrogen with a price rise that also cheapens diesel.")

    println(io, "\nBASELINE POLICY SUPPORT IN $YEAR (M USD/yr, current policy)")
    println(io, "-"^112)
    @printf(io, "  %-17s %11s %11s %11s %11s %11s\n",
            "Deployment", "total", "LCFS", "HRI", "45V", "HVIP")
    println(io, "-"^112)
    for (r, res) in zip(rows, results)
        @printf(io, "  %-17s %11.1f %11.1f %11.1f %11.1f %11.1f\n",
                r.name, r.support, res.base.lcfs * r.kg / 1e6, res.base.hri * r.kg / 1e6,
                res.base.v45 * r.kg / 1e6, hvip_per_truck(YEAR) * r.new_tr / 1e6)
    end

    println(io, "\nTOTAL POLICY SUPPORT IN $YEAR IF ALL THREE MEASURES ARE TAKEN (M USD/yr)")
    println(io, "  Baseline is current policy; the three measures are additional to it. The fuel")
    println(io, "  subsidy is the amount that closes the REMAINING gap, so the all-in figure is")
    println(io, "  what it costs to reach diesel parity in $YEAR — not a menu to choose from.")
    println(io, "-"^112)
    @printf(io, "  %-17s %11s %11s %11s %11s %11s\n",
            "Deployment", "baseline", "+HVIP ext", "+LCFS@250", "+fuel sub", "ALL-IN")
    println(io, "-"^112)
    for r in rows
        @printf(io, "  %-17s %11.1f %11.1f %11.1f %11.1f %11.1f\n",
                r.name, r.support, r.cost_hvip, r.cost_lcfs, r.cost_sub_net,
                r.support + r.cost_hvip + r.cost_lcfs + r.cost_sub_net)
    end
    println(io, "-"^112)
    # Funder split. HVIP and 45V are taxpayer-funded, LCFS and HRI come out of the
    # fuel market; the closing subsidy has no funder assigned — that is the choice
    # this table is meant to inform, so it is kept in its own column.
    for (r, res) in zip(rows, results)
        tax  = res.base.v45 * r.kg / 1e6 + hvip_per_truck(YEAR) * r.new_tr / 1e6 + r.cost_hvip
        fuel = (res.base.lcfs + res.base.hri) * r.kg / 1e6 + r.cost_lcfs
        @printf(io, "  %-17s taxpayer %7.1f | fuel-market %7.1f | unassigned (closing sub) %7.1f\n",
                r.name, tax, fuel, r.cost_sub_net)
    end

    println(io, "\nFLEET BASIS IN $YEAR")
    println(io, "-"^112)
    for r in rows
        @printf(io, "  %-17s %.0f trucks in service, %.0f new, %.2f million kg H₂/yr\n",
                r.name, r.trucks, r.new_tr, r.kg / 1e6)
    end
    println(io, "="^112)
end

report(stdout)
let path = joinpath(OUT_DIR, "tab_2030_policy_gap.txt")
    open(report, path, "w")
    println("\nSaved → $path")
end
