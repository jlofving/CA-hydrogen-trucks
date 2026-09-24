# =============================================================================
# VALIDATION CASE: 2026 DELIVERED H2 PRICE, NO NEW STATIONS
# =============================================================================
# A single-year, no-new-infrastructure run, meant to be held against observed
# 2026 conditions rather than used in the paper. Everything that makes the
# forward projection uncertain is stripped out, so the output is one number
# rather than a distribution:
#
#   · end_year = 2026, so only the current assessment year is simulated.
#   · Station list is cut to the entries that are operational TODAY. No
#     probabilistic openings, so Station_5 (FID), Station_11 and the 2027-28
#     pipeline are all excluded.
#   · Endogenous stations cannot contribute in any case: the model opens them
#     at `current_year + rand(2:4)`, i.e. 2028 at the earliest.
#
# What is left — the build trigger, p_invest, commissioning lead times — has
# nothing to act on inside a one-year horizon, so the run is deterministic. The
# script asserts this by re-running under N_CHECK seeds and reporting the spread,
# which should be exactly zero. If it ever stops being zero, something in the
# 2026 path has become stochastic and this case needs revisiting.
#
# TWO CAPEX TREATMENTS are reported, because the right comparator depends on
# what the real-world figure represents:
#
#   as-built    Each existing station carries full annualised new-build CAPEX,
#               exactly as the forward model treats it. This is the internally
#               consistent number — the same basis as manuscript fig 5.
#   CAPEX sunk  The same run with `capex = 0` on the existing stations, i.e.
#               their capital is already spent and not being recovered from the
#               dispensed price. Closer to what an already-built, partly
#               grant-funded station actually charges.
#
# CAVEATS for anyone comparing this to a posted price:
#   · This is a cost build-up, not a retail price. No retailer margin, no taxes.
#   · The modelled per-station CAPEX is EXTRAPOLATED below the cost reference
#     table, which starts at 2000 kg/day; the existing stations are 1200 kg/day.
#     The extrapolated value is printed so it can be checked against actuals.
#   · Station utilisation counts TRUCK demand only. Bus and car demand is in the
#     production-side volume but not in the station denominator, so station
#     fixed costs are spread over the truck throughput alone.
#
# Run from project root:
#   julia --project=figures figures/val_2026_baseline.jl
# =============================================================================

using Statistics
using Random
using Printf

cd(joinpath(@__DIR__, ".."))
include("../hydrogen truck deployment.jl")
include("config_defaults.jl")

const YEAR    = 2026
const SEED    = 42
const N_CHECK = parse(Int, get(ENV, "VAL_N_CHECK", "200"))   # determinism probe
const OUT_DIR = joinpath(@__DIR__, "out")
mkpath(OUT_DIR)

# ── Baseline configuration ────────────────────────────────────────────────────
# Identical to fig_price_breakdown.jl / fig_sensitivity.jl apart from the
# horizon, so the 2026 column is comparable with the manuscript figures.
BASE_KW = (
    h2_pathway_id                   = "current_mix",
    expansion_pathway_id            = "electrolysis",
    use_utilization_pricing         = true,
    utilization_transport_cost      = 1.0,
    use_lcfs                        = true,
    enable_45v                      = true,
    bus_demand_scenario             = "growing",
    end_year                        = YEAR,
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

cfg_full = build_config(; BASE_KW...)

# Only what is operational today. `already_operational` is set by the loader from
# status == "operational", so this follows config/stations_config.json.
const OPS     = [s for s in cfg_full.station_data if s.already_operational]
const DROPPED = [s for s in cfg_full.station_data if !s.already_operational]
const NAMEPLATE = sum(s.capacity_kg_per_day for s in OPS; init = 0.0)

isempty(OPS) && error("No operational stations in config/stations_config.json — nothing to validate against.")

# ── Run one variant ───────────────────────────────────────────────────────────
function run_case(station_list; n = 1, seed = SEED)
    Random.seed!(seed)
    mc = run_monte_carlo(merge(cfg_full, (station_data = station_list,)), n)
    (price = mc[1][:, 1], base = mc[5][1, 1], capex = mc[6][1, 1],
     capex_gov = mc[7][1, 1], om = mc[8][1, 1], lcfs = mc[9][1, 1],
     trucks = mc[3][1, 1], capacity = mc[4][1, 1], n_stat = mc[2][1, 1],
     prod_cap = mc[11][1, 1], ci = mc[15][1, 1], hri = mc[17][1, 1],
     util = mc[18][1, 1], v45 = mc[19][1, 1], warnings = mc[10])
end

as_built = run_case(OPS)
sunk     = run_case([merge(s, (capex = 0.0,)) for s in OPS])

# Determinism probe — the spread across seeds should be exactly zero.
probe      = run_case(OPS; n = N_CHECK, seed = 7).price
probe_span = maximum(probe) - minimum(probe)

# ── Demand decomposition ──────────────────────────────────────────────────────
# Station utilisation is truck-only, so back the served volume out of it rather
# than recomputing from the fleet (which would silently miss the first-year ramp).
truck_kg_day = as_built.util * NAMEPLATE
other_kg_day = bus_demand_kg_day(YEAR, cfg_full.bus_demand_scenario) +
               car_demand_kg_day(YEAR, cfg_full.car_demand_scenario)

function report(io)
    println(io, "2026 VALIDATION CASE — DELIVERED H₂ PRICE WITH NO NEW STATIONS")
    println(io, "="^78)
    println(io, "Single assessment year, station list frozen at what is operational today.")
    println(io, "Baseline configuration otherwise identical to manuscript fig 5.")
    println(io, "="^78)

    println(io, "\nSTATIONS IN SERVICE")
    println(io, "-"^78)
    for s in OPS
        @printf(io, "  %-12s %-30s %6d kg/day  %-8s CAPEX \$%.2fM\n",
                s.id, s.name, s.capacity_kg_per_day, s.storage_type, s.capex / 1e6)
    end
    @printf(io, "  %-12s %-30s %6.0f kg/day\n", "", "nameplate total", NAMEPLATE)
    println(io, "\n  excluded (not operational):")
    for s in DROPPED
        @printf(io, "    %-12s %-22s planned %d, %6d kg/day\n",
                s.id, s.status, s.planned_opening_year, s.capacity_kg_per_day)
    end
    println(io, "  NOTE: the CAPEX above is extrapolated below the cost reference table,")
    println(io, "        which starts at 2000 kg/day. Check it against an actual build cost.")

    println(io, "\nPRICE BUILD-UP (\$/kg H₂)")
    println(io, "-"^78)
    @printf(io, "  %-34s %12s %12s\n", "", "as-built", "CAPEX sunk")
    @printf(io, "  %-34s %12.4f %12.4f\n", "H₂ production + transport", as_built.base, sunk.base)
    @printf(io, "  %-34s %12.4f %12.4f\n", "station CAPEX (company-funded)", as_built.capex, sunk.capex)
    @printf(io, "  %-34s %12.4f %12.4f\n", "station O&M", as_built.om, sunk.om)
    @printf(io, "  %-34s %12.4f %12.4f\n", "LCFS credit (−)", -as_built.lcfs, -sunk.lcfs)
    @printf(io, "  %-34s %12.4f %12.4f\n", "HRI credit (−)", -as_built.hri, -sunk.hri)
    @printf(io, "  %-34s %12.4f %12.4f\n", "45V credit (−)", -as_built.v45, -sunk.v45)
    println(io, "  " * "-"^72)
    @printf(io, "  %-34s %12.4f %12.4f\n", "DELIVERED PRICE", as_built.price[1], sunk.price[1])
    @printf(io, "  %-34s %12.4f %12.4f\n", "  of which station fixed cost",
            as_built.capex + as_built.om, sunk.capex + sunk.om)
    println(io, "\n  Station CAPEX also has a government-funded share, zero here because")
    @printf(io, "  public_funding_percentage is 0 for every operational station (\$%.4f/kg).\n",
            as_built.capex_gov)

    println(io, "\nVOLUMES AND UTILISATION")
    println(io, "-"^78)
    @printf(io, "  %-38s %12.0f\n", "trucks in service", as_built.trucks)
    @printf(io, "  %-38s %12.0f kg/day\n", "truck H₂ demand (incl. first-year ramp)", truck_kg_day)
    @printf(io, "  %-38s %12.0f kg/day\n", "bus + car demand", other_kg_day)
    @printf(io, "  %-38s %12.0f kg/day\n", "station nameplate capacity", NAMEPLATE)
    @printf(io, "  %-38s %11.1f %%\n", "station utilisation (truck demand only)", 100 * as_built.util)
    @printf(io, "  %-38s %12.0f kg/day\n", "production capacity", as_built.prod_cap)
    @printf(io, "  %-38s %11.1f %%\n", "production utilisation (all demand)",
            100 * (truck_kg_day + other_kg_day) / as_built.prod_cap)
    @printf(io, "  %-38s %12.2f gCO₂e/MJ\n", "weighted H₂ carbon intensity", as_built.ci)

    println(io, "\nDETERMINISM CHECK")
    println(io, "-"^78)
    @printf(io, "  %d runs, different seed: min %.4f  max %.4f  spread %.6f\n",
            N_CHECK, minimum(probe), maximum(probe), probe_span)
    println(io, probe_span == 0.0 ?
        "  Zero spread, as expected — nothing stochastic can act inside a 2026 horizon." :
        "  *** NON-ZERO SPREAD: the 2026 path has become stochastic; this case needs review. ***")
    println(io, "  warnings: ", isempty(as_built.warnings) ? "none" : join(as_built.warnings, "; "))

    println(io, "\nUSING THIS FOR VALIDATION")
    println(io, "-"^78)
    println(io, "  · Cost build-up, not a posted price — no retail margin, no taxes.")
    println(io, "  · Compare production + transport against delivered-to-station cost.")
    println(io, "  · Compare the as-built total only if the real stations are recovering")
    println(io, "    their capital; otherwise the CAPEX-sunk column is the comparator.")
    println(io, "="^78)
end

report(stdout)
let path = joinpath(OUT_DIR, "val_2026_baseline.txt")
    open(report, path, "w")
    println("\nSaved → $path")
end
