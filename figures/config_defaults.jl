# =============================================================================
# CONFIG DEFAULTS FOR PUBLICATION FIGURES
# =============================================================================

# ── Shared diesel reference constants ────────────────────────────────────────
# Every figure that draws a diesel comparison used to carry its own
# `const DIESEL_MPG = 7.0`, `DIESEL_P_LO = 4.80` and `DIESEL_P_HI = 5.80`.
# Those numbers now live in config/tco_config.json and are loaded once here, so
# the suite cannot drift apart. Figure scripts alias them:
#
#     const DIESEL_MPG  = TCO_DIESEL_MPG
#     const DIESEL_P_LO = TCO_DIESEL_P_LO
#     const DIESEL_P_HI = TCO_DIESEL_P_HI
#
# Both scenarios are California retail PUMP PRICES ($/gal). Figures that work in
# $/mile convert them with the fleet-average fuel economy ($/gal ÷ mpg). ATRI's
# own fuel line item ($0.482/mile ≈ $3.58/gal) is a national average operating
# cost, not a California pump price, and is deliberately NOT used as the diesel
# reference — only its non-fuel components feed the TCO stack.
using Printf

"""
    usd_per_gal(p) -> String

Format a diesel pump price for a legend or axis label: `usd_per_gal(4.80)` →
`"\$4.80/gal"`. Always two decimals, so the low and high scenarios line up
("\$4.80/gal", "\$5.80/gal") instead of Julia's bare `\$5.8`.
"""
usd_per_gal(p) = @sprintf("\$%.2f/gal", p)

if !@isdefined(TCO_DIESEL_MPG)
    let raw = JSON.parsefile(joinpath(@__DIR__, "..", "config", "tco_config.json")),
        d   = raw["diesel_tco_usd_per_mile"],
        p   = raw["diesel_price_usd_per_gallon"]
        global const TCO_DIESEL_MPG     = Float64(d["miles_per_gallon"])   # miles/gallon
        global const TCO_DIESEL_P_LO    = Float64(p["low"])                # $/gal
        global const TCO_DIESEL_P_HI    = Float64(p["high"])               # $/gal
    end
end

"""
    load_truck_scenario(name) -> NamedTuple

Load a named truck deployment scenario from config/trucks_config.json.
Available names: "limited_dep", "high_dep", "pessimistic" (or any name you add).

Returns a NamedTuple (initial_trucks, schedule) compatible with build_config's
truck_deployment_schedule argument.

Usage:
    cfg = build_config(
        use_truck_deployment_schedule = true,
        truck_deployment_schedule     = load_truck_scenario("optimistic"),
    )
"""
function load_truck_scenario(name::String)
    data = JSON.parsefile(TRUCK_CONFIG_FILE)
    scenarios = get(data, "truck_deployment_scenarios", nothing)
    if isnothing(scenarios) || !haskey(scenarios, name)
        available = isnothing(scenarios) ? "none" : join(keys(scenarios), ", ")
        error("Truck scenario '$name' not found. Available: $available")
    end
    s = scenarios[name]
    schedule_dict = Dict{Int,Int}(entry["year"] => entry["trucks_added"] for entry in s["schedule"])
    return (initial_trucks = s["initial_trucks"], schedule = schedule_dict)
end

"""
    load_lcfs_price_scenario(name) -> Dict{Int,Float64}

Load a named LCFS price scenario from config/lcfs_config.json.
Available names: "no_change", "Small increase", "high_inc" (or any name you add).

Returns a Dict{Int,Float64} mapping year => price that can be passed directly
to build_config's h2_price_schedule equivalent for LCFS.

Usage:
    cfg = build_config(
        use_lcfs                = true,
        use_lcfs_price_schedule = true,
        lcfs_price_schedule_dict = load_lcfs_price_scenario("high"),
    )
"""
function load_lcfs_price_scenario(name::String)
    data = JSON.parsefile(LCFS_CONFIG_FILE)
    scenarios = get(data, "lcfs_price_scenarios", nothing)
    if isnothing(scenarios) || !haskey(scenarios, name)
        available = isnothing(scenarios) ? "none" : join(keys(scenarios), ", ")
        error("LCFS price scenario '$name' not found. Available: $available")
    end
    s = scenarios[name]
    return Dict{Int,Float64}(entry["year"] => Float64(entry["price_usd"]) for entry in s["schedule"])
end
# Provides build_config(; kwargs...) — a thin wrapper around the full model
# NamedTuple that pre-fills every field from the same defaults used by the
# model server, while letting figure scripts override only what they care about.
#
# Usage:
#   include("config_defaults.jl")
#   cfg = build_config(electrolyzer_capex_per_kw = 800.0, end_year = 2040)
#   results = run_monte_carlo(cfg, 1000)
# =============================================================================

"""
    build_config(; kwargs...) -> NamedTuple

Build a full simulation config NamedTuple with sensible defaults, loading
station and LCFS configuration from the standard JSON files.

All keyword arguments override the defaults. The working directory must be the
rollout model root (where the JSON config files live).

═══ H2 PRODUCTION PATHWAY ═══════════════════════════════════════════════════
  h2_pathway_id              Initial pathway: "electrolysis", "natural_gas_smr_liquid",
                             "dairy_biomethane_smr", "biomethane_smr", etc.
  expansion_pathway_id       Pathway for capacity expansions (nothing = same as initial)

═══ H2 PRICING MODE (pick one) ══════════════════════════════════════════════
  use_utilization_pricing = true   SMR-style: price from capacity utilization
  use_h2_price_curve = true        Linear/exp curve between start and end price
  use_h2_price_schedule = true     Predetermined price per year (pass h2_price_schedule dict)
  (none of above)                  Fixed price = h2_production_transport_cost

  With utilization pricing + electrolysis_pricing_enabled = true:
    CAPEX-based electrolysis cost, optionally blended with SMR

═══ ELECTROLYSIS CAPEX PRICING ══════════════════════════════════════════════
  electrolysis_pricing_enabled     (true / false)
  electrolyzer_capex_per_kw        (\$/kW, e.g. 1000)
  electrolysis_electricity_source  "grid" or "solar"
  electricity_cost_per_kwh         Grid price (\$/kWh, e.g. 0.05)
  grid_connection_cost_per_kw      Grid connection (\$/kW, default 75)
  solar_capex_per_kw               Solar panel + BOS cost (\$/kW, e.g. 800)
  solar_capacity_factor            Fraction of time at full output (e.g. 0.25)
  solar_operating_hours            Effective daily hours for grid-connected (default 10)
  solar_lifetime                   Solar system lifetime in years (default 25)

═══ LEARNING RATES (Wright's Law) ═══════════════════════════════════════════
  electrolyzer_learning_rate       Stack cost LR (default 0.233 = 23.3% per doubling)
  electrolyzer_stack_fraction      Share of ELY CAPEX that is fast-learning stack (default 0.60)
  bop_learning_rate                BoP cost LR (default 0.04)
  solar_panel_learning_rate        Solar panel LR (default 0.267 = 26.7% per doubling)
  solar_panel_fraction             Share of solar CAPEX that is fast-learning panel (default 0.80)
  solar_bop_learning_rate          Solar BoP LR (default 0.04)
  capex_vintage                    true (default): every production facility keeps the
                                   technology cost of its OPENING year, so the
                                   fleet-average CAPEX is a mix of vintages that lags
                                   the learning curve. false: all capacity is valued at
                                   the CURRENT year's technology cost (no-vintage
                                   counterfactual — see figures/fig_vintage_effect.jl)

═══ POLICY ══════════════════════════════════════════════════════════════════
  use_lcfs                         LCFS credits on/off (default true)
  use_lcfs_price_curve             Use linearly interpolated LCFS price (default false)
  lcfs_start_price                 LCFS price in start_year (\$/credit, default 60)
  lcfs_end_price                   LCFS price in end_year (\$/credit, default 100)
  lcfs_price_curve_type            "linear", "exponential", or "logarithmic"
  use_lcfs_price_schedule          Use dict-based LCFS price schedule (default false)
  enable_45v                       45V tax credit on/off (default false)
  tax_credit_45v_end_year          Last commissioning year eligible for 45V (default 2031)

═══ DEMAND SCENARIOS ════════════════════════════════════════════════════════
  bus_demand_scenario              "growing" (8→90 t/day by 2040) or "flat" (8 t/day)
  car_demand_scenario              "flat_5t" (only option currently, 5 t/day flat)

═══ TRUCKS ══════════════════════════════════════════════════════════════════
  use_truck_deployment_schedule    Use schedule from config/trucks_config.json (default true)
  max_trucks / initial_trucks      Fleet size limits
  truck_uptime_year_1/2/default    Fleet utilization by simulation year — 2026 / 2027 /
                                   2028 onward (defaults: 0.30 / 0.60 / 1.00)

═══ ECONOMICS ═══════════════════════════════════════════════════════════════
  discount_rate                    Annualization rate (default 0.07)
  transportation_cost_per_kg       Added transport cost when NOT using utilization pricing

NOTE: Station data is always loaded from config/stations_config.json — that file
is the only source, so editing it is the only way to change the station pipeline
the figures see.
"""
function build_config(;
    # ── Time range ─────────────────────────────────────────────────────────
    start_year::Int            = START_YEAR,
    end_year::Int              = END_YEAR,

    # ── H2 pathway ─────────────────────────────────────────────────────────
    h2_pathway_id              = H2_PATHWAY_ID,
    expansion_pathway_id       = nothing,

    # ── H2 pricing mode ────────────────────────────────────────────────────
    use_h2_price_schedule      = false,
    use_h2_price_curve         = false,
    use_utilization_pricing    = false,
    h2_price_schedule          = nothing,
    h2_price_curve_params      = nothing,
    h2_production_transport_cost = H2_PRODUCTION_TRANSPORT_COST,
    utilization_transport_cost::Float64 = 1.0,

    # ── Electrolysis CAPEX pricing ──────────────────────────────────────────
    electrolysis_pricing_enabled         = false,
    electrolyzer_capex_per_kw::Float64   = 1000.0,
    electrolysis_electricity_source      = "grid",
    electricity_cost_per_kwh::Float64    = 0.05,
    grid_connection_cost_per_kw::Float64 = 75.0,
    solar_operating_hours::Float64       = 10.0,
    solar_capex_per_kw::Float64          = 800.0,
    solar_capacity_factor::Float64       = 0.25,
    solar_lifetime::Int                  = 25,

    # ── Learning rates ─────────────────────────────────────────────────────
    electrolyzer_learning_rate::Float64  = 0.233,
    electrolyzer_stack_fraction::Float64 = 0.60,
    bop_learning_rate::Float64           = 0.04,
    solar_panel_fraction::Float64        = 0.80,
    solar_panel_learning_rate::Float64   = 0.267,
    solar_bop_learning_rate::Float64     = 0.04,
    # CAPEX vintaging: true (default) = each production facility keeps the
    # technology cost of its opening year, so the fleet-average CAPEX is a mix
    # of vintages. false = all capacity is valued at the current year's
    # technology cost (no-vintage counterfactual, fig_vintage_effect.jl).
    capex_vintage::Bool                  = true,

    # ── Hydrogen leakage (electrolytic H2 only) ─────────────────────────────
    enable_h2_leakage::Bool                 = true,
    h2_leakage_start_fraction::Float64      = 0.15,
    h2_leakage_floor_fraction::Float64      = 0.02,
    h2_leakage_start_year::Int              = 2026,
    h2_leakage_floor_year::Int              = 2035,

    # ── Demand scenarios ────────────────────────────────────────────────────
    bus_demand_scenario  = "growing",
    car_demand_scenario  = "flat_5t",

    # ── Trucks ─────────────────────────────────────────────────────────────
    max_trucks::Int                     = MAX_TRUCKS,
    initial_trucks::Int                 = INITIAL_TRUCKS,
    truck_uptime_year_1::Float64        = TRUCK_UPTIME_YEAR_1,
    truck_uptime_year_2::Float64        = TRUCK_UPTIME_YEAR_2,
    truck_uptime_default::Float64       = TRUCK_UPTIME_DEFAULT,
    h2_per_truck_per_day::Float64       = H2_PER_TRUCK_PER_DAY,
    operating_days_per_year::Int        = OPERATING_DAYS_PER_YEAR,
    use_truck_deployment_schedule       = USE_TRUCK_DEPLOYMENT_SCHEDULE,
    truck_deployment_schedule           = nothing,  # loaded below if needed

    # ── Economics ──────────────────────────────────────────────────────────
    discount_rate::Float64              = DISCOUNT_RATE,
    transportation_cost_per_kg::Float64 = TRANSPORTATION_COST_PER_KG,

    # ── Policy ─────────────────────────────────────────────────────────────
    use_lcfs                            = USE_LCFS,
    use_lcfs_price_schedule             = USE_LCFS_PRICE_SCHEDULE,
    use_lcfs_price_curve                = USE_LCFS_PRICE_CURVE,
    lcfs_start_price::Float64           = LCFS_START_PRICE,
    lcfs_end_price::Float64             = LCFS_END_PRICE,
    lcfs_price_curve_type               = LCFS_PRICE_CURVE_TYPE,
    # Pass a Dict{Int,Float64} directly (e.g. from load_lcfs_price_scenario())
    # Automatically enables use_lcfs_price_schedule when non-nothing
    lcfs_price_schedule_dict            = nothing,
    enable_45v                          = false,
    tax_credit_45v_end_year::Int        = 2031,

    # ── Station deployment probability ─────────────────────────────────────
    station_prob_type = STATION_PROB_TYPE,
    truck_prob_type   = TRUCK_PROB_TYPE,

    # ── Capacity investment behaviour ───────────────────────────────────────
    # trigger = floor + span·draw; foresight = years investors look ahead.
    # CURRENT MODE (2026-06-15): STOCHASTIC trigger ~ Uniform[0.60, 0.80],
    # drawn once per MC run (floor = 0.60, span = 0.20, capacity_trigger_uniform
    # = true), with the market-maturity adjustment OFF (adaptive_trigger = false).
    # To restore the DETERMINISTIC maturity-adjusted trigger (0.60 → 0.80 cap):
    # set span = 0.0, capacity_trigger_uniform = false, adaptive_trigger = true.
    capacity_trigger_floor::Float64    = 0.60,
    capacity_trigger_span::Float64     = 0.20,
    capacity_trigger_uniform::Bool     = true,   # draw is Uniform (not Beta) on [floor, floor+span]
    station_foresight_years::Int       = 3,   # = median build lead time (rand 2:4); prevents high_dep util overshoot >100%
    production_foresight_years::Int    = 3,
    # Market-maturity adjustment: the effective build trigger rises as demand
    # growth decelerates, so investors build tighter in a mature/predictable
    # market. Endogenous (keys off demand growth, not a hard-coded year).
    # OFF in the current stochastic-trigger mode.
    adaptive_trigger::Bool             = false,
    trigger_growth_hi::Float64         = 0.25,   # >= this annual growth → no lift
    trigger_growth_lo::Float64         = 0.05,   # <= this annual growth → full lift
    trigger_maturity_lift::Float64     = 0.20,   # max upward shift to the trigger as the market matures (0.60 → 0.80)
    trigger_cap::Float64               = 0.80,   # effective trigger never exceeds this (~20% buffer kept)
    # Investment-commitment probability p_invest distribution shape. Default draws
    # from a Beta(3,3) on the per-archetype interval (bell-shaped, mass near the mean).
    # Set true to draw from a flat Uniform over the same interval (wider spread,
    # same mean) — used to test sensitivity to the commitment-probability shape.
    p_invest_uniform::Bool             = false,

    # ── Misc ───────────────────────────────────────────────────────────────
    projection_start_year = nothing,
)
    # Load station config from file (always uses config/stations_config.json)
    station_config   = load_station_config(STATION_CONFIG_FILE)
    station_data     = station_config.stations
    station_prob_config = station_config.prob_config
    cost_ref_data    = station_config.cost_ref_data

    # Load truck schedule if requested
    _truck_schedule = if use_truck_deployment_schedule && isnothing(truck_deployment_schedule)
        load_truck_deployment_schedule(TRUCK_CONFIG_FILE)
    else
        truck_deployment_schedule
    end

    # Load LCFS config
    lcfs_cfg = if use_lcfs
        base = load_lcfs_config(LCFS_CONFIG_FILE)
        # Inject a named price-schedule dict if provided
        if !isnothing(lcfs_price_schedule_dict)
            # Extend schedule to cover end_year using last price
            schedule = copy(lcfs_price_schedule_dict)
            if !isempty(schedule)
                last_year = maximum(keys(schedule))
                last_price = schedule[last_year]
                for yr in (last_year + 1):end_year
                    schedule[yr] = last_price
                end
            end
            (
                eei                = base.eei,
                mj_per_kg_h2       = base.mj_per_kg_h2,
                credit_price       = base.credit_price,
                diesel_ci_schedule = base.diesel_ci_schedule,
                h2_pathways        = base.h2_pathways,
                lcfs_price_schedule = schedule,
                production_facilities = base.production_facilities,
                initial_production = base.initial_production,
                electrolysis_grid_ci = base.electrolysis_grid_ci,
            )
        else
            base
        end
    else
        nothing
    end

    # Resolve LCFS schedule flag — auto-enable when a dict was supplied
    _use_lcfs_price_schedule = use_lcfs_price_schedule || !isnothing(lcfs_price_schedule_dict)

    # Build LCFS price curve params if curve mode requested
    _lcfs_price_curve_params = if use_lcfs_price_curve
        (start_price = lcfs_start_price, end_price = lcfs_end_price, curve_type = lcfs_price_curve_type)
    else
        nothing
    end

    return (
        # Station
        use_station_config_file      = true,
        station_data                 = station_data,
        station_cost_ref_data        = cost_ref_data,
        station_prob_config          = station_prob_config,
        projection_start_year        = projection_start_year,

        # Truck
        use_truck_deployment_schedule = use_truck_deployment_schedule,
        truck_deployment_schedule     = _truck_schedule,
        max_trucks                    = max_trucks,
        initial_trucks                = initial_trucks,
        truck_uptime_year_1           = truck_uptime_year_1,
        truck_uptime_year_2           = truck_uptime_year_2,
        truck_uptime_default          = truck_uptime_default,
        h2_per_truck_per_day          = h2_per_truck_per_day,
        operating_days_per_year       = operating_days_per_year,
        h2_per_truck_per_year         = h2_per_truck_per_day * operating_days_per_year,

        # H2 pricing
        use_h2_price_schedule         = use_h2_price_schedule,
        use_h2_price_curve            = use_h2_price_curve,
        use_utilization_pricing       = use_utilization_pricing,
        h2_price_schedule             = h2_price_schedule,
        h2_price_curve_params         = h2_price_curve_params,
        h2_production_transport_cost  = h2_production_transport_cost,
        utilization_transport_cost    = utilization_transport_cost,

        # Electrolysis
        electrolysis_pricing_enabled  = electrolysis_pricing_enabled,
        electrolyzer_capex_per_kw     = electrolyzer_capex_per_kw,
        electrolysis_electricity_source = electrolysis_electricity_source,
        electricity_cost_per_kwh      = electricity_cost_per_kwh,
        grid_connection_cost_per_kw   = grid_connection_cost_per_kw,
        solar_operating_hours         = solar_operating_hours,
        solar_capex_per_kw            = solar_capex_per_kw,
        solar_capacity_factor         = solar_capacity_factor,
        solar_lifetime                = solar_lifetime,

        # Learning rates
        electrolyzer_learning_rate    = electrolyzer_learning_rate,
        electrolyzer_stack_fraction   = electrolyzer_stack_fraction,
        bop_learning_rate             = bop_learning_rate,
        solar_panel_fraction          = solar_panel_fraction,
        solar_panel_learning_rate     = solar_panel_learning_rate,
        solar_bop_learning_rate       = solar_bop_learning_rate,
        capex_vintage                 = capex_vintage,

        # Capacity investment behaviour
        capacity_trigger_floor        = capacity_trigger_floor,
        capacity_trigger_span         = capacity_trigger_span,
        capacity_trigger_uniform      = capacity_trigger_uniform,
        station_foresight_years       = station_foresight_years,
        production_foresight_years    = production_foresight_years,
        adaptive_trigger              = adaptive_trigger,
        trigger_growth_hi             = trigger_growth_hi,
        trigger_growth_lo             = trigger_growth_lo,
        trigger_maturity_lift         = trigger_maturity_lift,
        trigger_cap                   = trigger_cap,
        p_invest_uniform              = p_invest_uniform,

        # Hydrogen leakage (electrolytic only)
        enable_h2_leakage             = enable_h2_leakage,
        h2_leakage_start_fraction     = h2_leakage_start_fraction,
        h2_leakage_floor_fraction     = h2_leakage_floor_fraction,
        h2_leakage_start_year         = h2_leakage_start_year,
        h2_leakage_floor_year         = h2_leakage_floor_year,

        # Demand
        bus_demand_scenario           = bus_demand_scenario,
        car_demand_scenario           = car_demand_scenario,

        # Policy
        use_lcfs                      = use_lcfs,
        lcfs_config                   = lcfs_cfg,
        h2_pathway_id                 = h2_pathway_id,
        expansion_pathway_id          = expansion_pathway_id,
        use_lcfs_price_schedule       = _use_lcfs_price_schedule,
        use_lcfs_price_curve          = use_lcfs_price_curve,
        lcfs_price_curve_params       = _lcfs_price_curve_params,
        enable_45v                    = enable_45v,
        tax_credit_45v_end_year       = tax_credit_45v_end_year,

        # Time
        start_year                    = start_year,
        end_year                      = end_year,

        # Station fixed params
        max_stations                  = MAX_STATIONS,
        initial_stations              = INITIAL_STATIONS,
        station_capex                 = STATION_CAPEX,
        station_lifetime              = STATION_LIFETIME,
        om_cost_per_station           = OM_COST_PER_STATION,

        # Economics
        discount_rate                 = discount_rate,
        transportation_cost_per_kg    = transportation_cost_per_kg,

        # Deployment probability
        station_prob_type             = station_prob_type,
        station_prob_base             = STATION_PROB_BASE,
        station_prob_slope            = STATION_PROB_SLOPE,
        station_prob_steepness        = STATION_PROB_STEEPNESS,
        station_prob_midpoint         = STATION_PROB_MIDPOINT,
        station_prob_log_scale        = STATION_PROB_LOG_SCALE,
        station_prob_log_factor       = STATION_PROB_LOG_FACTOR,
        station_prob_exp_rate         = STATION_PROB_EXP_RATE,

        truck_prob_type               = truck_prob_type,
        truck_prob_base               = TRUCK_PROB_BASE,
        truck_prob_slope              = TRUCK_PROB_SLOPE,
        truck_prob_steepness          = TRUCK_PROB_STEEPNESS,
        truck_prob_midpoint           = TRUCK_PROB_MIDPOINT,
        truck_prob_midpoint_std       = TRUCK_PROB_MIDPOINT_STD,
        truck_prob_log_scale          = TRUCK_PROB_LOG_SCALE,
        truck_prob_log_factor         = TRUCK_PROB_LOG_FACTOR,
        truck_prob_exp_rate           = TRUCK_PROB_EXP_RATE,
    )
end
