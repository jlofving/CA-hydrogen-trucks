#=
HYDROGEN TRUCK ROLLOUT TECHNO-ECONOMIC MONTE CARLO SIMULATION

This program performs Monte Carlo simulations of hydrogen truck deployment
and calculates hydrogen prices over time based on station capex, O&M costs,
and hydrogen production/transport costs.

METHODOLOGY DOCUMENTATION:
The model framework, equations, assumptions and validation are described in the
Methods section of the accompanying article. The METHODOLOGY REFERENCE comments
below name the subsection that each piece of code implements.
Key methodology sections:
- Monte Carlo simulation framework (probabilistic deployment modeling)
- Infrastructure deployment (stations, trucks, production facilities)
- Hydrogen production costing (SMR utilization, electrolysis CAPEX-based)
- Policy instruments (LCFS credits, HRI bonus credits)
- Economic analysis (levelized cost calculation, public funding treatment)

INSTRUCTIONS:
1. Modify the parameters in the "USER CONFIGURATION" section below
2. Run the program using: julia "hydrogen truck deployment.jl"
3. Results will be displayed and plots will be generated

REQUIREMENTS:
- Julia 1.6 or higher
- Packages: Plots, Statistics, Random, Printf, JSON
- Install packages with: using Pkg; Pkg.add(["Plots", "Statistics", "Random", "JSON"])
- Printf is part of the standard library and doesn't need separate installation
=#

using Random
using Statistics
using Distributions
using LinearAlgebra   # for \ in cubic OLS fits
# Plots is only needed for local visualization functions, not the server
const _PLOTS_AVAILABLE = try; @eval using Plots; true; catch; false; end
using Printf
using JSON

#==============================================================================
USER CONFIGURATION
All parameters are stored in config/ — edit the JSON files there.
For standalone execution, the main file to edit is config/model_defaults.json.
==============================================================================#

# Configuration file paths (relative to model root directory)
const STATION_CONFIG_FILE = "config/stations_config.json"
const TRUCK_CONFIG_FILE   = "config/trucks_config.json"
const H2_CONFIG_FILE      = "config/trucks_config.json"
const LCFS_CONFIG_FILE    = "config/lcfs_config.json"

# Load simulation defaults from config file
const _MD = JSON.parsefile("config/model_defaults.json")

# Simulation settings
const N_MONTE_CARLO_RUNS    = _MD["simulation"]["n_monte_carlo_runs"]
const START_YEAR            = _MD["simulation"]["start_year"]
const END_YEAR              = _MD["simulation"]["end_year"]
const RANDOM_SEED           = _MD["simulation"]["random_seed"]

# Economic parameters
const DISCOUNT_RATE                = Float64(_MD["economics"]["discount_rate"])
const H2_PRODUCTION_TRANSPORT_COST = Float64(_MD["economics"]["h2_production_transport_cost"])

# H2 production pathway defaults
const H2_PATHWAY_ID        = _MD["h2_pathways"]["initial_pathway_id"]
const EXPANSION_PATHWAY_ID = _MD["h2_pathways"]["expansion_pathway_id"]

# Truck fleet defaults
const MAX_TRUCKS               = _MD["truck_defaults"]["max_trucks"]
const INITIAL_TRUCKS           = _MD["truck_defaults"]["initial_trucks"]
const H2_PER_TRUCK_PER_DAY     = Float64(_MD["truck_defaults"]["h2_per_truck_per_day"])
const OPERATING_DAYS_PER_YEAR  = _MD["truck_defaults"]["operating_days_per_year"]
const TRANSPORTATION_COST_PER_KG = Float64(_MD["truck_defaults"]["transportation_cost_per_kg"])
const TRUCK_UPTIME_YEAR_1      = Float64(_MD["truck_defaults"]["uptime"]["year_1"])
const TRUCK_UPTIME_YEAR_2      = Float64(_MD["truck_defaults"]["uptime"]["year_2"])
const TRUCK_UPTIME_DEFAULT     = Float64(_MD["truck_defaults"]["uptime"]["default"])
const H2_PER_TRUCK_PER_YEAR    = H2_PER_TRUCK_PER_DAY * OPERATING_DAYS_PER_YEAR

# H2 pricing mode for standalone execution
const USE_H2_PRICE_SCHEDULE = _MD["h2_pricing"]["use_schedule"]
const USE_H2_PRICE_CURVE    = _MD["h2_pricing"]["use_curve"]
const H2_START_PRICE        = Float64(_MD["h2_pricing"]["start_price"])
const H2_END_PRICE          = Float64(_MD["h2_pricing"]["end_price"])
const H2_PRICE_CURVE_TYPE   = _MD["h2_pricing"]["curve_type"]

# LCFS (Low Carbon Fuel Standard) parameters
const USE_LCFS                = _MD["lcfs"]["enabled"]
const USE_LCFS_PRICE_SCHEDULE = _MD["lcfs"]["use_price_schedule"]
const USE_LCFS_PRICE_CURVE    = _MD["lcfs"]["use_price_curve"]
const LCFS_START_PRICE        = Float64(_MD["lcfs"]["start_price"])
const LCFS_END_PRICE          = Float64(_MD["lcfs"]["end_price"])
const LCFS_PRICE_CURVE_TYPE   = _MD["lcfs"]["curve_type"]

# HRI (Hydrogen Refueling Infrastructure) LCFS credit parameters
const HRI_UTILIZATION_THRESHOLD = Float64(_MD["hri"]["utilization_threshold"])
const HRI_STATION_UPTIME        = Float64(_MD["hri"]["station_uptime"])

# Station projection after user-defined pipeline ends
const AUTO_PROJECT_STATIONS  = _MD["station_projection"]["auto_project"]
const STATION_CAPACITY_BUFFER = Float64(_MD["station_projection"]["capacity_buffer"])

# Deployment mode flags
const USE_STATION_CONFIG_FILE       = _MD["station_config"]["use_station_config_file"]
const USE_TRUCK_DEPLOYMENT_SCHEDULE = _MD["station_config"]["use_truck_deployment_schedule"]

# Production cost mode — not user-configurable, controls internal model behavior
# "exogenous" = use H2 base price; "endogenous" = calculate from facility CAPEX
const PRODUCTION_COST_MODE = "exogenous"

# Stochastic fallback mode parameters (only active when USE_STATION_CONFIG_FILE or
# USE_TRUCK_DEPLOYMENT_SCHEDULE is false — currently not in use)
const MAX_STATIONS          = _MD["stochastic_fallback"]["max_stations"]
const INITIAL_STATIONS      = _MD["stochastic_fallback"]["initial_stations"]
const STATION_CAPEX         = Float64(_MD["stochastic_fallback"]["station_capex"])
const STATION_LIFETIME      = _MD["stochastic_fallback"]["station_lifetime"]
const OM_COST_PER_STATION   = Float64(_MD["stochastic_fallback"]["om_cost_per_station"])
const STATION_PROB_TYPE     = _MD["stochastic_fallback"]["station_prob"]["type"]
const STATION_PROB_BASE     = Float64(_MD["stochastic_fallback"]["station_prob"]["base"])
const STATION_PROB_SLOPE    = Float64(_MD["stochastic_fallback"]["station_prob"]["slope"])
const STATION_PROB_STEEPNESS = Float64(_MD["stochastic_fallback"]["station_prob"]["steepness"])
const STATION_PROB_MIDPOINT = Float64(_MD["stochastic_fallback"]["station_prob"]["midpoint"])
const STATION_PROB_LOG_SCALE = Float64(_MD["stochastic_fallback"]["station_prob"]["log_scale"])
const STATION_PROB_LOG_FACTOR = Float64(_MD["stochastic_fallback"]["station_prob"]["log_factor"])
const STATION_PROB_EXP_RATE = Float64(_MD["stochastic_fallback"]["station_prob"]["exp_rate"])
const TRUCK_PROB_TYPE       = _MD["stochastic_fallback"]["truck_prob"]["type"]
const TRUCK_PROB_BASE       = Float64(_MD["stochastic_fallback"]["truck_prob"]["base"])
const TRUCK_PROB_SLOPE      = Float64(_MD["stochastic_fallback"]["truck_prob"]["slope"])
const TRUCK_PROB_STEEPNESS  = Float64(_MD["stochastic_fallback"]["truck_prob"]["steepness"])
const TRUCK_PROB_MIDPOINT   = Float64(_MD["stochastic_fallback"]["truck_prob"]["midpoint"])
const TRUCK_PROB_MIDPOINT_STD = Float64(_MD["stochastic_fallback"]["truck_prob"]["midpoint_std"])
const TRUCK_PROB_LOG_SCALE  = Float64(_MD["stochastic_fallback"]["truck_prob"]["log_scale"])
const TRUCK_PROB_LOG_FACTOR = Float64(_MD["stochastic_fallback"]["truck_prob"]["log_factor"])
const TRUCK_PROB_EXP_RATE   = Float64(_MD["stochastic_fallback"]["truck_prob"]["exp_rate"])

#==============================================================================
HELPER FUNCTIONS
==============================================================================#

"""
    calculate_annuity_factor(lifetime, discount_rate)

Calculate the capital recovery factor (CRF) for annuitizing capital costs.

METHODOLOGY REFERENCE: see the article Methods, "Economic Analysis"
This implements the standard capital recovery factor formula:
    CRF = r(1 + r)^n / ((1 + r)^n - 1)

Used to convert upfront capital expenditures into equivalent annual costs,
enabling comparison with annual O&M costs and spreading infrastructure
investments over hydrogen sales.

# Arguments
- `lifetime`: Asset lifetime in years (stations: 10y, electrolyzers: 20y, solar: 25y)
- `discount_rate`: Annual discount rate (e.g., 0.07 for 7%)

# Returns
- Capital recovery factor (dimensionless)

# Example
For a 10-year station with 7% discount rate:
    CRF = 0.1424 (i.e., \$1M upfront = \$142,400/year)
"""
function calculate_annuity_factor(lifetime, discount_rate)
    if discount_rate == 0.0
        return 1.0 / lifetime
    end
    r = discount_rate
    n = lifetime
    crf = (r * (1 + r)^n) / ((1 + r)^n - 1)
    return crf
end

"""
    calculate_electrolyzer_capacity_kw(capacity_kg_day, operating_hours_per_day=24.0)

Calculate required electrolyzer capacity in kW for given H2 production.
Each kg H2 requires 55 kWh.
- Grid-powered (24 hrs/day): kW = (kg/day × 55) / 24
- Solar-powered (e.g., 10 hrs/day): kW = (kg/day × 55) / 10
"""
function calculate_electrolyzer_capacity_kw(capacity_kg_day::Float64, operating_hours_per_day::Float64=24.0)
    kwh_per_kg_h2 = 55.0
    return (capacity_kg_day * kwh_per_kg_h2) / operating_hours_per_day
end

"""
    electrolyzer_manufacturing_capacity_gw(year)

Cubic OLS fit to global electrolyzer cumulative installed capacity (GW).
Source: IEA Global Hydrogen Review 2025 (2025 is estimate); 2030 from IEA NZE.
Data: 2020→0.30, 2021→0.55, 2022→0.70, 2023→1.35, 2024→2.00, 2025→4.95, 2030→65
Fit: C(t) = a·t³ + b·t² + c·t + d, where t = year − 2020, coefficients from OLS.
Returns 0.30 GW for years at or before 2020.
"""
const _EMC_COEFF = let
    ts = Float64.([0, 1, 2, 3, 4, 5, 10])   # t = year - 2020
    ys = Float64.([0.30, 0.55, 0.70, 1.35, 2.00, 4.95, 65.0])
    V  = hcat(ts.^3, ts.^2, ts, ones(length(ts)))
    V \ ys
end
function electrolyzer_manufacturing_capacity_gw(year::Int)::Float64
    t = Float64(year - 2020)
    t <= 0.0 && return 0.30
    return _EMC_COEFF[1]*t^3 + _EMC_COEFF[2]*t^2 + _EMC_COEFF[3]*t + _EMC_COEFF[4]
end

"""
    electrolyzer_capex_for_year(opening_year, base_capex_per_kw_2025, stack_fraction,
                                stack_learning_rate, bop_learning_rate)

Calculate site CAPEX for a given opening year using split learning curves.
The total CAPEX is split into electrolyzer stack (fast learning) and balance of plant (slow learning).

Formula:
  CAPEX_year = CAPEX_2025 × [
    stack_fraction × (C_year/C_2025)^(-α_stack) +
    (1 - stack_fraction) × (C_year/C_2025)^(-α_bop)
  ]
where α = log(1/(1-LR)) / log(2)

Arguments:
- opening_year: Year the facility opens
- base_capex_per_kw_2025: Total site CAPEX in 2025 (\$/kW)
- stack_fraction: Share of CAPEX that is electrolyzer stack (e.g., 0.60)
- stack_learning_rate: Learning rate for the stack component (e.g., 0.20)
- bop_learning_rate: Learning rate for balance-of-plant (construction, land, etc.) (e.g., 0.04)
"""
function electrolyzer_capex_for_year(
    opening_year::Int,
    base_capex_per_kw_2025::Float64,
    stack_fraction::Float64,
    stack_learning_rate::Float64,
    bop_learning_rate::Float64
)::Float64
    ref_capacity = electrolyzer_manufacturing_capacity_gw(2025)
    year_capacity = electrolyzer_manufacturing_capacity_gw(opening_year)
    ratio = year_capacity / ref_capacity

    # Stack component (electrolyzer hardware)
    stack_capex = if stack_learning_rate > 0.0 && stack_learning_rate < 1.0
        alpha = log(1.0 / (1.0 - stack_learning_rate)) / log(2.0)
        base_capex_per_kw_2025 * stack_fraction * ratio^(-alpha)
    else
        base_capex_per_kw_2025 * stack_fraction
    end

    # Balance-of-plant component (construction, land, grid connection, etc.)
    bop_fraction = 1.0 - stack_fraction
    bop_capex = if bop_learning_rate > 0.0 && bop_learning_rate < 1.0
        alpha_bop = log(1.0 / (1.0 - bop_learning_rate)) / log(2.0)
        base_capex_per_kw_2025 * bop_fraction * ratio^(-alpha_bop)
    else
        base_capex_per_kw_2025 * bop_fraction
    end

    return stack_capex + bop_capex
end

let
    _ts = Float64.([2015,2016,2017,2018,2019,2020,2021,2022,2023,2024,2030] .- 2015)
    _ys = Float64.([231.6,310.4,413.0,517.3,636.5,789.7,953.2,1185.1,1611.5,2116.7,6698.8])
    _V  = hcat(_ts.^3, _ts.^2, _ts, ones(length(_ts)))
    _θ  = _V \ _ys
    global _spv_a, _spv_b, _spv_c, _spv_d = _θ[1], _θ[2], _θ[3], _θ[4]
end
"""
    solar_pv_capacity_gw(year)

Cubic OLS fit to global solar PV cumulative capacity (GW), IEA Net Zero Scenario.
Historical data 2015–2024 (observed); NZE projection 2030.
Returns 231.6 GW for years at or before 2015.
"""
function solar_pv_capacity_gw(year::Int)::Float64
    t = Float64(year - 2015)
    t <= 0.0 && return 231.6
    return _spv_a * t^3 + _spv_b * t^2 + _spv_c * t + _spv_d
end

"""
    solar_capex_for_year(opening_year, base_capex_per_kw_2025, panel_fraction,
                         panel_learning_rate, bop_learning_rate)

Calculate solar installation CAPEX for a given year using split learning curves.
The total solar CAPEX is split into PV panels (fast learning) and balance-of-plant (slow learning).

Formula:
  CAPEX_year = CAPEX_2025 × [
    panel_fraction × (C_year/C_2025)^(-α_panel) +
    (1 - panel_fraction) × (C_year/C_2025)^(-α_bop)
  ]

Arguments:
- opening_year: Year the facility opens
- base_capex_per_kw_2025: Total solar CAPEX in 2025 (\$/kW_solar)
- panel_fraction: Share of CAPEX that is PV panels (e.g., 0.80)
- panel_learning_rate: Learning rate for the panel component (e.g., 0.20)
- bop_learning_rate: Learning rate for balance-of-plant (e.g., 0.04)
"""
function solar_capex_for_year(
    opening_year::Int,
    base_capex_per_kw_2025::Float64,
    panel_fraction::Float64,
    panel_learning_rate::Float64,
    bop_learning_rate::Float64
)::Float64
    ref_capacity = solar_pv_capacity_gw(2025)
    year_capacity = solar_pv_capacity_gw(opening_year)
    ratio = year_capacity / ref_capacity

    panel_capex = if panel_learning_rate > 0.0 && panel_learning_rate < 1.0
        alpha = log(1.0 / (1.0 - panel_learning_rate)) / log(2.0)
        base_capex_per_kw_2025 * panel_fraction * ratio^(-alpha)
    else
        base_capex_per_kw_2025 * panel_fraction
    end

    bop_fraction = 1.0 - panel_fraction
    bop_capex = if bop_learning_rate > 0.0 && bop_learning_rate < 1.0
        alpha_bop = log(1.0 / (1.0 - bop_learning_rate)) / log(2.0)
        base_capex_per_kw_2025 * bop_fraction * ratio^(-alpha_bop)
    else
        base_capex_per_kw_2025 * bop_fraction
    end

    return panel_capex + bop_capex
end

"""
    bus_demand_kg_day(year, scenario)

Daily hydrogen demand from bus fleets for a given year and scenario.

Scenarios:
- "growing": piecewise-linear fit to projected bus fleet expansion
  (2026: 8000, 2028: 12000, 2030: 25000, 2035: 55000, 2040: 90000 kg/day)
- "flat": constant 8000 kg/day
"""
function bus_demand_kg_day(year::Int, scenario::String)::Float64
    if scenario == "flat"
        return 8000.0
    end
    # "growing": piecewise linear between data points
    points = [(2026, 8000.0), (2028, 12000.0), (2030, 25000.0), (2035, 55000.0), (2040, 90000.0)]
    if year <= points[1][1]
        return points[1][2]
    end
    if year >= points[end][1]
        return points[end][2]
    end
    for i in 1:length(points)-1
        y0, d0 = points[i]
        y1, d1 = points[i+1]
        if year >= y0 && year <= y1
            t = Float64(year - y0) / Float64(y1 - y0)
            return d0 + t * (d1 - d0)
        end
    end
    return 8000.0
end

"""
    car_demand_kg_day(year, scenario)

Daily hydrogen demand from passenger fuel-cell vehicles for a given year and scenario.

Scenarios:
- "flat_5t": constant 5000 kg/day
"""
function car_demand_kg_day(year::Int, scenario::String)::Float64
    return 5000.0  # only one scenario for now
end

"""
    calculate_electrolysis_cost_grid(total_electrolyzer_capacity_kw, total_annual_demand_kg,
                                     electrolyzer_capex_per_kw, electricity_cost_per_kwh,
                                     grid_connection_cost_per_kw, discount_rate, lifetime)

Calculate electrolysis production cost using grid electricity.

METHODOLOGY REFERENCE: see the article Methods, "Electrolysis CAPEX-Based Pricing"
Implements the grid-powered electrolysis cost formula:
    C_elec,grid = (A_elec + A_grid) / D_annual + (C_electricity × 55 kWh/kg)

Where:
- A_elec = annualized electrolyzer CAPEX (using CRF)
- A_grid = annualized grid connection infrastructure CAPEX
- D_annual = total annual H2 demand (spreads fixed costs over production)
- 55 kWh/kg = electricity consumption for H2 production via electrolysis

This cost structure reflects that electrolysis is capital-intensive (CAPEX-dominated)
rather than fuel-cost-dominated like SMR. The CAPEX per kg decreases with higher
utilization (more demand to spread fixed costs over).

# Arguments
- `total_electrolyzer_capacity_kw`: Combined capacity of all electrolysis facilities (kW)
- `total_annual_demand_kg`: Truck demand + static other demand (kg/year)
- `electrolyzer_capex_per_kw`: Capital cost per kW (\$/kW), typically \$800-3000/kW
- `electricity_cost_per_kwh`: Grid electricity price (\$/kWh), typically \$0.05-0.20/kWh
- `grid_connection_cost_per_kw`: Grid connection infrastructure cost per kW (\$/kW)
- `discount_rate`: For annualization (e.g., 0.07)
- `lifetime`: Electrolyzer lifetime (years, typically 20)
"""
function calculate_electrolysis_cost_grid(
    total_electrolyzer_capacity_kw::Float64,
    total_annual_demand_kg::Float64,
    electrolyzer_capex_per_kw::Float64,
    electricity_cost_per_kwh::Float64,
    grid_connection_cost_per_kw::Float64,
    discount_rate::Float64,
    lifetime::Int
)
    if total_annual_demand_kg == 0.0
        return 0.0
    end

    # Calculate total electrolyzer CAPEX
    total_electrolyzer_capex = total_electrolyzer_capacity_kw * electrolyzer_capex_per_kw

    # Calculate total grid connection CAPEX
    total_grid_connection_capex = total_electrolyzer_capacity_kw * grid_connection_cost_per_kw

    # Annualize CAPEX (both electrolyzer and grid connection use same lifetime)
    annuity_factor = calculate_annuity_factor(lifetime, discount_rate)
    annualized_electrolyzer_capex = total_electrolyzer_capex * annuity_factor
    annualized_grid_connection_capex = total_grid_connection_capex * annuity_factor

    # CAPEX cost per kg (spread over actual demand, like stations)
    capex_per_kg = (annualized_electrolyzer_capex + annualized_grid_connection_capex) / total_annual_demand_kg

    # Electricity cost component (55 kWh per kg H2)
    kwh_per_kg_h2 = 55.0
    electricity_per_kg = electricity_cost_per_kwh * kwh_per_kg_h2

    return capex_per_kg + electricity_per_kg
end

"""
    calculate_electrolysis_cost_solar(total_electrolyzer_capacity_kw, total_annual_demand_kg,
                                      electrolyzer_capex_per_kw, solar_capex_per_kw,
                                      solar_capacity_factor, solar_operating_hours,
                                      discount_rate, electrolyzer_lifetime, solar_lifetime)

Calculate electrolysis production cost using dedicated solar power.

METHODOLOGY REFERENCE: see the article Methods, "Electrolysis CAPEX-Based Pricing"
Implements the solar-powered electrolysis cost formula:
    C_elec,solar = (A_elec + A_solar) / D_annual

Solar capacity calculation accounts for operating hours and capacity factor:
    P_solar = (P_electrolyzer × h_op) / (CF × 24)

Energy balance ensures: P_solar × CF × 24 = P_electrolyzer × h_op
This accounts for intermittency: solar produces during daylight hours (e.g., 10 hrs/day)
but capacity factor accounts for weather, panel efficiency, and day/night cycle over 24 hours.

Different annualization lifetimes reflect technology differences:
- Electrolyzer: 20 years (electrochemical degradation)
- Solar PV: 25 years (longer-lived technology)

# Arguments
- `total_electrolyzer_capacity_kw`: Combined capacity of all electrolysis facilities (kW)
- `total_annual_demand_kg`: Truck demand + static other demand (kg/year)
- `electrolyzer_capex_per_kw`: Electrolyzer capital cost (\$/kW)
- `solar_capex_per_kw`: Solar installation capital cost (\$/kW), typically \$700-1600/kW
- `solar_capacity_factor`: Solar capacity factor over 24 hours (e.g., 0.25 = 25%)
- `solar_operating_hours`: Hours per day electrolyzer operates with solar (e.g., 10)
- `discount_rate`: For annualization (e.g., 0.07)
- `electrolyzer_lifetime`: Electrolyzer lifetime (years, typically 20)
- `solar_lifetime`: Solar lifetime (years, typically 25)
"""
function calculate_electrolysis_cost_solar(
    total_electrolyzer_capacity_kw::Float64,
    total_annual_demand_kg::Float64,
    electrolyzer_capex_per_kw::Float64,
    solar_capex_per_kw::Float64,
    solar_capacity_factor::Float64,
    solar_operating_hours::Float64,
    discount_rate::Float64,
    electrolyzer_lifetime::Int,
    solar_lifetime::Int
)
    if total_annual_demand_kg == 0.0 || solar_capacity_factor == 0.0
        return 0.0
    end

    # Calculate required solar capacity accounting for operating hours
    # Energy balance: solar_capacity × CF × 24 = electrolyzer_capacity × operating_hours
    solar_capacity_kw = (total_electrolyzer_capacity_kw * solar_operating_hours) / (solar_capacity_factor * 24.0)

    # Calculate total CAPEX for both systems
    total_electrolyzer_capex = total_electrolyzer_capacity_kw * electrolyzer_capex_per_kw
    total_solar_capex = solar_capacity_kw * solar_capex_per_kw

    # Annualize both CAPEX components (different lifetimes)
    electrolyzer_annuity = calculate_annuity_factor(electrolyzer_lifetime, discount_rate)
    solar_annuity = calculate_annuity_factor(solar_lifetime, discount_rate)

    annualized_electrolyzer = total_electrolyzer_capex * electrolyzer_annuity
    annualized_solar = total_solar_capex * solar_annuity

    # Calculate cost per kg (spread over actual demand, like stations)
    return (annualized_electrolyzer + annualized_solar) / total_annual_demand_kg
end


"""
    deployment_probability(year_index, total_years, prob_type, params)

Calculate deployment probability for a given year.

# Arguments
- `year_index`: Current year index (0 = first year, total_years-1 = last year)
- `total_years`: Total number of years in simulation
- `prob_type`: Type of probability function ("linear", "sigmoid", "logarithmic", or "exponential")
- `params`: NamedTuple with function parameters

# Returns
- Probability value between 0 and 1
"""
function deployment_probability(year_index, total_years, prob_type, params)
    # Normalize year_index to [0, 1] scale
    year_progress = year_index / (total_years - 1)

    if prob_type == "linear"
        prob = params.base + params.slope * year_progress
        return min(1.0, max(0.0, prob))
    elseif prob_type == "sigmoid"
        prob = 1.0 / (1.0 + exp(-params.steepness * (year_progress - params.midpoint)))
        return prob
    elseif prob_type == "logarithmic"
        # Logarithmic growth: starts fast, then slows down
        prob = params.base + params.log_scale * log(1.0 + params.log_factor * year_progress)
        return min(1.0, max(0.0, prob))
    elseif prob_type == "exponential"
        # Exponential growth: starts slow, then accelerates
        prob = params.base * exp(params.exp_rate * year_progress)
        return min(1.0, max(0.0, prob))
    else
        error("Unknown probability type: $prob_type")
    end
end

"""
    linear_interpolate_extrapolate(x, x_data, y_data)

Perform linear interpolation or extrapolation to find y value for given x.

# Arguments
- `x`: The x value to interpolate/extrapolate for
- `x_data`: Array of known x values (must be sorted)
- `y_data`: Array of known y values corresponding to x_data

# Returns
- Interpolated or extrapolated y value
"""
function linear_interpolate_extrapolate(x, x_data, y_data)
    n = length(x_data)

    # If x is below the range, extrapolate using first two points
    if x <= x_data[1]
        if n < 2
            return y_data[1]
        end
        slope = (y_data[2] - y_data[1]) / (x_data[2] - x_data[1])
        return y_data[1] + slope * (x - x_data[1])
    end

    # If x is above the range, extrapolate using last two points
    if x >= x_data[end]
        if n < 2
            return y_data[end]
        end
        slope = (y_data[end] - y_data[end-1]) / (x_data[end] - x_data[end-1])
        return y_data[end] + slope * (x - x_data[end])
    end

    # Interpolate between two points
    for i in 1:(n-1)
        if x >= x_data[i] && x <= x_data[i+1]
            slope = (y_data[i+1] - y_data[i]) / (x_data[i+1] - x_data[i])
            return y_data[i] + slope * (x - x_data[i])
        end
    end

    # Fallback (should never reach here)
    return y_data[end]
end

"""
    fit_polynomial(x_data, y_data, degree)

Fit a polynomial of specified degree to the data using least squares.

# Arguments
- `x_data`: Array of x values
- `y_data`: Array of y values
- `degree`: Polynomial degree (2 for quadratic, 3 for cubic, etc.)

# Returns
- Array of coefficients [c0, c1, c2, ...] where y = c0 + c1*x + c2*x^2 + ...
"""
function fit_polynomial(x_data, y_data, degree)
    n = length(x_data)

    # Create Vandermonde matrix
    A = zeros(n, degree + 1)
    for i in 1:n
        for j in 0:degree
            A[i, j+1] = x_data[i]^j
        end
    end

    # Solve least squares: A * coeffs = y_data
    # Using normal equations: (A'A) * coeffs = A' * y_data
    coeffs = (A' * A) \ (A' * y_data)

    return coeffs
end

"""
    eval_polynomial(x, coeffs)

Evaluate polynomial at x using coefficients.

# Arguments
- `x`: Value to evaluate at
- `coeffs`: Polynomial coefficients [c0, c1, c2, ...]

# Returns
- y = c0 + c1*x + c2*x^2 + ...
"""
function eval_polynomial(x, coeffs)
    result = 0.0
    for (i, c) in enumerate(coeffs)
        result += c * x^(i-1)
    end
    return result
end

"""
    polynomial_fit_curve(x, x_data, y_data, degree)

Fit polynomial to data and evaluate at x.

# Arguments
- `x`: The x value to evaluate at
- `x_data`: Array of known x values
- `y_data`: Array of known y values
- `degree`: Polynomial degree (2 for quadratic, 3 for cubic)

# Returns
- Fitted y value at x
"""
function polynomial_fit_curve(x, x_data, y_data, degree)
    coeffs = fit_polynomial(x_data, y_data, degree)
    return eval_polynomial(x, coeffs)
end

"""
    calculate_station_costs(capacity_kg_per_day, storage_type, cost_ref_data)

Calculate station capex and O&M costs based on capacity and storage type.
- Capex: Uses polynomial curve fitting (degree 2 or 3) for non-linear scaling
- O&M: Uses linear interpolation/extrapolation

# Arguments
- `capacity_kg_per_day`: Station capacity in kg/day
- `storage_type`: Either "gaseous" or "liquid"
- `cost_ref_data`: Reference cost data NamedTuple

# Returns
- Tuple of (capex, om_cost_per_year)
"""
function calculate_station_costs(capacity_kg_per_day, storage_type, cost_ref_data)
    # Get reference data for this storage type
    if storage_type == "gaseous"
        ref_capacities = cost_ref_data.gaseous_capacities
        ref_capex = cost_ref_data.gaseous_capex
        ref_om    = cost_ref_data.gaseous_om
        ref_om_30 = cost_ref_data.gaseous_om_30pct
        ref_om_50 = cost_ref_data.gaseous_om_50pct
    elseif storage_type == "liquid"
        ref_capacities = cost_ref_data.liquid_capacities
        ref_capex = cost_ref_data.liquid_capex
        ref_om    = cost_ref_data.liquid_om
        ref_om_30 = cost_ref_data.liquid_om_30pct
        ref_om_50 = cost_ref_data.liquid_om_50pct
    else
        error("Unknown storage type: $storage_type. Must be 'gaseous' or 'liquid'")
    end

    # Determine polynomial degree based on number of data points
    # Use degree 2 (quadratic) for 3-4 points, degree 3 (cubic) for 5+ points
    poly_degree = length(ref_capacities) >= 5 ? 3 : 2

    # CAPEX: polynomial curve fitting for non-linear cost scaling
    capex = polynomial_fit_curve(capacity_kg_per_day, ref_capacities, ref_capex, poly_degree)

    # O&M at each utilization level (linear interpolation/extrapolation in capacity dimension)
    om_at_30pct = linear_interpolate_extrapolate(capacity_kg_per_day, ref_capacities, ref_om_30)
    om_at_50pct = linear_interpolate_extrapolate(capacity_kg_per_day, ref_capacities, ref_om_50)
    om_at_80pct = linear_interpolate_extrapolate(capacity_kg_per_day, ref_capacities, ref_om)

    return (capex, om_at_30pct, om_at_50pct, om_at_80pct)
end

"""
    interpolate_om_at_utilization(om_at_30, om_at_50, om_at_80, utilization)

Interpolate (or extrapolate) annual O&M cost given the three reference utilization levels.
Piecewise linear between 30%-50% and 50%-80%; linear extrapolation outside that range.

# Arguments
- `om_at_30`: O&M cost at 30% utilization (USD/year)
- `om_at_50`: O&M cost at 50% utilization (USD/year)
- `om_at_80`: O&M cost at 80% utilization (USD/year)
- `utilization`: Current utilization fraction (0-1)

# Returns
- Interpolated O&M cost (USD/year)
"""
function interpolate_om_at_utilization(om_at_30, om_at_50, om_at_80, utilization)
    if utilization <= 0.30
        # Extrapolate below 30% using slope from 30%-50% segment
        slope = (om_at_50 - om_at_30) / (0.50 - 0.30)
        return om_at_30 + slope * (utilization - 0.30)
    elseif utilization <= 0.50
        t = (utilization - 0.30) / (0.50 - 0.30)
        return om_at_30 + t * (om_at_50 - om_at_30)
    elseif utilization <= 0.80
        t = (utilization - 0.50) / (0.80 - 0.50)
        return om_at_50 + t * (om_at_80 - om_at_50)
    else
        # Extrapolate above 80% using slope from 50%-80% segment
        slope = (om_at_80 - om_at_50) / (0.80 - 0.50)
        return om_at_80 + slope * (utilization - 0.80)
    end
end

"""
    load_station_config(filepath)

Load station configuration from JSON file.

# Arguments
- `filepath`: Path to the JSON configuration file

# Returns
- NamedTuple containing:
  - stations: Array of station NamedTuples
  - prob_config: Deployment probability configuration
  - cost_ref_data: Cost reference data for interpolation
"""
function load_station_config(filepath)
    config_data = JSON.parsefile(filepath)

    # Load cost reference data
    gaseous_ref = config_data["station_cost_reference_data"]["gaseous"]
    liquid_ref = config_data["station_cost_reference_data"]["liquid"]

    cost_ref_data = (
        gaseous_capacities = Float64.(gaseous_ref["capacity_kg_per_day"]),
        gaseous_capex      = Float64.(gaseous_ref["capex_usd"]),
        gaseous_om         = Float64.(gaseous_ref["om_cost_per_year_usd"]),
        gaseous_om_30pct   = Float64.(gaseous_ref["om_30pct_per_year_usd"]),
        gaseous_om_50pct   = Float64.(gaseous_ref["om_50pct_per_year_usd"]),
        liquid_capacities  = Float64.(liquid_ref["capacity_kg_per_day"]),
        liquid_capex       = Float64.(liquid_ref["capex_usd"]),
        liquid_om          = Float64.(liquid_ref["om_cost_per_year_usd"]),
        liquid_om_30pct    = Float64.(liquid_ref["om_30pct_per_year_usd"]),
        liquid_om_50pct    = Float64.(liquid_ref["om_50pct_per_year_usd"])
    )

    # Load stations and calculate costs based on capacity and storage type
    stations = []
    for station_data in config_data["stations"]
        capacity = station_data["capacity_kg_per_day"]
        storage_type = station_data["storage_type"]

        # Calculate costs using interpolation/extrapolation
        capex, om_at_30pct, om_at_50pct, om_at_80pct = calculate_station_costs(capacity, storage_type, cost_ref_data)

        # Convert status to boolean - "operational" means already deployed
        status = get(station_data, "status", "in planning")
        # Backward compatibility: check for old already_operational field
        if haskey(station_data, "already_operational")
            already_operational = station_data["already_operational"]
        else
            already_operational = (lowercase(status) == "operational")
        end

        push!(stations, (
            id = station_data["id"],
            name = station_data["name"],
            planned_opening_year = station_data["planned_opening_year"],
            capacity_kg_per_day = capacity,
            storage_type = storage_type,
            capex = capex,
            om_cost_per_year = om_at_80pct,
            om_cost_30pct_per_year = om_at_30pct,
            om_cost_50pct_per_year = om_at_50pct,
            already_operational = already_operational,
            status = status,
            public_funding_percentage = Float64(get(station_data, "public_funding_percentage", 0.0))
        ))
    end

    prob_config_data = config_data["deployment_probability_config"]

    # Load probability by status (new format) or fall back to old format
    if haskey(prob_config_data, "probability_by_status")
        # Keys are normalised to lowercase because `calculate_station_opening_probability`
        # lowercases the station status before looking it up. Without this the JSON key
        # "FID" never matched and silently fell through to the 0.3 default instead of the
        # 0.8 configured for it.
        prob_by_status = Dict{String, Float64}(
            lowercase(String(k)) => Float64(v)
            for (k, v) in prob_config_data["probability_by_status"]
        )
        prob_config = (
            max_delay_years = prob_config_data["max_delay_years"],
            probability_by_status = prob_by_status
        )
    else
        # Backward compatibility with old format
        prob_config = (
            prob_in_planned_year = prob_config_data["probability_in_planned_year"],
            prob_in_subsequent_years = prob_config_data["probability_in_subsequent_years"],
            max_delay_years = prob_config_data["max_delay_years"],
            probability_by_status = nothing
        )
    end

    return (stations = stations, prob_config = prob_config, cost_ref_data = cost_ref_data)
end

"""
    load_truck_deployment_schedule(filepath)

Load predetermined truck deployment schedule from JSON configuration file.

# Arguments
- `filepath`: Path to configuration JSON file

# Returns
- NamedTuple with truck deployment schedule data
"""
function load_truck_deployment_schedule(filepath)
    config_data = JSON.parsefile(filepath)

    if !haskey(config_data, "truck_deployment_schedule")
        error("Configuration file missing 'truck_deployment_schedule' section")
    end

    truck_config = config_data["truck_deployment_schedule"]

    # Parse schedule into a dictionary mapping year => trucks_added
    schedule_dict = Dict{Int, Int}()
    for entry in truck_config["schedule"]
        schedule_dict[entry["year"]] = entry["trucks_added"]
    end

    return (
        initial_trucks = truck_config["initial_trucks"],
        schedule = schedule_dict
    )
end

# Trucks added in a given year, EXTRAPOLATING beyond the last authored schedule
# year by continuing the recent growth trend (a constant annual increment
# estimated from the final segment of the schedule). This lets demand exist a few
# years past the assessment horizon so that investment foresight — and the
# deployment loop itself when the horizon is extended — see a sensible
# truck-demand projection rather than an abrupt drop to zero.
#
# At or before the last authored year the authored value is returned unchanged,
# so a horizon equal to the schedule's last year reproduces the original results.
function scheduled_trucks_added(schedule::AbstractDict, year::Int)::Int
    isempty(schedule) && return 0
    last_year = maximum(keys(schedule))
    year <= last_year && return get(schedule, year, 0)
    # Extrapolate: constant annual increment from the final `lookback`-year segment.
    lookback = 3
    ref_year = last_year - lookback
    last_add = get(schedule, last_year, 0)
    ref_add  = get(schedule, ref_year, get(schedule, minimum(keys(schedule)), 0))
    slope    = (last_add - ref_add) / lookback
    return max(0, round(Int, last_add + slope * (year - last_year)))
end

# Total trucks deployed via the schedule (initial fleet excluded) from the first
# authored year through `through_year`, including extrapolated additions for any
# years beyond the schedule. Used to size the deployment array. For
# `through_year` equal to the schedule's last authored year this equals
# `sum(values(schedule))`, matching the original sizing.
function total_scheduled_additions(schedule::AbstractDict, through_year::Int)::Int
    isempty(schedule) && return 0
    lo = minimum(keys(schedule))
    total = 0
    for y in lo:through_year
        total += scheduled_trucks_added(schedule, y)
    end
    return total
end

"""
    load_h2_price_schedule(filepath)

Load predetermined H2 price schedule from JSON configuration file.

# Arguments
- `filepath`: Path to configuration JSON file

# Returns
- Dict mapping year => price
"""
function load_h2_price_schedule(filepath)
    config_data = JSON.parsefile(filepath)

    if !haskey(config_data, "h2_price_schedule")
        error("Configuration file missing 'h2_price_schedule' section")
    end

    h2_config = config_data["h2_price_schedule"]

    # Parse schedule into a dictionary mapping year => price
    schedule_dict = Dict{Int, Float64}()
    for entry in h2_config["schedule"]
        schedule_dict[entry["year"]] = entry["price_usd_per_kg"]
    end

    return schedule_dict
end

"""
    calculate_h2_price_from_curve(year_idx, n_years, start_price, end_price, curve_type)

Calculate H2 price for a given year based on start/end prices and curve type.

# Arguments
- `year_idx`: Current year index (1 = first year)
- `n_years`: Total number of years
- `start_price`: Starting H2 price (\$/kg)
- `end_price`: Ending H2 price (\$/kg)
- `curve_type`: Type of curve ("linear", "exponential", "logarithmic")

# Returns
- H2 price for the given year
"""
function calculate_h2_price_from_curve(year_idx, n_years, start_price, end_price, curve_type)
    # Normalize to [0, 1] scale
    progress = (year_idx - 1) / (n_years - 1)

    if curve_type == "linear"
        # Linear interpolation
        price = start_price + (end_price - start_price) * progress
    elseif curve_type == "exponential"
        # Exponential decay
        # price = start * (end/start)^progress
        ratio = end_price / start_price
        price = start_price * (ratio ^ progress)
    elseif curve_type == "logarithmic"
        # Logarithmic: fast initial decrease, slowing over time
        # Inverse of exponential
        price = end_price + (start_price - end_price) * (1 - progress) ^ 2
    else
        # Default to linear
        price = start_price + (end_price - start_price) * progress
    end

    return price
end

"""
    get_h2_base_price(year, year_idx, n_years, config, production_facilities, daily_demand_kg)

Get the H2 production/transport cost for a given year based on pricing mode.

# Arguments
- `year`: Current simulation year
- `year_idx`: Current year index (1 = first year)
- `n_years`: Total number of years in simulation
- `config`: Configuration NamedTuple
- `production_facilities`: Array of production facilities (for utilization-based pricing)
- `daily_demand_kg`: Daily truck demand in kg (for utilization-based pricing)

# Returns
- H2 base price (\$/kg) for production and transport
"""
function get_h2_base_price(year, year_idx, n_years, config, production_facilities=nothing, daily_demand_kg=0.0)
    if config.use_utilization_pricing
        # Check if production_facilities is valid
        if isnothing(production_facilities) || isempty(production_facilities)
            return (config.h2_production_transport_cost, 0.0)  # Fallback for no facilities
        end

        # Mode 4: Utilization-based pricing (SMR only)
        # Calculate total production capacity
        total_production_capacity = sum(f.capacity_kg_day for f in production_facilities)

        # Calculate total demand: trucks + bus + car demand
        other_demand_kg = bus_demand_kg_day(year, config.bus_demand_scenario) +
                          car_demand_kg_day(year, config.car_demand_scenario)
        total_demand_kg = daily_demand_kg + other_demand_kg

        # Calculate utilization
        if total_production_capacity > 0
            utilization = total_demand_kg / total_production_capacity
        else
            utilization = 0.0
        end

        # Check if the pathway is SMR technology
        # Get the primary pathway from facilities (use the most common or latest)
        is_smr = false
        if !isempty(production_facilities)
            # Check if any facility is SMR type
            for facility in production_facilities
                if facility.technology_type == "smr"
                    is_smr = true
                    break
                end
            end
        end

        # Calculate total annual demand for CAPEX spreading
        total_annual_h2_usage = total_demand_kg * 365.0  # Convert daily to annual

        if utilization > 0.0
            # Calculate SMR cost from utilization-based formula
            # METHODOLOGY REFERENCE: see the article Methods, "SMR Utilization-Based Pricing"
            # Empirical power law: C_SMR = 4.644 + 8.889 × (1 - u)²
            # At 30% utilization: \$9.00/kg; At 80% utilization: \$5.00/kg
            # Captures efficiency penalties of underutilized SMR plants
            smr_production_cost = 4.644 + 8.889 * (1.0 - utilization)^2

            # If electrolysis pricing enabled, blend SMR and electrolysis costs
            if config.electrolysis_pricing_enabled
                production_cost, credit_45v = calculate_blended_production_cost(
                    production_facilities,
                    smr_production_cost,
                    total_annual_h2_usage,
                    year,
                    config
                )
            else
                # Pure SMR pricing
                production_cost = smr_production_cost
                credit_45v = 0.0
            end

            return (production_cost + config.utilization_transport_cost, credit_45v)
        else
            # No utilization (early years or no facilities)
            if config.electrolysis_pricing_enabled
                production_cost, credit_45v = calculate_blended_production_cost(
                    production_facilities,
                    config.h2_production_transport_cost,  # Fallback SMR cost
                    total_annual_h2_usage,
                    year,
                    config
                )
                return (production_cost + config.utilization_transport_cost, credit_45v)
            else
                return (config.h2_production_transport_cost, 0.0)
            end
        end
    elseif config.use_h2_price_schedule
        # Mode 1: Use predetermined schedule
        if haskey(config.h2_price_schedule, year)
            return (config.h2_price_schedule[year], 0.0)
        else
            # Year not in schedule - use the last available price
            max_year = maximum(keys(config.h2_price_schedule))
            return (config.h2_price_schedule[max_year], 0.0)
        end
    elseif config.use_h2_price_curve
        # Mode 2: Calculate from curve
        return (calculate_h2_price_from_curve(
            year_idx,
            n_years,
            config.h2_price_curve_params.start_price,
            config.h2_price_curve_params.end_price,
            config.h2_price_curve_params.curve_type
        ), 0.0)
    else
        # Mode 3: Use fixed price
        return (config.h2_production_transport_cost, 0.0)
    end
end

#==============================================================================
PRODUCTION CAPACITY MODELING (Update 4)
==============================================================================#

"""
Production facility structure for tracking hydrogen production capacity.

# Fields
- `id`: Unique identifier for the facility
- `capacity_kg_day`: Daily production capacity (kg/day)
- `pathway`: Hydrogen production pathway ID (e.g., "electrolysis", "dairy_biomethane_smr")
- `ci`: Carbon intensity (gCO2e/MJ) from pathway
- `technology_type`: Technology type ("smr" or "electrolysis")
- `capex`: Capital expenditure (\$)
- `annual_om`: Annual operations & maintenance cost (\$/year)
- `lifetime`: Facility lifetime (years)
- `opening_year`: Year the facility opens
- `is_initial`: true for initial facility, false for expansion facilities
"""
struct ProductionFacility
    id::String
    capacity_kg_day::Float64
    pathway::String
    ci::Float64
    technology_type::String
    capex::Float64
    annual_om::Float64
    lifetime::Int
    opening_year::Int
    is_initial::Bool
end

# Fraction of hydrogen lost to leakage across the supply chain in a given year,
# applied to ELECTROLYTIC hydrogen only (SMR pricing already embeds its own losses).
#
# Default trajectory: 15% in 2026, decreasing linearly to 2% in 2035, held flat at
# 2% thereafter. Overridable via config fields (read defensively so configs that do
# not define them fall back to these defaults):
#   enable_h2_leakage          (Bool,  default true)
#   h2_leakage_start_fraction  (default 0.15)
#   h2_leakage_floor_fraction  (default 0.02)
#   h2_leakage_start_year      (default 2026)
#   h2_leakage_floor_year      (default 2035)
#
# To DELIVER 1 kg of electrolytic H2, 1/(1-L) kg must be PRODUCED; since every
# electrolytic cost component (electricity per kg, plus electrolyzer/solar/grid
# capacity) scales with production volume, the delivered $/kg is divided by (1-L).
function electrolysis_leakage_fraction(year, config)
    enabled = hasproperty(config, :enable_h2_leakage) ? config.enable_h2_leakage : true
    enabled || return 0.0

    start_frac = hasproperty(config, :h2_leakage_start_fraction) ? config.h2_leakage_start_fraction : 0.15
    floor_frac = hasproperty(config, :h2_leakage_floor_fraction) ? config.h2_leakage_floor_fraction : 0.02
    start_year = hasproperty(config, :h2_leakage_start_year)     ? config.h2_leakage_start_year     : 2026
    floor_year = hasproperty(config, :h2_leakage_floor_year)     ? config.h2_leakage_floor_year     : 2035

    year <= start_year && return start_frac
    year >= floor_year && return floor_frac
    return start_frac + (floor_frac - start_frac) * (year - start_year) / (floor_year - start_year)
end

"""
    calculate_blended_production_cost(production_facilities, smr_cost_per_kg,
                                      total_annual_demand_kg, year, config)

Calculate blended production cost for mixed SMR/electrolysis deployments.

METHODOLOGY REFERENCE: see the article Methods, "Technology Blending for Mixed Deployments"
Implements capacity-weighted cost blending:
    C_prod = (Q_SMR × C_SMR + Q_elec,45V × C_elec,45V + Q_elec,non × C_elec) / Q_total

This handles technology transition scenarios (e.g., initial SMR + later electrolysis expansion).
Each technology's cost is calculated independently, then averaged by production capacity.

45V Tax Credit Implementation (United States Inflation Reduction Act):
- Provides \$3/kg subsidy for clean hydrogen from electrolysis
- Eligibility requires ALL of:
    • clean electricity — grid-powered electrolysis carries CA-grid CO₂
      emissions and never qualifies; only solar (renewable) electrolysis can
    • commissioned 2026–`tax_credit_45v_end_year`
    • within the facility's first 10 operating years (credit then expires)
- Applied as: C_elec,45V = max(0, C_elec - 3.0)
- Creates temporal incentive window for accelerated electrolysis deployment

The function separates electrolysis facilities into:
1. 45V-eligible: receive \$3/kg subsidy
2. Non-eligible (grid-powered, commissioned outside the window, or past their
   first 10 operating years): no subsidy

# Arguments
- `production_facilities`: Vector of ProductionFacility
- `smr_cost_per_kg`: Pre-calculated SMR cost (\$/kg) from utilization formula
- `total_annual_demand_kg`: Total demand for spreading electrolysis CAPEX
- `year`: Current simulation year (for 45V eligibility check)
- `config`: Configuration with electrolysis parameters
"""
function calculate_blended_production_cost(
    production_facilities::Vector{ProductionFacility},
    smr_cost_per_kg::Float64,
    total_annual_demand_kg::Float64,
    year::Int,
    config
)
    smr_capacity = 0.0
    electrolysis_45v_eligible_capacity_kw = 0.0
    electrolysis_non_eligible_capacity_kw = 0.0
    electrolysis_facility_lifetime = 20  # Default, will use first facility's lifetime if available

    # For electrolysis, capacity depends on operating hours:
    # - Grid-powered operates 24 hrs/day
    # - Solar-powered operates only during daylight (e.g., 10 hrs/day)
    electrolysis_operating_hours = config.electrolysis_electricity_source == "solar" ? config.solar_operating_hours : 24.0

    # CAPEX VINTAGING (technology-cost year for each facility's capital).
    # By default a facility keeps the technology cost of the year it opened, so
    # the fleet-average CAPEX below is a mix of vintages that lags the learning
    # curve. Setting `config.capex_vintage = false` values ALL capacity at the
    # CURRENT year's technology cost instead — the no-vintage counterfactual
    # plotted by figures/fig_vintage_effect.jl. Read defensively so configs that
    # predate the flag (e.g. the model server's) keep the vintaged default.
    # Only the CAPEX year moves; 45V eligibility below still keys off the real
    # commissioning year, since that is policy rather than technology cost.
    use_capex_vintage = hasproperty(config, :capex_vintage) ? config.capex_vintage : true
    capex_year(facility) = use_capex_vintage ? facility.opening_year : year

    # 45V eligibility predicate (IRA clean-hydrogen production tax credit):
    #   • feature flag enabled
    #   • clean electricity only — grid-powered electrolysis carries CA-grid CO₂
    #     emissions (133.6 gCO2e/MJ) and therefore never qualifies
    #   • commissioned within the 2026–tax_credit_45v_end_year window
    #   • credit runs for the facility's first 10 operating years only
    is_45v_eligible(facility) = config.enable_45v &&
        config.electrolysis_electricity_source != "grid" &&
        (facility.opening_year >= 2026) &&
        (facility.opening_year <= config.tax_credit_45v_end_year) &&
        ((year - facility.opening_year) < 10)

    # Separate facilities by type and 45V eligibility
    for facility in production_facilities
        if facility.technology_type == "smr"
            smr_capacity += facility.capacity_kg_day
        elseif facility.technology_type == "electrolysis"
            electrolysis_facility_lifetime = facility.lifetime
            capacity_kw = calculate_electrolyzer_capacity_kw(facility.capacity_kg_day, electrolysis_operating_hours)

            if is_45v_eligible(facility)
                electrolysis_45v_eligible_capacity_kw += capacity_kw
            else
                electrolysis_non_eligible_capacity_kw += capacity_kw
            end
        end
    end

    total_electrolysis_capacity_kw = electrolysis_45v_eligible_capacity_kw + electrolysis_non_eligible_capacity_kw
    total_capacity = smr_capacity + sum(f.capacity_kg_day for f in production_facilities if f.technology_type == "electrolysis"; init=0.0)

    if total_capacity == 0.0
        return (config.h2_production_transport_cost, 0.0)  # Fallback
    end

    # Electrolysis facilities only serve their share of total demand.
    # Spreading CAPEX over total demand would artificially reduce cost per kg in mixed deployments.
    electrolysis_capacity_kg_day = sum(f.capacity_kg_day for f in production_facilities if f.technology_type == "electrolysis"; init=0.0)
    electrolysis_annual_demand_kg = electrolysis_capacity_kg_day / total_capacity * total_annual_demand_kg

    # Compute capacity-weighted effective CAPEX per kW using learning curve
    effective_electrolyzer_capex_per_kw = if total_electrolysis_capacity_kw > 0.0
        total_weighted_capex = 0.0
        for facility in production_facilities
            if facility.technology_type == "electrolysis"
                cap_kw = calculate_electrolyzer_capacity_kw(facility.capacity_kg_day, electrolysis_operating_hours)
                year_capex = electrolyzer_capex_for_year(capex_year(facility), config.electrolyzer_capex_per_kw, config.electrolyzer_stack_fraction, config.electrolyzer_learning_rate, config.bop_learning_rate)
                total_weighted_capex += cap_kw * year_capex
            end
        end
        total_weighted_capex / total_electrolysis_capacity_kw
    else
        config.electrolyzer_capex_per_kw
    end

    # Calculate base electrolysis cost (before 45V subsidy)
    base_electrolysis_cost_per_kg = 0.0

    if total_electrolysis_capacity_kw > 0.0
        if config.electrolysis_electricity_source == "grid"
            base_electrolysis_cost_per_kg = calculate_electrolysis_cost_grid(
                total_electrolysis_capacity_kw,
                electrolysis_annual_demand_kg,
                effective_electrolyzer_capex_per_kw,
                config.electricity_cost_per_kwh,
                config.grid_connection_cost_per_kw,
                config.discount_rate,
                electrolysis_facility_lifetime
            )
        elseif config.electrolysis_electricity_source == "solar"
            # Compute capacity-weighted effective solar CAPEX using learning curve
            effective_solar_capex_per_kw = if total_electrolysis_capacity_kw > 0.0
                total_weighted_solar = 0.0
                for facility in production_facilities
                    if facility.technology_type == "electrolysis"
                        cap_kw = calculate_electrolyzer_capacity_kw(facility.capacity_kg_day, electrolysis_operating_hours)
                        year_solar = solar_capex_for_year(capex_year(facility), config.solar_capex_per_kw, config.solar_panel_fraction, config.solar_panel_learning_rate, config.solar_bop_learning_rate)
                        total_weighted_solar += cap_kw * year_solar
                    end
                end
                total_weighted_solar / total_electrolysis_capacity_kw
            else
                config.solar_capex_per_kw
            end

            base_electrolysis_cost_per_kg = calculate_electrolysis_cost_solar(
                total_electrolysis_capacity_kw,
                electrolysis_annual_demand_kg,
                effective_electrolyzer_capex_per_kw,
                effective_solar_capex_per_kw,
                config.solar_capacity_factor,
                config.solar_operating_hours,
                config.discount_rate,
                electrolysis_facility_lifetime,
                config.solar_lifetime
            )
        else
            base_electrolysis_cost_per_kg = config.h2_production_transport_cost
        end
    end

    # Hydrogen leakage: electrolytic H₂ must over-produce to compensate for
    # downstream losses (15% in 2026 → 2% in 2035, then held). SMR price already
    # embeds its own losses, so this multiplier is applied to electrolysis only,
    # and BEFORE the 45V subsidy (the $3/kg credit stays fixed per delivered kg).
    leak = electrolysis_leakage_fraction(year, config)
    if leak > 0.0
        base_electrolysis_cost_per_kg /= (1.0 - leak)
    end

    # Apply 45V subsidy to eligible facilities ($3/kg)
    tax_credit_45v = 3.0  # USD per kg
    electrolysis_45v_cost_per_kg = max(0.0, base_electrolysis_cost_per_kg - tax_credit_45v)
    electrolysis_non_eligible_cost_per_kg = base_electrolysis_cost_per_kg

    # Get production capacities by group
    electrolysis_45v_production_capacity = sum(f.capacity_kg_day for f in production_facilities
        if f.technology_type == "electrolysis" && is_45v_eligible(f); init=0.0)
    electrolysis_non_eligible_production_capacity = sum(f.capacity_kg_day for f in production_facilities
        if f.technology_type == "electrolysis" && !is_45v_eligible(f); init=0.0)

    # Blend costs by capacity-weighted average
    if total_capacity > 0.0
        weighted_cost = (
            smr_capacity * smr_cost_per_kg +
            electrolysis_45v_production_capacity * electrolysis_45v_cost_per_kg +
            electrolysis_non_eligible_production_capacity * electrolysis_non_eligible_cost_per_kg
        ) / total_capacity
        # 45V credit per kg of total output (actual credit applied, capped at cost)
        actual_credit_per_elec_kg = base_electrolysis_cost_per_kg - electrolysis_45v_cost_per_kg
        credit_45v_per_kg = electrolysis_45v_production_capacity * actual_credit_per_elec_kg / total_capacity
        return (weighted_cost, credit_45v_per_kg)
    else
        return (config.h2_production_transport_cost, 0.0)
    end
end

"""
    effective_pathway_ci(pathway, lcfs_config, electricity_source)

Carbon intensity (gCO2e/MJ) a facility on `pathway` actually carries.

For electrolysis the CI follows the electricity source rather than the pathway
record: grid-powered electrolysis uses `lcfs_config.electrolysis_grid_ci`
(high-carbon CA grid), while solar electrolysis keeps the pathway value
(0 for the renewable pathway). All non-electrolysis pathways use `pathway.ci`.
"""
function effective_pathway_ci(pathway, lcfs_config, electricity_source::String)
    if pathway.technology_type == "electrolysis" && electricity_source == "grid"
        return lcfs_config.electrolysis_grid_ci
    end
    return pathway.ci
end

"""
    create_initial_production(pathway_id, lcfs_config, initial_prod_config, start_year, electricity_source)

Create the initial production facility (60 tpd default).

# Arguments
- `pathway_id`: Hydrogen production pathway ID for initial facility
- `lcfs_config`: LCFS configuration NamedTuple
- `initial_prod_config`: Initial production configuration from lcfs_config
- `start_year`: Simulation start year
- `electricity_source`: "grid" or "solar"; selects the effective electrolysis CI

# Returns
- ProductionFacility for the initial 60 tpd facility
"""
function create_initial_production(pathway_id::String, lcfs_config, initial_prod_config, start_year::Int, electricity_source::String)
    pathway = lcfs_config.h2_pathways[pathway_id]
    capacity_tpd = initial_prod_config.capacity_tpd
    capacity_kg_day = capacity_tpd * 1000.0  # Convert tons to kg

    # Get facility costs based on technology type
    fac_costs = lcfs_config.production_facilities[Symbol(pathway.technology_type)]

    # Use configured CAPEX or default to sunk cost (0)
    capex = initial_prod_config.capex_usd
    annual_om = capex * fac_costs.annual_om_fraction

    return ProductionFacility(
        "initial_production",
        capacity_kg_day,
        pathway_id,
        effective_pathway_ci(pathway, lcfs_config, electricity_source),
        pathway.technology_type,
        capex,
        annual_om,
        fac_costs.lifetime_years,
        start_year,
        true
    )
end

"""
    check_and_expand_production!(facilities, pending_production, foresight_demand_kg,
                                  current_year, foresight_year, expansion_pathway, lcfs_config,
                                  trigger_threshold, p_invest)

Check if production capacity expansion is needed based on foresight demand and, if so,
stochastically commit new facilities to the pending list with a construction lead time.

METHODOLOGY REFERENCE: see the article Methods, "Capacity Expansion Logic"

Expansion logic:
1. Compute capacity available at foresight_year: active facilities + pending that open by then.
2. If foresight_demand / foresight_capacity >= trigger_threshold, signal a capacity gap.
3. Draw Bernoulli(p_invest): if success, commit new facilities opening at current_year + rand(2:4).
4. New facilities go into pending_production (not active yet) until their open_year arrives.

Both trigger_threshold and p_invest are drawn once per Monte Carlo run, representing a
consistent investor type (risk appetite, capital availability) across the simulation.

# Arguments
- `facilities`: Active production facilities (read-only here)
- `pending_production`: Mutable list of (facility, open_year) pairs; new decisions appended here
- `foresight_demand_kg`: Expected daily H2 demand at foresight_year (kg/day)
- `current_year`: Current simulation year
- `foresight_year`: Year used for foresight capacity and demand check
- `expansion_pathway`: Pathway ID for new facilities
- `lcfs_config`: LCFS configuration NamedTuple
- `trigger_threshold`: Utilization level at which a capacity gap is signalled (drawn per MC run)
- `p_invest`: Probability of committing to investment when gap is signalled (drawn per MC run)
- `electricity_source`: "grid" or "solar"; selects the effective electrolysis CI for new facilities
"""
function check_and_expand_production!(
    facilities::Vector{ProductionFacility},
    pending_production::Vector{Tuple{ProductionFacility, Int}},
    foresight_demand_kg::Float64,
    current_year::Int,
    foresight_year::Int,
    expansion_pathway::String,
    lcfs_config,
    trigger_threshold::Float64,
    p_invest::Float64,
    electricity_source::String,
)
    if foresight_demand_kg <= 0.0
        return
    end

    # Capacity available at foresight_year: active + ALL committed pending
    # (investor knows about everything already committed to build, regardless of open year)
    active_capacity  = sum(f.capacity_kg_day for f in facilities; init = 0.0)
    pending_capacity = sum(f.capacity_kg_day for (f, _) in pending_production; init = 0.0)
    total_foresight_capacity = active_capacity + pending_capacity

    if total_foresight_capacity == 0.0
        return
    end

    expected_utilization = foresight_demand_kg / total_foresight_capacity
    if expected_utilization < trigger_threshold
        return
    end

    # Capacity gap signalled — stochastic investment decision
    rand() < p_invest || return

    required_capacity = foresight_demand_kg / trigger_threshold
    deficit = required_capacity - total_foresight_capacity
    deficit <= 0.0 && return

    pathway   = lcfs_config.h2_pathways[expansion_pathway]
    tech_type = pathway.technology_type
    fac_costs = lcfs_config.production_facilities[Symbol(tech_type)]
    facility_size = fac_costs.capacity_kg_day
    n_new = Int(ceil(deficit / facility_size))

    for i in 1:n_new
        open_year = current_year + rand(2:4)   # individual lead time per facility
        fac = ProductionFacility(
            "expansion_$(current_year)_$(i)",
            facility_size,
            expansion_pathway,
            effective_pathway_ci(pathway, lcfs_config, electricity_source),
            tech_type,
            fac_costs.capex_usd,
            fac_costs.capex_usd * fac_costs.annual_om_fraction,
            fac_costs.lifetime_years,
            open_year,
            false,
        )
        push!(pending_production, (fac, open_year))
    end
end

# Effective build trigger, adjusted for market maturity.
#
# In a fast-growing, uncertain market operators keep a larger capacity buffer:
# they build well ahead of demand, which keeps the effective trigger (and hence
# realized utilization) low. As demand growth decelerates and the market becomes
# predictable, they build closer to demand and the trigger creeps up toward a cap.
#
# The maturity signal is the implied ANNUAL demand growth over the foresight
# window (foresight_demand / current_demand, annualized). It is fully endogenous,
# so investors "slow down" whenever growth tails off — around 2040 in the current
# scenarios, but it adapts to any demand trajectory rather than keying off a year.
#
# growth >= trigger_growth_hi  → no lift (cautious, big buffer)
# growth <= trigger_growth_lo  → full lift (mature, build tight)
function maturity_adjusted_trigger(base_trigger, current_demand, foresight_demand,
                                   foresight_years, config)
    (hasproperty(config, :adaptive_trigger) ? config.adaptive_trigger : true) || return base_trigger
    (current_demand <= 0.0 || foresight_demand <= 0.0 || foresight_years <= 0) && return base_trigger
    g_hi = hasproperty(config, :trigger_growth_hi)     ? config.trigger_growth_hi     : 0.25
    g_lo = hasproperty(config, :trigger_growth_lo)     ? config.trigger_growth_lo     : 0.05
    lift = hasproperty(config, :trigger_maturity_lift) ? config.trigger_maturity_lift : 0.20
    cap  = hasproperty(config, :trigger_cap)           ? config.trigger_cap           : 0.85
    growth   = (foresight_demand / current_demand)^(1.0 / foresight_years) - 1.0
    maturity = clamp((g_hi - growth) / (g_hi - g_lo), 0.0, 1.0)  # 0 fast-growing, 1 mature
    return clamp(base_trigger + maturity * lift, 0.0, cap)
end

"""
    calculate_weighted_ci(facilities)

Calculate capacity-weighted average carbon intensity across all production facilities.

# Arguments
- `facilities`: Vector of ProductionFacility

# Returns
- Weighted average CI (gCO2e/MJ)
"""
function calculate_weighted_ci(facilities::Vector{ProductionFacility})::Float64
    total_capacity = sum(f.capacity_kg_day for f in facilities)

    if total_capacity == 0.0
        return 0.0
    end

    weighted_ci = sum(f.capacity_kg_day * f.ci for f in facilities) / total_capacity
    return weighted_ci
end

"""
    calculate_production_cost_per_kg(facilities, total_annual_demand_kg, discount_rate)

Calculate endogenous production cost per kg based on facility CAPEX and O&M.

# Arguments
- `facilities`: Vector of ProductionFacility
- `total_annual_demand_kg`: Total annual hydrogen demand (kg/year)
- `discount_rate`: Discount rate for annuitization

# Returns
- Production cost (\$/kg)
"""
function calculate_production_cost_per_kg(facilities::Vector{ProductionFacility},
                                         total_annual_demand_kg::Float64,
                                         discount_rate::Float64)::Float64
    if total_annual_demand_kg == 0.0
        return 0.0
    end

    total_annualized_cost = 0.0
    for f in facilities
        # Annualize CAPEX using standard formula
        annuity_factor = calculate_annuity_factor(f.lifetime, discount_rate)
        annualized_capex = f.capex * annuity_factor
        total_annualized_cost += annualized_capex + f.annual_om
    end

    return total_annualized_cost / total_annual_demand_kg
end

"""
    load_lcfs_config(filename)

Load LCFS (Low Carbon Fuel Standard) configuration from JSON file.

# Arguments
- `filename`: Path to LCFS configuration JSON file

# Returns
- NamedTuple containing:
  - eei: Energy Economy Improvement factor
  - mj_per_kg_h2: MJ energy content per kg of H2
  - credit_price: LCFS credit price (\$/credit)
  - diesel_ci_schedule: Dict mapping year => diesel CI (gCO2e/MJ)
  - h2_pathways: Dict mapping pathway ID => (name, CI)
"""
function load_lcfs_config(filename)
    try
        data = JSON.parsefile(filename)

        # Parse LCFS parameters
        params = data["lcfs_parameters"]
        eei = Float64(params["eei"])
        mj_per_kg_h2 = Float64(params["mj_per_kg_h2"])
        credit_price = Float64(params["default_credit_price_usd"])

        # Parse diesel CI schedule
        diesel_schedule = data["diesel_ci_schedule"]["schedule"]
        diesel_ci_dict = Dict{Int, Float64}()
        for entry in diesel_schedule
            year = Int(entry["year"])
            ci = Float64(entry["ci"])
            diesel_ci_dict[year] = ci
        end

        # Parse hydrogen pathways
        pathways_data = data["hydrogen_pathways"]["pathways"]
        h2_pathways = Dict{String, NamedTuple{(:name, :ci, :description, :technology_type), Tuple{String, Float64, String, String}}}()
        for pathway in pathways_data
            id = pathway["id"]
            name = pathway["name"]
            ci = Float64(pathway["ci"])
            description = pathway["description"]
            tech_type = String(pathway["technology_type"])
            h2_pathways[id] = (name = name, ci = ci, description = description, technology_type = tech_type)
        end

        # Parse LCFS price schedule (optional)
        lcfs_price_schedule_dict = nothing
        if haskey(data, "lcfs_price_schedule") && haskey(data["lcfs_price_schedule"], "schedule")
            lcfs_price_schedule_dict = Dict{Int, Float64}()
            for entry in data["lcfs_price_schedule"]["schedule"]
                year = Int(entry["year"])
                price = Float64(entry["price_usd"])
                lcfs_price_schedule_dict[year] = price
            end
        end

        # Parse LCFS price curve (optional)
        lcfs_price_curve = nothing
        if haskey(data, "lcfs_price_curve")
            curve_data = data["lcfs_price_curve"]
            lcfs_price_curve = (
                start_price = Float64(curve_data["start_price_usd"]),
                end_price = Float64(curve_data["end_price_usd"]),
                curve_type = String(curve_data["curve_type"])
            )
        end

        # Parse production facilities (optional - for Update 4)
        production_facilities = nothing
        if haskey(data, "production_facilities")
            fac_data = data["production_facilities"]
            production_facilities = (
                smr = (
                    capacity_kg_day = Float64(fac_data["smr"]["capacity_kg_day"]),
                    capex_usd = Float64(fac_data["smr"]["capex_usd"]),
                    annual_om_fraction = Float64(fac_data["smr"]["annual_om_fraction"]),
                    lifetime_years = Int(fac_data["smr"]["lifetime_years"])
                ),
                electrolysis = (
                    capacity_kg_day = Float64(fac_data["electrolysis"]["capacity_kg_day"]),
                    capex_usd = Float64(fac_data["electrolysis"]["capex_usd"]),
                    annual_om_fraction = Float64(fac_data["electrolysis"]["annual_om_fraction"]),
                    lifetime_years = Int(fac_data["electrolysis"]["lifetime_years"])
                )
            )
        end

        # Grid-electrolysis carbon intensity (gCO2e/MJ). Applied when
        # electrolysis_electricity_source == "grid"; solar electrolysis keeps the
        # pathway CI (0 for the renewable pathway). Defaults to 0 if absent.
        electrolysis_grid_ci = 0.0
        if haskey(data, "electrolysis_grid_ci")
            grid_ci_entry = data["electrolysis_grid_ci"]
            electrolysis_grid_ci = grid_ci_entry isa Number ?
                Float64(grid_ci_entry) : Float64(grid_ci_entry["ci"])
        end

        # Parse initial production (optional - for Update 4)
        initial_production = nothing
        if haskey(data, "initial_production")
            init_data = data["initial_production"]
            initial_production = (
                capacity_tpd = Float64(init_data["capacity_tpd"]),
                capex_usd = Float64(init_data["capex_usd"])
            )
        end

        return (
            eei = eei,
            mj_per_kg_h2 = mj_per_kg_h2,
            credit_price = credit_price,
            diesel_ci_schedule = diesel_ci_dict,
            h2_pathways = h2_pathways,
            lcfs_price_schedule = lcfs_price_schedule_dict,
            lcfs_price_curve = lcfs_price_curve,
            production_facilities = production_facilities,
            initial_production = initial_production,
            electrolysis_grid_ci = electrolysis_grid_ci
        )
    catch e
        error("Failed to load LCFS configuration from $filename: $e")
    end
end

"""
    get_diesel_ci(year, lcfs_config)

Get diesel carbon intensity for a given year.

# Arguments
- `year`: The year to get diesel CI for
- `lcfs_config`: LCFS configuration NamedTuple

# Returns
- Diesel CI in gCO2e/MJ for the given year (uses last available year if beyond schedule)
"""
function get_diesel_ci(year, lcfs_config)
    diesel_schedule = lcfs_config.diesel_ci_schedule

    if haskey(diesel_schedule, year)
        return diesel_schedule[year]
    else
        # Use last available year's CI if beyond schedule
        max_year = maximum(keys(diesel_schedule))
        if year > max_year
            return diesel_schedule[max_year]
        else
            # Use first available year's CI if before schedule
            min_year = minimum(keys(diesel_schedule))
            return diesel_schedule[min_year]
        end
    end
end

"""
    get_lcfs_credit_price(year, year_idx, n_years, config)

Get LCFS credit price for a given year based on configuration mode.

# Arguments
- `year`: Current year
- `year_idx`: Index of current year (1-based)
- `n_years`: Total number of years in simulation
- `config`: Configuration NamedTuple

# Returns
- LCFS credit price in \$/credit for the given year

# Modes
1. If use_lcfs_price_schedule = true: Use predetermined schedule from config file
2. If use_lcfs_price_curve = true: Calculate price using curve (linear, exponential, logarithmic)
3. Otherwise: Use fixed default credit price from LCFS config
"""
function get_lcfs_credit_price(year, year_idx, n_years, config)
    if config.use_lcfs_price_schedule
        # Mode 1: Use predetermined schedule from config file
        schedule = config.lcfs_config.lcfs_price_schedule
        if haskey(schedule, year)
            return schedule[year]
        else
            # Use last available year's price if beyond schedule
            max_year = maximum(keys(schedule))
            if year > max_year
                return schedule[max_year]
            else
                # Use first available year's price if before schedule
                min_year = minimum(keys(schedule))
                return schedule[min_year]
            end
        end
    elseif config.use_lcfs_price_curve
        # Mode 2: Use dynamic curve (linear, exponential, or logarithmic)
        curve_params = config.lcfs_price_curve_params
        return calculate_h2_price_from_curve(
            year_idx,
            n_years,
            curve_params.start_price,
            curve_params.end_price,
            curve_params.curve_type
        )
    else
        # Mode 3: Use fixed default price
        return config.lcfs_config.credit_price
    end
end

"""
    calculate_lcfs_credit(year, year_idx, n_years, h2_pathway_id, config; weighted_ci=nothing)

Calculate LCFS (Low Carbon Fuel Standard) credit or deficit per kg of H2 sold.

METHODOLOGY REFERENCE: see the article Methods, "Low Carbon Fuel Standard (LCFS) Credits"
Implements the California LCFS credit formula:
    LCFS_credit = (CI_diesel - CI_H2/EEI) × MJ_H2/kg / 1,000,000 × P_credit

Where:
- CI_diesel = diesel baseline carbon intensity (gCO2e/MJ), year-dependent schedule
  (CARB schedule: 94.71 gCO2e/MJ in 2024 → 10.57 gCO2e/MJ in 2045)
- CI_H2 = hydrogen pathway carbon intensity (gCO2e/MJ)
- EEI = Energy Economy Improvement factor = 1.9 (H2 vs diesel efficiency advantage)
- MJ_H2/kg = 120 MJ/kg (lower heating value of hydrogen)
- P_credit = LCFS credit price (USD/credit), user-configurable by year

Formula explanation:
1. CI_H2/EEI adjusts H2 carbon intensity for efficiency advantage over diesel
2. (CI_diesel - CI_H2/EEI) gives carbon intensity reduction vs diesel baseline
3. Multiply by MJ_H2/kg and divide by 1,000,000 to get metric tons CO2e per kg H2
4. Multiply by credit price to get USD per kg H2

Interpretation:
- Positive value = credit income (reduces H2 price to consumers)
- Negative value = credit deficit (increases H2 price)
- Very low-carbon pathways (e.g., dairy biomethane-SMR at -300 gCO2e/MJ)
  can generate substantial credits (>\$10/kg)

# Arguments
- `year`: Current year
- `year_idx`: Index of current year (1-based)
- `n_years`: Total number of years in simulation
- `h2_pathway_id`: Hydrogen production pathway ID (e.g., "electrolysis", "natural_gas_smr_gaseous")
- `config`: Configuration NamedTuple
- `weighted_ci`: Optional weighted CI from production facilities (for mixed deployments).
                If provided, overrides pathway CI to account for blended production.

# Returns
- LCFS credit value in \$/kg H2 (positive = income, negative = deficit)
"""
function calculate_lcfs_credit(year, year_idx, n_years, h2_pathway_id, config; weighted_ci=nothing)
    lcfs_config = config.lcfs_config

    # Get diesel CI for this year
    diesel_ci = get_diesel_ci(year, lcfs_config)

    # Get H2 CI - use weighted CI if provided (Update 4), otherwise use pathway CI
    if weighted_ci !== nothing
        h2_ci = weighted_ci
    else
        # Get H2 pathway CI (backward compatible)
        if !haskey(lcfs_config.h2_pathways, h2_pathway_id)
            error("Unknown hydrogen pathway: $h2_pathway_id")
        end
        h2_ci = lcfs_config.h2_pathways[h2_pathway_id].ci
    end

    # Get LCFS credit price for this year (dynamic based on configuration)
    credit_price = get_lcfs_credit_price(year, year_idx, n_years, config)

    # Calculate credit value
    # (Diesel_CI - H2_CI/EEI) × MJ_per_kg_H2 / 1,000,000 × Credit_Price
    credit_value_per_kg = (diesel_ci - h2_ci / lcfs_config.eei) *
                          lcfs_config.mj_per_kg_h2 / 1_000_000.0 *
                          credit_price

    return credit_value_per_kg
end

"""
    calculate_station_opening_probability(current_year, planned_year, station_status, prob_config)

Calculate probability that a station opens in the current year given its planned opening year and status.

# Arguments
- `current_year`: Current simulation year
- `planned_year`: Year station was planned to open
- `station_status`: Station project status (operational, under construction, FID, project development, in planning)
- `prob_config`: Configuration NamedTuple with probability parameters

# Returns
- Probability between 0 and 1, or 0 if beyond max delay
"""
function calculate_station_opening_probability(current_year, planned_year, station_status, prob_config)
    years_since_planned = current_year - planned_year

    # Not yet planned to open
    if years_since_planned < 0
        return 0.0
    end

    # Beyond maximum delay - station project abandoned
    if years_since_planned > prob_config.max_delay_years
        return 0.0
    end

    # Get probability based on station status
    if !isnothing(prob_config.probability_by_status)
        # New format: use status-specific probability
        status_key = lowercase(station_status)
        return get(prob_config.probability_by_status, status_key, 0.3)  # Default to 30% if status not found
    else
        # Old format: backward compatibility
        if years_since_planned == 0
            return prob_config.prob_in_planned_year
        else
            return prob_config.prob_in_subsequent_years
        end
    end
end

#==============================================================================
SIMULATION FUNCTIONS
==============================================================================#

"""
    run_single_simulation(config)

Run a single Monte Carlo simulation of hydrogen truck rollout.

METHODOLOGY REFERENCE: see the article Methods, "Monte Carlo Simulation Framework"
This function implements one complete simulation run. Monte Carlo ensemble runs this
function N times (typically 100-1000) with probabilistic variation in deployment timing.

Simulation algorithm for each year:
1. Evaluate station deployment probabilities (status-based)
2. Deploy trucks (schedule or probabilistic function)
3. Retire trucks exceeding 7-year lifetime
4. Calculate hydrogen demand = trucks × consumption × uptime
5. Check production capacity; expand if utilization > 60%
6. Calculate station utilization = demand / capacity
7. Compute cost components (production, station CAPEX/O&M)
8. Calculate LCFS and HRI credits
9. Compute final levelized H2 price: LCOH = production + infra - credits

Key stochastic elements:
- Station commissioning delays (probability varies by project status)
- Truck adoption variation (probabilistic mode only; schedule mode is deterministic)

Deterministic elements (no Monte Carlo variation):
- Costs (CAPEX, O&M, electricity prices)
- Efficiencies (truck consumption, electrolyzer efficiency)
- Policy parameters (LCFS formulas, credit prices)
- Physical constraints (capacity limits, lifetimes)

The simulation starts with existing infrastructure (initial stations/trucks already deployed),
then stochastically deploys additional infrastructure following configured probabilities.

# Arguments
- `config`: NamedTuple containing all simulation parameters

# Returns
- Tuple of (h2_prices, n_stations, n_trucks, station_capacity, h2_base_prices, infrastructure_costs, lcfs_credits, warnings, production_capacity, production_facilities_count, weighted_ci, production_cost, hri_credits, station_utilization):
  - h2_prices: Array of total hydrogen prices (\$/kg) for each year
  - n_stations: Array of number of deployed stations for each year
  - n_trucks: Array of number of deployed trucks for each year
  - station_capacity: Array of total station capacity (kg/day) for each year
  - h2_base_prices: Array of base H2 production/transport costs (\$/kg) for each year
  - infrastructure_costs: Array of station infrastructure costs (\$/kg) for each year
  - lcfs_credits: Array of LCFS credits (\$/kg) for each year (positive = income/reduction)
  - warnings: Array of warning messages
  - production_capacity: Array of total production capacity (kg/day) for each year
  - production_facilities_count: Array of number of production facilities for each year
  - weighted_ci: Array of capacity-weighted average CI (gCO2e/MJ) for each year
  - production_cost: Array of endogenous production cost (\$/kg) for each year, or 0 if exogenous mode
  - hri_credits: Array of HRI credits (\$/kg) for each year
  - station_utilization: Array of station utilization (fraction) for each year
"""
function run_single_simulation(config; event_sink=nothing)
    n_years = config.end_year - config.start_year + 1
    years = config.start_year:config.end_year

    # Arrays to store results
    h2_prices = zeros(Float64, n_years)
    n_stations_per_year = zeros(Int, n_years)
    n_trucks_per_year = zeros(Int, n_years)
    station_capacity_per_year = zeros(Float64, n_years)  # Total daily H2 capacity (kg/day)
    h2_base_prices = zeros(Float64, n_years)  # Base H2 production/transport cost per year
    infrastructure_costs = zeros(Float64, n_years)  # Station infrastructure CAPEX cost per kg per year (company portion)
    infrastructure_costs_government = zeros(Float64, n_years)  # Station infrastructure CAPEX cost per kg per year (government-funded portion)
    infrastructure_costs_om = zeros(Float64, n_years)  # Station O&M cost per kg per year
    lcfs_credits = zeros(Float64, n_years)  # LCFS credit income per kg per year
    warnings = String[]  # Track warnings during simulation

    # Production facility tracking arrays (Update 4)
    production_capacity_per_year = zeros(Float64, n_years)  # Total production capacity (kg/day)
    production_facilities_count_per_year = zeros(Int, n_years)  # Number of production facilities
    smr_facilities_count_per_year = zeros(Int, n_years)  # Number of SMR facilities
    electrolysis_facilities_count_per_year = zeros(Int, n_years)  # Number of electrolysis facilities
    weighted_ci_per_year = zeros(Float64, n_years)  # Capacity-weighted average CI
    production_cost_per_year = zeros(Float64, n_years)  # Production cost ($/kg) if endogenous mode

    # 45V tax credit tracking
    credits_45v = zeros(Float64, n_years)  # 45V tax credit per kg per year

    # HRI credit tracking (Update 1)
    hri_credits = zeros(Float64, n_years)  # HRI credit income per kg per year
    station_utilization = zeros(Float64, n_years)  # Station utilization (fraction) per year
    # Effective (market-maturity-adjusted) build trigger θ_eff per year, fraction.
    # Deterministic given the demand path; NaN in years/modes where it is not evaluated.
    eff_trigger_prod_per_year    = fill(NaN, n_years)  # production θ_eff
    eff_trigger_station_per_year = fill(NaN, n_years)  # refuelling-station θ_eff

    # Calculate annuity factor once (constant over simulation)
    annuity_factor = calculate_annuity_factor(config.station_lifetime, config.discount_rate)

    # Branch based on station deployment mode
    if config.use_station_config_file
        # MODE 1: Station configuration file mode
        # Load station data and track which stations are deployed
        stations_deployed = falses(length(config.station_data))

        # Mark already operational stations as deployed
        for (i, station) in enumerate(config.station_data)
            if station.already_operational
                stations_deployed[i] = true
            end
        end

        # Initialize truck deployment
        truck_midpoint_this_run = clamp(
            config.truck_prob_midpoint + randn() * config.truck_prob_midpoint_std,
            0.0, 1.0
        )

        # Determine effective max trucks: if using schedule, calculate max from schedule; otherwise use MAX_TRUCKS
        effective_max_trucks = if config.use_truck_deployment_schedule
            # Total trucks from schedule (initial fleet + additions through the
            # horizon, including extrapolated additions beyond the authored years)
            config.truck_deployment_schedule.initial_trucks +
                total_scheduled_additions(config.truck_deployment_schedule.schedule, config.end_year)
        else
            config.max_trucks
        end

        # Track truck deployment years (0 = not deployed, >0 = year deployed)
        truck_deployment_years = zeros(Int, effective_max_trucks)

        # Use initial trucks from schedule if using predetermined mode, otherwise use config
        initial_trucks_count = if config.use_truck_deployment_schedule
            config.truck_deployment_schedule.initial_trucks
        else
            config.initial_trucks
        end
        if initial_trucks_count > 0
            # Mark initial trucks as deployed in the start year
            truck_deployment_years[1:min(initial_trucks_count, effective_max_trucks)] .= config.start_year
        end

        # Initialize production facilities (Update 4)
        # ALWAYS track production facilities regardless of pricing mode
        production_facilities = ProductionFacility[]
        initial_facility = create_initial_production(
            config.h2_pathway_id,
            config.lcfs_config,
            config.lcfs_config.initial_production,
            config.start_year,
            config.electrolysis_electricity_source
        )
        push!(production_facilities, initial_facility)
        if !isnothing(event_sink)
            push!(event_sink, (kind = :production, id = initial_facility.id,
                               capacity = initial_facility.capacity_kg_day,
                               year = initial_facility.opening_year,
                               tech = initial_facility.technology_type,
                               lifetime = initial_facility.lifetime,
                               is_initial = initial_facility.is_initial))
        end

        # Determine expansion pathway (Issue 2)
        # Use expansion_pathway_id if specified, otherwise same as initial pathway
        expansion_pathway = isnothing(config.expansion_pathway_id) ? config.h2_pathway_id : config.expansion_pathway_id

        # Station projection setup (Update 3)
        # Use user-specified projection start year, or calculate from station data
        if !isnothing(config.projection_start_year)
            last_user_station_year = config.projection_start_year
        else
            last_user_station_year = maximum([s.planned_opening_year for s in config.station_data])
        end
        projected_station_capacity = 0.0  # Capacity from graduated auto-projected stations (kg/day), cumulative
        projected_station_capex = 0.0     # Total CAPEX for graduated projected stations ($), cumulative
        # Note: projected station O&M is recalculated each year at current utilization (not accumulated)

        # ── Investment parameters ─────────────────────────────────────────────────────
        # trigger_threshold: utilization level at which a capacity gap is signalled.
        #   DETERMINISTIC base = floor (default 0.60, span = 0); the maturity
        #   adjustment lifts the EFFECTIVE trigger toward the cap (0.80) as demand
        #   growth slows. If span > 0 it reverts to a stochastic Beta(3,3) base draw
        #   on [floor, floor+span] (drawn once per MC run).
        trig_floor   = hasproperty(config, :capacity_trigger_floor)   ? config.capacity_trigger_floor   : 0.60
        trig_span    = hasproperty(config, :capacity_trigger_span)    ? config.capacity_trigger_span    : 0.0
        trig_uniform = hasproperty(config, :capacity_trigger_uniform) ? config.capacity_trigger_uniform : false
        trigger_threshold = trig_span > 0 ?
            trig_floor + trig_span * (trig_uniform ? rand() : rand(Beta(3, 3))) :
            trig_floor
        # p_invest: probability of committing to investment when a gap is signalled.
        # Scheduled trucks: demand is known but not perfectly certain → [0.60, 0.80]
        # Probabilistic trucks: genuine market uncertainty → [0.40, 0.95]
        # Shape: Beta(3,3) (bell, mass near mean) by default; flat Uniform on the same
        # interval if config.p_invest_uniform (same mean, wider spread).
        p_inv_uniform = hasproperty(config, :p_invest_uniform) ? config.p_invest_uniform : false
        p_inv_shape() = p_inv_uniform ? rand() : rand(Beta(3, 3))
        p_invest = config.use_truck_deployment_schedule ?
            0.60 + 0.20 * p_inv_shape() :
            0.40 + 0.55 * p_inv_shape()
        # Foresight horizons: how many years ahead investors look when sizing capacity.
        # Shortening them lets current utilization climb closer to the trigger.
        station_foresight_years    = hasproperty(config, :station_foresight_years)    ? config.station_foresight_years    : 2
        production_foresight_years = hasproperty(config, :production_foresight_years) ? config.production_foresight_years : 3

        # Pending capacity lists — decisions taken now, capacity arrives after lead time
        # pending_production: Vector of (ProductionFacility, open_year)
        pending_production = Vector{Tuple{ProductionFacility, Int}}()
        # pending_stations: Vector of (capacity_kg_day, open_year, capex)
        pending_stations = Vector{Tuple{Float64, Int, Float64}}()

        # Simulate each year
        for year_idx in 1:n_years
            current_year = config.start_year + year_idx - 1

            # ── Flush pending production facilities that are now open ─────────────
            ready_prod = filter(p -> p[2] <= current_year, pending_production)
            for (fac, _) in ready_prod
                push!(production_facilities, fac)
                if !isnothing(event_sink)
                    push!(event_sink, (kind = :production, id = fac.id,
                                       capacity = fac.capacity_kg_day,
                                       year = fac.opening_year,
                                       tech = fac.technology_type,
                                       lifetime = fac.lifetime,
                                       is_initial = fac.is_initial))
                end
            end
            filter!(p -> p[2] > current_year, pending_production)

            # ── Flush pending stations that are now open ──────────────────────────
            for (cap, yr, capex) in pending_stations
                if yr <= current_year
                    projected_station_capacity += cap
                    projected_station_capex    += capex
                end
            end
            filter!(p -> p[2] > current_year, pending_stations)

            # ── Station foresight (2 years ahead) ────────────────────────────────
            # Foresight is NOT capped at the assessment horizon: investors look a
            # fixed number of years ahead, into extrapolated demand, even in the
            # final assessment years (otherwise late-horizon costs blow up).
            station_foresight_year = current_year + station_foresight_years
            # Retirement-aware: count only trucks that will STILL be operating at the
            # foresight year (deployed within the prior 7 years), so the projection
            # nets out the retirement of today's fleet rather than assuming it persists.
            station_foresight_trucks = sum(truck_deployment_years .> (station_foresight_year - 7))
            if config.use_truck_deployment_schedule
                for y in (current_year + 1):station_foresight_year
                    station_foresight_trucks += scheduled_trucks_added(config.truck_deployment_schedule.schedule, y)
                end
            end
            foresight_daily_demand_total =
                station_foresight_trucks * config.h2_per_truck_per_day * config.truck_uptime_default +
                bus_demand_kg_day(station_foresight_year, config.bus_demand_scenario) +
                car_demand_kg_day(station_foresight_year, config.car_demand_scenario)
            # Current-year total demand on the same basis as the foresight, used to
            # gauge demand growth for the maturity-adjusted station trigger.
            station_current_demand_total =
                sum(truck_deployment_years .> 0) * config.h2_per_truck_per_day * config.truck_uptime_default +
                bus_demand_kg_day(current_year, config.bus_demand_scenario) +
                car_demand_kg_day(current_year, config.car_demand_scenario)

            # ── Production foresight (3 years ahead) ─────────────────────────────
            # Foresight is NOT capped at the assessment horizon (see station note).
            production_foresight_year = current_year + production_foresight_years
            # Retirement-aware (see station note): only trucks still operating at the foresight year.
            production_foresight_trucks = sum(truck_deployment_years .> (production_foresight_year - 7))
            if config.use_truck_deployment_schedule
                for y in (current_year + 1):production_foresight_year
                    production_foresight_trucks += scheduled_trucks_added(config.truck_deployment_schedule.schedule, y)
                end
            end
            production_foresight_demand_total =
                production_foresight_trucks * config.h2_per_truck_per_day * config.truck_uptime_default +
                bus_demand_kg_day(production_foresight_year, config.bus_demand_scenario) +
                car_demand_kg_day(production_foresight_year, config.car_demand_scenario)

            # Retire trucks that are 7+ years old
            truck_lifetime_years = 7
            for i in 1:effective_max_trucks
                if truck_deployment_years[i] > 0  # Truck is deployed
                    truck_age = current_year - truck_deployment_years[i]
                    if truck_age >= truck_lifetime_years
                        truck_deployment_years[i] = 0  # Retire the truck
                    end
                end
            end

            # Deploy new stations based on planned opening year + probability
            for (i, station) in enumerate(config.station_data)
                if !stations_deployed[i]
                    # Get station status (default to "in planning" if not present)
                    station_status = haskey(station, :status) ? station.status : "in planning"

                    open_prob = calculate_station_opening_probability(
                        current_year,
                        station.planned_opening_year,
                        station_status,
                        config.station_prob_config
                    )
                    if rand() < open_prob
                        stations_deployed[i] = true
                    end
                end
            end

            # Deploy new trucks based on mode (predetermined schedule or probabilistic)
            if config.use_truck_deployment_schedule
                # MODE: Predetermined schedule - deploy exact number of trucks specified for this year
                trucks_to_add = scheduled_trucks_added(config.truck_deployment_schedule.schedule, current_year)

                if trucks_to_add > 0
                    # Add trucks
                    trucks_added_this_year = 0
                    for i in 1:effective_max_trucks
                        if truck_deployment_years[i] == 0 && trucks_added_this_year < trucks_to_add
                            truck_deployment_years[i] = current_year
                            trucks_added_this_year += 1
                        end
                    end
                elseif trucks_to_add < 0
                    # Remove trucks (negative value = manual removal, e.g., sold or taken off market)
                    trucks_to_remove = abs(trucks_to_add)
                    trucks_removed_this_year = 0
                    # Remove oldest trucks first
                    for i in 1:effective_max_trucks
                        if truck_deployment_years[i] > 0 && trucks_removed_this_year < trucks_to_remove
                            truck_deployment_years[i] = 0  # Remove this truck
                            trucks_removed_this_year += 1
                        end
                    end
                end
            else
                # MODE: Probabilistic deployment using standard probability function
                truck_prob_params = (
                    base = config.truck_prob_base,
                    slope = config.truck_prob_slope,
                    steepness = config.truck_prob_steepness,
                    midpoint = truck_midpoint_this_run,
                    log_scale = config.truck_prob_log_scale,
                    log_factor = config.truck_prob_log_factor,
                    exp_rate = config.truck_prob_exp_rate
                )
                truck_prob = deployment_probability(year_idx - 1, n_years,
                                                   config.truck_prob_type,
                                                   truck_prob_params)

                for i in 1:effective_max_trucks
                    if truck_deployment_years[i] == 0 && rand() < truck_prob
                        truck_deployment_years[i] = current_year
                    end
                end
            end

            # Count active stations and trucks
            n_active_stations = sum(stations_deployed)
            n_active_trucks = sum(truck_deployment_years .> 0)

            # Compute truck uptime early (needed for preliminary demand estimate below)
            truck_uptime_current = if year_idx == 1
                config.truck_uptime_year_1
            elseif year_idx == 2
                config.truck_uptime_year_2
            else
                config.truck_uptime_default
            end

            # PASS 1: Sum CAPEX and capacity from deployed user stations (O&M deferred)
            total_station_capex = 0.0
            total_station_capex_company = 0.0
            total_station_capex_government = 0.0
            total_capacity = 0.0
            for (i, station) in enumerate(config.station_data)
                if stations_deployed[i]
                    funding_fraction = station.public_funding_percentage / 100.0
                    total_station_capex += station.capex
                    total_station_capex_company += station.capex * (1.0 - funding_fraction)
                    total_station_capex_government += station.capex * funding_fraction
                    total_capacity += station.capacity_kg_per_day
                end
            end

            # Station projection — foresight-based stochastic investment decisions
            # Checks from year 1 (no gate on last_user_station_year); the capacity at the
            # foresight horizon naturally includes the user pipeline, so early investment
            # decisions are only triggered when that pipeline falls short.
            if AUTO_PROJECT_STATIONS && foresight_daily_demand_total > 0

                projected_station_size = 8000.0   # kg/day — liquid H2, 8 t/day
                liquid_8000_capex      = 14500000.0   # USD

                # Capacity the investor can see:
                #   - All user-pipeline stations (deployed + all planned, any year)
                #   - Graduated projected stations (projected_station_capacity)
                #   - ALL pending projected stations already committed (regardless of open year)
                # The investor knows the full pipeline and all their own prior commitments.
                pipeline_known = sum(
                    s.capacity_kg_per_day
                    for s in config.station_data;
                    init = 0.0
                )
                pending_all = sum(cap for (cap, _, _) in pending_stations; init = 0.0)
                station_capacity_at_foresight = pipeline_known +
                                                projected_station_capacity +
                                                pending_all

                if station_capacity_at_foresight > 0.0
                    # Same market-maturity adjustment as production: build tighter as growth slows.
                    eff_station_trigger = maturity_adjusted_trigger(trigger_threshold, station_current_demand_total,
                                                                    foresight_daily_demand_total,
                                                                    station_foresight_years, config)
                    eff_trigger_station_per_year[year_idx] = eff_station_trigger
                    expected_station_util = foresight_daily_demand_total / station_capacity_at_foresight
                    if expected_station_util >= eff_station_trigger && rand() < p_invest
                        required_station_capacity = foresight_daily_demand_total / eff_station_trigger
                        deficit = required_station_capacity - station_capacity_at_foresight
                        if deficit > 0.0
                            num_to_add = ceil(Int, deficit / projected_station_size)
                            for _ in 1:num_to_add
                                open_year = current_year + rand(2:4)   # individual lead time per station
                                push!(pending_stations, (projected_station_size, open_year, liquid_8000_capex))
                                if !isnothing(event_sink)
                                    push!(event_sink, (kind = :station_projected,
                                                       capacity = projected_station_size,
                                                       year = open_year))
                                end
                            end
                        end
                    end
                end
            end

            # Add projected capacity and CAPEX to totals
            total_capacity += projected_station_capacity
            total_station_capex += projected_station_capex

            # Split projected CAPEX into company and government portions
            if length(config.station_data) > 0
                avg_funding_pct_for_projected = mean([s.public_funding_percentage for s in config.station_data])
                projected_funding_fraction_final = avg_funding_pct_for_projected / 100.0
            else
                projected_funding_fraction_final = 0.0
            end
            total_station_capex_company += projected_station_capex * (1.0 - projected_funding_fraction_final)
            total_station_capex_government += projected_station_capex * projected_funding_fraction_final

            annualized_station_capex_company = total_station_capex_company * annuity_factor
            annualized_station_capex_government = total_station_capex_government * annuity_factor

            # Compute preliminary utilization from current capacity and estimated demand
            # This is used to interpolate utilization-dependent O&M costs
            preliminary_daily_demand = n_active_trucks * config.h2_per_truck_per_day * truck_uptime_current
            preliminary_utilization = total_capacity > 0 ? preliminary_daily_demand / total_capacity : 0.0

            # PASS 2: Sum O&M from deployed user stations using utilization-interpolated values
            total_om_cost = 0.0
            for (i, station) in enumerate(config.station_data)
                if stations_deployed[i]
                    om = interpolate_om_at_utilization(
                        station.om_cost_30pct_per_year,
                        station.om_cost_50pct_per_year,
                        station.om_cost_per_year,
                        preliminary_utilization
                    )
                    total_om_cost += om
                end
            end

            # Add projected station O&M (recalculated each year at current utilization)
            # Projected stations are 8000 kg/day liquid type
            if projected_station_capacity > 0
                n_projected_stations = round(Int, projected_station_capacity / 8000.0)
                liquid_8000_om_30 = linear_interpolate_extrapolate(8000.0, config.station_cost_ref_data.liquid_capacities, config.station_cost_ref_data.liquid_om_30pct)
                liquid_8000_om_50 = linear_interpolate_extrapolate(8000.0, config.station_cost_ref_data.liquid_capacities, config.station_cost_ref_data.liquid_om_50pct)
                liquid_8000_om_80 = linear_interpolate_extrapolate(8000.0, config.station_cost_ref_data.liquid_capacities, config.station_cost_ref_data.liquid_om)
                om_per_projected_station = interpolate_om_at_utilization(liquid_8000_om_30, liquid_8000_om_50, liquid_8000_om_80, preliminary_utilization)
                total_om_cost += n_projected_stations * om_per_projected_station
            end

            # Store deployment numbers. Active stations = deployed config-file
            # stations + auto-projected stations (8 t/day each).
            n_stations_per_year[year_idx] = n_active_stations + round(Int, projected_station_capacity / 8000.0)
            n_trucks_per_year[year_idx] = n_active_trucks
            station_capacity_per_year[year_idx] = total_capacity

            # Calculate H2 usage and price
            truck_uptime = if year_idx == 1
                config.truck_uptime_year_1
            elseif year_idx == 2
                config.truck_uptime_year_2
            else
                config.truck_uptime_default
            end

            total_annual_h2_usage = n_active_trucks * config.h2_per_truck_per_year * truck_uptime

            # Production capacity tracking (Update 4)
            # ALWAYS track production regardless of pricing mode
            # Calculate daily demand (truck-only, used for price calculation)
            daily_demand = total_annual_h2_usage / config.operating_days_per_year
            # Total daily demand includes bus and car demand (for expansion trigger)
            daily_demand_total = daily_demand + bus_demand_kg_day(current_year, config.bus_demand_scenario) +
                                               car_demand_kg_day(current_year, config.car_demand_scenario)

            # Check and expand production capacity if needed (foresight-based, stochastic).
            # Trigger is raised as demand growth decelerates (mature market builds tighter).
            eff_trigger = maturity_adjusted_trigger(trigger_threshold, daily_demand_total,
                                                    production_foresight_demand_total,
                                                    production_foresight_years, config)
            eff_trigger_prod_per_year[year_idx] = eff_trigger
            check_and_expand_production!(
                production_facilities,
                pending_production,
                production_foresight_demand_total,
                current_year,
                production_foresight_year,
                expansion_pathway,
                config.lcfs_config,
                eff_trigger,
                p_invest,
                config.electrolysis_electricity_source,
            )

            # Calculate capacity-weighted CI
            weighted_ci = calculate_weighted_ci(production_facilities)

            # Track production outputs (ALWAYS - these are physical facts about the system)
            production_capacity_per_year[year_idx] = sum(f.capacity_kg_day for f in production_facilities)
            production_facilities_count_per_year[year_idx] = length(production_facilities)
            smr_facilities_count_per_year[year_idx] = count(f -> f.technology_type == "smr", production_facilities)
            electrolysis_facilities_count_per_year[year_idx] = count(f -> f.technology_type == "electrolysis", production_facilities)
            weighted_ci_per_year[year_idx] = weighted_ci

            # Calculate station utilization and HRI credits (Update 1)
            # METHODOLOGY REFERENCE: see the article Methods, "Hydrogen Refueling Infrastructure (HRI) Bonus Credits"
            # HRI mechanism provides LCFS credits for underutilized station capacity (<62.5% utilization)
            # This incentivizes infrastructure deployment ahead of demand, addressing the "chicken-and-egg" problem
            if total_capacity > 0 && total_annual_h2_usage > 0
                daily_demand = total_annual_h2_usage / config.operating_days_per_year
                utilization = daily_demand / total_capacity
                station_utilization[year_idx] = utilization

                # HRI credit for underutilized capacity up to 62.5%
                # Formula: HRI_credit = [(0.625 - u) × Q_total × 365 × 0.92 × LCFS_value] / D_annual
                if config.use_lcfs && utilization < HRI_UTILIZATION_THRESHOLD
                    # Get weighted CI for LCFS calculation
                    # HRI credits use the actual capacity-weighted production CI
                    # (so grid electrolysis carries its high CI), not just the
                    # initial pathway's static value.
                    weighted_ci_for_hri = weighted_ci_per_year[year_idx]

                    # Calculate LCFS credit value per kg for this year
                    lcfs_credit_value_per_kg = calculate_lcfs_credit(
                        current_year, year_idx, n_years, config.h2_pathway_id, config;
                        weighted_ci=weighted_ci_for_hri
                    )

                    # Underutilized fraction (e.g., if 50% utilized, underutilized = 62.5% - 50% = 12.5%)
                    underutilized_fraction = HRI_UTILIZATION_THRESHOLD - utilization

                    # Annual underutilized capacity (kg/year)
                    annual_underutilized_capacity = underutilized_fraction * total_capacity * 365.0 * HRI_STATION_UPTIME

                    # Total HRI credits from underutilized capacity
                    total_hri_credits = annual_underutilized_capacity * lcfs_credit_value_per_kg

                    # HRI credit per sold kg
                    hri_credits[year_idx] = total_hri_credits / total_annual_h2_usage
                else
                    hri_credits[year_idx] = 0.0
                end
            else
                station_utilization[year_idx] = 0.0
                hri_credits[year_idx] = 0.0
            end

            if total_annual_h2_usage > 0
                # METHODOLOGY REFERENCE: see the article Methods, "Hydrogen Price Components"
                # Final LCOH = C_prod + C_infra,CAPEX + C_infra,O&M - LCFS_credit - HRI_credit
                # Each component calculated separately below

                # Determine H2 base price: use production cost if endogenous, otherwise use configured price
                if PRODUCTION_COST_MODE == "endogenous"
                    production_cost = calculate_production_cost_per_kg(
                        production_facilities,
                        total_annual_h2_usage,
                        config.discount_rate
                    )
                    h2_base_price = production_cost + config.transportation_cost_per_kg
                    production_cost_per_year[year_idx] = h2_base_price
                else
                    h2_base_price, credit_45v_this_year = get_h2_base_price(current_year, year_idx, n_years, config, production_facilities, daily_demand)
                    production_cost_per_year[year_idx] = 0.0  # Not calculated
                    credits_45v[year_idx] = credit_45v_this_year
                end

                # Company CAPEX portion - included in H2 price
                infrastructure_cost_per_kg_capex = annualized_station_capex_company / total_annual_h2_usage

                # Government CAPEX portion - NOT in H2 price (informational)
                infrastructure_cost_per_kg_government = annualized_station_capex_government / total_annual_h2_usage

                # O&M cost - included in H2 price (separate from CAPEX)
                infrastructure_cost_per_kg_om = total_om_cost / total_annual_h2_usage

                # Store price components
                h2_base_prices[year_idx] = h2_base_price
                infrastructure_costs[year_idx] = infrastructure_cost_per_kg_capex
                infrastructure_costs_government[year_idx] = infrastructure_cost_per_kg_government
                infrastructure_costs_om[year_idx] = infrastructure_cost_per_kg_om

                # Calculate LCFS credit/deficit if enabled
                if config.use_lcfs
                    # Use weighted CI if available (Update 4), otherwise use pathway CI
                    # LCFS credits use the actual capacity-weighted production CI
                    # (so grid electrolysis carries its high CI), not just the
                    # initial pathway's static value.
                    weighted_ci_for_lcfs = weighted_ci_per_year[year_idx]
                    lcfs_credit_per_kg = calculate_lcfs_credit(
                        current_year, year_idx, n_years, config.h2_pathway_id, config;
                        weighted_ci=weighted_ci_for_lcfs
                    )
                    lcfs_credits[year_idx] = lcfs_credit_per_kg
                    # Include HRI credits in final price (Update 1)
                    # Total infrastructure = CAPEX (company) + O&M
                    h2_prices[year_idx] = h2_base_price + infrastructure_cost_per_kg_capex + infrastructure_cost_per_kg_om - lcfs_credit_per_kg - hri_credits[year_idx]
                else
                    lcfs_credits[year_idx] = 0.0
                    # Include HRI credits even when LCFS is disabled (Update 1)
                    # Total infrastructure = CAPEX (company) + O&M
                    h2_prices[year_idx] = h2_base_price + infrastructure_cost_per_kg_capex + infrastructure_cost_per_kg_om - hri_credits[year_idx]
                end
            else
                # Zero trucks: default to $40/kg to avoid division by zero
                h2_prices[year_idx] = 40.0
                h2_base_prices[year_idx] = 40.0
                infrastructure_costs[year_idx] = 0.0
                infrastructure_costs_government[year_idx] = 0.0
                infrastructure_costs_om[year_idx] = 0.0
                lcfs_credits[year_idx] = 0.0
                hri_credits[year_idx] = 0.0
                station_utilization[year_idx] = 0.0
                warning_msg = "Year $current_year: Zero trucks deployed, H2 price defaulted to \$40/kg"
                push!(warnings, warning_msg)
            end
        end

    else
        # MODE 2: Stochastic deployment mode (original logic)
        # Add random variation to truck deployment midpoint for this run
        truck_midpoint_this_run = clamp(
            config.truck_prob_midpoint + randn() * config.truck_prob_midpoint_std,
            0.0, 1.0
        )

        # Initialize tracking arrays
        stations_deployed = falses(config.max_stations)

        # Determine effective max trucks: if using schedule, calculate max from schedule; otherwise use MAX_TRUCKS
        effective_max_trucks = if config.use_truck_deployment_schedule
            # Total trucks from schedule (initial fleet + additions through the
            # horizon, including extrapolated additions beyond the authored years)
            config.truck_deployment_schedule.initial_trucks +
                total_scheduled_additions(config.truck_deployment_schedule.schedule, config.end_year)
        else
            config.max_trucks
        end

        # Track truck deployment years (0 = not deployed, >0 = year deployed)
        truck_deployment_years = zeros(Int, effective_max_trucks)

        # Set initial stations and trucks as already deployed
        if config.initial_stations > 0
            stations_deployed[1:min(config.initial_stations, config.max_stations)] .= true
        end
        # Use initial trucks from schedule if using predetermined mode, otherwise use config
        initial_trucks_count = if config.use_truck_deployment_schedule
            config.truck_deployment_schedule.initial_trucks
        else
            config.initial_trucks
        end
        if initial_trucks_count > 0
            # Mark initial trucks as deployed in the start year
            truck_deployment_years[1:min(initial_trucks_count, effective_max_trucks)] .= config.start_year
        end

        # Initialize production facilities (Update 4)
        # ALWAYS track production facilities regardless of pricing mode
        production_facilities = ProductionFacility[]
        initial_facility = create_initial_production(
            config.h2_pathway_id,
            config.lcfs_config,
            config.lcfs_config.initial_production,
            config.start_year,
            config.electrolysis_electricity_source
        )
        push!(production_facilities, initial_facility)
        if !isnothing(event_sink)
            push!(event_sink, (kind = :production, id = initial_facility.id,
                               capacity = initial_facility.capacity_kg_day,
                               year = initial_facility.opening_year,
                               tech = initial_facility.technology_type,
                               lifetime = initial_facility.lifetime,
                               is_initial = initial_facility.is_initial))
        end

        # Determine expansion pathway (Issue 2)
        # Use expansion_pathway_id if specified, otherwise same as initial pathway
        expansion_pathway = isnothing(config.expansion_pathway_id) ? config.h2_pathway_id : config.expansion_pathway_id

        # Investment parameters (same logic as MODE 1): deterministic trigger when span = 0
        trig_floor   = hasproperty(config, :capacity_trigger_floor)   ? config.capacity_trigger_floor   : 0.60
        trig_span    = hasproperty(config, :capacity_trigger_span)    ? config.capacity_trigger_span    : 0.0
        trig_uniform = hasproperty(config, :capacity_trigger_uniform) ? config.capacity_trigger_uniform : false
        trigger_threshold = trig_span > 0 ?
            trig_floor + trig_span * (trig_uniform ? rand() : rand(Beta(4, 3))) :
            trig_floor
        p_inv_uniform = hasproperty(config, :p_invest_uniform) ? config.p_invest_uniform : false
        p_invest = config.use_truck_deployment_schedule ?
            0.60 + 0.20 * (p_inv_uniform ? rand() : rand(Beta(3, 3))) :
            0.40 + 0.55 * (p_inv_uniform ? rand() : rand(Beta(3, 2)))
        production_foresight_years = hasproperty(config, :production_foresight_years) ? config.production_foresight_years : 3
        pending_production = Vector{Tuple{ProductionFacility, Int}}()

        # Simulate each year
        for year_idx in 1:n_years
            current_year = config.start_year + year_idx - 1

            # Flush pending production facilities that are now open
            ready_prod = filter(p -> p[2] <= current_year, pending_production)
            for (fac, _) in ready_prod
                push!(production_facilities, fac)
                if !isnothing(event_sink)
                    push!(event_sink, (kind = :production, id = fac.id,
                                       capacity = fac.capacity_kg_day,
                                       year = fac.opening_year,
                                       tech = fac.technology_type,
                                       lifetime = fac.lifetime,
                                       is_initial = fac.is_initial))
                end
            end
            filter!(p -> p[2] > current_year, pending_production)

            # Production foresight demand (4 years ahead)
            # Foresight is NOT capped at the assessment horizon (see station note).
            production_foresight_year = current_year + production_foresight_years
            # Retirement-aware (see station note): only trucks still operating at the foresight year.
            production_foresight_trucks = sum(truck_deployment_years .> (production_foresight_year - 7))
            if config.use_truck_deployment_schedule
                for y in (current_year + 1):production_foresight_year
                    production_foresight_trucks += scheduled_trucks_added(config.truck_deployment_schedule.schedule, y)
                end
            end
            production_foresight_demand_total =
                production_foresight_trucks * config.h2_per_truck_per_day * config.truck_uptime_default +
                bus_demand_kg_day(production_foresight_year, config.bus_demand_scenario) +
                car_demand_kg_day(production_foresight_year, config.car_demand_scenario)

            # Retire trucks that are 7+ years old
            truck_lifetime_years = 7
            for i in 1:effective_max_trucks
                if truck_deployment_years[i] > 0  # Truck is deployed
                    truck_age = current_year - truck_deployment_years[i]
                    if truck_age >= truck_lifetime_years
                        truck_deployment_years[i] = 0  # Retire the truck
                    end
                end
            end
            # Determine deployment probabilities for this year
            station_prob_params = (
                base = config.station_prob_base,
                slope = config.station_prob_slope,
                steepness = config.station_prob_steepness,
                midpoint = config.station_prob_midpoint,
                log_scale = config.station_prob_log_scale,
                log_factor = config.station_prob_log_factor,
                exp_rate = config.station_prob_exp_rate
            )
            current_year = config.start_year + year_idx - 1

            # Calculate station deployment probability
            station_prob = deployment_probability(year_idx - 1, n_years,
                                                 config.station_prob_type,
                                                 station_prob_params)

            # Deploy new stations (each non-deployed station has independent chance)
            for i in 1:config.max_stations
                if !stations_deployed[i] && rand() < station_prob
                    stations_deployed[i] = true
                end
            end

            # Deploy new trucks based on mode (predetermined schedule or probabilistic)
            if config.use_truck_deployment_schedule
                # MODE: Predetermined schedule - deploy exact number of trucks specified for this year
                trucks_to_add = scheduled_trucks_added(config.truck_deployment_schedule.schedule, current_year)

                if trucks_to_add > 0
                    # Add trucks
                    trucks_added_this_year = 0
                    for i in 1:effective_max_trucks
                        if truck_deployment_years[i] == 0 && trucks_added_this_year < trucks_to_add
                            truck_deployment_years[i] = current_year
                            trucks_added_this_year += 1
                        end
                    end
                elseif trucks_to_add < 0
                    # Remove trucks (negative value = manual removal, e.g., sold or taken off market)
                    trucks_to_remove = abs(trucks_to_add)
                    trucks_removed_this_year = 0
                    # Remove oldest trucks first
                    for i in 1:effective_max_trucks
                        if truck_deployment_years[i] > 0 && trucks_removed_this_year < trucks_to_remove
                            truck_deployment_years[i] = 0  # Remove this truck
                            trucks_removed_this_year += 1
                        end
                    end
                end
            else
                # MODE: Probabilistic deployment
                truck_prob_params = (
                    base = config.truck_prob_base,
                    slope = config.truck_prob_slope,
                    steepness = config.truck_prob_steepness,
                    midpoint = truck_midpoint_this_run,
                    log_scale = config.truck_prob_log_scale,
                    log_factor = config.truck_prob_log_factor,
                    exp_rate = config.truck_prob_exp_rate
                )
                truck_prob = deployment_probability(year_idx - 1, n_years,
                                                   config.truck_prob_type,
                                                   truck_prob_params)

                # Deploy new trucks (each non-deployed truck has independent chance)
                for i in 1:effective_max_trucks
                    if truck_deployment_years[i] == 0 && rand() < truck_prob
                        truck_deployment_years[i] = current_year
                    end
                end
            end

            # Count active stations and trucks
            n_active_stations = sum(stations_deployed)
            n_active_trucks = sum(truck_deployment_years .> 0)

            # Store deployment numbers for this year
            n_stations_per_year[year_idx] = n_active_stations
            n_trucks_per_year[year_idx] = n_active_trucks
            # In stochastic mode, assume default capacity of 8000 kg/day per station
            station_capacity_per_year[year_idx] = n_active_stations * 8000.0

            # Calculate costs and hydrogen usage
            total_station_capex = n_active_stations * config.station_capex
            # In probabilistic mode, assume 0% public funding (no station-specific data available)
            total_station_capex_company = total_station_capex
            total_station_capex_government = 0.0
            annualized_station_capex_company = total_station_capex_company * annuity_factor
            annualized_station_capex_government = 0.0
            om_cost = n_active_stations * config.om_cost_per_station

            # Determine truck uptime based on year
            truck_uptime = if year_idx == 1
                config.truck_uptime_year_1
            elseif year_idx == 2
                config.truck_uptime_year_2
            else
                config.truck_uptime_default
            end

            total_annual_h2_usage = n_active_trucks * config.h2_per_truck_per_year * truck_uptime

            # Production capacity tracking (Update 4)
            # ALWAYS track production regardless of pricing mode
            # Calculate daily demand (truck-only, used for price calculation)
            daily_demand = total_annual_h2_usage / config.operating_days_per_year
            # Total daily demand includes bus and car demand (for expansion trigger)
            daily_demand_total = daily_demand + bus_demand_kg_day(current_year, config.bus_demand_scenario) +
                                               car_demand_kg_day(current_year, config.car_demand_scenario)

            # Check and expand production capacity if needed (foresight-based, stochastic).
            # Trigger is raised as demand growth decelerates (mature market builds tighter).
            eff_trigger = maturity_adjusted_trigger(trigger_threshold, daily_demand_total,
                                                    production_foresight_demand_total,
                                                    production_foresight_years, config)
            eff_trigger_prod_per_year[year_idx] = eff_trigger
            check_and_expand_production!(
                production_facilities,
                pending_production,
                production_foresight_demand_total,
                current_year,
                production_foresight_year,
                expansion_pathway,
                config.lcfs_config,
                eff_trigger,
                p_invest,
                config.electrolysis_electricity_source,
            )

            # Calculate capacity-weighted CI
            weighted_ci = calculate_weighted_ci(production_facilities)

            # Track production outputs (ALWAYS - these are physical facts about the system)
            production_capacity_per_year[year_idx] = sum(f.capacity_kg_day for f in production_facilities)
            production_facilities_count_per_year[year_idx] = length(production_facilities)
            smr_facilities_count_per_year[year_idx] = count(f -> f.technology_type == "smr", production_facilities)
            electrolysis_facilities_count_per_year[year_idx] = count(f -> f.technology_type == "electrolysis", production_facilities)
            weighted_ci_per_year[year_idx] = weighted_ci

            # Calculate station utilization and HRI credits (Update 1)
            # In stochastic mode, station capacity is 8000 kg/day per station
            total_capacity_mode2 = n_active_stations * 8000.0
            if total_capacity_mode2 > 0 && total_annual_h2_usage > 0
                daily_demand = total_annual_h2_usage / config.operating_days_per_year
                utilization = daily_demand / total_capacity_mode2
                station_utilization[year_idx] = utilization

                # HRI credit for underutilized capacity up to 62.5%
                if config.use_lcfs && utilization < HRI_UTILIZATION_THRESHOLD
                    # Get weighted CI for LCFS calculation
                    # HRI credits use the actual capacity-weighted production CI
                    # (so grid electrolysis carries its high CI), not just the
                    # initial pathway's static value.
                    weighted_ci_for_hri = weighted_ci_per_year[year_idx]

                    # Calculate LCFS credit value per kg for this year
                    lcfs_credit_value_per_kg = calculate_lcfs_credit(
                        current_year, year_idx, n_years, config.h2_pathway_id, config;
                        weighted_ci=weighted_ci_for_hri
                    )

                    # Underutilized fraction (e.g., if 50% utilized, underutilized = 62.5% - 50% = 12.5%)
                    underutilized_fraction = HRI_UTILIZATION_THRESHOLD - utilization

                    # Annual underutilized capacity (kg/year)
                    annual_underutilized_capacity = underutilized_fraction * total_capacity_mode2 * 365.0 * HRI_STATION_UPTIME

                    # Total HRI credits from underutilized capacity
                    total_hri_credits = annual_underutilized_capacity * lcfs_credit_value_per_kg

                    # HRI credit per sold kg
                    hri_credits[year_idx] = total_hri_credits / total_annual_h2_usage
                else
                    hri_credits[year_idx] = 0.0
                end
            else
                station_utilization[year_idx] = 0.0
                hri_credits[year_idx] = 0.0
            end

            # Calculate hydrogen price
            if total_annual_h2_usage > 0
                # Determine H2 base price: use production cost if endogenous, otherwise use configured price
                if PRODUCTION_COST_MODE == "endogenous"
                    production_cost = calculate_production_cost_per_kg(
                        production_facilities,
                        total_annual_h2_usage,
                        config.discount_rate
                    )
                    h2_base_price = production_cost + config.transportation_cost_per_kg
                    production_cost_per_year[year_idx] = h2_base_price
                else
                    h2_base_price, credit_45v_this_year = get_h2_base_price(current_year, year_idx, n_years, config, production_facilities, daily_demand)
                    production_cost_per_year[year_idx] = 0.0  # Not calculated
                    credits_45v[year_idx] = credit_45v_this_year
                end

                # Company CAPEX portion - included in H2 price
                infrastructure_cost_per_kg_capex = annualized_station_capex_company / total_annual_h2_usage

                # Government CAPEX portion - NOT in H2 price (informational)
                infrastructure_cost_per_kg_government = annualized_station_capex_government / total_annual_h2_usage

                # O&M cost - included in H2 price (separate from CAPEX)
                infrastructure_cost_per_kg_om = om_cost / total_annual_h2_usage

                # Store price components
                h2_base_prices[year_idx] = h2_base_price
                infrastructure_costs[year_idx] = infrastructure_cost_per_kg_capex
                infrastructure_costs_government[year_idx] = infrastructure_cost_per_kg_government
                infrastructure_costs_om[year_idx] = infrastructure_cost_per_kg_om

                # Calculate LCFS credit/deficit if enabled
                if config.use_lcfs
                    # Use weighted CI if available (Update 4), otherwise use pathway CI
                    # LCFS credits use the actual capacity-weighted production CI
                    # (so grid electrolysis carries its high CI), not just the
                    # initial pathway's static value.
                    weighted_ci_for_lcfs = weighted_ci_per_year[year_idx]
                    lcfs_credit_per_kg = calculate_lcfs_credit(
                        current_year, year_idx, n_years, config.h2_pathway_id, config;
                        weighted_ci=weighted_ci_for_lcfs
                    )
                    lcfs_credits[year_idx] = lcfs_credit_per_kg
                    # Include HRI credits in final price (Update 1)
                    # Total infrastructure = CAPEX (company) + O&M
                    h2_prices[year_idx] = h2_base_price + infrastructure_cost_per_kg_capex + infrastructure_cost_per_kg_om - lcfs_credit_per_kg - hri_credits[year_idx]
                else
                    lcfs_credits[year_idx] = 0.0
                    # Include HRI credits even when LCFS is disabled (Update 1)
                    # Total infrastructure = CAPEX (company) + O&M
                    h2_prices[year_idx] = h2_base_price + infrastructure_cost_per_kg_capex + infrastructure_cost_per_kg_om - hri_credits[year_idx]
                end
            else
                # Zero trucks: default to $40/kg to avoid division by zero
                h2_prices[year_idx] = 40.0
                h2_base_prices[year_idx] = 40.0
                infrastructure_costs[year_idx] = 0.0
                infrastructure_costs_government[year_idx] = 0.0
                infrastructure_costs_om[year_idx] = 0.0
                lcfs_credits[year_idx] = 0.0
                hri_credits[year_idx] = 0.0
                station_utilization[year_idx] = 0.0
                warning_msg = "Year $current_year: Zero trucks deployed, H2 price defaulted to \$40/kg"
                push!(warnings, warning_msg)
            end
        end
    end

    return (
        h2_prices, n_stations_per_year, n_trucks_per_year, station_capacity_per_year,
        h2_base_prices, infrastructure_costs, infrastructure_costs_government, infrastructure_costs_om, lcfs_credits, warnings,
        production_capacity_per_year, production_facilities_count_per_year,
        smr_facilities_count_per_year, electrolysis_facilities_count_per_year,
        weighted_ci_per_year, production_cost_per_year,
        hri_credits, station_utilization, credits_45v,
        eff_trigger_prod_per_year, eff_trigger_station_per_year
    )
end

"""
    run_monte_carlo(config, n_runs)

Run multiple Monte Carlo simulations.

# Arguments
- `config`: NamedTuple containing all simulation parameters
- `n_runs`: Number of Monte Carlo runs to perform

# Returns
- Tuple of (price_results, station_results, truck_results, capacity_results, base_price_results, infrastructure_results, lcfs_results, all_warnings, production_capacity_results, production_facilities_count_results, weighted_ci_results, production_cost_results, hri_credits_results, station_utilization_results):
  - price_results: Matrix (n_runs × n_years) of total hydrogen prices
  - station_results: Matrix (n_runs × n_years) of station deployments
  - truck_results: Matrix (n_runs × n_years) of truck deployments
  - capacity_results: Matrix (n_runs × n_years) of station capacities
  - base_price_results: Matrix (n_runs × n_years) of base H2 production/transport costs
  - infrastructure_results: Matrix (n_runs × n_years) of infrastructure costs
  - lcfs_results: Matrix (n_runs × n_years) of LCFS credits
  - all_warnings: Array of unique warning messages from all simulations
  - production_capacity_results: Matrix (n_runs × n_years) of production capacity (kg/day) (Update 4)
  - production_facilities_count_results: Matrix (n_runs × n_years) of number of production facilities (Update 4)
  - weighted_ci_results: Matrix (n_runs × n_years) of capacity-weighted CI (gCO2e/MJ) (Update 4)
  - production_cost_results: Matrix (n_runs × n_years) of endogenous production cost (\$/kg) or 0 (Update 4)
  - hri_credits_results: Matrix (n_runs × n_years) of HRI credits (\$/kg) (Update 1)
  - station_utilization_results: Matrix (n_runs × n_years) of station utilization (fraction) (Update 1)
"""
function run_monte_carlo(config, n_runs)
    n_years = config.end_year - config.start_year + 1
    price_results = zeros(Float64, n_runs, n_years)
    station_results = zeros(Int, n_runs, n_years)
    truck_results = zeros(Int, n_runs, n_years)
    capacity_results = zeros(Float64, n_runs, n_years)
    base_price_results = zeros(Float64, n_runs, n_years)
    infrastructure_results = zeros(Float64, n_runs, n_years)
    infrastructure_government_results = zeros(Float64, n_runs, n_years)
    infrastructure_om_results = zeros(Float64, n_runs, n_years)
    lcfs_results = zeros(Float64, n_runs, n_years)
    all_warnings = Set{String}()  # Use Set to collect unique warnings

    # Production tracking matrices (Update 4)
    production_capacity_results = zeros(Float64, n_runs, n_years)
    production_facilities_count_results = zeros(Int, n_runs, n_years)
    smr_facilities_count_results = zeros(Int, n_runs, n_years)
    electrolysis_facilities_count_results = zeros(Int, n_runs, n_years)
    weighted_ci_results = zeros(Float64, n_runs, n_years)
    production_cost_results = zeros(Float64, n_runs, n_years)

    # HRI credit tracking matrices (Update 1)
    hri_credits_results = zeros(Float64, n_runs, n_years)
    station_utilization_results = zeros(Float64, n_runs, n_years)

    # 45V tax credit tracking
    credits_45v_results = zeros(Float64, n_runs, n_years)

    # Effective build trigger θ_eff (market-maturity-adjusted), fraction
    eff_trigger_prod_results    = fill(NaN, n_runs, n_years)
    eff_trigger_station_results = fill(NaN, n_runs, n_years)

    println("Running $n_runs Monte Carlo simulations...")
    for run in 1:n_runs
        if run % 10 == 0
            println("  Completed $run/$n_runs runs")
        end
        h2_prices, n_stations, n_trucks, capacities, h2_base_prices, infrastructure_costs, infrastructure_costs_government, infrastructure_costs_om, lcfs_credits, warnings,
            production_capacity, production_facilities_count, smr_facilities_count, electrolysis_facilities_count, weighted_ci, production_cost,
            hri_credits, station_utilization, credits_45v,
            eff_trigger_prod, eff_trigger_station = run_single_simulation(config)
        price_results[run, :] = h2_prices
        station_results[run, :] = n_stations
        truck_results[run, :] = n_trucks
        capacity_results[run, :] = capacities
        base_price_results[run, :] = h2_base_prices
        infrastructure_results[run, :] = infrastructure_costs
        infrastructure_government_results[run, :] = infrastructure_costs_government
        infrastructure_om_results[run, :] = infrastructure_costs_om
        lcfs_results[run, :] = lcfs_credits

        # Store production results (Update 4)
        production_capacity_results[run, :] = production_capacity
        production_facilities_count_results[run, :] = production_facilities_count
        smr_facilities_count_results[run, :] = smr_facilities_count
        electrolysis_facilities_count_results[run, :] = electrolysis_facilities_count
        weighted_ci_results[run, :] = weighted_ci
        production_cost_results[run, :] = production_cost

        # Store HRI results (Update 1)
        hri_credits_results[run, :] = hri_credits
        station_utilization_results[run, :] = station_utilization

        # Store 45V credits
        credits_45v_results[run, :] = credits_45v

        # Store effective build trigger θ_eff
        eff_trigger_prod_results[run, :]    = eff_trigger_prod
        eff_trigger_station_results[run, :] = eff_trigger_station

        # Collect warnings
        for warning in warnings
            push!(all_warnings, warning)
        end
    end
    println("  Completed $n_runs/$n_runs runs")

    # Convert Set to sorted Array for consistent output
    warnings_array = sort(collect(all_warnings))

    return (
        price_results, station_results, truck_results, capacity_results,
        base_price_results, infrastructure_results, infrastructure_government_results, infrastructure_om_results, lcfs_results, warnings_array,
        production_capacity_results, production_facilities_count_results,
        smr_facilities_count_results, electrolysis_facilities_count_results,
        weighted_ci_results, production_cost_results,
        hri_credits_results, station_utilization_results, credits_45v_results,
        eff_trigger_prod_results, eff_trigger_station_results
    )
end

#==============================================================================
ANALYSIS FUNCTIONS
==============================================================================#

"""
    analyze_results(results, years)

Analyze Monte Carlo results and compute summary statistics.

# Arguments
- `results`: Matrix (n_runs × n_years) of hydrogen prices
- `years`: Array of year values

# Returns
- NamedTuple containing summary statistics
"""
function analyze_results(results, years)
    n_runs, n_years = size(results)

    # Calculate statistics for each year
    means = zeros(n_years)
    medians = zeros(n_years)
    stds = zeros(n_years)
    p10 = zeros(n_years)
    p25 = zeros(n_years)
    p75 = zeros(n_years)
    p90 = zeros(n_years)

    for year_idx in 1:n_years
        year_data = results[:, year_idx]
        # Filter out NaN values (years with no trucks)
        valid_data = filter(!isnan, year_data)

        if !isempty(valid_data)
            means[year_idx] = mean(valid_data)
            medians[year_idx] = median(valid_data)
            stds[year_idx] = std(valid_data)
            p10[year_idx] = quantile(valid_data, 0.10)
            p25[year_idx] = quantile(valid_data, 0.25)
            p75[year_idx] = quantile(valid_data, 0.75)
            p90[year_idx] = quantile(valid_data, 0.90)
        else
            means[year_idx] = NaN
            medians[year_idx] = NaN
            stds[year_idx] = NaN
            p10[year_idx] = NaN
            p25[year_idx] = NaN
            p75[year_idx] = NaN
            p90[year_idx] = NaN
        end
    end

    return (
        years = years,
        mean = means,
        median = medians,
        std = stds,
        p10 = p10,
        p25 = p25,
        p75 = p75,
        p90 = p90
    )
end

"""
    print_summary(stats)

Print summary statistics to console.
"""
function print_summary(stats)
    println("\n" * "="^80)
    println("HYDROGEN PRICE SUMMARY STATISTICS (\$/kg)")
    println("="^80)
    println()
    println("Year    Mean    Median    Std     P10     P25     P75     P90")
    println("-"^70)

    for i in eachindex(stats.years)
        if !isnan(stats.mean[i])
            @printf("%4d   %6.2f   %6.2f   %6.2f   %6.2f   %6.2f   %6.2f   %6.2f\n",
                    stats.years[i], stats.mean[i], stats.median[i], stats.std[i],
                    stats.p10[i], stats.p25[i], stats.p75[i], stats.p90[i])
        else
            println("$(stats.years[i])   No trucks deployed")
        end
    end
    println()
end

"""
    plot_results(price_results, station_results, truck_results, capacity_results, stats, config)

Create visualization of Monte Carlo results.
"""
function plot_results(price_results, station_results, truck_results, capacity_results, stats, config)
    years = stats.years

    # Plot 1: Mean price with confidence bands
    p1 = plot(years, stats.mean,
              label="Mean",
              linewidth=3,
              legend=:topright,
              xlabel="Year",
              ylabel="Hydrogen Price (\$/kg)",
              title="Hydrogen Price Over Time (Monte Carlo Results)",
              size=(1000, 600))

    # Add percentile bands
    plot!(p1, years, stats.p10, fillrange=stats.p90,
          fillalpha=0.2, label="10th-90th Percentile",
          linewidth=0, color=:blue)
    plot!(p1, years, stats.p25, fillrange=stats.p75,
          fillalpha=0.3, label="25th-75th Percentile",
          linewidth=0, color=:blue)

    # Re-plot mean on top
    plot!(p1, years, stats.mean, label="", linewidth=3, color=:red)

    # Plot 2: Sample individual runs
    n_sample_runs = min(20, size(price_results, 1))
    p2 = plot(xlabel="Year",
              ylabel="Hydrogen Price (\$/kg)",
              title="Sample Individual Simulation Runs (n=$n_sample_runs)",
              legend=false,
              alpha=0.3,
              size=(1000, 600))

    for i in 1:n_sample_runs
        plot!(p2, years, price_results[i, :], color=:gray, linewidth=1)
    end

    # Add mean line
    plot!(p2, years, stats.mean, color=:red, linewidth=3, label="Mean")

    # Plot 3: Number of stations deployed over time
    mean_stations = vec(mean(station_results, dims=1))
    p10_stations = vec([quantile(filter(!isnan, station_results[:, i]), 0.10) for i in 1:size(station_results, 2)])
    p90_stations = vec([quantile(filter(!isnan, station_results[:, i]), 0.90) for i in 1:size(station_results, 2)])

    p3 = plot(years, mean_stations,
              label="Mean",
              linewidth=3,
              legend=:bottomright,
              xlabel="Year",
              ylabel="Number of Stations",
              title="Hydrogen Refueling Station Deployment",
              size=(1000, 600),
              color=:green)

    # Add confidence bands
    plot!(p3, years, p10_stations, fillrange=p90_stations,
          fillalpha=0.2, label="10th-90th Percentile",
          linewidth=0, color=:green)

    # Re-plot mean on top
    plot!(p3, years, mean_stations, label="", linewidth=3, color=:green)

    # Plot 4: Number of trucks deployed over time
    mean_trucks = vec(mean(truck_results, dims=1))
    p10_trucks = vec([quantile(filter(!isnan, truck_results[:, i]), 0.10) for i in 1:size(truck_results, 2)])
    p90_trucks = vec([quantile(filter(!isnan, truck_results[:, i]), 0.90) for i in 1:size(truck_results, 2)])

    p4 = plot(years, mean_trucks,
              label="Mean Trucks",
              linewidth=3,
              legend=:right,
              xlabel="Year",
              ylabel="Number of Trucks",
              title="Hydrogen Truck Deployment",
              size=(1000, 600),
              color=:purple)

    # Add confidence bands
    plot!(p4, years, p10_trucks, fillrange=p90_trucks,
          fillalpha=0.2, label="10th-90th Percentile",
          linewidth=0, color=:purple,
          legend=(0.75, 0.2))

    # Re-plot mean on top
    plot!(p4, years, mean_trucks, label="", linewidth=3, color=:purple)

    # Add secondary y-axis for truck uptime percentage
    # Create uptime array based on year index using config values
    n_years = length(years)
    uptime_percent = zeros(n_years)
    for i in 1:n_years
        if i == 1
            uptime_percent[i] = config.truck_uptime_year_1 * 100
        elseif i == 2
            uptime_percent[i] = config.truck_uptime_year_2 * 100
        else
            uptime_percent[i] = config.truck_uptime_default * 100
        end
    end

    # Add right y-axis with uptime
    plot!(twinx(p4), years, uptime_percent,
          label="Truck Uptime %",
          linewidth=2,
          linestyle=:dash,
          ylabel="Truck Uptime (%)",
          color=:orange,
          legend=(0.75, 0.9),
          ylims=(0, 100))

    # Plot 5: Station capacity vs truck demand
    mean_capacity = vec(mean(capacity_results, dims=1))
    p10_capacity = vec([quantile(filter(!isnan, capacity_results[:, i]), 0.10) for i in 1:size(capacity_results, 2)])
    p90_capacity = vec([quantile(filter(!isnan, capacity_results[:, i]), 0.90) for i in 1:size(capacity_results, 2)])

    # Calculate daily truck demand (number of trucks × 50 kg/day × uptime)
    mean_trucks_vec = vec(mean(truck_results, dims=1))

    # Create uptime array based on year
    n_years = length(years)
    uptime = zeros(n_years)
    for i in 1:n_years
        if i == 1
            uptime[i] = config.truck_uptime_year_1
        elseif i == 2
            uptime[i] = config.truck_uptime_year_2
        else
            uptime[i] = config.truck_uptime_default
        end
    end

    truck_demand = mean_trucks_vec .* config.h2_per_truck_per_day .* uptime

    p5 = plot(years, mean_capacity,
              label="Station Capacity (mean)",
              linewidth=3,
              xlabel="Year",
              ylabel="Hydrogen (kg/day)",
              title="Daily H2 Capacity vs Truck Demand",
              legend=(0.25, 0.9),
              color=:green)

    # Add capacity confidence bands
    plot!(p5, years, p10_capacity,
          fillrange=p90_capacity,
          fillalpha=0.2,
          label="Capacity (10th-90th percentile)",
          linewidth=0,
          color=:green)

    # Add truck demand
    plot!(p5, years, truck_demand,
          label="Truck Demand (trucks × $(config.h2_per_truck_per_day) kg/day × uptime)",
          linewidth=3,
          color=:purple,
          linestyle=:dash)

    # Calculate and add capacity utilization percentage on secondary y-axis
    utilization_percent = zeros(length(mean_capacity))
    for i in 1:length(mean_capacity)
        if mean_capacity[i] > 0
            utilization_percent[i] = (truck_demand[i] / mean_capacity[i]) * 100.0
        else
            utilization_percent[i] = 0.0
        end
    end

    plot!(twinx(p5), years, utilization_percent,
          label="Capacity Utilization %",
          linewidth=2,
          linestyle=:dot,
          ylabel="Utilization (%)",
          color=:orange,
          legend=(0.75, 0.9),
          ylims=(0, 120))

    # Combine all plots
    plot(p1, p2, p3, p4, p5, layout=(3, 2), size=(1600, 1800))

    savefig("images/hydrogen_rollout_monte_carlo.png")
    println("Plot saved as 'images/hydrogen_rollout_monte_carlo.png'")

    return p1
end

"""
    plot_example_run(price_results, station_results, truck_results, capacity_results, years, config, run_index)

Create a detailed plot of a single example simulation run showing stations, trucks, capacity, and price.
"""
function plot_example_run(price_results, station_results, truck_results, capacity_results, years, config, run_index=1)
    # Extract data for the selected run
    stations = station_results[run_index, :]
    trucks = truck_results[run_index, :]
    prices = price_results[run_index, :]
    capacities = capacity_results[run_index, :]

    # Plot 1: Number of stations over time
    p1 = plot(years, stations,
              linewidth=3,
              xlabel="Year",
              ylabel="Number of Stations",
              title="Hydrogen Refueling Station Deployment (Example Run)",
              legend=false,
              size=(1200, 400),
              color=:green,
              marker=:circle,
              markersize=5)

    # Plot 2: Number of trucks over time
    p2 = plot(years, trucks,
              linewidth=3,
              xlabel="Year",
              ylabel="Number of Trucks",
              title="Hydrogen Truck Deployment (Example Run)",
              legend=:right,
              label="Trucks",
              size=(1200, 400),
              color=:purple,
              marker=:circle,
              markersize=5)

    # Add secondary y-axis for truck uptime percentage using config values
    n_years = length(years)
    uptime_percent = zeros(n_years)
    for i in 1:n_years
        if i == 1
            uptime_percent[i] = config.truck_uptime_year_1 * 100
        elseif i == 2
            uptime_percent[i] = config.truck_uptime_year_2 * 100
        else
            uptime_percent[i] = config.truck_uptime_default * 100
        end
    end

    # Add right y-axis with uptime
    plot!(twinx(p2), years, uptime_percent,
          label="Truck Uptime %",
          linewidth=2,
          linestyle=:dash,
          ylabel="Truck Uptime (%)",
          color=:orange,
          legend=(0.88, 0.95),
          marker=:square,
          markersize=4,
          ylims=(0, 100))

    # Plot 3: Hydrogen price over time
    p3 = plot(years, prices,
              linewidth=3,
              xlabel="Year",
              ylabel="Hydrogen Price (\$/kg)",
              title="Hydrogen Price Evolution (Example Run)",
              legend=false,
              size=(1200, 400),
              color=:red,
              marker=:circle,
              markersize=5)

    # Plot 4: Station capacity vs truck demand
    # Calculate truck demand with uptime (convert percentage to decimal)
    truck_demand = trucks .* 50.0 .* (uptime_percent ./ 100.0)  # Daily demand in kg/day

    p4 = plot(years, capacities,
              label="Station Capacity",
              linewidth=3,
              xlabel="Year",
              ylabel="Hydrogen (kg/day)",
              title="Daily H2 Capacity vs Truck Demand (Example Run)",
              legend=:topleft,
              size=(1200, 400),
              color=:green,
              marker=:circle,
              markersize=5)

    plot!(p4, years, truck_demand,
          label="Truck Demand (trucks × 50 kg/day × uptime)",
          linewidth=3,
          color=:purple,
          linestyle=:dash,
          marker=:square,
          markersize=5)

    # Calculate and add capacity utilization percentage on secondary y-axis
    utilization_percent = zeros(length(capacities))
    for i in 1:length(capacities)
        if capacities[i] > 0
            utilization_percent[i] = (truck_demand[i] / capacities[i]) * 100.0
        else
            utilization_percent[i] = 0.0
        end
    end

    plot!(twinx(p4), years, utilization_percent,
          label="Capacity Utilization %",
          linewidth=2,
          linestyle=:dot,
          ylabel="Utilization (%)",
          color=:orange,
          legend=(0.85, 0.95),
          marker=:diamond,
          markersize=4,
          ylims=(0, 120))

    # Combine all four plots vertically
    plot(p1, p2, p3, p4, layout=(4, 1), size=(1200, 1600))

    savefig("images/hydrogen_example_run.png")
    println("Example run plot saved as 'images/hydrogen_example_run.png'")

    return p1, p2, p3
end

"""
    plot_om_cost_curves(config, output_dir="images")

Plot station O&M cost curves as a function of nameplate capacity for three utilization
levels (30%, 50%, 80%), for both gaseous and liquid station types. Reference data points
are overlaid as markers to show the underlying dataset.

Args:
- config: Configuration NamedTuple (must contain station_cost_ref_data)
- output_dir: Output directory for the saved plot (default: "images")
"""
function plot_om_cost_curves(config, output_dir="images")
    if !isdir(output_dir)
        mkpath(output_dir)
    end

    ref = config.station_cost_ref_data

    # Capacity range for smooth fitted curves (kg/day)
    cap_range = 1000:100:20000

    util_levels  = [0.30, 0.50, 0.80]
    util_labels  = ["30% utilization", "50% utilization", "80% utilization"]
    util_colors  = [:royalblue, :darkorange, :crimson]

    # ── Gaseous panel ────────────────────────────────────────────────────────
    p_gas = plot(
        title         = "Gaseous Station (GH₂) O&M Cost vs. Capacity",
        xlabel        = "Nameplate capacity (kg/day)",
        ylabel        = "Annual O&M cost (USD/year)",
        legend        = :topleft,
        grid          = true,
        left_margin   = 20Plots.mm,
        bottom_margin = 12Plots.mm,
        top_margin    = 8Plots.mm,
        right_margin  = 6Plots.mm
    )

    gas_om_refs = [ref.gaseous_om_30pct, ref.gaseous_om_50pct, ref.gaseous_om]

    for (om_ref, label, color) in zip(gas_om_refs, util_labels, util_colors)
        # Smooth fitted curve
        curve = [linear_interpolate_extrapolate(Float64(c), ref.gaseous_capacities, om_ref)
                 for c in cap_range]
        plot!(p_gas, collect(cap_range), curve,
              label  = label,
              color  = color,
              linewidth = 2)
        # Reference data points
        scatter!(p_gas, ref.gaseous_capacities, om_ref,
                 label  = "",
                 color  = color,
                 marker = :circle,
                 markersize = 6)
    end

    # ── Liquid panel ─────────────────────────────────────────────────────────
    p_liq = plot(
        title         = "Liquid Station (LH₂) O&M Cost vs. Capacity",
        xlabel        = "Nameplate capacity (kg/day)",
        ylabel        = "Annual O&M cost (USD/year)",
        legend        = :topleft,
        grid          = true,
        left_margin   = 20Plots.mm,
        bottom_margin = 12Plots.mm,
        top_margin    = 8Plots.mm,
        right_margin  = 6Plots.mm
    )

    liq_om_refs = [ref.liquid_om_30pct, ref.liquid_om_50pct, ref.liquid_om]

    for (om_ref, label, color) in zip(liq_om_refs, util_labels, util_colors)
        curve = [linear_interpolate_extrapolate(Float64(c), ref.liquid_capacities, om_ref)
                 for c in cap_range]
        plot!(p_liq, collect(cap_range), curve,
              label  = label,
              color  = color,
              linewidth = 2)
        scatter!(p_liq, ref.liquid_capacities, om_ref,
                 label  = "",
                 color  = color,
                 marker = :circle,
                 markersize = 6)
    end

    combined = plot(p_gas, p_liq, layout=(1, 2), size=(1600, 600), dpi=150)
    savefig(combined, joinpath(output_dir, "station_om_cost_curves.png"))
    println("O&M cost curves plot saved as '$(joinpath(output_dir, "station_om_cost_curves.png"))'")

    return combined
end

"""
    plot_station_cost_vs_utilization(infrastructure_results, infrastructure_om_results, station_utilization_results, years, config, output_dir="images")

Generate station infrastructure cost vs utilization plots.
Creates two versions:
- One with only 2026 data
- One with all three representative years (2026, middle year, last year)

Args:
- infrastructure_results: Matrix (runs × years) of infrastructure CAPEX cost per kg
- infrastructure_om_results: Matrix (runs × years) of infrastructure O&M cost per kg
- station_utilization_results: Matrix (runs × years) of station utilization fraction
- years: Vector of years
- config: Configuration NamedTuple
- output_dir: Output directory for plots (default: "images")
"""
function plot_station_cost_vs_utilization(infrastructure_results, infrastructure_om_results, station_utilization_results, years, config, output_dir="images")
    # Ensure output directory exists
    if !isdir(output_dir)
        mkpath(output_dir)
    end

    n_runs = size(infrastructure_results, 1)
    n_years = length(years)

    # Select representative year indices (same as web chart)
    year_indices = []
    if n_years > 0
        push!(year_indices, 1)  # First year
    end
    if n_years > 2
        push!(year_indices, div(n_years, 2) + 1)  # Middle year
    end
    if n_years > 1
        push!(year_indices, n_years)  # Last year
    end

    # Generate utilization range from 30% to 100%
    utilization_range = 30:5:100

    # Colors for different years
    colors = [:red, :blue, :green]

    # Prepare data for all years
    all_curves_data = []

    for (idx, year_idx) in enumerate(year_indices)
        year = years[year_idx]

        # Read directly from the plain Float64 matrices
        infra_capex_values = infrastructure_results[:, year_idx]
        infra_om_values    = infrastructure_om_results[:, year_idx]
        utilization_values = station_utilization_results[:, year_idx]

        # Filter out zero-demand runs (no trucks deployed)
        valid = (infra_capex_values .> 0) .| (infra_om_values .> 0)
        infra_capex_values = infra_capex_values[valid]
        infra_om_values    = infra_om_values[valid]
        utilization_values = utilization_values[valid]

        if isempty(infra_capex_values)
            @warn "Skipping year $year: no valid data"
            continue
        end

        # Use mean values
        infra_capex_per_kg = mean(infra_capex_values)
        infra_om_per_kg    = mean(infra_om_values)
        total_infra_per_kg = infra_capex_per_kg + infra_om_per_kg
        actual_utilization = mean(utilization_values)
        total_capacity     = 0.0  # Not available directly; unused below

        # Skip if no valid data
        if total_infra_per_kg == 0 || actual_utilization == 0 || total_capacity == 0
            @warn "Skipping year $year: missing data"
            continue
        end

        # Calculate K constant
        K = total_infra_per_kg * actual_utilization

        # Calculate cost for each utilization level
        costs = Float64[]
        utils_plot = Float64[]

        for util_pct in utilization_range
            util = util_pct / 100.0
            cost = K / util
            push!(costs, cost)
            push!(utils_plot, util_pct)
        end

        push!(all_curves_data, (
            year = year,
            utilizations = utils_plot,
            costs = costs,
            color = colors[min(idx, length(colors))]
        ))
    end

    if isempty(all_curves_data)
        @warn "No valid data for station cost vs utilization plot"
        return
    end

    # Plot 1: Only 2026 (first year)
    if length(all_curves_data) >= 1
        first_year_data = all_curves_data[1]

        p1 = plot(first_year_data.utilizations, first_year_data.costs,
                 label="$(first_year_data.year)",
                 xlabel="Station Utilization (%)",
                 ylabel="Station Infrastructure\nCost Contribution (USD/kg)",
                 title="Station Infrastructure Cost vs. Utilization - \$(first_year_data.year)",
                 linewidth=3,
                 color=first_year_data.color,
                 legend=:topright,
                 grid=true,
                 size=(800, 600),
                 dpi=300,
                 xlims=(30, 100),
                 ylims=(0, maximum(first_year_data.costs) * 1.1),
                 left_margin=15Plots.mm,
                 bottom_margin=8Plots.mm,
                 top_margin=10Plots.mm)

        filepath1 = joinpath(output_dir, "station_cost_vs_utilization_2026.png")
        savefig(p1, filepath1)
        println("Saved: $filepath1")
    end

    # Calculate max cost for y-axis limit
    max_cost = maximum([maximum(curve.costs) for curve in all_curves_data])

    # Plot 2: All three years
    p2 = plot(xlabel="Station Utilization (%)",
             ylabel="Station Infrastructure\nCost Contribution (USD/kg)",
             title="Station Infrastructure Cost vs. Utilization",
             legend=:topright,
             grid=true,
             size=(800, 600),
             dpi=300,
             xlims=(30, 100),
             ylims=(0, max_cost * 1.1),
             left_margin=15Plots.mm,
             bottom_margin=8Plots.mm,
             top_margin=10Plots.mm)

    # Add all curves
    for curve_data in all_curves_data
        plot!(p2, curve_data.utilizations, curve_data.costs,
              label="$(curve_data.year)",
              linewidth=3,
              color=curve_data.color)
    end

    filepath2 = joinpath(output_dir, "station_cost_vs_utilization_all_years.png")
    savefig(p2, filepath2)
    println("Saved: $filepath2")

    return p1, p2
end

#==============================================================================
MAIN EXECUTION
==============================================================================#

function main()
    # Set random seed for reproducibility
    if !isnothing(RANDOM_SEED)
        Random.seed!(RANDOM_SEED)
    end

    # Load station configuration
    station_config = load_station_config(STATION_CONFIG_FILE)
    station_data = station_config.stations
    station_prob_config = station_config.prob_config
    station_cost_ref_data = station_config.cost_ref_data

    # Load truck deployment schedule
    truck_schedule = load_truck_deployment_schedule(TRUCK_CONFIG_FILE)

    # Load H2 pricing configuration
    if USE_H2_PRICE_SCHEDULE
        h2_price_schedule = load_h2_price_schedule(H2_CONFIG_FILE)

        # Extend schedule to cover full simulation period if needed
        max_schedule_year = maximum(keys(h2_price_schedule))
        if END_YEAR > max_schedule_year
            last_price = h2_price_schedule[max_schedule_year]
            for year in (max_schedule_year + 1):END_YEAR
                h2_price_schedule[year] = last_price
            end
            println("  Extended H2 price schedule from $max_schedule_year to $END_YEAR using last price (\$$last_price/kg)")
        end

        h2_price_curve_params = nothing
    elseif USE_H2_PRICE_CURVE
        h2_price_schedule = nothing
        h2_price_curve_params = (
            start_price = H2_START_PRICE,
            end_price = H2_END_PRICE,
            curve_type = H2_PRICE_CURVE_TYPE
        )
    else
        h2_price_schedule = nothing
        h2_price_curve_params = nothing
    end

    # Load LCFS configuration if enabled
    if USE_LCFS
        lcfs_config = load_lcfs_config(LCFS_CONFIG_FILE)
    else
        lcfs_config = nothing
    end

    # Configure LCFS price parameters (if LCFS is enabled)
    lcfs_price_curve_params = nothing
    if USE_LCFS
        if USE_LCFS_PRICE_CURVE
            lcfs_price_curve_params = (
                start_price = LCFS_START_PRICE,
                end_price = LCFS_END_PRICE,
                curve_type = LCFS_PRICE_CURVE_TYPE
            )
        end
        # Note: LCFS price schedule is already loaded in lcfs_config if it exists in the file
    end

    # Validate Monte Carlo runs
    if N_MONTE_CARLO_RUNS > 3000
        error("N_MONTE_CARLO_RUNS ($N_MONTE_CARLO_RUNS) cannot exceed 3000 to prevent excessive computation time")
    end
    if N_MONTE_CARLO_RUNS < 1
        error("N_MONTE_CARLO_RUNS ($N_MONTE_CARLO_RUNS) must be at least 1")
    end

    if INITIAL_TRUCKS > MAX_TRUCKS
        error("INITIAL_TRUCKS ($INITIAL_TRUCKS) cannot exceed MAX_TRUCKS ($MAX_TRUCKS)")
    end
    if TRUCK_UPTIME_YEAR_1 <= 0.0 || TRUCK_UPTIME_YEAR_1 > 1.0
        error("TRUCK_UPTIME_YEAR_1 ($TRUCK_UPTIME_YEAR_1) must be between 0 and 1")
    end
    if TRUCK_UPTIME_YEAR_2 <= 0.0 || TRUCK_UPTIME_YEAR_2 > 1.0
        error("TRUCK_UPTIME_YEAR_2 ($TRUCK_UPTIME_YEAR_2) must be between 0 and 1")
    end
    if TRUCK_UPTIME_DEFAULT <= 0.0 || TRUCK_UPTIME_DEFAULT > 1.0
        error("TRUCK_UPTIME_DEFAULT ($TRUCK_UPTIME_DEFAULT) must be between 0 and 1")
    end
    if TRUCK_PROB_MIDPOINT_STD < 0.0
        error("TRUCK_PROB_MIDPOINT_STD ($TRUCK_PROB_MIDPOINT_STD) must be non-negative")
    end

    # Package configuration into NamedTuple
    config = (
        use_station_config_file = USE_STATION_CONFIG_FILE,
        station_data = station_data,
        station_cost_ref_data = station_cost_ref_data,
        station_prob_config = station_prob_config,
        use_truck_deployment_schedule = USE_TRUCK_DEPLOYMENT_SCHEDULE,
        truck_deployment_schedule = truck_schedule,
        use_h2_price_schedule = USE_H2_PRICE_SCHEDULE,
        use_h2_price_curve = USE_H2_PRICE_CURVE,
        h2_price_schedule = h2_price_schedule,
        h2_price_curve_params = h2_price_curve_params,
        use_lcfs = USE_LCFS,
        lcfs_config = lcfs_config,
        h2_pathway_id = H2_PATHWAY_ID,
        expansion_pathway_id = nothing,
        projection_start_year = nothing,
        use_utilization_pricing = false,
        static_other_demand_tpd = 0.0,
        utilization_transport_cost = 1.0,
        electrolysis_pricing_enabled = false,
        use_lcfs_price_schedule = USE_LCFS_PRICE_SCHEDULE,
        use_lcfs_price_curve = USE_LCFS_PRICE_CURVE,
        lcfs_price_curve_params = lcfs_price_curve_params,
        start_year = START_YEAR,
        end_year = END_YEAR,
        max_stations = MAX_STATIONS,
        initial_stations = INITIAL_STATIONS,
        max_trucks = MAX_TRUCKS,
        initial_trucks = INITIAL_TRUCKS,
        station_capex = STATION_CAPEX,
        station_lifetime = STATION_LIFETIME,
        om_cost_per_station = OM_COST_PER_STATION,
        h2_per_truck_per_day = H2_PER_TRUCK_PER_DAY,
        operating_days_per_year = OPERATING_DAYS_PER_YEAR,
        h2_per_truck_per_year = H2_PER_TRUCK_PER_YEAR,
        truck_uptime_year_1 = TRUCK_UPTIME_YEAR_1,
        truck_uptime_year_2 = TRUCK_UPTIME_YEAR_2,
        truck_uptime_default = TRUCK_UPTIME_DEFAULT,
        discount_rate = DISCOUNT_RATE,
        transportation_cost_per_kg = TRANSPORTATION_COST_PER_KG,
        h2_production_transport_cost = H2_PRODUCTION_TRANSPORT_COST,
        station_prob_type = STATION_PROB_TYPE,
        station_prob_base = STATION_PROB_BASE,
        station_prob_slope = STATION_PROB_SLOPE,
        station_prob_steepness = STATION_PROB_STEEPNESS,
        station_prob_midpoint = STATION_PROB_MIDPOINT,
        station_prob_log_scale = STATION_PROB_LOG_SCALE,
        station_prob_log_factor = STATION_PROB_LOG_FACTOR,
        station_prob_exp_rate = STATION_PROB_EXP_RATE,
        truck_prob_type = TRUCK_PROB_TYPE,
        truck_prob_base = TRUCK_PROB_BASE,
        truck_prob_slope = TRUCK_PROB_SLOPE,
        truck_prob_steepness = TRUCK_PROB_STEEPNESS,
        truck_prob_midpoint = TRUCK_PROB_MIDPOINT,
        truck_prob_midpoint_std = TRUCK_PROB_MIDPOINT_STD,
        truck_prob_log_scale = TRUCK_PROB_LOG_SCALE,
        truck_prob_log_factor = TRUCK_PROB_LOG_FACTOR,
        truck_prob_exp_rate = TRUCK_PROB_EXP_RATE
    )

    # Print configuration
    println("\n" * "="^80)
    println("HYDROGEN TRUCK ROLLOUT MONTE CARLO SIMULATION")
    println("="^80)
    println("\nConfiguration:")
    println("  Time period: $(config.start_year) - $(config.end_year)")
    println("  Monte Carlo runs: $N_MONTE_CARLO_RUNS")

    println("  Station deployment: Configuration file ($(STATION_CONFIG_FILE))")
    println("  Number of stations: $(length(config.station_data))")
    n_operational = sum(s.already_operational for s in config.station_data)
    println("    - Already operational: $n_operational")
    println("    - Planned: $(length(config.station_data) - n_operational)")
    if !isnothing(config.station_prob_config.probability_by_status)
        println("  Opening probability by status:")
        for (status, prob) in config.station_prob_config.probability_by_status
            println("    - $(status): $(prob * 100)%")
        end
    end
    println("  Max delay for station opening: $(config.station_prob_config.max_delay_years) years")

    println("  Truck deployment: Predetermined schedule ($(TRUCK_CONFIG_FILE))")
    println("  Initial trucks: $(config.truck_deployment_schedule.initial_trucks)")
    total_scheduled = total_scheduled_additions(config.truck_deployment_schedule.schedule, config.end_year)
    println("  Total trucks scheduled to be added: $total_scheduled")
    println("  H2 per truck: $(config.h2_per_truck_per_year) kg/year")
    println("  Truck uptime: Year 1=$(config.truck_uptime_year_1 * 100)%, Year 2=$(config.truck_uptime_year_2 * 100)%, Year 3+=$(config.truck_uptime_default * 100)%")
    println("  Discount rate: $(config.discount_rate * 100)%")

    if config.use_h2_price_schedule
        println("  H2 pricing mode: Predetermined schedule ($(H2_CONFIG_FILE))")
    elseif config.use_h2_price_curve
        println("  H2 pricing mode: Dynamic curve ($(config.h2_price_curve_params.curve_type))")
        println("  H2 price range: \$$(config.h2_price_curve_params.start_price)/kg → \$$(config.h2_price_curve_params.end_price)/kg")
    else
        println("  H2 pricing mode: Fixed cost")
        println("  H2 production+transport cost: \$$(config.h2_production_transport_cost)/kg")
    end

    if config.use_lcfs
        pathway_name = config.lcfs_config.h2_pathways[config.h2_pathway_id].name
        pathway_ci = config.lcfs_config.h2_pathways[config.h2_pathway_id].ci
        println("  LCFS credits: Enabled")
        println("  H2 production pathway: $pathway_name (CI = $pathway_ci gCO2e/MJ)")
        println("  LCFS credit price: \$$(config.lcfs_config.credit_price)/credit")
    else
        println("  LCFS credits: Disabled")
    end
    println()

    # Run Monte Carlo simulation
    price_results, station_results, truck_results, capacity_results,
        base_price_results, infrastructure_results, infrastructure_government_results, infrastructure_om_results, lcfs_results, warnings,
        production_capacity_results, production_facilities_count_results,
        smr_facilities_count_results, electrolysis_facilities_count_results,
        weighted_ci_results, production_cost_results,
        hri_credits_results, station_utilization_results, credits_45v_results = run_monte_carlo(config, N_MONTE_CARLO_RUNS)

    # Analyze results
    years = config.start_year:config.end_year
    stats = analyze_results(price_results, collect(years))

    # Print warnings if any
    if !isempty(warnings)
        println("\n" * "="^80)
        println("⚠️  WARNINGS")
        println("="^80)
        for warning in warnings
            println("  ⚠️  $warning")
        end
        println()
    end

    # Print summary
    print_summary(stats)

    # Create plots
    plot_results(price_results, station_results, truck_results, capacity_results, stats, config)

    # Create example run plot
    plot_example_run(price_results, station_results, truck_results, capacity_results, collect(years), config, 1)

    # Create station cost vs utilization plots
    plot_station_cost_vs_utilization(infrastructure_results, infrastructure_om_results, station_utilization_results, collect(years), config)

    # Create O&M cost curves (capacity × utilization surface)
    plot_om_cost_curves(config)

    println("\nSimulation complete!")
    println("="^80)

    return price_results, station_results, truck_results, stats
end

# Run the simulation (only when file is executed directly, not when included)
if abspath(PROGRAM_FILE) == @__FILE__
    price_results, station_results, truck_results, stats = main()
end
