# =============================================================================
# FIGURE: SCENARIO INPUTS — SPLIT INTO SEPARATE PANELS
# =============================================================================
# Renders each panel of fig_scenario_inputs as its own standalone figure:
#   fig_scenario_truck_deployment.{png,pdf} — cumulative H₂ truck fleet
#   fig_scenario_lcfs_price.{png,pdf}        — LCFS credit price scenarios
#
# No simulation required — data read directly from config JSON files.
#
# Run from project root:
#   julia --project figures/fig_scenario_inputs_split.jl
# =============================================================================

using CairoMakie
using JSON

include("pub_theme.jl")

cd(joinpath(@__DIR__, ".."))

const OUT_DIR = joinpath(@__DIR__, "out")
mkpath(OUT_DIR)

# ── Load config ───────────────────────────────────────────────────────────────
trucks_raw = JSON.parsefile(joinpath(@__DIR__, "..", "config", "trucks_config.json"))
lcfs_raw   = JSON.parsefile(joinpath(@__DIR__, "..", "config", "lcfs_config.json"))
model_raw  = JSON.parsefile(joinpath(@__DIR__, "..", "config", "model_defaults.json"))

# Truck panel runs to the model's assessment horizon (`end_year` in
# model_defaults.json) so the plotted fleet matches what the simulation sees.
const END_YEAR    = Int(model_raw["simulation"]["end_year"])
const YEARS_TRUCK = 2026:END_YEAR

# LCFS panel stays on the authored schedule window; prices are held flat beyond
# its last year, so extending the axis would only add a horizontal tail.
const YEARS_LCFS = 2026:2040

# ── Truck deployment scenarios ────────────────────────────────────────────────
truck_scenarios = [
    (key = "nolow_dep", label = "No/Low deployment", color = C_DEP_NOLOW, ls = :solid),
    (key = "limited_dep", label = "Limited deployment",    color = C_DEP_LIM,  ls = :solid),
    (key = "high_dep",  label = "High deployment",   color = C_DEP_HIGH,  ls = :solid),
]

# Mirrors `scheduled_trucks_added` (hydrogen truck deployment.jl:871-882): past the
# last authored year the model extrapolates additions at the constant annual
# increment of the final 3-year segment. Without this the curve would flatten at
# the schedule's end instead of following the trajectory actually simulated.
const EXTRAP_LOOKBACK = 3

function scheduled_added(sched::Dict{Int,Int}, year::Int)
    isempty(sched) && return 0
    last_year = maximum(keys(sched))
    year <= last_year && return get(sched, year, 0)
    ref_year = last_year - EXTRAP_LOOKBACK
    last_add = get(sched, last_year, 0)
    ref_add  = get(sched, ref_year, get(sched, minimum(keys(sched)), 0))
    slope    = (last_add - ref_add) / EXTRAP_LOOKBACK
    return max(0, round(Int, last_add + slope * (year - last_year)))
end

function cumulative_fleet(key::String)
    s     = trucks_raw["truck_deployment_scenarios"][key]
    init  = s["initial_trucks"]
    sched = Dict{Int,Int}(e["year"] => e["trucks_added"] for e in s["schedule"])
    fleet = Int[]
    total = init
    for yr in YEARS_TRUCK
        total += scheduled_added(sched, yr)
        push!(fleet, total)
    end
    return fleet
end

# In-service fleet: replicates the model's 7-year truck lifetime
# (hydrogen truck deployment.jl:2114-2123). The initial fleet is deployed at the
# first simulation year; a cohort added in year Y operates through Y+6 (retired
# at age ≥ 7). This nets out retirements, unlike the cumulative-deployed curve.
const TRUCK_LIFETIME_YEARS = 7

function operating_fleet(key::String)
    s     = trucks_raw["truck_deployment_scenarios"][key]
    init  = s["initial_trucks"]
    sched = Dict{Int,Int}(e["year"] => e["trucks_added"] for e in s["schedule"])
    start = first(YEARS_TRUCK)
    fleet = Int[]
    for yr in YEARS_TRUCK
        total = 0
        # initial fleet deployed at the start year
        if (yr - start) < TRUCK_LIFETIME_YEARS
            total += init
        end
        # scheduled additions, each cohort retired after its lifetime
        for Y in start:yr
            if (yr - Y) < TRUCK_LIFETIME_YEARS
                total += scheduled_added(sched, Y)
            end
        end
        push!(fleet, total)
    end
    return fleet
end

# ── LCFS price scenarios ──────────────────────────────────────────────────────
lcfs_scenarios = [
    (key = "no_change", label = "No change (\$65/credit)", color = colorant"#4472C4", ls = :solid),
    (key = "high_inc",  label = "High increase (→\$250)",  color = colorant"#ED7D31", ls = :solid),
]

function lcfs_prices(key::String)
    s     = lcfs_raw["lcfs_price_scenarios"][key]
    sched = Dict{Int,Float64}(e["year"] => Float64(e["price_usd"]) for e in s["schedule"])
    return [get(sched, yr, NaN) for yr in YEARS_LCFS]
end

# ── Shared sizing ─────────────────────────────────────────────────────────────
years_truck = collect(YEARS_TRUCK)
years_lcfs  = collect(YEARS_LCFS)

# ── Panel (a) standalone: cumulative truck fleet ──────────────────────────────
fig_a = Figure(size = (W_SINGLE, 52 * MM_TO_PT))
ax_a = Axis(
    fig_a[1, 1];
    xlabel             = "Year",
    ylabel             = "Cumulative H₂ truck fleet",
    xlabelsize         = 9,
    ylabelsize         = 9,
    xticklabelsize     = 8,
    yticklabelsize     = 8,
    xticks             = year_ticks(first(YEARS_TRUCK), END_YEAR),
    xticklabelrotation = π / 4,
    ytickformat        = v -> ["$(Int(round(x / 1000)))k" for x in v],
    limits             = ((2026, END_YEAR), (0, nothing)),
)

for ts in truck_scenarios
    deployed  = cumulative_fleet(ts.key)
    operating = operating_fleet(ts.key)
    # solid = cumulative deployed (scenario label); dashed = in-service (7-yr life)
    lines!(ax_a, years_truck, deployed;
           color = ts.color, linewidth = 1.8, linestyle = :solid, label = ts.label)
    scatter!(ax_a, years_truck, deployed; color = ts.color, markersize = 3)
    lines!(ax_a, years_truck, operating;
           color = ts.color, linewidth = 1.8, linestyle = :dash)
end

# Single combined legend: scenario colours + linestyle key (solid = cumulative
# deployed, dashed = in-service fleet with 7-yr lifetime)
legend_elems = [
    [LineElement(color = ts.color, linewidth = 1.8) for ts in truck_scenarios];
    LineElement(color = :black, linestyle = :solid, linewidth = 1.8);
    LineElement(color = :black, linestyle = :dash,  linewidth = 1.8);
]
legend_labels = [
    [ts.label for ts in truck_scenarios];
    "Cumulative deployed";
    "In-service (7-yr life)";
]
axislegend(ax_a, legend_elems, legend_labels;
           position     = :lt,
           framevisible = true,
           labelsize    = 7,
           rowgap       = 0,
           patchsize    = (14, 8),
)

save_pub("fig_scenario_truck_deployment", fig_a)

# ── Panel (b) standalone: LCFS credit price ───────────────────────────────────
fig_b = Figure(size = (W_SINGLE, 82 * MM_TO_PT))
ax_b = Axis(
    fig_b[1, 1];
    xlabel             = "Year",
    ylabel             = "LCFS credit price (USD credit⁻¹)",
    xlabelsize         = 9,
    ylabelsize         = 9,
    xticklabelsize     = 8,
    yticklabelsize     = 8,
    xticks             = year_ticks(first(YEARS_LCFS), last(YEARS_LCFS)),
    xticklabelrotation = π / 4,
    limits             = ((2026, last(YEARS_LCFS)), (0, 270)),
    yticks             = 0:50:250,
)

for ls in lcfs_scenarios
    prices = lcfs_prices(ls.key)
    lines!(ax_b, years_lcfs, prices;
           color = ls.color, linewidth = 1.8, linestyle = ls.ls, label = ls.label)
    scatter!(ax_b, years_lcfs, prices; color = ls.color, markersize = 3)
end

axislegend(ax_b;
           position     = (0.98, 0.82),
           framevisible = true,
           labelsize    = 7,
           rowgap       = 0,
           patchsize    = (14, 8),
)

save_pub("fig_scenario_lcfs_price", fig_b)
