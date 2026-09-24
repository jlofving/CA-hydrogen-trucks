# =============================================================================
# FIGURE: TOTAL TCO PREMIUM — H2 vs. DIESEL (M USD / YEAR), by deployment
# =============================================================================
# Total TCO premium (H₂ − diesel, M USD/yr), one panel per truck-deployment scenario:
#   (a) limited_dep   (b) high_dep
# Grouped bars at snapshot years (2030 / 2035 / 2040 / END_YEAR). Encodings:
#   • Colour = H₂ pathway (SMR, grid electrolysis, solar electrolysis).
#   • Within each pathway, two dodged bars = diesel price ($4.80 no outline,
#     $5.80 black outline). The solid lower segment is the policy-inclusive Total
#     premium (LCFS credit + HRI + 45V on H₂; HVIP voucher in capital; diesel
#     reference LCFS-adjusted, rising over time).
#   • Hatched upper segment = effect of removing ALL policy support, stacked on
#     the Total. Bar top = fully unsubsidized resource cost (LCFS+HRI+45V added
#     back, HVIP removed, diesel flat — matches fig_cost_premium_3row); the hatch
#     thickness is the total support (the same at $4.80 and $5.80).
# No SCC enters this figure — it is the private/adoption framing, not a social CBA.
#
# Run from project root:
#   julia --project figures/fig_tco_premium.jl
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
mkpath(OUT_DIR)

# ─────────────────────────────────────────────────────────────────────────────
# TCO parameters
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

# Unsubsidized (resource-cost) variant — gross truck capital (no HVIP voucher) and
# an LCOH with the per-kg credits (LCFS + HRI + 45V) added back. The "no policy"
# line uses this; matches fig_cost_premium_3row's premium definition.
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

const DIESEL_MPG  = TCO_DIESEL_MPG   # miles/gallon, from config/tco_config.json
const DIESEL_P_LO = TCO_DIESEL_P_LO  # $/gal, from config/tco_config.json
const DIESEL_P_HI = TCO_DIESEL_P_HI  # $/gal, from config/tco_config.json
const DIESEL_MJ_PER_GAL = 128.45
const CA_BLEND_CI = 0.66 * 43.74 + 0.06 * 38.49 + (1.0 - 0.66 - 0.06) * 90.0

# Flat diesel TCO ($/mile), no LCFS — used for the "− LCFS" line.
diesel_tco_flat(p_gal) = p_gal / DIESEL_MPG + d_capital + d_rm + d_tires + d_common

# ─────────────────────────────────────────────────────────────────────────────
# Pathways + config builder (kind × truck schedule)
# ─────────────────────────────────────────────────────────────────────────────
pathways = [
    (label = "SMR (current mix)",    color = C_SMR,   kind = :smr),
    (label = "Electrolysis — grid",  color = C_GRID,  kind = :grid),
    (label = "Electrolysis — solar", color = C_SOLAR, kind = :solar),
]
lcfs_prices = load_lcfs_price_scenario("no_change")

# LCFS-on-diesel adjustment (anchored 2026): the CA diesel blend swings from LCFS
# credits to deficits as the benchmark CI declines, so diesel grows more expensive
# over time. Used for the "total" line; the "− LCFS" line uses flat diesel instead.
const _benchmark_ci = Dict{Int,Float64}(
    Int(d["year"]) => Float64(d["ci"])
    for d in JSON.parsefile(joinpath(@__DIR__, "..", "config", "lcfs_config.json"))["diesel_ci_schedule"]["schedule"]
)
function diesel_lcfs_per_mile(yr::Int)
    ci_std = get(_benchmark_ci, yr, _benchmark_ci[maximum(filter(y -> y <= yr, keys(_benchmark_ci)))])
    lcfs_p = get(lcfs_prices, yr, lcfs_prices[maximum(filter(y -> y <= yr, keys(lcfs_prices)))])
    return (ci_std - CA_BLEND_CI) / 1e6 * DIESEL_MJ_PER_GAL * lcfs_p / DIESEL_MPG   # $/mile
end
const _lcfs_diesel_2026 = diesel_lcfs_per_mile(2026)
diesel_tco_lcfs(p_gal, yr::Int) =
    p_gal / DIESEL_MPG + (_lcfs_diesel_2026 - diesel_lcfs_per_mile(yr)) +
    d_capital + d_rm + d_tires + d_common

function make_cfg(kind, sched)
    base = (h2_pathway_id = "current_mix", use_utilization_pricing = true, utilization_transport_cost = 1.0,
            use_lcfs = true, enable_45v = true, bus_demand_scenario = "growing", end_year = END_YEAR,
            use_truck_deployment_schedule = true, truck_deployment_schedule = sched,
            lcfs_price_schedule_dict = lcfs_prices)
    if kind == :smr
        build_config(; base..., expansion_pathway_id = "current_mix", electrolysis_pricing_enabled = false)
    elseif kind == :grid
        build_config(; base..., expansion_pathway_id = "electrolysis", electrolysis_pricing_enabled = true,
            electrolysis_electricity_source = "grid", electrolyzer_capex_per_kw = 3000.0,
            electricity_cost_per_kwh = 0.2, electrolyzer_learning_rate = 0.233,
            electrolyzer_stack_fraction = 0.60, bop_learning_rate = 0.04)
    else
        build_config(; base..., expansion_pathway_id = "electrolysis", electrolysis_pricing_enabled = true,
            electrolysis_electricity_source = "solar", electrolyzer_capex_per_kw = 3000.0,
            solar_capex_per_kw = 1600.0, solar_capacity_factor = 0.25, solar_lifetime = 25,
            electrolyzer_learning_rate = 0.233, electrolyzer_stack_fraction = 0.60, bop_learning_rate = 0.04,
            solar_panel_learning_rate = 0.267, solar_panel_fraction = 0.80, solar_bop_learning_rate = 0.04)
    end
end

# Run the 3 pathways for a deployment schedule → TCO-premium summary per pathway
function premiums(scenario_key)
    sched = load_truck_scenario(scenario_key)
    map(pathways) do pw
        cfg = make_cfg(pw.kind, sched)
        Random.seed!(SEED)
        raw = run_monte_carlo(cfg, N_RUNS)
        price_r = raw[1];  lcfs_r = raw[9];  hri_r = raw[17];  v45_r = raw[19];  truck_r = Float64.(raw[3])
        years = collect(cfg.start_year : cfg.end_year);  n = length(years)
        mi_yr = truck_r .* Float64(H2_PER_TRUCK_PER_DAY) .* Float64(OPERATING_DAYS_PER_YEAR) .* miles_per_kg_h2
        # Unsubsidized H₂ price: add back ALL per-kg credits (LCFS + HRI + 45V).
        price_unsub = price_r .+ lcfs_r .+ hri_r .+ v45_r
        # Total @ $4.80: policy-inclusive, diesel LCFS-adjusted — all policy active.
        tot = [(h2_tco_prem(price_r[r,k], years[k]) - diesel_tco_lcfs(DIESEL_P_LO, years[k])) * mi_yr[r,k] / 1e6 for r in 1:N_RUNS, k in 1:n]
        # Total @ $5.80: same, higher diesel pump price (→ lower premium). Diesel-price comparison.
        tot_hi = [(h2_tco_prem(price_r[r,k], years[k]) - diesel_tco_lcfs(DIESEL_P_HI, years[k])) * mi_yr[r,k] / 1e6 for r in 1:N_RUNS, k in 1:n]
        # No policy: unsubsidized resource cost (LCFS+HRI+45V+HVIP all removed), diesel flat,
        # at each pump price so both diesel scenarios get a no-policy band.
        res    = [(h2_tco_unsub(price_unsub[r,k], years[k]) - diesel_tco_flat(DIESEL_P_LO)) * mi_yr[r,k] / 1e6 for r in 1:N_RUNS, k in 1:n]
        res_hi = [(h2_tco_unsub(price_unsub[r,k], years[k]) - diesel_tco_flat(DIESEL_P_HI)) * mi_yr[r,k] / 1e6 for r in 1:N_RUNS, k in 1:n]
        (years      = years,
         med        = [median(tot[:, k])    for k in 1:n],
         med_hi     = [median(tot_hi[:, k]) for k in 1:n],
         med_res    = [median(res[:, k])    for k in 1:n],
         med_res_hi = [median(res_hi[:, k]) for k in 1:n])
    end
end

println("Running 3 pathways × 2 deployments ($(N_RUNS) runs each)…")
res_demo = premiums("limited_dep");  println("  limited_dep done")
res_high = premiums("high_dep");   println("  high_dep done")
deploys  = [("(a)  Limited deployment", res_demo), ("(b)  High deployment", res_high)]

# ─────────────────────────────────────────────────────────────────────────────
# Bar layout — snapshot years on x; within each year, the 3 pathways (colour); within
# each pathway, two dodged bars for diesel $4.80 (no outline) and $5.80 (black outline).
# Each bar is stacked: the solid lower segment is the policy-inclusive Total premium,
# and the hatched upper segment is the additional cost when ALL policy support is
# removed (top of bar = unsubsidized resource cost; hatch thickness = total support).
# ─────────────────────────────────────────────────────────────────────────────
const SNAP_YEARS = [2030, 2035, 2040, END_YEAR]
snap_idx(r) = [findfirst(==(y), r.years) for y in SNAP_YEARS]

# Shared y-limit (the hatched no-policy top edge is highest) across the snapshot years.
ymax = maximum(max(r.med_res[k], r.med_res_hi[k])
               for D in (res_demo, res_high) for r in D for k in snap_idx(r)) * 1.08

# Bar geometry within a panel.
const YC       = Float64.(0:length(SNAP_YEARS)-1) .* 3.6   # one cluster centre per snapshot year
const PATH_OFF = [-0.95, 0.0, 0.95]                        # pathway offsets within a year cluster
const D_OFF    = [-0.20, 0.20]                             # diesel offsets within a pathway (4.80, 5.80)
const BW       = 0.36                                      # bar width

hatch(col) = Pattern('/'; linecolor = col, backgroundcolor = (col, 0.10),
                      width = 1.0f0, tilesize = (7, 7))

fig = Figure(size = (W_DOUBLE, 95 * MM_TO_PT))

path_elems    = [PolyElement(color = pw.color) for pw in pathways]
path_labels   = [pw.label for pw in pathways]
diesel_elems  = [PolyElement(color = (:gray, 0.55)),
                 PolyElement(color = (:gray, 0.55), strokecolor = :black, strokewidth = 1)]
diesel_labels = [@sprintf("Diesel \$%.2f/gal", DIESEL_P_LO), @sprintf("Diesel \$%.2f/gal", DIESEL_P_HI)]
seg_elems     = [PolyElement(color = (:gray, 0.55)),
                 PolyElement(color = hatch(:gray))]
seg_labels    = ["Total (policy-inclusive)", "Added cost w/o policy"]

for (j, (title, D)) in enumerate(deploys)
    ax = Axis(fig[1, j];
        title              = title,
        titlesize          = 8,
        xlabel             = "Year",
        ylabel             = j == 1 ? L"Total TCO premium (M USD yr$^{-1}$)" : "",
        xticks             = (YC, string.(SNAP_YEARS)),
        yticklabelsvisible = j == 1,
        limits             = ((YC[1] - 1.7, YC[end] + 1.7), (0, ymax)),
    )
    hlines!(ax, 0; color = (:black, 0.4), linewidth = 0.8, linestyle = :dot)
    for (pi, (pw, r)) in enumerate(zip(pathways, D))
        for (ci, k) in enumerate(snap_idx(r))
            reg = (r.med[k],     r.med_hi[k])          # policy-inclusive Total ($4.80, $5.80)
            top = (r.med_res[k], r.med_res_hi[k])      # unsubsidised resource cost (bar top)
            for di in 1:2
                xc  = YC[ci] + PATH_OFF[pi] + D_OFF[di]
                x0  = xc - BW/2
                skw = di == 2 ? 1.0 : 0.0              # $5.80 gets a black outline
                # Lower segment: solid Total premium.
                poly!(ax, Rect2f(x0, 0.0, BW, reg[di]);
                      color = pw.color, strokecolor = :black, strokewidth = skw)
                # Upper segment: hatched added cost when policy is removed.
                poly!(ax, Rect2f(x0, reg[di], BW, top[di] - reg[di]);
                      color = hatch(pw.color), strokecolor = :black, strokewidth = skw)
            end
        end
    end
    j == 1 && axislegend(ax, [path_elems, diesel_elems, seg_elems], [path_labels, diesel_labels, seg_labels],
        ["Pathway (colour)", "Diesel price (outline)", "Bar segment"];
        position = :lt, rowgap = 0, labelsize = 6, titlesize = 6.5, framevisible = true,
        titlegap = 1, groupgap = 4, patchsize = (12, 8))
end

colgap!(fig.layout, 6)
resize_to_layout!(fig)
save_pub("fig_tco_premium", fig)
