# =============================================================================
# FIGURE: GLOBAL SENSITIVITY — LATIN HYPERCUBE SAMPLE
# =============================================================================
# Global (all-at-once) counterpart to the one-at-a-time tornado grid in
# fig_sensitivity.jl / manuscript fig 10. Where that figure moves one parameter
# to each end of its bracket, this one samples the whole parameter space jointly
# and reports (a) the resulting output distribution and (b) how much of that
# distribution's variance each parameter explains.
#
# Metrics — the same two as fig 10, plus the timing claim the manuscript makes:
#   H₂ delivery price ($/kg)   |   H₂ truck TCO ($/mile)   |   diesel-parity year
#
# ── Two nested uncertainty sources ───────────────────────────────────────────
# The model is stochastic independently of any parameter in the sample:
# investment triggers, Bernoulli build decisions and 2–4 year facility lead
# times all draw randomly inside run_single_simulation. That is *aleatory*
# uncertainty; the parameters sampled below are *epistemic*. The two are kept
# separate rather than blended:
#
#   • Each LHS point is evaluated as the median of M_INNER stochastic runs, so
#     the point estimate is a clean function of the parameters.
#   • The seed is reset to SEED at every LHS point (common random numbers), so
#     differences between points are parametric, not Monte Carlo noise.
#   • The within-point spread is retained alongside the median, which lets the
#     summary report what share of total variance is deployment timing rather
#     than parameter ignorance — see the VARIANCE DECOMPOSITION block.
#
# ── Sampling ─────────────────────────────────────────────────────────────────
# Continuous parameters are triangular on the same endpoints fig 10 uses for its
# low/high variants, with the mode at the model default. Triangular rather than
# uniform because those endpoints are plausible extremes, not equally likely
# alternatives; the mode keeps the calibrated default as the central case.
#
# Two policy switches stay discrete and are sampled as ordered categories
# (45V: off < baseline < extended; bus demand: flat < growing), so they appear
# in the sensitivity ranking on the same footing as the continuous parameters.
# Their levels are given equal probability — an uninformative prior over policy
# states, not a forecast. Because the output distribution therefore mixes policy
# worlds, the summary also reports quantiles conditioned on the baseline policy
# state, and that conditional band is what the figure draws.
#
# ── Sensitivity measure ──────────────────────────────────────────────────────
# Standardised RANK regression coefficients (SRRC), which an LHS supports
# directly. Ranks rather than levels because several responses are monotone but
# distinctly non-linear (learning rates compound; the parity year is censored).
# The model R² is reported with every table: it is the share of output variance
# the rank-linear summary captures, and a low R² is the signal that the SRRCs
# understate interaction effects and should not be read as a full decomposition.
#
# Run from project root (needs the figures environment):
#   julia --project=figures figures/fig_sensitivity_lhs.jl
#   julia --project=figures figures/fig_sensitivity_lhs.jl high_dep
#
# Environment overrides for quick iteration:
#   LHS_N=200 LHS_M=25 julia --project=figures figures/fig_sensitivity_lhs.jl
# =============================================================================

using Statistics
using CairoMakie
using LaTeXStrings
using Random
using JSON
using Printf
using LinearAlgebra

cd(joinpath(@__DIR__, ".."))
include("../hydrogen truck deployment.jl")
include("config_defaults.jl")
include("pub_theme.jl")

# N_LHS × M_INNER model runs. At ~4.5 ms per run, 1500 × 100 is about 11 minutes.
# N is set well above the usual 10×k rule (k = 15 parameters) because the bands
# in panels (a)–(b) use only the baseline-policy subsample — 1/6 of the points,
# since 45V has three equiprobable levels and bus demand two — so ~250 points
# reach the quantiles that are actually drawn.
const N_LHS   = parse(Int, get(ENV, "LHS_N", "1500"))   # LHS points (epistemic)
const M_INNER = parse(Int, get(ENV, "LHS_M", "100"))    # stochastic runs per point
const SEED    = 42
const OUT_DIR = joinpath(@__DIR__, "out")
mkpath(OUT_DIR)

const DEP_SCENARIO = isempty(ARGS) ? get(ENV, "SENS_DEP", "limited_dep") : ARGS[1]
const DEP_SUFFIX   = DEP_SCENARIO == "limited_dep" ? "" : "_" * DEP_SCENARIO
const DEP_LABEL    = Dict("limited_dep" => "Limited deployment",
                          "high_dep"  => "High deployment",
                          "nolow_dep" => "No/low deployment")
dep_title() = get(DEP_LABEL, DEP_SCENARIO, DEP_SCENARIO)

println("Deployment scenario: $(dep_title())  [$DEP_SCENARIO]")
println("LHS: $N_LHS points × $M_INNER stochastic runs = $(N_LHS * M_INNER) simulations")

# ── TCO helpers ───────────────────────────────────────────────────────────────
# Mirrors the block in fig_sensitivity.jl, extended so that fuel economy and the
# discount rate are per-sample arguments rather than module constants. If this
# figure is kept, the two copies should be factored into a shared include.
let
    tco_raw = JSON.parsefile(joinpath(@__DIR__, "..", "config", "tco_config.json"))
    d   = tco_raw["diesel_tco_usd_per_mile"]
    ht  = tco_raw["hydrogen_truck"]
    id  = tco_raw["identical_cost_categories"]

    global _d_lease  = Float64(d["truck_lease_or_purchase"])
    global _d_rm     = Float64(d["repair_and_maintenance"])
    global _d_tires  = Float64(d["tires"])
    global _d_common = Float64(id["driver_and_other"])

    global _mpkg_base = Float64(ht["miles_per_kg_h2"])
    global _h2_tires  = Float64(ht["tires_multiplier"]) * _d_tires
    global _h2_common = _d_common
    global _rm_mult   = Float64(ht["repair_and_maintenance_multiplier"])

    global _tco_lr      = Float64(ht["learning"]["learning_rate"])
    global _tco_ref_yr  = Int(ht["learning"]["reference_year"])
    global _tco_purch   = Float64(ht["purchase_cost_usd"])
    global _tco_plat    = Float64(ht["platform_cost_usd"])

    global _sub_sched = Dict{Int,Float64}(
        Int(s["year"]) => Float64(s["subsidy_usd"]) for s in ht["subsidy_schedule"])

    obs = ht["learning"]["observed_fleet_stock"]
    oy  = Float64.([Int(r["year"])   for r in obs])
    os  = Float64.([Int(r["trucks"]) for r in obs])
    ts  = oy .- 2019.0
    V   = hcat(ts.^3, ts.^2, ts, ones(length(ts)))
    θ   = V \ os
    global _fsa, _fsb, _fsc, _fsd = θ[1], θ[2], θ[3], θ[4]
end

_sub(yr)  = _sub_sched[maximum(filter(y -> y <= yr, keys(_sub_sched)))]
_fs(yr)   = max(1.0, _fsa*(yr-2019)^3 + _fsb*(yr-2019)^2 + _fsc*(yr-2019) + _fsd)
_h2rm(yr) = _d_rm * (_rm_mult + (1.0 - _rm_mult) * clamp((yr - 2026) / 9.0, 0.0, 1.0))

# TCO per mile, with every sampled quantity passed in explicitly.
#   mpkg          — truck fuel economy (miles/kg H₂); also rescales annual miles
#   subsidy_scale — multiplier on the HVIP phase-out schedule (1.0 = as legislated)
#   discount_rate — capital recovery rate, shared with the production side
function h2_tco_per_mile(lcoh::Float64, yr::Int;
                          h2_per_day::Float64    = Float64(H2_PER_TRUCK_PER_DAY),
                          op_days::Int           = OPERATING_DAYS_PER_YEAR,
                          mpkg::Float64          = _mpkg_base,
                          purchase_cost::Float64 = _tco_purch,
                          truck_lr::Float64      = _tco_lr,
                          subsidy_scale::Float64 = 1.0,
                          discount_rate::Float64 = DISCOUNT_RATE)
    ann_miles = h2_per_day * op_days * mpkg
    fc_base   = purchase_cost - _tco_plat
    α         = log(1 / (1 - truck_lr)) / log(2)
    fc_yr     = fc_base * (_fs(yr) / _fs(_tco_ref_yr))^(-α)
    net_cost  = _tco_plat + fc_yr - subsidy_scale * _sub(yr)
    capital   = net_cost * calculate_annuity_factor(7, discount_rate) / ann_miles
    return lcoh / mpkg + capital + _h2rm(yr) + _h2_tires + _h2_common
end

# ── Diesel reference ──────────────────────────────────────────────────────────
# As in fig_sensitivity.jl, but the LCFS credit price is the sampled one rather
# than the fixed no_change scenario: the same credit price has to apply to the
# diesel blend and to hydrogen, or the LCFS parameter gets a one-sided effect.
# Anchored at 2026 — the pump prices are taken to embed 2026 LCFS conditions, so
# only the change relative to 2026 adjusts later years.
const DIESEL_MPG        = TCO_DIESEL_MPG
const DIESEL_MJ_PER_GAL = 128.45
const CA_BLEND_CI       = 0.66 * 43.74 + 0.06 * 38.49 + (1.0 - 0.66 - 0.06) * 90.0
const _benchmark_ci = Dict{Int,Float64}(
    Int(d["year"]) => Float64(d["ci"])
    for d in JSON.parsefile(joinpath(@__DIR__, "..", "config", "lcfs_config.json"))["diesel_ci_schedule"]["schedule"])

_ci_std(yr) = get(_benchmark_ci, yr,
                  _benchmark_ci[maximum(filter(y -> y <= yr, keys(_benchmark_ci)))])

# $/mile adjustment to the diesel pump price, + = diesel dearer than in 2026.
# Linear in the credit price, since the blend CI is fixed and only the benchmark
# declines: (CI_2026 − CI_yr) × MJ/gal × price / 1e6 / mpg.
diesel_lcfs_adj(yr::Int, lcfs_price::Float64) =
    (_ci_std(2026) - _ci_std(yr)) / 1e6 * DIESEL_MJ_PER_GAL * lcfs_price / DIESEL_MPG

diesel_tco_total(p_gal::Float64, yr::Int, lcfs_price::Float64) =
    p_gal / DIESEL_MPG + diesel_lcfs_adj(yr, lcfs_price) +
    _d_lease + _d_rm + _d_tires + _d_common

# ── Parameter space ───────────────────────────────────────────────────────────
# Continuous: triangular(a, mode = c, b). Endpoints match the fig 10 low/high
# variants wherever fig 10 has that parameter; modes are the model defaults.
const TRI_PARAMS = [
    (key = :annual_miles,  label = "Annual miles per truck",
     a =  74_000.0, c =  85_500.0, b =  97_000.0),
    (key = :mpkg,          label = "Truck fuel economy (mi/kg)",
     a = 0.9 * _mpkg_base, c = _mpkg_base, b = 1.1 * _mpkg_base),
    (key = :elec_capex,    label = "Electrolyzer CAPEX (\$/kW)",
     a =   2000.0, c =   3000.0, b =   4000.0),
    (key = :solar_capex,   label = "Solar panel CAPEX (\$/kW)",
     a =   1100.0, c =   1600.0, b =   2100.0),
    (key = :elec_lr,       label = "Electrolyzer learning rate",
     a =     0.15, c =    0.233, b =     0.30),
    (key = :solar_lr,      label = "Solar PV learning rate",
     a =     0.12, c =    0.267, b =     0.45),
    (key = :transport,     label = "H₂ transport cost (\$/kg)",
     a =     0.50, c =     1.00, b =     2.00),
    # fig 10 treats the LCFS as two named scenarios; here it is continuous, a
    # flat credit price over the whole horizon. Endpoints span a modest discount
    # to the no_change $65 baseline up to the $250 reached by the high_inc
    # scenario, which is the ceiling side of config/lcfs_config.json.
    (key = :lcfs_price,    label = "LCFS credit price (\$/credit)",
     a =     50.0, c =     65.0, b =    250.0),
    (key = :truck_cost,    label = "Truck purchase cost (\$)",
     a = 450_000.0, c = 600_000.0, b = 850_000.0),
    (key = :truck_lr,      label = "Truck FC learning rate",
     a =     0.10, c =     0.20, b =     0.30),
    # fig 10's high variant is "$240 000 throughout", which is not a scaling of
    # the phase-out schedule and so has no single mode. Sampled instead as a
    # multiplier on the legislated schedule, mode 1.0 = exactly the baseline.
    (key = :hvip_scale,    label = "HVIP subsidy (× schedule)",
     a =      0.0, c =      1.0, b =      1.5),
    (key = :discount,      label = "Discount rate",
     a =     0.04, c = DISCOUNT_RATE, b = 0.12),
    # Enters the parity comparison only — it does not affect the H₂ price or the
    # H₂ TCO. Mode at the $4.80 ATRI baseline, upper endpoint above the $5.80
    # high case of the standing diesel bracket.
    (key = :diesel_price,  label = "Diesel price (\$/gal)",
     a =     4.00, c = TCO_DIESEL_P_LO, b = 6.50),
]

# Discrete, sampled as ordered categories (least → most favourable to H₂), with
# equal probability per level. `base_level` is the index matching fig 10's
# baseline, used for the conditional-on-baseline-policy quantiles.
const CAT_PARAMS = [
    (key = :v45, label = "45V tax credit",
     levels = [:off, :base, :ext], probs = [1/3, 1/3, 1/3], base_level = 2),
    (key = :bus, label = "Bus H₂ demand growth",
     levels = [:flat, :growing], probs = [0.5, 0.5], base_level = 2),
]

const PARAM_LABELS = vcat([p.label for p in TRI_PARAMS], [p.label for p in CAT_PARAMS])
const N_PARAM      = length(PARAM_LABELS)

# ── Latin hypercube ───────────────────────────────────────────────────────────
# One stratified, independently permuted column per parameter: stratum i of
# column j receives a uniform draw from [(i-1)/n, i/n). Marginals are exact by
# construction; columns are independent (no rank correlation imposed).
function lhs_unit(rng, n::Int, k::Int)
    U = Matrix{Float64}(undef, n, k)
    for j in 1:k
        perm = randperm(rng, n)
        for i in 1:n
            U[i, j] = (perm[i] - 1 + rand(rng)) / n
        end
    end
    return U
end

# Inverse CDF of triangular(a, mode c, b). Handles c == a and c == b.
function tri_inv(u::Float64, a::Float64, c::Float64, b::Float64)
    b <= a && return a
    fc = (c - a) / (b - a)
    return u <= fc ? a + sqrt(u * (b - a) * (c - a)) :
                     b - sqrt((1 - u) * (b - a) * (b - c))
end

# Inverse CDF of a categorical with the given probabilities → level index.
function cat_inv(u::Float64, probs::Vector{Float64})
    acc = 0.0
    for (i, p) in enumerate(probs)
        acc += p
        u < acc && return i
    end
    return length(probs)
end

# ── Evaluate one LHS point ────────────────────────────────────────────────────
# A flat LCFS credit price across the whole horizon, which is the continuous
# generalisation of the flat $65 no_change baseline.
flat_lcfs(price::Float64) = Dict{Int,Float64}(y => price for y in START_YEAR:END_YEAR)

const BASE_KW = (
    h2_pathway_id                   = "current_mix",
    expansion_pathway_id            = "electrolysis",
    use_utilization_pricing         = true,
    use_lcfs                        = true,
    enable_45v                      = true,
    end_year                        = END_YEAR,
    use_truck_deployment_schedule   = true,
    truck_deployment_schedule       = load_truck_scenario(DEP_SCENARIO),
    electrolysis_pricing_enabled    = true,
    electrolysis_electricity_source = "solar",
    solar_capacity_factor           = 0.25,
    solar_lifetime                  = 25,
    electrolyzer_stack_fraction     = 0.60,
    bop_learning_rate               = 0.04,
    solar_panel_fraction            = 0.80,
    solar_bop_learning_rate         = 0.04,
)

const REPORT_YEARS = [2026, 2030, 2035, END_YEAR]

# First year the H₂ TCO falls to or below the diesel TCO, linearly interpolated
# between annual points. `nothing` when it never happens inside the window —
# never clamped to an endpoint, matching the convention in
# fig_societal_cost_benefit.jl.
function crossing_year(years, vals)
    for i in 2:length(vals)
        a, b = vals[i-1], vals[i]
        if a > 0 && b <= 0
            return years[i-1] + (a / (a - b)) * (years[i] - years[i-1])
        end
    end
    return nothing
end

function evaluate(s)
    sim = (
        electrolyzer_capex_per_kw   = s.elec_capex,
        solar_capex_per_kw          = s.solar_capex,
        electrolyzer_learning_rate  = s.elec_lr,
        solar_panel_learning_rate   = s.solar_lr,
        utilization_transport_cost  = s.transport,
        h2_per_truck_per_day        = s.annual_miles / (OPERATING_DAYS_PER_YEAR * s.mpkg),
        discount_rate               = s.discount,
        lcfs_price_schedule_dict    = flat_lcfs(s.lcfs_price),
        bus_demand_scenario         = s.bus === :flat ? "flat" : "growing",
        enable_45v                  = s.v45 !== :off,
    )
    s.v45 === :ext && (sim = merge(sim, (tax_credit_45v_end_year = 2038,)))

    cfg   = build_config(; merge(BASE_KW, sim)...)
    years = collect(cfg.start_year:cfg.end_year)

    # Common random numbers: the same stochastic draws at every LHS point, so
    # differences across points are parametric rather than Monte Carlo noise.
    # run_monte_carlo reports per-run progress, which at N_LHS points would be
    # millions of lines, so its stdout is dropped — the outer loop prints the
    # progress that matters.
    Random.seed!(SEED)
    raw     = redirect_stdout(devnull) do
        run_monte_carlo(cfg, M_INNER)
    end
    price_r = raw[1]

    h2pd = Float64(cfg.h2_per_truck_per_day)
    opd  = cfg.operating_days_per_year

    lcoh_med = Vector{Float64}(undef, length(years))
    lcoh_sd  = similar(lcoh_med)
    tco_med  = similar(lcoh_med)
    tco_sd   = similar(lcoh_med)
    for k in eachindex(years)
        col = @view price_r[:, k]
        lcoh_med[k] = median(col)
        lcoh_sd[k]  = std(col)
        tvec = [h2_tco_per_mile(col[r], years[k];
                                h2_per_day = h2pd, op_days = opd, mpkg = s.mpkg,
                                purchase_cost = s.truck_cost, truck_lr = s.truck_lr,
                                subsidy_scale = s.hvip_scale, discount_rate = s.discount)
                for r in 1:M_INNER]
        tco_med[k] = median(tvec)
        tco_sd[k]  = std(tvec)
    end

    gap    = [tco_med[k] - diesel_tco_total(s.diesel_price, years[k], s.lcfs_price)
              for k in eachindex(years)]
    parity = crossing_year(Float64.(years), gap)

    idx = [findfirst(==(y), years) for y in REPORT_YEARS]
    return (years = years,
            lcoh = lcoh_med, tco = tco_med,
            lcoh_at = lcoh_med[idx], tco_at = tco_med[idx],
            lcoh_sd_at = lcoh_sd[idx], tco_sd_at = tco_sd[idx],
            parity = parity)
end

# ── Draw the sample and run it ────────────────────────────────────────────────
rng = MersenneTwister(SEED)
U   = lhs_unit(rng, N_LHS, N_PARAM)

samples = Vector{NamedTuple}(undef, N_LHS)
X       = Matrix{Float64}(undef, N_LHS, N_PARAM)   # design matrix, for the SRRCs
for i in 1:N_LHS
    vals = Dict{Symbol,Any}()
    for (j, p) in enumerate(TRI_PARAMS)
        v = tri_inv(U[i, j], p.a, p.c, p.b)
        vals[p.key] = v
        X[i, j]     = v
    end
    for (jj, p) in enumerate(CAT_PARAMS)
        j = length(TRI_PARAMS) + jj
        li = cat_inv(U[i, j], p.probs)
        vals[p.key] = p.levels[li]
        X[i, j] = Float64(li)          # ordered levels → rank measures are valid
    end
    samples[i] = NamedTuple(vals)
end

println("Evaluating $N_LHS LHS points…")
results = Vector{Any}(undef, N_LHS)
let t0 = time()
    for i in 1:N_LHS
        results[i] = evaluate(samples[i])
        if i % 25 == 0 || i == N_LHS
            el = time() - t0
            @printf("  %4d/%d   %.0f s elapsed, %.0f s remaining\n",
                    i, N_LHS, el, el / i * (N_LHS - i))
        end
    end
end
println("Sample complete.\n")

const YEARS   = results[1].years
const N_YEARS = length(YEARS)

# Output matrices: rows = LHS points, columns = years
lcoh_all = [results[i].lcoh[k] for i in 1:N_LHS, k in 1:N_YEARS]
tco_all  = [results[i].tco[k]  for i in 1:N_LHS, k in 1:N_YEARS]
parity   = [isnothing(results[i].parity) ? NaN : results[i].parity for i in 1:N_LHS]

# Baseline policy state, for the conditional band the figure draws.
base_mask = [samples[i].v45 === CAT_PARAMS[1].levels[CAT_PARAMS[1].base_level] &&
             samples[i].bus === CAT_PARAMS[2].levels[CAT_PARAMS[2].base_level]
             for i in 1:N_LHS]

# ── Rank statistics ───────────────────────────────────────────────────────────
# Average ranks, ties shared. Implemented here because StatsBase is not a
# dependency of the figures environment.
function tiedrank(v::AbstractVector{<:Real})
    n = length(v)
    p = sortperm(v)
    r = Vector{Float64}(undef, n)
    i = 1
    while i <= n
        j = i
        while j < n && v[p[j+1]] == v[p[i]]
            j += 1
        end
        avg = (i + j) / 2
        for t in i:j
            r[p[t]] = avg
        end
        i = j + 1
    end
    return r
end

"""
    srrc(X, y) -> (β, R²)

Standardised rank regression coefficients: rank-transform every input column and
the response, standardise, and regress. β[j] is the change in the response's
standardised rank per standard deviation of parameter j's rank, holding the
others fixed — comparable across parameters regardless of units.

R² is the share of response-rank variance the rank-linear model explains. It is
the honesty check on the whole table: at high R² the βs are a near-complete
variance decomposition, at low R² they miss interactions and non-monotonicity.
Columns with no variation (a parameter held fixed) are dropped to keep the
normal equations non-singular, and reported with β = 0.
"""
function srrc(X::AbstractMatrix, y::AbstractVector)
    keep = [std(@view X[:, j]) > 0 for j in 1:size(X, 2)]
    n    = length(y)
    R    = hcat([tiedrank(@view X[:, j]) for j in findall(keep)]...)
    Z    = (R .- mean(R, dims = 1)) ./ std(R, dims = 1)
    ry   = tiedrank(y)
    sy   = std(ry)
    sy == 0 && return zeros(size(X, 2)), 0.0
    zy   = (ry .- mean(ry)) ./ sy
    A    = hcat(ones(n), Z)
    coef = A \ zy
    res  = zy .- A * coef
    R2   = 1 - sum(abs2, res) / sum(abs2, zy)
    β    = zeros(size(X, 2))
    β[findall(keep)] = coef[2:end]
    return β, R2
end

# Spearman rank correlation — the marginal (unconditioned) counterpart to SRRC.
function spearman(x::AbstractVector, y::AbstractVector)
    rx, ry = tiedrank(x), tiedrank(y)
    sx, sy = std(rx), std(ry)
    (sx == 0 || sy == 0) && return 0.0
    return mean((rx .- mean(rx)) .* (ry .- mean(ry))) / (sx * sy) * (length(x) / (length(x) - 1))
end

const REPORT_IDX = [findfirst(==(y), YEARS) for y in REPORT_YEARS]

srrc_lcoh = Dict{Int,Tuple{Vector{Float64},Float64}}()
srrc_tco  = Dict{Int,Tuple{Vector{Float64},Float64}}()
for (j, yr) in enumerate(REPORT_YEARS)
    k = REPORT_IDX[j]
    srrc_lcoh[yr] = srrc(X, @view lcoh_all[:, k])
    srrc_tco[yr]  = srrc(X, @view tco_all[:, k])
end

# Parity year: censored above at END_YEAR for the samples that never reach it.
# Ranking them beyond the last achieved year is order-preserving, so the rank
# statistics stay valid on the full sample; the censored share is reported with
# the table so a reader can see how much of it is "never" rather than "later".
parity_censored = isnan.(parity)
parity_rankable = [isnan(p) ? Float64(END_YEAR) + 1.0 : p for p in parity]
srrc_parity     = srrc(X, parity_rankable)

# ── Variance decomposition ────────────────────────────────────────────────────
# Epistemic  — variance of the per-point medians (parameters).
# Aleatory   — mean within-point variance (deployment timing at fixed parameters).
# The aleatory term is the variance of a single stochastic run, not of the
# M_INNER-run median; the median that this figure plots carries roughly 1/M of it.
function var_split(getter_med, getter_sd)
    epi = var(getter_med)
    ale = mean(abs2, getter_sd)
    return epi, ale, epi / (epi + ale)
end

# ── Summary tables ────────────────────────────────────────────────────────────
qs(v, p) = quantile(filter(isfinite, v), p)

function write_summary(io)
    println(io, "GLOBAL SENSITIVITY — LATIN HYPERCUBE  —  $(dep_title())")
    println(io, "="^108)
    @printf(io, "  %d LHS points × %d stochastic runs = %d simulations, seed %d\n",
            N_LHS, M_INNER, N_LHS * M_INNER, SEED)
    println(io, "  Continuous parameters: triangular(low, mode = model default, high).")
    println(io, "  Policy switches: ordered categories, equal probability per level.")
    println(io, "  Each point = median of the stochastic runs, common random numbers across points.")
    println(io, "-"^108)
    @printf(io, "  %-32s %14s %14s %14s\n", "Parameter", "Low", "Mode", "High")
    println(io, "-"^108)
    for p in TRI_PARAMS
        @printf(io, "  %-32s %14.4g %14.4g %14.4g\n", p.label, p.a, p.c, p.b)
    end
    for p in CAT_PARAMS
        @printf(io, "  %-32s %44s\n", p.label,
                join(string.(p.levels), " / ") * "  (baseline: " *
                string(p.levels[p.base_level]) * ")")
    end
    println(io, "="^108)

    # ── Output distributions ─────────────────────────────────────────────────
    for (nm, M, unit, fmt) in (("H₂ DELIVERY PRICE", lcoh_all, "USD/kg", "%12.3f"),
                               ("TRUCK TCO",         tco_all,  "USD/mile", "%12.4f"))
        println(io, "\n$nm — distribution across the sample ($unit)")
        println(io, "-"^108)
        @printf(io, "  %-6s %12s %12s %12s %12s %12s   %12s\n",
                "Year", "p5", "p25", "median", "p75", "p95", "median|base")
        println(io, "-"^108)
        for k in 1:N_YEARS
            col = @view M[:, k]
            row = [qs(col, q) for q in (0.05, 0.25, 0.50, 0.75, 0.95)]
            bs  = median(col[base_mask])
            @printf(io, "  %-6d", YEARS[k])
            for v in row
                Printf.format(io, Printf.Format(fmt), v)
            end
            print(io, "   ")
            Printf.format(io, Printf.Format(fmt), bs)
            println(io)
        end
        println(io, "  median|base = median over the baseline policy state only (45V baseline, growing bus demand).")
    end

    # ── Variance decomposition ───────────────────────────────────────────────
    println(io, "\nVARIANCE DECOMPOSITION — parameters vs deployment stochasticity")
    println(io, "-"^108)
    println(io, "  Epistemic = variance of the per-point medians (what the LHS varies).")
    println(io, "  Aleatory  = mean within-point variance of a single run (investment triggers, lead times).")
    println(io, "-"^108)
    @printf(io, "  %-22s %-6s %14s %14s %12s\n",
            "Metric", "Year", "Epistemic sd", "Aleatory sd", "Epi. share")
    println(io, "-"^108)
    for (j, yr) in enumerate(REPORT_YEARS)
        for (nm, fld_med, fld_sd) in (("H₂ price (USD/kg)", :lcoh_at, :lcoh_sd_at),
                                       ("TCO (USD/mile)",    :tco_at,  :tco_sd_at))
            med = [getfield(results[i], fld_med)[j] for i in 1:N_LHS]
            sd  = [getfield(results[i], fld_sd)[j]  for i in 1:N_LHS]
            epi, ale, share = var_split(med, sd)
            @printf(io, "  %-22s %-6d %14.4g %14.4g %11.0f%%\n",
                    nm, yr, sqrt(epi), sqrt(ale), 100 * share)
        end
    end

    # ── Parity year ──────────────────────────────────────────────────────────
    ach = .!parity_censored
    println(io, "\nDIESEL-PARITY YEAR — first year H₂ truck TCO ≤ diesel TCO")
    println(io, "-"^108)
    @printf(io, "  reaches parity by %d: %d/%d points (%.0f%%)\n",
            END_YEAR, count(ach), N_LHS, 100 * count(ach) / N_LHS)
    if any(ach)
        pv = parity[ach]
        @printf(io, "  among those:  p5 %.1f   p25 %.1f   median %.1f   p75 %.1f   p95 %.1f\n",
                qs(pv, 0.05), qs(pv, 0.25), qs(pv, 0.50), qs(pv, 0.75), qs(pv, 0.95))
    end
    bp = parity[base_mask]
    @printf(io, "  baseline policy state only: %.0f%% reach parity, median %s\n",
            100 * count(!isnan, bp) / max(count(base_mask), 1),
            any(!isnan, bp) ? @sprintf("%.1f", median(filter(!isnan, bp))) : "n/a")
    println(io, "  Censored points are ranked beyond the last achieved year in the table below.")

    # ── Sensitivity tables ───────────────────────────────────────────────────
    println(io, "\n" * "="^108)
    println(io, "STANDARDISED RANK REGRESSION COEFFICIENTS (SRRC)")
    println(io, "  Sign = direction of effect. |SRRC| = relative strength, comparable across parameters.")
    println(io, "  R² = share of output-rank variance the rank-linear model explains; low R² ⇒ interactions matter.")
    println(io, "="^108)

    function srrc_block(title, pairs)
        println(io, "\n$title")
        println(io, "-"^108)
        print(io, rpad("  Parameter", 34))
        for (lab, _) in pairs
            print(io, lpad(lab, 12))
        end
        println(io)
        println(io, "-"^108)
        # order rows by the largest |SRRC| in the last column of the block
        last_β = pairs[end][2][1]
        order  = sortperm(abs.(last_β), rev = true)
        for i in order
            print(io, rpad("  " * PARAM_LABELS[i], 34))
            for (_, (β, _)) in pairs
                print(io, lpad(@sprintf("%+.3f", β[i]), 12))
            end
            println(io)
        end
        println(io, "-"^108)
        print(io, rpad("  R²", 34))
        for (_, (_, R2)) in pairs
            print(io, lpad(@sprintf("%.3f", R2), 12))
        end
        println(io)
    end

    srrc_block("H₂ delivery price (USD/kg)",
               [(string(yr), srrc_lcoh[yr]) for yr in REPORT_YEARS])
    srrc_block("Truck TCO (USD/mile)",
               [(string(yr), srrc_tco[yr]) for yr in REPORT_YEARS])
    srrc_block("Diesel-parity year (later = positive)",
               [("parity", srrc_parity)])

    println(io, "="^108)
end

write_summary(stdout)
let path = joinpath(OUT_DIR, "fig_sensitivity_lhs_values$(DEP_SUFFIX).txt")
    open(io -> write_summary(io), path, "w")
    println("\nSaved → $path")
end

# ─────────────────────────────────────────────────────────────────────────────
# Figure
# ─────────────────────────────────────────────────────────────────────────────
# Row 1  output distribution over time, as nested quantile bands
# Row 2  SRRC tornado at END_YEAR, one panel per metric
# Row 3  parity-year distribution + its SRRC tornado
# ─────────────────────────────────────────────────────────────────────────────
const C_BAND  = C_SOLAR                  # solar electrolysis is the pathway throughout
const C_POS   = colorant"#C0392B"        # SRRC raises the cost
const C_NEG   = colorant"#1565C0"        # SRRC lowers the cost

# Nested 90 % / 50 % bands with the median line — the house "ranges are bands"
# encoding, conditioned on the baseline policy state so the band is one policy
# world rather than a mixture.
function band_panel!(ax, M, mask; label = "")
    lo90 = [qs(@view(M[mask, k]), 0.05) for k in 1:N_YEARS]
    lo50 = [qs(@view(M[mask, k]), 0.25) for k in 1:N_YEARS]
    med  = [qs(@view(M[mask, k]), 0.50) for k in 1:N_YEARS]
    hi50 = [qs(@view(M[mask, k]), 0.75) for k in 1:N_YEARS]
    hi90 = [qs(@view(M[mask, k]), 0.95) for k in 1:N_YEARS]
    yrs  = Float64.(YEARS)
    band!(ax, yrs, lo90, hi90; color = (C_BAND, 0.18))
    band!(ax, yrs, lo50, hi50; color = (C_BAND, 0.35))
    lines!(ax, yrs, med; color = C_BAND, linewidth = 1.6)
    return (lo90, hi90)
end

# Horizontal bar chart of SRRCs, largest magnitude at the top, signed colour.
function srrc_panel!(ax, β; show_yticks = true, labels = PARAM_LABELS, nshow = 10)
    order = sortperm(abs.(β), rev = true)[1:min(nshow, length(β))]
    order = reverse(order)                     # largest ends up at the top
    n     = length(order)
    for (rank, i) in enumerate(order)
        poly!(ax, Rect(min(0.0, β[i]), rank - 0.32, abs(β[i]), 0.64);
              color = (β[i] > 0 ? C_POS : C_NEG, 0.85), strokewidth = 0)
    end
    vlines!(ax, [0.0]; color = :black, linewidth = 0.8)
    ax.yticks = (collect(1.0:Float64(n)), labels[order])
    CairoMakie.ylims!(ax, 0.4, n + 0.6)
    ax.yticklabelsvisible = show_yticks
    m = maximum(abs.(β[order])) * 1.15
    CairoMakie.xlims!(ax, -m, m)
end

fig = Figure(size = (W_DOUBLE, 200 * MM_TO_PT))

# ── Row 1: distributions over time ───────────────────────────────────────────
ax_a = Axis(fig[1, 1];
    title = "(a)  H₂ delivery price", titlesize = FS_TITLE, titlefont = :bold,
    titlealign = :left,
    xlabel = "Year", ylabel = L"H$_2$ price (USD kg$^{-1}$)",
    xlabelsize = FS_LABEL, ylabelsize = FS_LABEL,
    xticklabelsize = FS_TICK, yticklabelsize = FS_TICK)
band_panel!(ax_a, lcoh_all, base_mask)

ax_b = Axis(fig[1, 2];
    title = "(b)  H₂ truck TCO", titlesize = FS_TITLE, titlefont = :bold,
    titlealign = :left,
    xlabel = "Year", ylabel = L"TCO (USD mile$^{-1}$)",
    xlabelsize = FS_LABEL, ylabelsize = FS_LABEL,
    xticklabelsize = FS_TICK, yticklabelsize = FS_TICK)
band_panel!(ax_b, tco_all, base_mask)
# Diesel reference at the two standing bracket prices, at the sample's median
# LCFS credit price so the two sides share one credit price.
let lp = median([s.lcfs_price for s in samples])
    for (p, ls) in ((TCO_DIESEL_P_LO, LS_DIESEL_LO), (TCO_DIESEL_P_HI, LS_DIESEL_HI))
        lines!(ax_b, Float64.(YEARS),
               [diesel_tco_total(p, y, lp) for y in YEARS];
               color = C_DIESEL, linewidth = 1.1, linestyle = ls)
    end
end

# ── Row 2: SRRC tornados at END_YEAR ─────────────────────────────────────────
ax_c = Axis(fig[2, 1];
    title = "(c)  H₂ price — SRRC, $(END_YEAR)", titlesize = FS_TITLE,
    titlefont = :bold, titlealign = :left,
    xlabel = "SRRC", xlabelsize = FS_LABEL,
    xticklabelsize = FS_TICK, yticklabelsize = 6)
srrc_panel!(ax_c, srrc_lcoh[END_YEAR][1])

ax_d = Axis(fig[2, 2];
    title = "(d)  Truck TCO — SRRC, $(END_YEAR)", titlesize = FS_TITLE,
    titlefont = :bold, titlealign = :left,
    xlabel = "SRRC", xlabelsize = FS_LABEL,
    xticklabelsize = FS_TICK, yticklabelsize = 6)
srrc_panel!(ax_d, srrc_tco[END_YEAR][1])

# ── Row 3: parity year ───────────────────────────────────────────────────────
ax_e = Axis(fig[3, 1];
    title = "(e)  Diesel-parity year", titlesize = FS_TITLE,
    titlefont = :bold, titlealign = :left,
    xlabel = "Year H₂ TCO ≤ diesel TCO", ylabel = "LHS points",
    xlabelsize = FS_LABEL, ylabelsize = FS_LABEL,
    xticklabelsize = FS_TICK, yticklabelsize = FS_TICK)
if any(.!parity_censored)
    hist!(ax_e, parity[.!parity_censored];
          bins = 20, color = (C_BAND, 0.65), strokewidth = 0.3, strokecolor = :white)
end
# The censored share is the headline number for this panel and cannot be drawn
# as a bar, so it is stated in the corner instead.
CairoMakie.text!(ax_e, 0.03, 0.95;
      text = @sprintf("no parity by %d: %.0f%% of points", END_YEAR,
                      100 * count(parity_censored) / N_LHS),
      space = :relative, align = (:left, :top), fontsize = FS_ANNOT, color = C_DIESEL)

ax_f = Axis(fig[3, 2];
    title = "(f)  Parity year — SRRC", titlesize = FS_TITLE,
    titlefont = :bold, titlealign = :left,
    xlabel = "SRRC  (positive = later parity)", xlabelsize = FS_LABEL,
    xticklabelsize = FS_TICK, yticklabelsize = 6)
srrc_panel!(ax_f, srrc_parity[1])

Legend(fig[4, 1:2],
    [PolyElement(color = (C_BAND, 0.35)), PolyElement(color = (C_BAND, 0.18)),
     LineElement(color = C_BAND, linewidth = 1.6),
     LineElement(color = C_DIESEL, linestyle = LS_DIESEL_LO, linewidth = 1.1),
     LineElement(color = C_DIESEL, linestyle = LS_DIESEL_HI, linewidth = 1.1),
     PolyElement(color = (C_POS, 0.85)), PolyElement(color = (C_NEG, 0.85))],
    ["50 % of LHS points", "90 % of LHS points", "Median",
     @sprintf("Diesel TCO (\$%.2f/gal)", TCO_DIESEL_P_LO),
     @sprintf("Diesel TCO (\$%.2f/gal)", TCO_DIESEL_P_HI),
     "Raises cost / delays parity", "Lowers cost / advances parity"];
    orientation = :horizontal, nbanks = 3, tellwidth = false,
    framevisible = false, labelsize = FS_LEGEND)

Label(fig[5, 1:2];
    text = string(N_LHS, " Latin hypercube points × ", M_INNER,
                  " stochastic runs; solar electrolysis, ", dep_title(),
                  ". Bands in (a)–(b) are conditioned on the baseline policy state. ",
                  @sprintf("SRRC R²: %.2f (H₂ price), %.2f (TCO), %.2f (parity year).",
                           srrc_lcoh[END_YEAR][2], srrc_tco[END_YEAR][2], srrc_parity[2])),
    fontsize = FS_LEGEND, color = C_DIESEL, tellwidth = false,
    word_wrap = true, halign = :left, justification = :left)

colgap!(fig.layout, 10)
rowgap!(fig.layout, 6)
resize_to_layout!(fig)

# Not listed in fig_routing.jl, so this lands in out/old/ via FIG_DEFAULT_SUBDIR
# — the convention for figures the paper does not cite. Retained rather than
# deleted because the global sensitivity numbers it prints (see the values file)
# stand on their own, and because it is the only place the discount rate is
# treated as uncertain.
save_pub("fig_sensitivity_lhs$(DEP_SUFFIX)", fig)
