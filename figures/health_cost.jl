# =============================================================================
# AVOIDED AIR-QUALITY DAMAGE — shared health-cost basis for the societal figures
# =============================================================================
# One definition of "dollars of avoided health damage per displaced diesel mile",
# used by every figure that nets a health benefit against the cost premium:
#
#   fig_societal_cost_benefit.jl   manuscript figs 11 + 12
#   fig_abatement_cost.jl          manuscript fig 13 (net MAC)
#   fig_societal_benefit.jl        appendix
#
# Before this file each of those scripts carried its own transcribed copy of the
# ENV_COST_DELTA table, so a change of valuation had to be applied in four
# places and could silently disagree between panels of the same paper.
#
# TWO BASES ARE AVAILABLE. Select with the HEALTH_BASIS environment variable:
#
#   cobra    (DEFAULT) EPA COBRA, applied through the bivariate proxy below.
#   envcost            Per-mile factors from "Env Cost Factors 20apr LF.xlsx".
#
#   HEALTH_BASIS=envcost julia --project=figures figures/fig_abatement_cost.jl
#
# The two differ by roughly 4.7x (COBRA is the smaller), so they are never mixed
# within a figure suite — run all of the scripts above on the same basis.
# =============================================================================

if !@isdefined(HEALTH_BASIS)

const HEALTH_BASIS = lowercase(get(ENV, "HEALTH_BASIS", "cobra"))
HEALTH_BASIS in ("cobra", "envcost") ||
    error("HEALTH_BASIS=\"$HEALTH_BASIS\" not recognised — use \"cobra\" or \"envcost\"")

# ─────────────────────────────────────────────────────────────────────────────
# Tailpipe emission factors — Class 8 diesel truck (g/mile)
# ─────────────────────────────────────────────────────────────────────────────
# Avoided outright by a fuel-cell truck, so these double as the per-mile
# emission *reductions* that COBRA is fed.
const NOX_G_PER_MILE  = 0.06    # NOx   (g/mile)
const PM25_G_PER_MILE = 0.0015  # PM2.5 (g/mile)

const METRIC_T_TO_SHORT_T = 1.10231   # COBRA works in US short tons

# ─────────────────────────────────────────────────────────────────────────────
# BASIS 1 — EPA COBRA (Co-Benefits Risk Assessment), bivariate proxy
# ─────────────────────────────────────────────────────────────────────────────
# benefit [M USD/yr] = α_NOx × NOx [short t/yr] + α_PM25 × PM2.5 [short t/yr]
#
# NOx and PM2.5 are perfectly collinear in this model (both scale with fleet
# miles), so their separate damage coefficients cannot be recovered from a
# combined COBRA run. They come instead from two isolated single-pollutant runs
# for 2037, each divided by the tonnage that run was given. See the calibration
# notes and the emission-changes CSV export in fig_cost_premium.jl.
const COBRA_ALPHA_NOX  = 3.5  / 33.18  # = 0.1055 M USD / short-ton NOx   (COBRA 2037, NOx-only run)
const COBRA_ALPHA_PM25 = 0.59 /  0.83  # = 0.7108 M USD / short-ton PM2.5 (COBRA 2037, PM2.5-only run)

# Combined COBRA runs, kept for provenance and for the drift check below.
# year => (total annual health benefit M USD/yr, NOx short t/yr fed to that run)
const COBRA_RESULTS = Dict{Int, Tuple{Float64,Float64}}(
    2026 => (0.14,  1.16),  2027 => (0.15,  1.22),  2028 => (0.19,  1.51),
    2029 => (0.30,  2.39),  2030 => (0.37,  2.97),  2031 => (0.51,  4.13),
    2032 => (0.73,  5.88),  2033 => (0.94,  7.62),  2034 => (1.40, 11.64),
    2035 => (2.10, 17.17),  2036 => (3.10, 25.03),  2037 => (4.10, 33.18),
    2038 => (5.40, 43.65),  2039 => (7.00, 56.46),  2040 => (8.80, 71.01),
)

# The proxy is linear in fleet miles and its coefficients are year-invariant, so
# it collapses to a single $/mile figure — held FLAT in real terms. Unlike the
# envcost table it carries no assumption about the diesel fleet cleaning up over
# time; the emission factors above are the whole of its year dependence.
const COBRA_HEALTH_USD_PER_MILE = let
    g_to_short_t = 1e-6 * METRIC_T_TO_SHORT_T   # g → metric tonne → US short ton
    (COBRA_ALPHA_NOX  * NOX_G_PER_MILE  * g_to_short_t +
     COBRA_ALPHA_PM25 * PM25_G_PER_MILE * g_to_short_t) * 1e6   # M USD → USD
end

# ─────────────────────────────────────────────────────────────────────────────
# BASIS 2 — "Env Cost Factors 20apr LF.xlsx"  (legacy; HEALTH_BASIS=envcost)
# ─────────────────────────────────────────────────────────────────────────────
# Per-mile avoided air-quality damage (NOx + PM2.5 + NH3) for an HDV switching
# from diesel to ZEV:  Δ = column E (diesel HDV) − column K (ZEV HDV), sheet
# "Env cost parameters". Transcribed so no script needs to read the workbook.
# Broader than COBRA (it includes NH3) and it declines ~2%/yr as the diesel
# fleet is assumed to get cleaner.
const ENV_COST_DELTA_USD_PER_MILE = Dict{Int,Float64}(
    2015=>0.0480283801,2016=>0.0470866472,2017=>0.0461633796,2018=>0.0452582153,
    2019=>0.0443707993,2020=>0.0435007836,2021=>0.0426478271,2022=>0.0418115952,
    2023=>0.0409917600,2024=>0.0401880000,2025=>0.0394000000,2026=>0.0386274510,
    2027=>0.0378700500,2028=>0.0371275000,2029=>0.0363995098,2030=>0.0356857939,
    2031=>0.0349860725,2032=>0.0343000710,2033=>0.0336275206,2034=>0.0329681575,
    2035=>0.0323217230,2036=>0.0316879637,2037=>0.0310666311,2038=>0.0304574815,
    2039=>0.0298602760,2040=>0.0292747804,2041=>0.0287007651,2042=>0.0281380050,
    2043=>0.0275862794,2044=>0.0270453719,2045=>0.0265150705,2046=>0.0259951672,
    2047=>0.0254854580,2048=>0.0249857432,2049=>0.0244958266,2050=>0.0240155163,
)

# ─────────────────────────────────────────────────────────────────────────────
# Public interface
# ─────────────────────────────────────────────────────────────────────────────

"""
    health_cost_per_mile(year) -> Float64

Avoided air-quality damage in USD per displaced diesel mile, on whichever basis
`HEALTH_BASIS` selects. Multiply by fleet miles for an annual benefit.

Errors rather than extrapolating when the envcost table has no entry for
`year` — a missing factor means the assessment horizon moved past the
transcribed sheet, which should be noticed rather than silently filled in.
"""
health_cost_per_mile(year::Int) =
    HEALTH_BASIS == "cobra" ? COBRA_HEALTH_USD_PER_MILE :
    get(ENV_COST_DELTA_USD_PER_MILE, year) do
        error("No env-cost factor for year $year in ENV_COST_DELTA_USD_PER_MILE")
    end

"""
    health_basis_label() -> String

One-line description of the active basis, for the values files so a saved table
records which valuation produced it.
"""
health_basis_label() =
    HEALTH_BASIS == "cobra" ?
        @sprintf("EPA COBRA (NOx + PM2.5), %.5f USD/mile flat", COBRA_HEALTH_USD_PER_MILE) :
        "Env Cost Factors 20apr LF.xlsx (NOx + PM2.5 + NH3), year-dependent"

"""
    health_basis_banner()

Print the active basis to stdout, and — on the COBRA basis — check the fleet
tonnages the stored COBRA runs were given against what the model produces now.
`nox_short_tons` is a year => short-tons/yr mapping for the *current* run; pass
`nothing` to skip the drift check. A few percent of drift is expected and
harmless (the proxy rescales with current miles); tens of percent means the
calibration runs should be redone.
"""
function health_basis_banner(nox_short_tons = nothing)
    println("Health-cost basis: $(uppercase(HEALTH_BASIS)) — $(health_basis_label())")
    if HEALTH_BASIS == "cobra" && nox_short_tons !== nothing
        drift = [(yr, now_t, tup[2]) for (yr, tup) in sort(collect(COBRA_RESULTS))
                 for now_t in (get(nox_short_tons, yr, nothing),) if now_t !== nothing]
        if !isempty(drift)
            worst = maximum(abs(n - c) / c for (_, n, c) in drift)
            @printf("  calibration drift vs stored COBRA runs: max %.1f%% on NOx tonnage\n",
                    100 * worst)
            worst > 0.25 && @warn "COBRA calibration tonnages have drifted >25% — re-run the COBRA model"
        end
    end
    println()
end

end  # @isdefined guard
