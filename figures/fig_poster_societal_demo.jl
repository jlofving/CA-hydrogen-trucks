# =============================================================================
# POSTER FIGURE — SOCIETAL COST-BENEFIT, LIMITED DEPLOYMENT  (1 row x 3 panels)
# =============================================================================
# Poster version of manuscript figures 11 and 12, keeping only the "Suggested
# demo" deployment column of each:
#
#   (a) resource-cost premium vs. avoided damages   (= manuscript fig 11a)
#   (b) cumulative net societal cost                (= manuscript fig 12a)
#   (c) net societal cost per mile                  (= manuscript fig 12c)
#
# No figure title and no scenario names in the panel titles — the panels carry
# bare letters, because a poster supplies its own heading and the deployment is
# stated once in the caption block next to it. Type and line weights are scaled
# up from the paper defaults for reading at a distance.
#
# Encoding is unchanged from the manuscript figures, so the poster and the paper
# teach the same legend: SCC 1.5-2.5% is a translucent BAND with the 2.0% centre
# as its line; diesel pump price is LINE STYLE in (a), where only the green
# premium depends on it, and COLOUR in (b)/(c), where it drives the net cost.
#
# The median flow series are READ FROM THE CACHE written by
# fig_societal_cost_benefit.jl, so the poster reproduces the published panels
# exactly rather than drawing a second, independently-seeded Monte Carlo. If the
# cache is missing — or the model or configs changed since it was written — run
# the manuscript script first, which refreshes it:
#
#   julia --project=figures figures/fig_societal_cost_benefit.jl
#   julia --project=figures figures/fig_poster_societal_demo.jl
# =============================================================================

using CairoMakie
using JSON
using Printf
import Serialization

cd(joinpath(@__DIR__, ".."))
include("pub_theme.jl")

const OUT_DIR = joinpath(@__DIR__, "out")

# ── Poster typography / weights ───────────────────────────────────────────────
# Bumped from the 9 pt publication base; everything else (colours, styles,
# save routine) still comes from pub_theme.jl.
const FS_POSTER   = 13
const LW_POSTER   = 3.0     # data lines
const LW_POSTER_2 = 2.4     # the thinner of a solid/dashed pair
update_theme!(
    fontsize = FS_POSTER,
    Axis = (titlesize = FS_POSTER + 2, xlabelsize = FS_POSTER + 1,
            ylabelsize = FS_POSTER + 1,
            xticklabelsize = FS_POSTER, yticklabelsize = FS_POSTER),
    Legend = (labelsize = FS_POSTER, titlesize = FS_POSTER, patchsize = (20, 12),
              patchlabelgap = 6, rowgap = 3, padding = (6, 6, 4, 4)),
)

# ── Diesel pump prices (labels only) ─────────────────────────────────────────
# Read from config, never hard-coded — see config/tco_config.json.
let p = JSON.parsefile(joinpath(@__DIR__, "..", "config", "tco_config.json"))["diesel_price_usd_per_gallon"]
    global const DIESEL_P_LO = Float64(p["low"])
    global const DIESEL_P_HI = Float64(p["high"])
end

# ─────────────────────────────────────────────────────────────────────────────
# ANALYSIS  —  mirrors fig_societal_cost_benefit.jl; keep the two in sync
# ─────────────────────────────────────────────────────────────────────────────
# Only the post-Monte-Carlo layer is repeated here: the cached flows already
# carry the median premium, health, CO₂ and mileage streams, so nothing in the
# TCO or deployment model is re-evaluated.
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

benefit(F, rate)      = F.health .+ [scc_per_ton(F.years[k]; rate = rate) * F.co2_t[k] / 1e6 for k in eachindex(F.years)]
permile(F, tcp, rate) = (tcp .- benefit(F, rate)) .* 1e6 ./ F.miles
net_stream(F, dfield, rate) = getfield(F, dfield) .- benefit(F, rate)
discount_series(years, vals, r, t0) = [v / (1 + r)^(y - t0) for (y, v) in zip(years, vals)]
function cumulative_integral(years, vals)
    cum = zeros(length(vals));  s = 0.0
    for i in 2:length(vals)
        s += 0.5 * (vals[i-1] + vals[i]) * (years[i] - years[i-1])
        cum[i] = s
    end
    return cum
end

const Y0, Y1, DBASE = 2026, 2045, 2026
const SCC_LO_RATE, SCC_MID_RATE, SCC_HI_RATE = "2.5%", "2.0%", "1.5%"   # 2.5% ⇒ lowest SCC ⇒ highest net cost
const PV_RATE = 0.02

diesels = [(DIESEL_P_LO, :tcp_lo, C_NET_LO), (DIESEL_P_HI, :tcp_hi, C_NET_HI)]

function cum_trajectory(F, dfield, scc_rate, pv)
    idx = findall(y -> Y0 <= y <= Y1, F.years)
    yrs = F.years[idx]
    net = net_stream(F, dfield, scc_rate)[idx]
    dv  = discount_series(yrs, net, pv, DBASE)
    return Float64.(yrs), cumulative_integral(yrs, dv)
end

# ─────────────────────────────────────────────────────────────────────────────
# Cached flows — limited deployment only
# ─────────────────────────────────────────────────────────────────────────────
const N_RUNS, SEED = 1000, 42
const FLOW_CACHE = joinpath(OUT_DIR, "fig_societal_cost_benefit_flows_$(N_RUNS)_$(SEED).jls")
isfile(FLOW_CACHE) || error("""
    Missing flow cache: $FLOW_CACHE
    Run the manuscript script first (it writes the cache):
      julia --project=figures figures/fig_societal_cost_benefit.jl""")
F = first(Serialization.deserialize(FLOW_CACHE))   # (limited_dep, high_dep)

const XT = year_ticks(Y0, Y1)
padlim(v) = let p = 0.05 * (maximum(v) - minimum(v)); (minimum(v) - p, maximum(v) + p) end

# ─────────────────────────────────────────────────────────────────────────────
# Panel drawing  (same encodings as manuscript figs 11 / 12, heavier strokes)
# ─────────────────────────────────────────────────────────────────────────────
c_premium = C_SOLAR    # solar-electrolysis resource-cost premium (unsubsidized)
c_benefit = C_SCC      # avoided damages / SCC (pink)

zero_line!(ax) = hlines!(ax, 0; color = C_ZERO_LINE, linewidth = 1.2, linestyle = :dot)

function draw_costbenefit!(ax, F)
    y = F.years
    zero_line!(ax)
    band!(ax, y, benefit(F, SCC_LO_RATE), benefit(F, SCC_HI_RATE); color = (c_benefit, 0.18))
    lines!(ax, y, benefit(F, SCC_MID_RATE); color = c_benefit, linewidth = LW_POSTER)
    lines!(ax, y, F.tcp_lo; color = c_premium, linewidth = LW_POSTER,   linestyle = LS_DIESEL_LO)
    lines!(ax, y, F.tcp_hi; color = c_premium, linewidth = LW_POSTER_2, linestyle = LS_DIESEL_HI)
end

function draw_cumulative!(ax, F)
    zero_line!(ax)
    for (_, dfield, col) in diesels
        yrs, c_mid = cum_trajectory(F, dfield, SCC_MID_RATE, PV_RATE)
        _,   c_lo  = cum_trajectory(F, dfield, SCC_LO_RATE,  PV_RATE)
        _,   c_hi  = cum_trajectory(F, dfield, SCC_HI_RATE,  PV_RATE)
        band!(ax, yrs, min.(c_lo, c_hi), max.(c_lo, c_hi); color = (col, 0.18))
        lines!(ax, yrs, c_mid; color = col, linewidth = LW_POSTER)
    end
end

function draw_permile!(ax, F)
    y = F.years
    zero_line!(ax)
    for (_, dfield, col) in diesels
        tcp = getfield(F, dfield)
        band!(ax, y, permile(F, tcp, SCC_LO_RATE), permile(F, tcp, SCC_HI_RATE); color = (col, 0.18))
        lines!(ax, y, permile(F, tcp, SCC_MID_RATE); color = col, linewidth = LW_POSTER)
    end
end

# Per-panel y limits from the demo data alone. The manuscript figures share y
# across the deployment columns so the two scenarios stay comparable; here there
# is only one scenario, so each panel may use its full frame.
cb_lim  = padlim(vcat(benefit(F, SCC_LO_RATE), benefit(F, SCC_HI_RATE), F.tcp_lo, F.tcp_hi))
cum_lim = padlim(vcat([cum_trajectory(F, d, r, PV_RATE)[2]
                       for (_, d, _) in diesels, r in (SCC_LO_RATE, SCC_MID_RATE, SCC_HI_RATE)]...))
pm_lim  = padlim(vcat([permile(F, getfield(F, d), r)
                       for (_, d, _) in diesels, r in (SCC_LO_RATE, SCC_HI_RATE)]...))

# ─────────────────────────────────────────────────────────────────────────────
# Figure
# ─────────────────────────────────────────────────────────────────────────────
const W_POSTER = 350 * MM_TO_PT
fig = Figure(size = (W_POSTER, 110 * MM_TO_PT), figure_padding = (6, 14, 6, 6))

panels = [
    ("a", "M USD yr⁻¹",                 cb_lim,  draw_costbenefit!),
    ("b", "Cumulative net cost (M USD)", cum_lim, draw_cumulative!),
    ("c", "Net cost (USD mi⁻¹)",         pm_lim,  draw_permile!),
]
for (j, (letter, ylab, ylim, draw!)) in enumerate(panels)
    ax = Axis(fig[1, j];
        title = "($letter)", xlabel = "Year", ylabel = ylab,
        xticks = XT, limits = ((Y0, Y1), ylim))
    draw!(ax, F)
end
text!(fig.content[2], 0.03, 0.04;
    text = @sprintf("discounted to %d at %.0f%%/yr", DBASE, 100 * PV_RATE),
    space = :relative, align = (:left, :bottom), fontsize = FS_POSTER - 2, color = :black)

# One legend for all three panels: the green/pink pair belongs to (a), the
# blue/orange pair to (b) and (c). Group titles say which, so the row reads
# left-to-right in the same order as the panels.
leg_groups = Vector{Vector}([
    [LineElement(color = c_premium, linewidth = 3, linestyle = LS_DIESEL_LO),
     LineElement(color = c_premium, linewidth = 3, linestyle = LS_DIESEL_HI)],
    [[[LineElement(color = c_benefit, linewidth = 3), PolyElement(color = (c_benefit, 0.18))]]],
    [[LineElement(color = C_NET_LO, linewidth = 3), PolyElement(color = (C_NET_LO, 0.18))],
     [LineElement(color = C_NET_HI, linewidth = 3), PolyElement(color = (C_NET_HI, 0.18))]],
])
leg_labels = Vector{Vector{String}}([
    [@sprintf("vs. diesel \$%.2f/gal", DIESEL_P_LO), @sprintf("vs. diesel \$%.2f/gal", DIESEL_P_HI)],
    ["SCC 2.0% (shading 1.5–2.5%)"],
    [@sprintf("Diesel \$%.2f/gal", DIESEL_P_LO), @sprintf("Diesel \$%.2f/gal", DIESEL_P_HI)],
])
leg_titles = ["(a) Cost premium, hydrogen trucks",
              "(a) Avoided damages (health + CO₂)",
              "(b), (c) Net societal cost"]
Legend(fig[2, 1:3], leg_groups, leg_labels, leg_titles;
    orientation = :horizontal, nbanks = 2, titlefont = :bold,
    tellheight = true, tellwidth = false, colgap = 20, framevisible = true)

rowsize!(fig.layout, 1, Fixed(80 * MM_TO_PT))
colgap!(fig.layout, 26); rowgap!(fig.layout, 6)
resize_to_layout!(fig)
save_pub("fig_poster_societal_demo", fig; subdir = "poster")
