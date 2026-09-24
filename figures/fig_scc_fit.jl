#!/usr/bin/env julia
# Reference: EPA (2023) SC-CO₂ (2024 USD) for the three near-term Ramsey
# discount rates (1.5%, 2.0%, 2.5%), with the piecewise-linear interpolation the
# model uses. The interpolation passes exactly through the published decade
# points; the model horizon (2026–2045) is interior. Shows how much the SCC
# discount-rate choice matters (the 1.5% value is ~2× the 2.5% value).
using CairoMakie
using Printf

include("pub_theme.jl")

# EPA (2023) SC-CO₂, published in 2020 USD t⁻¹ CO₂, by emission year.
allyears = [2020, 2030, 2040, 2050, 2060, 2070, 2080]
scc2020 = Dict(
    "1.5%" => [340.0, 380.0, 430.0, 480.0, 530.0, 570.0, 600.0],
    "2.0%" => [190.0, 230.0, 270.0, 310.0, 350.0, 380.0, 410.0],
    "2.5%" => [120.0, 140.0, 170.0, 200.0, 230.0, 260.0, 280.0],
)
const DEFLATOR = 1.18   # US GDP implicit price deflator, 2020 → 2024 USD
rates = ["1.5%", "2.0%", "2.5%"]

# Piecewise-linear interpolation (2024 USD), identical to scc_per_ton in the
# cost-premium / abatement figures.
function scc_per_ton(year, rate)
    xs = allyears;  ys = scc2020[rate]
    yr = clamp(year, xs[1], xs[end])
    i  = min(searchsortedlast(xs, yr), length(xs) - 1)
    t  = (yr - xs[i]) / (xs[i+1] - xs[i])
    return (ys[i] + t * (ys[i+1] - ys[i])) * DEFLATOR
end

# Console report: interpolated values at the horizon endpoints
println("EPA SC-CO₂ piecewise interpolation (2024 USD / tCO₂e)")
println("="^54)
@printf("%-6s %14s %14s\n", "rate", "2024USD@2026", "2024USD@2045")
println("-"^54)
for r in rates
    @printf("%-6s %14.1f %14.1f\n", r, scc_per_ton(2026, r), scc_per_ton(2045, r))
end
println()

# ── Figure ────────────────────────────────────────────────────────────────────
fig = Figure(size = (W_DOUBLE, 115 * MM_TO_PT))
ax = Axis(fig[1, 1];
    title          = "EPA SC-CO₂ (2024 USD), piecewise-linear interpolation by discount rate",
    titlesize      = 9,
    xlabel         = "Emission year",
    ylabel         = "SCC  (2024 USD per tonne CO₂)",
    xticks         = 2020:10:2080,
    limits         = ((2018, 2082), (100, 760)),
)

vspan!(ax, 2026, 2045; color = (:steelblue, 0.10))
text!(ax, 2035.5, 730; text = "model horizon", align = (:center, :center),
      fontsize = 7, color = (:steelblue, 0.9))

ratecolor = SCC_SHADES   # single-hue pink ramp (light = 2.5%, dark = 1.5%)
for r in rates
    c = ratecolor[r]
    lines!(ax, allyears, scc2020[r] .* DEFLATOR; color = c, linewidth = 2,
           label = "$r discount rate")
    scatter!(ax, allyears, scc2020[r] .* DEFLATOR; color = c, markersize = 8)
end
axislegend(ax; position = :lt, labelsize = 7, framevisible = true)

outdir = joinpath(@__DIR__, "out")
isdir(outdir) || mkdir(outdir)
save_pub("fig_scc_fit", fig; dir = outdir)
