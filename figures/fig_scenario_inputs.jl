# =============================================================================
# FIGURE: SCENARIO INPUTS — TRUCK DEPLOYMENT & LCFS CREDIT PRICE
# =============================================================================
# Two-panel figure:
#   (a) Cumulative H₂ truck fleet — three deployment scenarios
#       (No/Low, Limited Deployment, High Deployment)
#   (b) LCFS credit price — two price scenarios
#       (No Change, High Increase)
#
# No simulation required — data read directly from config JSON files.
#
# Run from project root:
#   julia --project figures/fig_scenario_inputs.jl
# =============================================================================

using CairoMakie
using JSON

cd(joinpath(@__DIR__, ".."))

include(joinpath(@__DIR__, "fig_routing.jl"))   # figure → manuscript/appendix/old routing
const OUT_DIR = joinpath(@__DIR__, "out")
mkpath(OUT_DIR)

# ── Load config ───────────────────────────────────────────────────────────────
trucks_raw = JSON.parsefile(joinpath(@__DIR__, "..", "config", "trucks_config.json"))
lcfs_raw   = JSON.parsefile(joinpath(@__DIR__, "..", "config", "lcfs_config.json"))

const YEARS = 2026:2040

# ── Truck deployment scenarios ────────────────────────────────────────────────
truck_scenarios = [
    (key = "nolow_dep", label = "No/Low deployment", color = colorant"#C0504D", ls = :solid),
    (key = "limited_dep", label = "Limited deployment",    color = colorant"#4472C4", ls = :solid),
    (key = "high_dep",  label = "High deployment",   color = colorant"#70AD47", ls = :solid),
]

function cumulative_fleet(key::String)
    s     = trucks_raw["truck_deployment_scenarios"][key]
    init  = s["initial_trucks"]
    sched = Dict{Int,Int}(e["year"] => e["trucks_added"] for e in s["schedule"])
    fleet = Int[]
    total = init
    for yr in YEARS
        total += get(sched, yr, 0)
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
    return [get(sched, yr, NaN) for yr in YEARS]
end

# ── Figure ────────────────────────────────────────────────────────────────────
MM_TO_PT  = 1 / 0.352778
years_vec = collect(YEARS)

fig = Figure(
    size     = (183 * MM_TO_PT, 82 * MM_TO_PT),
    fontsize = 8,
)

# ── Panel (a): cumulative truck fleet ─────────────────────────────────────────
ax_a = Axis(
    fig[1, 1];
    xlabel             = "Year",
    ylabel             = "Cumulative H₂ truck fleet",
    xlabelsize         = 9,
    ylabelsize         = 9,
    xticklabelsize     = 8,
    yticklabelsize     = 8,
    xticks             = years_vec[1:2:end],
    xticklabelrotation = π / 4,
    ytickformat        = v -> ["$(Int(round(x / 1000)))k" for x in v],
    limits             = ((2026, 2040), (0, nothing)),
)

text!(ax_a, 0.03, 0.97;
      text     = "(a)",
      space    = :relative,
      align    = (:left, :top),
      fontsize = 8,
      font     = :bold,
      color    = :black,
)

for ts in truck_scenarios
    fleet = cumulative_fleet(ts.key)
    lines!(ax_a, years_vec, fleet;
           color     = ts.color,
           linewidth = 1.8,
           linestyle = ts.ls,
           label     = ts.label,
    )
    scatter!(ax_a, years_vec, fleet; color = ts.color, markersize = 3)
end

axislegend(ax_a;
           position     = (0.02, 0.82),
           framevisible = true,
           labelsize    = 7,
           rowgap       = 0,
           patchsize    = (14, 8),
)

# ── Panel (b): LCFS credit price ──────────────────────────────────────────────
ax_b = Axis(
    fig[1, 2];
    xlabel             = "Year",
    ylabel             = "LCFS credit price (USD credit⁻¹)",
    xlabelsize         = 9,
    ylabelsize         = 9,
    xticklabelsize     = 8,
    yticklabelsize     = 8,
    xticks             = years_vec[1:2:end],
    xticklabelrotation = π / 4,
    limits             = ((2026, 2040), (0, 270)),
    yticks             = 0:50:250,
)

text!(ax_b, 0.03, 0.97;
      text     = "(b)",
      space    = :relative,
      align    = (:left, :top),
      fontsize = 8,
      font     = :bold,
      color    = :black,
)

for ls in lcfs_scenarios
    prices = lcfs_prices(ls.key)
    lines!(ax_b, years_vec, prices;
           color     = ls.color,
           linewidth = 1.8,
           linestyle = ls.ls,
           label     = ls.label,
    )
    scatter!(ax_b, years_vec, prices; color = ls.color, markersize = 3)
end

axislegend(ax_b;
           position     = (0.98, 0.82),
           framevisible = true,
           labelsize    = 7,
           rowgap       = 0,
           patchsize    = (14, 8),
)

# ── Layout & save ─────────────────────────────────────────────────────────────
colgap!(fig.layout, 10)

out_pdf = fig_path(OUT_DIR, "fig_scenario_inputs.pdf")
out_png = fig_path(OUT_DIR, "fig_scenario_inputs.png")
save(out_pdf, fig; pt_per_unit = 1)
println("Saved → $out_pdf")
save(out_png, fig; px_per_unit = 300 / 72)
println("Saved → $out_png")
