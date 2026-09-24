# Bus H2 demand profile (2026–2040)
# Run from figures directory:  julia --project . fig_bus_demand_curve.jl

using CairoMakie

include(joinpath(@__DIR__, "fig_routing.jl"))   # figure → manuscript/appendix/old routing
const OUT_DIR = joinpath(@__DIR__, "out")
mkpath(OUT_DIR)

# ── Anchor data ───────────────────────────────────────────────────────────────
anchor_years  = [2026, 2028, 2030, 2035, 2040]
anchor_t_day  = [8.0, 12.0, 25.0, 55.0, 90.0]   # t/day

# ── Piecewise-linear function ─────────────────────────────────────────────────
function bus_demand_t_day(year)
    pts = collect(zip(anchor_years, anchor_t_day))
    year <= pts[1][1] && return pts[1][2]
    year >= pts[end][1] && return pts[end][2]
    for i in 1:length(pts)-1
        y0, d0 = pts[i]; y1, d1 = pts[i+1]
        if y0 <= year <= y1
            return d0 + (d1 - d0) * (year - y0) / (y1 - y0)
        end
    end
    return pts[1][2]
end

years_fine = 2026:2040

# ── Figure ────────────────────────────────────────────────────────────────────
MM_TO_PT = 1 / 0.352778
fig = Figure(size = (120 * MM_TO_PT, 75 * MM_TO_PT), fontsize = 8)

ax = Axis(fig[1, 1];
    xlabel             = "Year",
    ylabel             = "Bus H₂ demand (t/day)",
    xlabelsize         = 7,
    ylabelsize         = 7,
    xticklabelsize     = 7,
    yticklabelsize     = 7,
    xticks             = 2026:2:2040,
    xticklabelrotation = π / 4,
    limits             = ((2026, 2040), (0, nothing)),
)

lines!(ax, collect(Float64.(years_fine)), bus_demand_t_day.(years_fine);
       color = colorant"#4472C4", linewidth = 1.5)

scatter!(ax, Float64.(anchor_years), anchor_t_day;
         color = :black, markersize = 6)

out = fig_path(OUT_DIR, "fig_bus_demand_curve.png")
save(out, fig; px_per_unit = 300 / 72)
println("Saved → $out")
