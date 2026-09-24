# =============================================================================
# FIGURE: INVESTMENT TIMELINE  (Gantt-style, 2 stacked panels)
# =============================================================================
# Each refuelling station and each hydrogen-production facility is drawn as a
# horizontal bar that starts in the year the asset is built and runs for its
# operating lifetime (clipped to the assessment horizon). The THICKNESS of each
# bar is proportional to that asset's daily H2 capacity, and bars are stacked in
# build-year order so the vertical envelope reads as cumulative installed
# capacity over time.
#
#   Top panel    — refuelling stations  (coloured by deployment status)
#   Bottom panel — hydrogen production   (coloured by technology)
#
# Refuelling stations come straight from the committed pipeline in
# config/stations_config.json (their planned opening year + capacity), which is
# deterministic. Auto-projected stations and production facilities are generated
# stochastically by the model, so they are captured from a single representative
# Monte Carlo run (fixed seed) via the `event_sink` hook in run_single_simulation.
#
# Run from the rollout model root directory:
#   julia --project figures/fig_investment_timeline.jl
# =============================================================================

using Statistics
using CairoMakie
using Random

cd(joinpath(@__DIR__, ".."))
include("../hydrogen truck deployment.jl")
include("config_defaults.jl")

const SEED    = 42
include(joinpath(@__DIR__, "fig_routing.jl"))   # figure → manuscript/appendix/old routing
const OUT_DIR = joinpath(@__DIR__, "out")
mkpath(OUT_DIR)

# ── Configuration & single representative run ───────────────────────────────────
cfg        = build_config()
start_year = cfg.start_year
end_year   = cfg.end_year

# Capture per-asset build events from one representative simulation.
events = NamedTuple[]
Random.seed!(SEED)
run_single_simulation(cfg; event_sink = events)

# ── Assemble station rows ────────────────────────────────────────────────────────
# Committed pipeline (deterministic): one row per station in the config file.
station_life = cfg.station_lifetime
stations = NamedTuple[]
for s in cfg.station_data
    push!(stations, (
        label    = haskey(s, :name) ? s.name : s.id,
        capacity = Float64(s.capacity_kg_per_day),
        year     = Int(s.planned_opening_year),
        lifetime = station_life,
        status   = haskey(s, :status) ? s.status : "in planning",
    ))
end
# Auto-projected stations (stochastic, from the representative run). 8 t/day each.
let n = 0
    for e in events
        if e.kind === :station_projected && e.year <= end_year
            n += 1
            push!(stations, (
                label    = "Projected $n",
                capacity = Float64(e.capacity),
                year     = Int(e.year),
                lifetime = station_life,
                status   = "auto-projected",
            ))
        end
    end
end

# ── Assemble production rows ──────────────────────────────────────────────────────
production = NamedTuple[]
for e in events
    if e.kind === :production && e.year <= end_year
        push!(production, (
            label    = e.is_initial ? "Initial" : e.id,
            capacity = Float64(e.capacity),
            year     = Int(e.year),
            lifetime = Int(e.lifetime),
            status   = e.tech,                # "smr" or "electrolysis"
            initial  = e.is_initial,
        ))
    end
end

# ── Colours ───────────────────────────────────────────────────────────────────────
station_colors = Dict(
    "operational"         => "#1B5E20",
    "under construction"  => "#388E3C",
    "FID"                 => "#66BB6A",
    "project development" => "#A5D6A7",
    "in planning"         => "#C8E6C9",
    "auto-projected"      => "#BDBDBD",
)
station_order = ["operational", "under construction", "FID",
                 "project development", "in planning", "auto-projected"]
status_label = Dict(
    "operational"         => "Operational",
    "under construction"  => "Under construction",
    "FID"                 => "FID (committed)",
    "project development" => "Project development",
    "in planning"         => "In planning",
    "auto-projected"      => "Auto-projected (model)",
)

tech_colors = Dict(
    "electrolysis" => "#4472C4",
    "smr"          => "#ED7D31",
)
tech_label = Dict("electrolysis" => "Electrolysis", "smr" => "SMR (natural gas)")

color_for(status, colormap) = get(colormap, status, "#9E9E9E")

# ── Helper: draw one stacked Gantt panel ───────────────────────────────────────────
# rows sorted by build year (then capacity desc); each row is a lane whose height
# equals its capacity and which spans [year, min(year+lifetime, end_year+1)].
function draw_gantt!(ax, rows, colormap)
    sorted = sort(rows, by = r -> (r.year, -r.capacity))
    total  = sum(r -> r.capacity, sorted; init = 0.0)
    y0 = 0.0
    prev_cap = NaN
    for r in sorted
        x0 = Float64(r.year)
        x1 = min(Float64(r.year + r.lifetime), Float64(end_year + 1))
        h  = r.capacity
        is_initial = haskey(r, :initial) && r.initial
        poly!(ax, Rect2f(x0, y0, x1 - x0, h);
              color       = color_for(r.status, colormap),
              strokecolor = is_initial ? :black : (:white, 0.9),
              strokewidth = is_initial ? 1.4 : 0.6)
        # Capacity label (t/day) at the bar's left edge — only for lanes tall enough
        # to fit text, and only on the first of a run of identical capacities so a
        # stack of equal-sized projected stations is labelled once, not N times.
        if h >= 0.03 * total && abs(h - prev_cap) > 1.0
            text!(ax, x0 + 0.12, y0 + h / 2;
                  text = string(round(r.capacity / 1000; digits = 1), " t/d"),
                  align = (:left, :center), fontsize = 6.5, color = :black)
        end
        prev_cap = h
        y0 += h
    end
    return y0
end

# ── Figure ──────────────────────────────────────────────────────────────────────
MM_TO_PT = 1 / 0.352778
fig = Figure(
    size     = (183 * MM_TO_PT, 130 * MM_TO_PT),
    fontsize = 8,
    fonts    = (; regular = "Arial", bold = "Arial Bold"),
)

xticks = collect(start_year:2:end_year)

ax_top = Axis(fig[1, 1];
    title           = "Refuelling stations",
    titlealign      = :left,
    titlefont       = "Arial Bold",
    ylabel          = "Daily H₂ capacity (kg day⁻¹)",
    xticks          = xticks,
    xgridvisible     = true,
    ygridvisible     = false,
    xticklabelsvisible = false,
)
ax_bot = Axis(fig[2, 1];
    title           = "Hydrogen production",
    titlealign      = :left,
    titlefont       = "Arial Bold",
    xlabel          = "Year",
    ylabel          = "Daily H₂ capacity (kg day⁻¹)",
    xticks          = xticks,
    xgridvisible     = true,
    ygridvisible     = false,
)
linkxaxes!(ax_top, ax_bot)

top_h = draw_gantt!(ax_top, stations,   station_colors)
bot_h = draw_gantt!(ax_bot, production, tech_colors)

CairoMakie.xlims!(ax_top, start_year - 0.3, end_year + 1.3)
CairoMakie.xlims!(ax_bot, start_year - 0.3, end_year + 1.3)
CairoMakie.ylims!(ax_top, 0, top_h * 1.04)
CairoMakie.ylims!(ax_bot, 0, bot_h * 1.04)

# ── Legends ───────────────────────────────────────────────────────────────────────
present_status = [s for s in station_order if any(r -> r.status == s, stations)]
Legend(fig[1, 2],
    [PolyElement(color = station_colors[s]) for s in present_status],
    [status_label[s] for s in present_status],
    "Status"; framevisible = false, labelsize = 7, titlesize = 7, patchsize = (10, 10))

present_tech = [t for t in ("electrolysis", "smr") if any(r -> r.status == t, production)]
tech_elems  = [PolyElement(color = tech_colors[t]) for t in present_tech]
tech_labels = [tech_label[t] for t in present_tech]
push!(tech_elems,  PolyElement(color = :white, strokecolor = :black, strokewidth = 1.4))
push!(tech_labels, "Initial facility")
Legend(fig[2, 2], tech_elems, tech_labels, "Technology";
    framevisible = false, labelsize = 7, titlesize = 7, patchsize = (10, 10))

colsize!(fig.layout, 2, Relative(0.18))
rowgap!(fig.layout, 6)

# ── Save ────────────────────────────────────────────────────────────────────────
pdf_path = fig_path(OUT_DIR, "fig_investment_timeline.pdf")
png_path = fig_path(OUT_DIR, "fig_investment_timeline.png")
save(pdf_path, fig; pt_per_unit = 1)
println("Saved → $pdf_path")
save(png_path, fig; px_per_unit = 300 / 72)
println("Saved → $png_path")

println("\nStations drawn: $(length(stations))  (config: $(length(cfg.station_data)), projected: $(length(stations) - length(cfg.station_data)))")
println("Production facilities drawn: $(length(production))")
