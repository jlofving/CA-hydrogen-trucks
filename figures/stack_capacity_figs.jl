using CairoMakie
const FileIO = CairoMakie.FileIO

include(joinpath(@__DIR__, "fig_routing.jl"))   # figure → manuscript/appendix/old routing
const OUT_DIR = joinpath(@__DIR__, "out")

# Inputs are the two single-scenario capacity figures (routed to old/); the
# stacked result is the appendix figure.
top = FileIO.load(fig_path(OUT_DIR, "fig_capacity_vs_demand.png"))
bot = FileIO.load(fig_path(OUT_DIR, "fig_capacity_vs_demand_high.png"))

# images are (height, width); pad to equal width with white, then vcat (top over bottom)
w = max(size(top, 2), size(bot, 2))
white = oneunit(eltype(top))

function pad_width(img, w)
    h, iw = size(img)
    iw == w && return img
    canvas = fill(white, h, w)
    off = (w - iw) ÷ 2
    canvas[:, off+1:off+iw] .= img
    return canvas
end

top = pad_width(top, w)
bot = pad_width(bot, w)

combined = vcat(top, bot)
FileIO.save(fig_path(OUT_DIR, "fig_capacity_vs_demand_combined.png"), combined)
println("Saved fig_capacity_vs_demand_combined.png  ", size(combined))
