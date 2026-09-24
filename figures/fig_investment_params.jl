# =============================================================================
# FIGURE: INVESTMENT PARAMETERS
# =============================================================================
# Single-panel figure showing:
#   • Investment trigger threshold θ — STOCHASTIC, drawn once per Monte Carlo run
#     from a Uniform[0.60, 0.80] distribution (flat density).
#   • Investment probability p_invest — stochastic, Beta(3,3) scaled to [0.60, 0.80].
# Both parameters are drawn once per MC run and held fixed across the horizon.
#
# Run from project root:
#   julia --project=figures figures/fig_investment_params.jl
# =============================================================================

using CairoMakie

include(joinpath(@__DIR__, "fig_routing.jl"))   # figure → manuscript/appendix/old routing
const OUT_DIR = joinpath(@__DIR__, "out")
mkpath(OUT_DIR)

# ── Beta distribution helpers ─────────────────────────────────────────────────
function beta_pdf(x, a, b)
    B = exp(sum(log.(1:a-1)) + sum(log.(1:b-1)) - sum(log.(1:a+b-1)))
    return x^(a-1) * (1-x)^(b-1) / B
end

function scaled_beta_pdf(v, a, b, lo, hi)
    x = (v - lo) / (hi - lo)
    (x < 0 || x > 1) && return 0.0
    return beta_pdf(x, a, b) / (hi - lo)
end

# ── Figure ────────────────────────────────────────────────────────────────────
MM_TO_PT = 1 / 0.352778
c_blue   = colorant"#4472C4"
c_orange = colorant"#ED7D31"

fig = Figure(size = (90 * MM_TO_PT, 80 * MM_TO_PT), fontsize = 8)

ax = Axis(fig[1, 1];
    xlabel         = "Parameter value",
    ylabel         = "Probability density",
    xlabelsize     = 9,
    ylabelsize     = 9,
    xticklabelsize = 8,
    yticklabelsize = 8,
    limits         = ((0.35, 1.0), (0, nothing)),
)

text!(ax, 0.03, 0.97; text = "(a)", space = :relative, align = (:left, :top),
      fontsize = 8, font = :bold, color = :black)

xs = range(0.35, 1.0, length = 500)

# p_invest — stochastic Beta(3,3) on [0.60, 0.80]
pinv = scaled_beta_pdf.(xs, 3, 3, 0.60, 0.80)
lines!(ax, collect(xs), pinv;
       color = c_orange, linewidth = 1.8, label = "p_invest  [0.60–0.80]")

# Trigger θ — stochastic, Uniform[0.60, 0.80] (flat density), drawn once per run.
ymax = maximum(pinv)
uni_h = 1.0 / (0.80 - 0.60)   # uniform density height = 1/(hi-lo) = 5.0
band!(ax, [0.60, 0.80], [0.0, 0.0], [uni_h, uni_h];
      color = (c_blue, 0.18))
lines!(ax, [0.60, 0.60, 0.80, 0.80], [0.0, uni_h, uni_h, 0.0];
       color = c_blue, linewidth = 1.8, label = "Trigger θ  [0.60–0.80]")

axislegend(ax; position = :lt, labelsize = 7, framevisible = true,
           rowgap = -2, patchsize = (12, 8))

# ── Save ──────────────────────────────────────────────────────────────────────
out_pdf = fig_path(OUT_DIR, "fig_investment_params.pdf")
out_png = fig_path(OUT_DIR, "fig_investment_params.png")
save(out_pdf, fig; pt_per_unit = 1)
println("Saved → $out_pdf")
save(out_png, fig; px_per_unit = 300 / 72)
println("Saved → $out_png")
