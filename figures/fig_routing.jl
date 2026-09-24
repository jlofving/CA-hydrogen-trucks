# =============================================================================
# FIGURE OUTPUT ROUTING  —  which subfolder of figures/out each figure belongs in
# =============================================================================
# The manuscript and its appendix each have a dedicated folder so that a
# regenerated figure lands where it is cited, instead of leaving a stale copy
# behind. Anything not listed below is retired from the paper and goes to
# `old/`.
#
# To promote or retire a figure, move its name between the groups here — this
# table is the single place that decides destinations. Names are the output
# file's basename WITHOUT extension (i.e. what is passed to `save_pub`).
#
# Included by pub_theme.jl (so every `save_pub` caller gets routing for free)
# and directly by the scripts that write their PDF/PNG by hand via `fig_path`.
# Deliberately free of `set_theme!` and of any plotting dependency, so a script
# can include it without picking up the publication theme.
# =============================================================================

if !@isdefined(FIG_DESTINATIONS)

# ORDER MATTERS: this is the figure order in the manuscript, and it sets the
# numeric prefix on the output files (position 1 → fig_1_…, position 2 → fig_2_…).
# Reordering this list renumbers the files on the next run.
const FIG_MANUSCRIPT = [
    # Authored, not plotted: an editable PPTX schematic of the model structure,
    # built by figures/make_schematic.py. No Julia script writes it, but it is
    # listed here so this table remains the single source of figure numbering —
    # the Python script reads its number back out of this list.
    "fig_model_schematic",
    "fig_learning_curves",
    "fig_station_costs",
    "fig_scenario_truck_deployment",
    "fig_price_breakdown",
    "fig_utilization_impact",
    "fig_scenario_matrix_overlay",
    "fig_tco_comparison",
    # Sits behind the TCO figure it qualifies: what the price and TCO would be if
    # standing capacity were re-valued at each year's technology cost instead of
    # keeping its build-year CAPEX (figures/fig_vintage_effect.jl).
    "fig_vintage_effect",
    "fig_sensitivity_grid",
    # The former 6-panel fig_cost_premium_3row, split in two so each figure
    # carries one message. Both come from fig_societal_cost_benefit.jl (one
    # shared Monte Carlo), so their numbers reconcile.
    "fig_cost_premium_benefit",
    "fig_net_societal_cost",
    "fig_abatement_cost",
]

const FIG_APPENDIX = [
    "fig_background_demand",
    "fig_scc_fit",
    "fig_capacity_vs_demand_combined",
    "fig_tco_premium",
    "fig_societal_benefit",
    # The policy support stripped out of the social figures' resource-cost
    # premium, shown by funder against the damages it buys. Cited alongside the
    # subsidy-accounting note, so it belongs with the paper rather than in old/.
    "fig_policy_expenditure",
    # High-deployment counterpart of manuscript fig 10, from
    # `fig_sensitivity.jl high_dep`. The tornado variant is not routed here and
    # so lands in old/, matching where the baseline tornado goes.
    "fig_sensitivity_grid_high_dep",
]

const FIG_DESTINATIONS = merge(
    Dict(n => "manuscript" for n in FIG_MANUSCRIPT),
    Dict(n => "appendix"   for n in FIG_APPENDIX),
)

# Figures absent from the table are no longer cited in the paper.
const FIG_DEFAULT_SUBDIR = "old"

# Manuscript figure number, taken from the position in FIG_MANUSCRIPT.
const FIG_NUMBER = Dict(n => i for (i, n) in enumerate(FIG_MANUSCRIPT))

end  # @isdefined guard

"""
    fig_subdir(name) -> String

Subfolder of `figures/out` that the figure `name` (basename, no extension)
belongs in: "manuscript", "appendix", or "old" for anything unlisted.
"""
fig_subdir(name::AbstractString) = get(FIG_DESTINATIONS, String(name), FIG_DEFAULT_SUBDIR)

"""
    fig_basename(name) -> String

Output basename for the figure `name`. Manuscript figures get their citation
number spliced in after the `fig_` prefix — `"fig_learning_curves"` →
`"fig_2_learning_curves"` — so the files sort and read in manuscript order.
Appendix and retired figures keep their name unchanged.
"""
function fig_basename(name::AbstractString)
    i = get(FIG_NUMBER, String(name), nothing)
    i === nothing && return String(name)
    stem = startswith(name, "fig_") ? name[5:end] : name
    return "fig_$(i)_$(stem)"
end

"""
    fig_path(base_dir, filename) -> String

Full output path for `filename` (e.g. "fig_truck_cost.pdf"), routed into the
correct subfolder of `base_dir`, numbered via [`fig_basename`](@ref) if it is a
manuscript figure, and with that folder created. Drop-in replacement for
`joinpath(OUT_DIR, filename)` in scripts that save by hand.
"""
function fig_path(base_dir::AbstractString, filename::AbstractString)
    name, ext = splitext(filename)
    dir = joinpath(base_dir, fig_subdir(name))
    mkpath(dir)
    return joinpath(dir, fig_basename(name) * ext)
end
