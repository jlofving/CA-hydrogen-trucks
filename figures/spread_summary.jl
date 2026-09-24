# =============================================================================
# SPREAD SUMMARY — one table of medians and interquartile ranges
# =============================================================================
# Condenses the three per-figure Monte Carlo spread dumps into a single digest
# at the reporting years, so the headline numbers can be quoted without opening
# three files and reading across twenty rows each:
#
#   LCOH  delivered hydrogen fuel cost   fig_scenario_matrix_overlay_spread.txt
#   TCO   truck total cost of ownership  fig_tco_comparison_spread.txt
#   MAC   net CO2 abatement cost         fig_abatement_cost_spread.txt
#
# Each cell is the median with the interquartile range beside it, expressed as
# a percentage of that median. The three are the same uncertainty seen through
# three denominators, which is the point of putting them side by side: the
# spread that reads as ~10% on a fuel cost is ~3% once diluted into cost per
# mile and 20-30% once divided by a near-constant abated tonnage.
#
# WHAT THE SPREAD IS. Only the infrastructure investment mechanics vary across
# runs — the build trigger, p_invest, whether a commitment fires in a given
# year, the 2-4 year build lead time, and probabilistic station openings. Every
# economic input is held at a point value, and the truck fleet follows a fixed
# schedule. So this is the spread from WHEN capacity arrives and in what lumps,
# not a confidence interval on cost. Parameter uncertainty is fig_sensitivity.jl.
#
# This script reads the three dumps rather than re-running the model, so it is
# instant but stale if they are: regenerate them first, or run the suite.
#
#   julia --project=figures figures/fig_scenario_matrix_overlay.jl
#   julia --project=figures figures/fig_tco_comparison.jl
#   julia --project=figures figures/fig_abatement_cost.jl
#   julia --project=figures figures/spread_summary.jl
# =============================================================================

using Printf

const OUT_DIR = joinpath(@__DIR__, "out")
const YEARS   = [2026, 2030, 2035, 2040, 2045]

"""
    parse_dump(file, is_header, med_col, pct_col) -> Vector{Pair}

Pull (scenario => year => (median, iqr_pct)) out of one spread dump. Column
positions are passed in because the three dumps do not share a layout; they are
1-based indices into the whitespace-split data row with bare "%" tokens dropped.
Order of appearance is preserved so the summary reads in the same order as the
figures. Throws if the file is missing or yields nothing, rather than quietly
emitting an empty table.
"""
function parse_dump(file, is_header, med_col, pct_col)
    path = joinpath(OUT_DIR, file)
    isfile(path) || error("missing $path — run the figure script that writes it first " *
                          "(see the header of this file)")
    rows = Pair{String,Dict{Int,Tuple{Float64,Float64}}}[]
    idx  = Dict{String,Int}()
    cur  = ""
    for line in eachline(path)
        s = strip(line)
        (isempty(s) || all(c -> c in ('=', '-', ' '), s)) && continue
        if occursin(r"^\s+\d{4}\s", line)
            isempty(cur) && continue
            f = filter(!=("%"), split(line))
            y = parse(Int, f[1])
            y in YEARS || continue
            haskey(idx, cur) || (push!(rows, cur => Dict{Int,Tuple{Float64,Float64}}()); idx[cur] = length(rows))
            rows[idx[cur]].second[y] = (parse(Float64, f[med_col]), parse(Float64, f[pct_col]))
        elseif is_header(s)
            cur = s
        end
    end
    isempty(rows) && error("parsed no data rows from $path — its layout has changed; " *
                           "check the column indices in this script")
    return rows
end

# Column indices below track the @printf formats in the three writing scripts.
f7  = parse_dump("fig_scenario_matrix_overlay_spread.txt",
                 s -> occursin('|', s) && occursin("LCFS", s), 4, 8)
f8  = parse_dump("fig_tco_comparison_spread.txt",
                 s -> occursin('|', s) && occursin("DEPLOYMENT", s), 3, 6)
f13 = parse_dump("fig_abatement_cost_spread.txt",
                 s -> occursin('—', s) && occursin("diesel", s), 3, 7)

# ── Scenario labels, shortened to fit one line ────────────────────────────────
short(s) = replace(s, "Electrolysis — grid" => "Grid elec.", "Electrolysis — solar" => "Solar elec.",
                      "SMR (2026 mix)" => "SMR", "SMR (current mix)" => "SMR",
                      " deployment" => "", " DEPLOYMENT" => "")
function label7(k)
    p = strip.(split(short(k), '|'));  "$(p[2]) · $(p[1])"
end
function label8(k)
    p = strip.(split(short(k), '|'));  "$(p[2]) · $(titlecase(lowercase(p[1])))"
end
function label13(k)
    p = strip.(split(short(k), '—'))
    "$(titlecase(lowercase(p[1]))) · $(p[2])"
end

function table(io, title, unit, rows, label, fmt)
    println(io)
    println(io, "$title  —  $unit")
    println(io, "-"^96)
    @printf(io, " %-28s %13s %13s %13s %13s %13s\n", "Scenario", YEARS...)
    println(io, " " * "-"^94)
    for (k, v) in rows
        @printf(io, " %-28s", label(k))
        for y in YEARS
            if haskey(v, y)
                m, p = v[y]
                @printf(io, " %13s", string(fmt(m), " (", round(Int, p), "%)"))
            else
                @printf(io, " %13s", "—")
            end
        end
        println(io)
    end
end

function report(io)
    println(io, "="^96)
    println(io, " MONTE CARLO SPREAD SUMMARY — median (IQR as % of median)")
    println(io, "="^96)
    println(io, " Condensed from the three per-figure spread dumps at the reporting years.")
    println(io, " Spread source: investment timing only — build trigger, p_invest, 2-4 yr lead")
    println(io, " time, probabilistic station openings. Economic inputs are point values and")
    println(io, " the truck fleet is scheduled, so this is not a confidence interval on cost.")
    println(io, "="^96)

    table(io, "LCOH — delivered hydrogen fuel cost    [fig 7]",  "USD/kg",    f7,  label7,  m -> @sprintf("%.2f", m))
    table(io, "TCO — hydrogen truck cost of ownership [fig 8]",  "USD/mile",  f8,  label8,  m -> @sprintf("%.3f", m))
    table(io, "MAC — net CO2 abatement cost           [fig 13]", "USD/tCO2e", f13, label13, m -> @sprintf("%.0f", m))

    println(io)
    println(io, "="^96)
    println(io, " NOTE  The No/Low deployment rows of the LCOH table run off the figure's")
    println(io, " y-axis from 2033 and reach the model's zero-fleet sentinel by 2037. See the")
    println(io, " header of fig_scenario_matrix_overlay_spread.txt before quoting them.")
    println(io, "="^96)
end

report(stdout)
let path = joinpath(OUT_DIR, "spread_summary.txt")
    open(report, path, "w")
    println("\nSaved → $path")
end
