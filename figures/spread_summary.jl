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

# Reporting years. Deliberately sparse: the point of this file is the shape of
# the trajectory and how the spread behaves along it, which three points carry
# as well as five. Every year is in the per-figure dumps.
const YEARS = [2026, 2035, 2045]

"""
    parse_dump(file, is_header, cols) -> Vector{Pair}

Pull (scenario => year => (p25, median, p75, iqr_pct)) out of one spread dump.
`cols` is a NamedTuple of 1-based indices into the whitespace-split data row
with bare "%" tokens dropped; the three dumps do not share a layout, so the
positions are passed in rather than guessed. Order of appearance is preserved
so the summary reads in the same order as the figures. Throws if the file is
missing or yields nothing, rather than quietly emitting an empty table.
"""
function parse_dump(file, is_header, cols)
    path = joinpath(OUT_DIR, file)
    isfile(path) || error("missing $path — run the figure script that writes it first " *
                          "(see the header of this file)")
    rows = Pair{String,Dict{Int,NTuple{4,Float64}}}[]
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
            haskey(idx, cur) ||
                (push!(rows, cur => Dict{Int,NTuple{4,Float64}}()); idx[cur] = length(rows))
            rows[idx[cur]].second[y] = (parse(Float64, f[cols.p25]), parse(Float64, f[cols.med]),
                                        parse(Float64, f[cols.p75]), parse(Float64, f[cols.pct]))
        elseif is_header(s)
            cur = s
        end
    end
    isempty(rows) && error("parsed no data rows from $path — its layout has changed; " *
                           "check the column indices in this script")
    return rows
end

# Column indices track the @printf formats in the three writing scripts.
f7  = parse_dump("fig_scenario_matrix_overlay_spread.txt",
                 s -> occursin('|', s) && occursin("LCFS", s),
                 (p25 = 3, med = 4, p75 = 5, pct = 8))
f8  = parse_dump("fig_tco_comparison_spread.txt",
                 s -> occursin('|', s) && occursin("DEPLOYMENT", s),
                 (p25 = 2, med = 3, p75 = 4, pct = 6))
f13 = parse_dump("fig_abatement_cost_spread.txt",
                 s -> occursin('—', s) && occursin("diesel", s),
                 (p25 = 2, med = 3, p75 = 4, pct = 7))

# The No/Low deployment scenario is excluded. Its fleet empties inside the
# horizon: from 2033 the fixed station cost is spread over a collapsing volume
# and the price runs off the figure's y-axis, and from 2037 the model has no
# fleet to price at all and reports a $40/kg sentinel. Neither is a cost that
# belongs in a summary table. The full rows, with that explanation, stay in
# fig_scenario_matrix_overlay_spread.txt.
const SKIP_SCENARIO = "No/Low"
filter!(kv -> !occursin(SKIP_SCENARIO, kv.first), f7)
isempty(f7) && error("every LCOH scenario was filtered out — check SKIP_SCENARIO")

# ── Scenario labels, shortened to fit one line ────────────────────────────────
short(s) = replace(s, "Electrolysis — grid" => "Grid elec.", "Electrolysis — solar" => "Solar elec.",
                      "SMR (2026 mix)" => "SMR", "SMR (current mix)" => "SMR",
                      " deployment" => "", " DEPLOYMENT" => "")
label7(k)  = (p = strip.(split(short(k), '|')); "$(p[2]) · $(p[1])")
label8(k)  = (p = strip.(split(short(k), '|')); "$(p[2]) · $(titlecase(lowercase(p[1])))")
label13(k) = (p = strip.(split(short(k), '—')); "$(titlecase(lowercase(p[1]))) · $(p[2])")

const TABLES = (
    (title = "LCOH — delivered hydrogen fuel cost    [fig 7]",  unit = "USD/kg",
     rows = f7,  label = label7,  fmt = m -> @sprintf("%.2f", m)),
    (title = "TCO — hydrogen truck cost of ownership [fig 8]",  unit = "USD/mile",
     rows = f8,  label = label8,  fmt = m -> @sprintf("%.3f", m)),
    (title = "MAC — net CO2 abatement cost           [fig 13]", unit = "USD/tCO2e",
     rows = f13, label = label13, fmt = m -> @sprintf("%.0f", m)),
)

const STATS  = ("P25", "median", "P75", "IQR %")
const CAVEAT = "Spread is investment timing only (build trigger, p_invest, 2–4 yr lead time, " *
               "probabilistic station openings). Economic inputs are point values and the truck " *
               "fleet is scheduled, so this is not a confidence interval on cost."
const EXCLUDED_NOTE = "No/Low deployment is omitted: its fleet empties during the horizon, " *
                      "so from 2033 its cost runs off the figure's y-axis and from 2037 it is the " *
                      "model's zero-fleet sentinel rather than a cost. See " *
                      "fig_scenario_matrix_overlay_spread.txt for those rows."

"""
    wrap(text, width) -> Vector{String}

Greedy word wrap, so the shared note strings can be reused verbatim in the HTML
and TSV (where the reader's window wraps them) while still respecting the
fixed-width rules in the .txt.
"""
function wrap(text, width)
    lines, cur = String[], ""
    for w in split(text)
        if isempty(cur)
            cur = w
        elseif length(cur) + 1 + length(w) <= width
            cur *= " " * w
        else
            push!(lines, cur); cur = w
        end
    end
    isempty(cur) || push!(lines, cur)
    return lines
end

# Four statistics per year, so the fixed-width table is wider than the usual
# 96-column dumps. Sized to the content rather than to a rule it cannot meet.
const LW    = 24                            # scenario label field
const CW    = 9                             # one statistic field
const GW    = length(STATS) * (CW + 1)      # one year group, incl. its separators
const TOTAL = 1 + LW + length(YEARS) * GW   # full row width

"Centre `s` in a field of `w` characters, for the year headers that span a group."
function centre(s, w)
    t = string(s); pad = max(0, w - length(t))
    return " "^(pad ÷ 2) * t * " "^(pad - pad ÷ 2)
end

cells(v, fmt) = v === nothing ? fill("—", 4) :
                [fmt(v[1]), fmt(v[2]), fmt(v[3]), string(round(Int, v[4])) * "%"]

function table(io, t)
    println(io)
    println(io, "$(t.title)  —  $(t.unit)")
    println(io, "-"^TOTAL)
    print(io, " ", " "^LW)
    for y in YEARS; print(io, centre(y, GW)); end
    println(io)
    @printf(io, " %-*s", LW, "Scenario")
    for _ in YEARS, s in STATS; @printf(io, " %*s", CW, s); end
    println(io)
    println(io, " ", "-"^(TOTAL - 1))
    for (k, v) in t.rows
        @printf(io, " %-*s", LW, t.label(k))
        for y in YEARS, c in cells(get(v, y, nothing), t.fmt)
            @printf(io, " %*s", CW, c)
        end
        println(io)
    end
end

function report(io)
    println(io, "="^TOTAL)
    println(io, " MONTE CARLO SPREAD SUMMARY — P25, median, P75 and IQR as % of median")
    println(io, "="^TOTAL)
    for l in wrap("Condensed from the three per-figure spread dumps at the reporting years. " *
                  CAVEAT, TOTAL - 2)
        println(io, " ", l)
    end
    println(io, "="^TOTAL)
    for t in TABLES; table(io, t); end
    println(io)
    println(io, "="^TOTAL)
    for l in wrap(EXCLUDED_NOTE, TOTAL - 2); println(io, " ", l); end
    println(io, "="^TOTAL)
end

# ─────────────────────────────────────────────────────────────────────────────
# Paste-ready variants
# ─────────────────────────────────────────────────────────────────────────────
# The fixed-width table above is for reading in a terminal; pasted into a word
# processor it arrives as one monospace blob. These two are for getting the same
# numbers into a document as an actual table:
#
#   .html  open in a browser, select the table, copy, paste into Word. Arrives
#          as a native Word table, borders and all. One step, no dialog.
#   .tsv   paste into Word, select it, then Insert ▸ Table ▸ Convert Text to
#          Table with tabs as the separator. Also opens directly in Excel.

# Column widths for the pasted table, in points.
const LABEL_PT = 96
const NUM_PT   = 30

function write_tsv(io)
    for t in TABLES
        println(io, t.title, "\t", t.unit)
        print(io, "Scenario")
        for y in YEARS, s in STATS; print(io, "\t", y, " ", s); end
        println(io)
        for (k, v) in t.rows
            print(io, t.label(k))
            for c in Iterators.flatten(cells(get(v, y, nothing), t.fmt) for y in YEARS)
                print(io, "\t", c)
            end
            println(io)
        end
        println(io)
    end
    println(io, CAVEAT)
    println(io, EXCLUDED_NOTE)
end

function write_html(io)
    # Widths are given in points, per column, and the layout is fixed. Word sizes
    # a pasted table from the CSS it is given and otherwise falls back to
    # stretching every column to the page width, which is what makes the cells
    # look oversized. 13 columns at these widths come to roughly 460 pt, which
    # fits A4 portrait with normal margins.
    println(io, """<!DOCTYPE html><meta charset="utf-8">
<title>Monte Carlo spread summary</title>
<style>
 body  { font: 10pt/1.3 "Calibri", sans-serif; margin: 2em; }
 table { border-collapse: collapse; table-layout: fixed; width: auto;
         margin-bottom: 1.6em; font-size: 8.5pt; }
 caption { caption-side: top; text-align: left; font-weight: bold;
           font-size: 9.5pt; padding-bottom: .3em; white-space: nowrap; }
 th, td { border: 0.5pt solid #999; padding: 0 3pt; white-space: nowrap;
          overflow: hidden; line-height: 1.25; }
 th    { background: #eee; font-weight: normal; text-align: center; }
 td.n  { text-align: right; }
 td.m  { text-align: right; font-weight: bold; }
 td.q  { text-align: right; color: #555; }
 p.note { font-size: 8pt; color: #555; max-width: 40em; }
</style>
<h2 style="font-size:12pt">Monte Carlo spread summary — P25, median, P75</h2>""")
    for t in TABLES
        println(io, "<table>")
        println(io, "<caption>", t.title, " — ", t.unit, "</caption>")
        print(io, "<colgroup><col style=\"width:", LABEL_PT, "pt\">")
        for _ in YEARS, _ in STATS; print(io, "<col style=\"width:", NUM_PT, "pt\">"); end
        println(io, "</colgroup>")
        print(io, "<tr><th rowspan=\"2\">Scenario</th>")
        for y in YEARS; print(io, "<th colspan=\"", length(STATS), "\">", y, "</th>"); end
        println(io, "</tr>")
        print(io, "<tr>")
        for _ in YEARS, s in STATS; print(io, "<th>", replace(s, " " => "&#160;"), "</th>"); end
        println(io, "</tr>")
        for (k, v) in t.rows
            print(io, "<tr><td>", t.label(k), "</td>")
            for y in YEARS
                c = cells(get(v, y, nothing), t.fmt)
                print(io, "<td class=\"n\">", c[1], "</td><td class=\"m\">", c[2],
                          "</td><td class=\"n\">", c[3], "</td><td class=\"q\">", c[4], "</td>")
            end
            println(io, "</tr>")
        end
        println(io, "</table>")
    end
    println(io, "<p class=\"note\">", CAVEAT, "</p>")
    println(io, "<p class=\"note\">", EXCLUDED_NOTE, "</p>")
end

report(stdout)
for (name, writer) in (("spread_summary.txt",  report),
                       ("spread_summary.tsv",  write_tsv),
                       ("spread_summary.html", write_html))
    path = joinpath(OUT_DIR, name)
    open(writer, path, "w")
    println("Saved → $path")
end
