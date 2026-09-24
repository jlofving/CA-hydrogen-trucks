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

# The No/Low deployment scenario is excluded from this summary. Its fleet empties
# inside the horizon: from 2033 the fixed station cost is spread over a collapsing
# volume and the price runs off the figure's y-axis, and from 2037 the model has
# no fleet to price at all and reports a $40/kg sentinel. Neither is a cost that
# belongs in a summary table. The full rows, with that explanation, stay in
# fig_scenario_matrix_overlay_spread.txt.
const SKIP_SCENARIO = "No/Low"
filter!(kv -> !occursin(SKIP_SCENARIO, kv.first), f7)
isempty(f7) && error("every LCOH scenario was filtered out — check SKIP_SCENARIO")

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

const TABLES = (
    (title = "LCOH — delivered hydrogen fuel cost    [fig 7]",  unit = "USD/kg",
     rows = f7,  label = label7,  fmt = m -> @sprintf("%.2f", m)),
    (title = "TCO — hydrogen truck cost of ownership [fig 8]",  unit = "USD/mile",
     rows = f8,  label = label8,  fmt = m -> @sprintf("%.3f", m)),
    (title = "MAC — net CO2 abatement cost           [fig 13]", unit = "USD/tCO2e",
     rows = f13, label = label13, fmt = m -> @sprintf("%.0f", m)),
)

const EXCLUDED_NOTE = "No/Low deployment is omitted: its fleet empties during the horizon, " *
                      "so from 2033 its cost runs off the figure's y-axis and from 2037 it is the " *
                      "model's zero-fleet sentinel rather than a cost. See " *
                      "fig_scenario_matrix_overlay_spread.txt for those rows."

const CAVEAT = "Spread is investment timing only (build trigger, p_invest, 2–4 yr lead time, " *
               "probabilistic station openings). Economic inputs are point values and the truck " *
               "fleet is scheduled, so this is not a confidence interval on cost."

function report(io)
    println(io, "="^96)
    println(io, " MONTE CARLO SPREAD SUMMARY — median (IQR as % of median)")
    println(io, "="^96)
    println(io, " Condensed from the three per-figure spread dumps at the reporting years.")
    println(io, " Spread source: investment timing only — build trigger, p_invest, 2-4 yr lead")
    println(io, " time, probabilistic station openings. Economic inputs are point values and")
    println(io, " the truck fleet is scheduled, so this is not a confidence interval on cost.")
    println(io, "="^96)

    for t in TABLES
        table(io, t.title, t.unit, t.rows, t.label, t.fmt)
    end

    println(io)
    println(io, "="^96)
    for l in wrap(EXCLUDED_NOTE, 94); println(io, " ", l); end
    println(io, "="^96)
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
#
# Both carry median and IQR in separate columns rather than "12.68 (10%)" in
# one, so the numbers stay sortable and formattable once they land.

cell(v, fmt) = v === nothing ? ("—", "—") : (fmt(v[1]), string(round(Int, v[2])))

"""
    wrap(text, width) -> Vector{String}

Greedy word wrap, so the shared note strings can be reused verbatim in the HTML
and TSV (where the reader's window does the wrapping) while still respecting the
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

function write_tsv(io)
    for t in TABLES
        println(io, t.title, "\t", t.unit)
        print(io, "Scenario")
        for y in YEARS; print(io, "\t", y, " median\t", y, " IQR %"); end
        println(io)
        for (k, v) in t.rows
            print(io, t.label(k))
            for y in YEARS
                m, p = cell(get(v, y, nothing), t.fmt)
                print(io, "\t", m, "\t", p)
            end
            println(io)
        end
        println(io)
    end
    println(io, CAVEAT)
    println(io, EXCLUDED_NOTE)
end

function write_html(io)
    println(io, """<!DOCTYPE html><meta charset="utf-8">
<title>Monte Carlo spread summary</title>
<style>
 body  { font: 11pt/1.4 "Calibri", sans-serif; margin: 2em; }
 table { border-collapse: collapse; margin-bottom: 2em; }
 caption { caption-side: top; text-align: left; font-weight: bold; padding-bottom: .4em; }
 th, td { border: 1px solid #999; padding: 3px 8px; }
 th    { background: #eee; }
 td.n  { text-align: right; }
 td.q  { text-align: right; color: #555; }
 p.note { font-size: 9pt; color: #555; max-width: 46em; }
</style>
<h2>Monte Carlo spread summary — median and interquartile range</h2>""")
    for t in TABLES
        println(io, "<table>")
        println(io, "<caption>", t.title, " — ", t.unit, "</caption>")
        print(io, "<tr><th rowspan=\"2\">Scenario</th>")
        for y in YEARS; print(io, "<th colspan=\"2\">", y, "</th>"); end
        println(io, "</tr>")
        print(io, "<tr>")
        for _ in YEARS; print(io, "<th>median</th><th>IQR&#160;%</th>"); end
        println(io, "</tr>")
        for (k, v) in t.rows
            print(io, "<tr><td>", t.label(k), "</td>")
            for y in YEARS
                m, p = cell(get(v, y, nothing), t.fmt)
                print(io, "<td class=\"n\">", m, "</td><td class=\"q\">", p, "</td>")
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
