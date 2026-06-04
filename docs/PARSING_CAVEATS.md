# Parsing Caveats & Heuristics

## LlamaParse Quirks
- Table cells sometimes have escaped bold: `\*\*COBS 4.12A.44\*\*` — we strip these
- Rule type appears in 6+ formats: `<u>R</u>`, `<mark>G</mark>`, `**R**`, `**COBS 4.12A.21 R**` (inside bold), plain `R`, or `<span>` tags
- Table rows have no trailing `|` — we parse line-by-line, split on `|`
- Non-rule tables (fee schedules, etc.) exist — we validate rule_id + type before accepting a row

## Merge Bugs & the Rule Splitter
Parser's cross-page merging sometimes swallows subsequent rules. `rule_splitter.py` is a safety net:
- **Pass 1**: Split at `**bold rule IDs**` in text
- **Pass 2**: For rules >3K chars, also split at non-bold rule IDs at line starts
- Result: 4,413 → 5,753 rules (+1,340 freed)
- Can't fix: rules at part boundaries (COBS has 10 parts, MCOB has 6 — ~14 boundary edges)

## Regex Gotchas
- Rule ID pattern must be 3-segment explicit (`\d+[A-Z]?\.\d+[A-Z]?\.\d+[A-Z]*`), NOT greedy `[\d.]+[A-Z]*` which mismatches on IDs like `4.12A.9B`
- Cross-refs strip trailing R/G/E/D after digits: `re.sub(r'(?<=\d)[RGDE]$', '', num)` — preserves alpha suffixes (1.1.4A stays)
- XREF_RE allows `*`, `(` before sourcebook name (refs in italics: `*CMCOB 2.1.7R*`)

## ID Format
| Field | Format | Example |
|---|---|---|
| `rule_id` | Base ID, no type | `COBS 2.1.1` |
| `rule_type` | Separate field | `R` |
| `cross_references` | Base IDs | `["COBS 2.1.1", "CASS 5.5.14"]` |
| Display/citation | `rule_id + rule_type` | `COBS 2.1.1R` |
| D/E suffix ambiguity | Minor — `CASS 6.1.6D` could be rule 6D or type Direction. Affects handful of rules. |

## Known Gaps
- **Empty rule_type**: 1,144/5,753 (20%) — mostly splitter-freed rules where type wasn't in the format
- **Cross-ref match rate**: 84% match a known rule_id. 16% are external refs (SYSC, GEN, SUP etc.) or missed rules
- **External sourcebooks**: 175 refs to 17 books not in our set — preserved in text, not in graph
