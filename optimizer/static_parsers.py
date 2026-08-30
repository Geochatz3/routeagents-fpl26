"""Static, stateless parsers for Vivado report text.

Extracted from dcp_optimizer.py — the only production
extraction sanctioned before the final-round deadline, because this is the one span
that passes every qualification gate: no free globals, no os.environ, no logger, no
clock, no randomness, no filesystem, no self-state, no decorators, and it cannot
mutate its input (str).

THE BODIES BELOW ARE VERBATIM. Do not reformat, rename, reorder dict keys, or "tidy"
them. Their exact behaviour on malformed input is depended on: `dcp_optimizer` reads
WNS through these at five call sites, and a changed default or key order silently
alters an optimization trajectory that costs an hour of Vivado per design to observe.

IMPORT CONTRACT — read before touching dcp_optimizer's import of this module.
`dcp_optimizer` imports these HARD, not with the soft `try/except -> None` pattern its
optional helpers use. That is deliberate: a None fallback would raise
`TypeError: 'NoneType' object is not callable` at a call site mid-run on the eval box,
which is strictly worse than failing loudly at import. The names must also stay
module-globals in `dcp_optimizer` and its call sites must keep using the BARE name, so
that `mock.patch("dcp_optimizer.parse_timing_summary_static")` (5 tests) still
intercepts every call. Re-exporting is part of the contract, not a convenience.
"""


def parse_timing_summary_static(timing_report: str) -> dict:
    """
    Parse timing summary report to extract WNS, TNS, and failing endpoints.
    Returns dict with keys: wns, tns, failing_endpoints
    
    Parses the Design Timing Summary table:
        WNS(ns)      TNS(ns)  TNS Failing Endpoints  ...
        -------      -------  ---------------------  ...
         -0.099       -1.449                     42  ...
    
    This is a shared utility function used by both FPGAOptimizer and FPGAOptimizerTest.
    """
    result = {
        "wns": None,
        "tns": None,
        "failing_endpoints": None
    }
    
    lines = timing_report.split('\n')
    
    # Find the line with "WNS(ns)" header
    header_idx = -1
    for i, line in enumerate(lines):
        if 'WNS(ns)' in line and 'TNS(ns)' in line:
            header_idx = i
            break
    
    if header_idx == -1:
        return result
    
    # The data line should be 2 lines after the header (skipping the dashes line)
    # Format: whitespace + values separated by whitespace
    data_idx = header_idx + 2
    if data_idx >= len(lines):
        return result
    
    data_line = lines[data_idx].strip()
    if not data_line:
        return result
    
    # Split by whitespace and extract first 3 values: WNS, TNS, TNS Failing Endpoints
    parts = data_line.split()
    if len(parts) >= 3:
        try:
            result["wns"] = float(parts[0])
            result["tns"] = float(parts[1])
            result["failing_endpoints"] = int(parts[2])
        except (ValueError, IndexError):
            # If parsing fails, leave as None
            pass
    
    return result
