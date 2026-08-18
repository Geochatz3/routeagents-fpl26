# place_gate_probe.tcl — R&D probe (NOT wired into the agent).
#
# Question (2026-06-12): does post-PLACE WNS rank-predict post-ROUTE WNS
# across place directives (multi-fidelity culling)? decisions.jsonl cannot
# answer (wns_after is the tracer's running value, identical across stages
# — artifact found jun12). If the ranking holds, ILS cycles gain a
# place-gate: measure after place_design, skip route+phys_opt when the
# post-place WNS already trails best by more than a margin — the same
# philosophy as the existing post-route phys_opt skip, one stage earlier.
# Output: PAIR lines (directive, wns_place, wns_route, dt_place, dt_route)
# for offline rank/margin analysis.
#
# Usage: vivado -mode batch -source place_gate_probe.tcl -tclargs <in.dcp>

proc wns_now {} {
    set clk [get_clocks -quiet *fpl26contest*]
    if {[llength $clk] > 0} {
        set tp [get_timing_paths -quiet -max_paths 1 -setup -to [lindex $clk 0]]
    } else {
        set tp [get_timing_paths -quiet -max_paths 1 -slack_lesser_than 999]
    }
    if {[llength $tp] == 0} { return 0.0 }
    return [get_property SLACK [lindex $tp 0]]
}
proc tic {} { return [clock seconds] }

set in_dcp [lindex $argv 0]
# The ILS rotation's place directives (winners first), i.e. exactly the
# candidates a place-gate would be culling between.
set directives {Explore ExtraTimingOpt AltSpreadLogic_high ExtraNetDelay_high SSI_SpreadLogic_high EarlyBlockPlacement}

foreach d $directives {
    if {[catch {
        open_checkpoint $in_dcp
        set t0 [tic]
        place_design -unplace
        place_design -directive $d
        set wp [wns_now]
        set dtp [expr {[clock seconds] - $t0}]
        set t1 [tic]
        route_design
        set wr [wns_now]
        set dtr [expr {[clock seconds] - $t1}]
        puts "PROBE_PAIR directive=$d wns_place=$wp wns_route=$wr dt_place=${dtp}s dt_route=${dtr}s"
    } err]} {
        puts "PROBE_PAIR_ERROR directive=$d err=$err"
    }
}
puts "PROBE_DONE=1"
