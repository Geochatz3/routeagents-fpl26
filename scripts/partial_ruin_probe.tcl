# partial_ruin_probe.tcl — R&D probe (NOT wired into the agent).
#
# Question (2026-06-11): full-unplace ILS is size-gated off above 300k cells,
# leaving boom_soc/ispd16 recipe-only (+2.9/+14.7, our weakest designs). If
# unplacing ONLY the worst-paths' cells + incremental re-place gives a cycle at
# a fraction of full-ruin cost, ILS becomes viable on the large class.
#
# Measures per step: wall seconds, cells touched, WNS before/after, route
# status. Pure probe — writes out_dcp only if strictly improved + routed.
#
# Usage: set argv {in.dcp out.dcp [n_paths]}; source partial_ruin_probe.tcl
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
proc whs_now {} {
    set tp [get_timing_paths -quiet -hold -max_paths 1 -slack_lesser_than 999]
    if {[llength $tp] == 0} { return 99.0 }
    return [get_property SLACK [lindex $tp 0]]
}
proc fully_routed {} {
    set rs [report_route_status -return_string]
    set routable -1; set fully -2; set errs 0
    foreach line [split $rs "\n"] {
        if {[regexp {# of routable nets[^0-9]*([0-9]+)} $line -> v]} { set routable $v }
        if {[regexp {# of fully routed nets[^0-9]*([0-9]+)} $line -> v]} { set fully $v }
        if {[regexp {# of nets with routing errors[^0-9]*([0-9]+)} $line -> v]} { set errs $v }
    }
    return [expr {$routable == $fully && $errs == 0}]
}
proc tic {} { return [clock seconds] }
proc lap {t0 label} { puts "PROBE_T ${label}=[expr {[clock seconds] - $t0}]s" }

if {$argc < 2} { puts "PROBE_VERDICT=INVALID reason=usage"; exit 1 }
set in_dcp  [lindex $argv 0]
set out_dcp [lindex $argv 1]
set n_paths 50
if {$argc >= 3} { set n_paths [lindex $argv 2] }

if {[catch {
    set t [tic]; open_checkpoint $in_dcp; lap $t open
    set base [wns_now]
    puts "PROBE_BASE_WNS=$base (whs [whs_now])"

    # Worst-N setup paths -> their primitive cells (the ruin neighborhood).
    set t [tic]
    set paths [get_timing_paths -quiet -max_paths $n_paths -nworst 1 -setup]
    # Cells via path PINS (get_cells -of <path> is not supported); whitelist
    # movable fabric primitives (don't rip GTs/BUFGs/IO out of a routed design).
    set pcells [get_cells -quiet -of_objects [get_pins -quiet -of_objects $paths]]
    set cells [filter -quiet $pcells {IS_PRIMITIVE && \
               (PRIMITIVE_GROUP == "REGISTER" || PRIMITIVE_GROUP == "LUT" || \
                PRIMITIVE_GROUP == "CARRY" || PRIMITIVE_GROUP == "MUXF" || \
                PRIMITIVE_GROUP == "FLOP_LATCH" || PRIMITIVE_GROUP == "CLB")}]
    puts "PROBE_RUIN_CELLS=[llength $cells] (from $n_paths paths)"
    if {[llength $cells] == 0} { puts "PROBE_VERDICT=INVALID reason=no_cells"; exit 0 }
    unplace_cell $cells
    lap $t ruin

    set t [tic]; place_design; lap $t place
    set t [tic]; route_design; lap $t route
    set t [tic]; phys_opt_design -directive AggressiveExplore; lap $t phys_opt

    set new [wns_now]
    set routed [fully_routed]
    set whs [whs_now]
    puts "PROBE_NEW_WNS=$new routed=$routed whs=$whs"
    if {$routed && $new > $base && $whs >= -0.001} {
        write_checkpoint -force $out_dcp
        puts "PROBE_VERDICT=IMPROVED delta=[expr {$new - $base}]"
    } elseif {$routed} {
        puts "PROBE_VERDICT=NO_GAIN"
    } else {
        puts "PROBE_VERDICT=BROKEN_ROUTE"
    }
} err]} {
    puts "PROBE_VERDICT=INVALID reason=$err"
}
exit 0
