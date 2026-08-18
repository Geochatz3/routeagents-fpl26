# DRILL A — placement-phase INTRA-clock overconstraint (jun13).
# Hypothesis: UG949 recommends set_clock_uncertainty -setup tightening DURING
# place/phys_opt (reset before route) — we only ever tested it POST-ROUTE
# (neutral). Tightening during placement should make the placer + phys_opt
# work HARDER on the scored clock's paths, yielding a better placement that
# survives the reset. Sweep the extra uncertainty.
# Starts from RAW input (full re-place). Writes out_dcp only if the best
# variant strictly beats a plain (no-overconstraint) re-place baseline AND is
# fully routed + hold-clean. Emits DRILLA_* lines.
# Usage: vivado -mode batch -source drill_place_overconstrain.tcl \
#          -tclargs <raw_in.dcp> <out.dcp>
#
# ⚠️ HAZARD (added jul29, learned the hard way). This script calls place_design
# -directive WITHOUT a preceding `route_design -unroute; place_design -unplace`.
# The organizer's benchmark DCPs are FULLY PLACED AND ROUTED, so against those inputs
# place_design is a NO-OP and Vivado says so:
#     INFO: [Place 30-281] No place-able instance is found; ... all instances are placed
# Every directive then measures only whatever route/phys_opt follows, and the branches
# come out bit-identical — which reads as "the directive does nothing" when in truth the
# directive never ran. A jul29 sweep hit exactly this: four different place directives all
# returned WNS -12.555 / WHS 0.006.
# If you run this on a placed input, add the unroute+unplace pair first. If you run it on a
# genuinely unplaced netlist, it is fine as written.

proc clk_obj {} {
    set c [get_clocks -quiet *fpl26contest*]
    if {[llength $c]} { return [lindex $c 0] }
    return [lindex [get_clocks] 0]
}
proc wns {} {
    set c [clk_obj]
    set p [get_timing_paths -quiet -setup -max_paths 1 -to $c]
    if {[llength $p]==0} { set p [get_timing_paths -quiet -setup -max_paths 1] }
    if {[llength $p]==0} { return 0.0 }
    return [get_property SLACK [lindex $p 0]]
}
proc whs {} {
    set p [get_timing_paths -quiet -hold -max_paths 1 -slack_lesser_than 999]
    if {[llength $p]==0} { return 99.0 }
    return [get_property SLACK [lindex $p 0]]
}
proc routed {} {
    set rs [report_route_status -return_string]
    return [expr {[string match "*fully routed*" $rs]}]
}

set in_dcp  [lindex $argv 0]
set out_dcp [lindex $argv 1]

# variant 0 = plain (baseline), then overconstrain sweeps
set variants {0.0 0.15 0.30}
set best_wns -1e9
set base_wns ""
foreach u $variants {
    if {[catch {
        open_checkpoint $in_dcp
        set c [clk_obj]
        set t0 [clock seconds]
        if {$u > 0} { set_clock_uncertainty -setup $u $c }
        place_design -directive Explore
        phys_opt_design -directive AggressiveExplore
        if {$u > 0} { set_clock_uncertainty -setup 0 $c }
        route_design -directive Explore
        phys_opt_design -directive Explore
        set w [wns]; set h [whs]; set r [routed]
        set dt [expr {[clock seconds]-$t0}]
        if {$u == 0.0} { set base_wns $w }
        puts "DRILLA u=$u wns=$w whs=$h routed=$r dt=${dt}s"
        if {$r && $h >= 0 && $w > $best_wns} {
            set best_wns $w
            if {$base_wns ne "" && $w > $base_wns + 0.002} {
                write_checkpoint -force $out_dcp
                puts "DRILLA_SAVED u=$u wns=$w (beats plain $base_wns)"
            }
        }
        close_design
    } e]} { puts "DRILLA_ERR u=$u : $e"; catch { close_design } }
}
puts "DRILLA_BASE=$base_wns DRILLA_BEST=$best_wns"
if {$base_wns ne "" && $best_wns > $base_wns + 0.002} {
    puts "DRILLA_VERDICT=IMPROVED gain=[expr {$best_wns-$base_wns}]"
} else {
    puts "DRILLA_VERDICT=NEUTRAL_OR_WORSE"
}
