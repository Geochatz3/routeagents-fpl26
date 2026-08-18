# DRILL G — hold-fix recovery on the fanout-polished ispd16 (jun15).
# The fanout polish gets ispd16 −7.752 -> −6.770 (+0.982ns ~+12.8MHz) but erodes
# hold to whs~0.000, so the strict-hold gate (and cheap-gate) currently SKIP it.
# Hypothesis: ispd16's hold-critical paths (short/fast) are DIFFERENT from its
# setup-critical paths (the −6.77 WNS), so phys_opt -hold_fix (LUT1 insertion on
# hold paths with positive setup slack, UG949) can restore whs >= +0.010 WITHOUT
# hurting the setup gain -> the +12.8MHz becomes shippable.
# Flow (from ispd16 OPTIMIZED): base -> AggressiveFanoutOpt -> reroute ->
#   recovery A: phys_opt -hold_fix -> reroute -> measure
#   recovery B (independent re-open of fanout state not kept; B re-derives): full
#     route_design (router prioritizes hold) -> measure
# Reports WNS+WHS at each stage. SUCCESS = a recovery with WNS > base+0.002 AND
# whs >= 0.010 AND routed.
# Usage: vivado -mode batch -source drill_holdfix_recovery.tcl -tclargs <opt_in.dcp> <out.dcp>

proc clk_obj {} {
    set c [get_clocks -quiet *fpl26contest*]
    if {[llength $c]} { return [lindex $c 0] }
    return [lindex [get_clocks] 0]
}
proc wns {} {
    set p [get_timing_paths -quiet -setup -max_paths 1 -to [clk_obj]]
    if {[llength $p]==0} { set p [get_timing_paths -quiet -setup -max_paths 1] }
    if {[llength $p]==0} { return 0.0 }
    return [get_property SLACK [lindex $p 0]]
}
proc whs {} {
    set p [get_timing_paths -quiet -hold -max_paths 1 -slack_lesser_than 999]
    if {[llength $p]==0} { return 99.0 }
    return [get_property SLACK [lindex $p 0]]
}
proc routed {} { return [expr {[string match "*fully routed*" [report_route_status -return_string]]}] }

set in_dcp  [lindex $argv 0]
set out_dcp [lindex $argv 1]

open_checkpoint $in_dcp
set base [wns]; set bh [whs]
puts "HOLDFIX_BASE wns=$base whs=$bh routed=[routed]"

# fanout polish -> reroute if needed
set t0 [clock seconds]
phys_opt_design -directive AggressiveFanoutOpt
if {![routed]} { route_design -directive Explore }
set wf [wns]; set hf [whs]
puts "HOLDFIX_FANOUT wns=$wf whs=$hf routed=[routed] dt=[expr {[clock seconds]-$t0}]s"

# recovery A: phys_opt -hold_fix (LUT1 insertion on hold paths) -> reroute
set t0 [clock seconds]
if {[catch {phys_opt_design -hold_fix} e]} { puts "HOLDFIX_A_ERR : $e" } else {
    if {![routed]} { route_design -directive Explore }
    set wa [wns]; set ha [whs]
    puts "HOLDFIX_A wns=$wa whs=$ha routed=[routed] dt=[expr {[clock seconds]-$t0}]s"
    if {[routed] && $wa > $base + 0.002 && $ha >= 0.010} {
        write_checkpoint -force $out_dcp
        puts "HOLDFIX_A_SAVED wns=$wa whs=$ha (setup gain preserved + hold recovered)"
    }
}
# recovery B: aggressive hold fix (more paths) as a fallback if A under-recovers
set t0 [clock seconds]
if {[catch {phys_opt_design -aggressive_hold_fix} e]} { puts "HOLDFIX_B_ERR : $e" } else {
    if {![routed]} { route_design -directive Explore }
    set wb [wns]; set hb [whs]
    puts "HOLDFIX_B(aggr) wns=$wb whs=$hb routed=[routed] dt=[expr {[clock seconds]-$t0}]s"
    if {[routed] && $wb > $base + 0.002 && $hb >= 0.010} {
        write_checkpoint -force $out_dcp
        puts "HOLDFIX_B_SAVED wns=$wb whs=$hb"
    }
}
puts "HOLDFIX_DONE base_wns=$base"
