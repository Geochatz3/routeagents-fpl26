# DRILL F — retiming breadth (jun15). Tests register-retiming levers NO prior
# drill touched (drill C only did `-retime -lut_opt`). `-interconnect_retime`
# (Vivado 2022.1+) is the modern replacement for the old high-fanout opts
# (UG904) and moves registers into interconnect to shorten the longest
# combinational path — the classic setup-ceiling breaker, and the leader's
# named hypothesis for v2's +32.5 ("structural retiming"). From a ship-quality
# OPTIMIZED routed DCP, try 3 variants independently, keep-best, never-worse
# (routed + hold-clean + strictly beats input WNS to save).
#   V1 phys_opt_design -interconnect_retime
#   V2 phys_opt_design -retime
#   V3 phys_opt_design -retime -interconnect_retime
# Retiming moves registers -> can unroute affected nets -> reroute before measure.
# Usage: vivado -mode batch -source drill_retime_breadth.tcl -tclargs <opt_in.dcp> <out.dcp>

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
set variants {{-interconnect_retime} {-retime} {-retime -interconnect_retime}}

open_checkpoint $in_dcp
set base [wns]; set bh [whs]; set br [routed]
puts "RETIME_BASE wns=$base whs=$bh routed=$br"
close_design

set best $base
set best_tag "baseline"
foreach opts $variants {
    if {[catch {
        open_checkpoint $in_dcp
        set t0 [clock seconds]
        if {[catch {eval phys_opt_design $opts} e]} { puts "RETIME_PHYSOPT_ERR opts={$opts} : $e"; close_design; continue }
        if {![routed]} { route_design -directive Explore }
        set w [wns]; set h [whs]; set r [routed]
        puts "RETIME var={$opts} wns=$w whs=$h routed=$r dt=[expr {[clock seconds]-$t0}]s base=$base"
        if {$r && $h >= 0 && $w > $best + 0.002} {
            set best $w; set best_tag $opts
            write_checkpoint -force $out_dcp
            puts "RETIME_SAVED var={$opts} wns=$w"
        }
        close_design
    } e]} { puts "RETIME_ERR opts={$opts} : $e"; catch { close_design } }
}
if {$best > $base + 0.002} {
    puts "RETIME_VERDICT=IMPROVED gain=[expr {$best-$base}] tag={$best_tag}"
} else {
    puts "RETIME_VERDICT=NEUTRAL_OR_WORSE base=$base best=$best"
}
puts "RETIME_DONE"
