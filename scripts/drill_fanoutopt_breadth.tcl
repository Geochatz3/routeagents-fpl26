# DRILL D breadth (jun14) — does AggressiveFanoutOpt (the one fanout lever that
# helped logicnets, +0.029ns GT-confirmed) generalize? Apply ONE post-route
# phys_opt -directive AggressiveFanoutOpt pass to a ship-quality OPTIMIZED DCP,
# never-worse gated (routed + hold-clean + strictly beats input WNS to save).
# Usage: vivado -mode batch -source drill_fanoutopt_breadth.tcl -tclargs <opt_in.dcp> <out.dcp>

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
if {[catch {
    open_checkpoint $in_dcp
    set base [wns]; set bh [whs]; set br [routed]
    puts "FOPT_BASE wns=$base whs=$bh routed=$br"
    set t0 [clock seconds]
    phys_opt_design -directive AggressiveFanoutOpt
    if {![routed]} { route_design -directive Explore }
    set w [wns]; set h [whs]; set r [routed]
    puts "FOPT_AFTER wns=$w whs=$h routed=$r dt=[expr {[clock seconds]-$t0}]s"
    if {$r && $h >= 0 && $w > $base + 0.002} {
        write_checkpoint -force $out_dcp
        puts "FOPT_VERDICT=IMPROVED gain=[expr {$w-$base}]"
    } else {
        puts "FOPT_VERDICT=NEUTRAL_OR_WORSE delta=[expr {$w-$base}]"
    }
} e]} { puts "FOPT_ERR : $e"; catch { close_design } }
puts "FOPT_DONE"
