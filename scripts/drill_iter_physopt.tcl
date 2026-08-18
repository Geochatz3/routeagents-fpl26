# DRILL C — iterative post-route phys_opt-until-dry (jun13).
# UG904: "use multiple consecutive runs of physical optimization to gradually
# reduce failing paths." We only ever do SINGLE phys_opt passes. Drill: from a
# ship-quality OPTIMIZED DCP, loop phys_opt (-retime -lut_opt, then Aggressive
# /Alternate directives) until a pass yields no WNS gain. keep-best each pass;
# never-worse. Emits DRILLC_* lines; writes out_dcp if it beats the input.
# Usage: vivado -mode batch -source drill_iter_physopt.tcl -tclargs <opt_in.dcp> <out.dcp>

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
    set base [wns]
    puts "DRILLC_BASE_WNS=$base (whs [whs])"
    set best $base
    set passes {{-retime -lut_opt} {-directive AggressiveExplore} {-directive AlternateReplication} {-directive Explore}}
    set i 0
    foreach opts $passes {
        incr i
        set t0 [clock seconds]
        if {[catch { eval phys_opt_design $opts } e]} { puts "DRILLC_PASS$i ERR=$e"; continue }
        set w [wns]; set h [whs]; set r [routed]
        puts "DRILLC_PASS$i opts={$opts} wns=$w whs=$h routed=$r dt=[expr {[clock seconds]-$t0}]s"
        if {$r && $h >= 0 && $w > $best + 0.002} {
            set best $w
            write_checkpoint -force $out_dcp
            puts "DRILLC_SAVED pass=$i wns=$w"
        } else {
            # no gain this pass -> stop (until-dry)
            if {$w <= $best + 0.002} { puts "DRILLC_DRY at pass $i"; break }
        }
    }
    puts "DRILLC_BEST=$best"
    if {$best > $base + 0.002} { puts "DRILLC_VERDICT=IMPROVED gain=[expr {$best-$base}]" } else { puts "DRILLC_VERDICT=NEUTRAL_OR_WORSE" }
} e]} { puts "DRILLC_FATAL=$e" }
