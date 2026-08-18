# GT-verify saved AggressiveFanoutOpt DCPs vs their inputs (jun14).
# Open-only (no re-place); report scored-clock WNS via report_timing_summary,
# route status, hold slack. Usage: -tclargs <label> <dcp>
proc clk_obj {} {
    set c [get_clocks -quiet *fpl26contest*]
    if {[llength $c]} { return [lindex $c 0] }
    return [lindex [get_clocks] 0]
}
proc gt_wns {} {
    set rpt [report_timing_summary -no_header -return_string]
    if {[regexp {WNS\(ns\)[^\n]*\n[-\s|]*\n\s*(-?\d+\.?\d*)} $rpt -> w]} { return $w }
    set p [get_timing_paths -quiet -setup -max_paths 1 -to [clk_obj]]
    if {[llength $p]==0} { return "NA" }
    return [get_property SLACK [lindex $p 0]]
}
proc whs {} {
    set p [get_timing_paths -quiet -hold -max_paths 1 -slack_lesser_than 999]
    if {[llength $p]==0} { return 99.0 }
    return [get_property SLACK [lindex $p 0]]
}
proc routed {} { return [expr {[string match "*fully routed*" [report_route_status -return_string]]}] }
set label [lindex $argv 0]
set dcp   [lindex $argv 1]
open_checkpoint $dcp
puts "GTCHK $label : period=[get_property PERIOD [clk_obj]] WNS=[gt_wns] whs=[whs] routed=[routed]"
close_design
