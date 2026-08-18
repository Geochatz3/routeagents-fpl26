# DRILL B — alternate place directives NOT in our ILS combo set (jun13).
# LASTMILE was found by trying a Vivado-native lever we didn't use. Same idea
# for placement: try place_design directives the router/combo set never
# exercises (ExtraPostPlacementOpt, WLDrivenBlockPlacement, ExtraNetDelay_low),
# each followed by route + phys_opt. Keep-best across directives. From RAW
# input. Emits DRILLB_* lines; writes out_dcp for the best routed+hold-clean.
# Usage: vivado -mode batch -source drill_altplace.tcl -tclargs <raw_in.dcp> <out.dcp>
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
set dirs {ExtraPostPlacementOpt WLDrivenBlockPlacement ExtraNetDelay_low}
set best_wns -1e9
foreach d $dirs {
    if {[catch {
        open_checkpoint $in_dcp
        set t0 [clock seconds]
        place_design -directive $d
        route_design -directive Explore
        phys_opt_design -directive AggressiveExplore
        set w [wns]; set h [whs]; set r [routed]
        puts "DRILLB dir=$d wns=$w whs=$h routed=$r dt=[expr {[clock seconds]-$t0}]s"
        if {$r && $h >= 0 && $w > $best_wns} {
            set best_wns $w
            write_checkpoint -force $out_dcp
            puts "DRILLB_SAVED dir=$d wns=$w"
        }
        close_design
    } e]} { puts "DRILLB_ERR dir=$d : $e"; catch { close_design } }
}
puts "DRILLB_BEST=$best_wns"
puts "DRILLB_VERDICT=DONE best_wns=$best_wns"
