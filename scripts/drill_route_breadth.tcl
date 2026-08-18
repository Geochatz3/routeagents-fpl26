# DRILL H — route-directive breadth (jun15). The router analog of the (closed)
# place-directive sweep: re-route the EXISTING optimized placement with route
# directives we never swept as a post-recipe polish, keep-best, never-worse.
# Cheap (route-only, no re-place). From a ship-quality OPTIMIZED DCP.
#   unroute -> route_design -directive X -> phys_opt Explore -> measure
# Variants: AggressiveExplore, NoTimingRelaxation, HigherDelayCost.
# Usage: vivado -mode batch -source drill_route_breadth.tcl -tclargs <opt_in.dcp> <out.dcp>
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
set dirs {AggressiveExplore NoTimingRelaxation HigherDelayCost}

open_checkpoint $in_dcp
set base [wns]; set bh [whs]
puts "ROUTE_BASE wns=$base whs=$bh routed=[routed]"
close_design
set best $base; set best_tag "baseline"
foreach d $dirs {
    if {[catch {
        open_checkpoint $in_dcp
        set t0 [clock seconds]
        route_design -unroute
        if {[catch {route_design -directive $d} e]} { puts "ROUTE_ERR dir=$d : $e"; close_design; continue }
        catch {phys_opt_design -directive Explore}
        set w [wns]; set h [whs]; set r [routed]
        puts "ROUTE dir=$d wns=$w whs=$h routed=$r dt=[expr {[clock seconds]-$t0}]s base=$base"
        if {$r && $h >= 0 && $w > $best + 0.002} {
            set best $w; set best_tag $d
            write_checkpoint -force $out_dcp
            puts "ROUTE_SAVED dir=$d wns=$w"
        }
        close_design
    } e]} { puts "ROUTE_OUTER_ERR dir=$d : $e"; catch { close_design } }
}
if {$best > $base + 0.002} {
    puts "ROUTE_VERDICT=IMPROVED gain=[expr {$best-$base}] tag=$best_tag"
} else {
    puts "ROUTE_VERDICT=NEUTRAL_OR_WORSE base=$base best=$best"
}
puts "ROUTE_DONE"
