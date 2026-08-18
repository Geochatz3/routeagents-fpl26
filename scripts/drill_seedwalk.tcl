# DRILL E — long keep-best placement-perturbation walk (jun14).
# Tests the #1 unexploited lever from TECHNIQUE_HUNT_CONCLUSION: "10s-100s of
# place Explore seeds". The 6-seed seedvar.tcl variance (spread 0.31ns) came
# from place_design -unplace; place_design in ONE session (placer not reset ->
# different basin each call) == the ILS full-unplace perturbation. So the lever
# is simply running that keep-best walk for MANY more cycles than production's
# ~3-cycle leftover budget. Measures whether best-of-N post-route WNS on
# logicnets approaches the leader's ~ -0.39 (our best-of-6 was -0.592 post-PLACE).
# Full post-route measure each cycle (route+phys_opt). Keep-best, never-worse,
# routed + hold-clean gated. Writes out_dcp = best so far (survives wall kill).
# Usage: vivado -mode batch -source drill_seedwalk.tcl -tclargs <in.dcp> <out.dcp> <N> <wall_s>

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

set in_dcp  [lindex $argv 0]
set out_dcp [lindex $argv 1]
set N       [lindex $argv 2]
set wall    [lindex $argv 3]
# placement directives to rotate for diversity (all full re-place via -unplace)
set dirs {Explore AltSpreadLogic_high SSI_SpreadLogic_high ExtraNetDelay_high EarlyBlockPlacement Default}
set t_start [clock seconds]

open_checkpoint $in_dcp
set best [gt_wns]; set br [routed]; set bh [whs]
puts "SEEDWALK base WNS=$best routed=$br whs=$bh"
if {$br && $bh >= 0} { write_checkpoint -force $out_dcp }
set accepts 0
for {set i 1} {$i <= $N} {incr i} {
    if {[expr {[clock seconds]-$t_start}] > $wall} { puts "SEEDWALK_WALL hit at cycle $i"; break }
    set pd [lindex $dirs [expr {($i-1) % [llength $dirs]}]]
    set t0 [clock seconds]
    if {[catch {
        place_design -unplace
        if {$pd eq "Default"} { place_design } else { place_design -directive $pd }
        route_design -directive Explore
        phys_opt_design -directive AggressiveExplore
        set w [gt_wns]; set h [whs]; set r [routed]
        puts "SEEDWALK $i place=$pd WNS=$w whs=$h routed=$r best=$best dt=[expr {[clock seconds]-$t0}]s"
        if {$r && $h >= 0 && $w ne "NA" && $w > $best + 0.002} {
            set best $w; incr accepts
            write_checkpoint -force $out_dcp
            puts "SEEDWALK_ACCEPT $i WNS=$best place=$pd"
        }
    } e]} { puts "SEEDWALK_ERR $i : $e" }
}
puts "SEEDWALK_DONE best=$best accepts=$accepts cycles_run=[expr {$i-1}] elapsed=[expr {[clock seconds]-$t_start}]s"
