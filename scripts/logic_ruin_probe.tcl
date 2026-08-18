# logic_ruin_probe.tcl — R&D probe (NOT wired): NETLIST-level ruin.
#
# Hypothesis (2026-06-11): the logicnets leader gap (+124.7 vs our ~+97) needs
# structure no placement directive reaches. Placement-only ruin keeps the LUT
# network fixed; `opt_design -directive AddRemap` remaps LUT combinations
# (depth reduction) — then a fresh place/route/phys_opt exploits the new
# structure. Prior evidence (technique hunt) tested opt_design Explore as a
# one-shot recipe move (neutral/regressed); THIS is AddRemap as a keep-best
# ruin level. Writes out only on improved + routed + hold-clean.
#
# Usage: set argv {in.dcp out.dcp [opt_directive] [place_directive]}; source me
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
proc lap {t0 label} { puts "LRUIN_T ${label}=[expr {[clock seconds] - $t0}]s" }
proc logic_levels {} {
    # Worst-path logic levels — the structural metric AddRemap should move.
    set tp [get_timing_paths -quiet -max_paths 1 -slack_lesser_than 999]
    if {[llength $tp] == 0} { return -1 }
    return [get_property LOGIC_LEVELS [lindex $tp 0]]
}

if {$argc < 2} { puts "LRUIN_VERDICT=INVALID reason=usage"; exit 1 }
set in_dcp  [lindex $argv 0]
set out_dcp [lindex $argv 1]
set opt_dir "AddRemap"
if {$argc >= 3} { set opt_dir [lindex $argv 2] }
set place_dir "Explore"
if {$argc >= 4} { set place_dir [lindex $argv 3] }

if {[catch {
    set t [tic]; open_checkpoint $in_dcp; lap $t open
    set base [wns_now]
    puts "LRUIN_BASE_WNS=$base (whs [whs_now]) logic_levels=[logic_levels]"

    set t [tic]
    route_design -unroute
    place_design -unplace
    lap $t unbuild
    set t [tic]; opt_design -directive $opt_dir; lap $t opt
    puts "LRUIN_POST_OPT logic_levels=[logic_levels]"
    set t [tic]; place_design -directive $place_dir; lap $t place
    set t [tic]; route_design -directive Explore; lap $t route
    set t [tic]; phys_opt_design -directive AggressiveExplore; lap $t phys_opt

    set new [wns_now]
    set routed [fully_routed]
    set whs [whs_now]
    puts "LRUIN_NEW_WNS=$new routed=$routed whs=$whs logic_levels=[logic_levels]"
    if {$routed && $new > $base && $whs >= -0.001} {
        write_checkpoint -force $out_dcp
        puts "LRUIN_VERDICT=IMPROVED delta=[expr {$new - $base}]"
    } elseif {$routed} {
        puts "LRUIN_VERDICT=NO_GAIN delta=[expr {$new - $base}]"
    } else {
        puts "LRUIN_VERDICT=BROKEN_ROUTE"
    }
} err]} {
    puts "LRUIN_VERDICT=INVALID reason=$err"
}
exit 0
