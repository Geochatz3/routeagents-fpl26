# incr_warmstart_probe.tcl — R&D probe (NOT wired into the agent).
#
# Question (2026-06-12): can Vivado's OWN incremental machinery act as a
# targeted ruin-recreate operator from a DCP-only state? Unlike manual
# route-ruin (refuted ad9fcec: router re-solve has less freedom), these
# re-enter the PLACER — the stage actually stuck in a local optimum — with
# the best DCP as reference:
#   W1 "incremental": unplace -> read_checkpoint -incremental SELF
#      -directive TimingClosure -> place_design (incremental, rips failing
#      paths, targets WNS 0) -> route_design (reuses reference routing).
#   W2 "lastmile": UG906 IDR Stage-3 recipe directly on the routed design:
#      phys_opt(clock/retime/lut) -> place_design -directive LastMile ->
#      phys_opt Explore -> route_design -directive Explore -> phys_opt Explore.
# Each variant restarts from the pristine input. out_dcp written only for
# the best strictly-improved + fully-routed + hold-clean result.
#
# Usage: vivado -mode batch -source incr_warmstart_probe.tcl \
#          -tclargs <in.dcp> <out.dcp> [variants=W1,W2]

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

if {$argc < 2} { puts "PROBE_VERDICT=INVALID reason=usage"; exit 1 }
set in_dcp  [lindex $argv 0]
set out_dcp [lindex $argv 1]
set variants {W1 W2}
if {$argc >= 3} { set variants [split [lindex $argv 2] ","] }

set best_gain 0.0
set base ""
foreach v $variants {
    if {[catch {
        set t0 [tic]
        open_checkpoint $in_dcp
        if {$base eq ""} {
            set base [wns_now]
            puts "PROBE_BASE_WNS=$base (whs [whs_now])"
        }
        if {$v eq "W1"} {
            place_design -unplace
            read_checkpoint -incremental $in_dcp -directive TimingClosure
            place_design
            route_design
        } else {
            # UG906 last-mile sequence (entry criteria ignored on purpose —
            # we want the raw mechanism, not the packaged gate).
            phys_opt_design -clock_opt -retime -lut_opt
            place_design -directive LastMile
            phys_opt_design -directive Explore
            route_design -directive Explore
            phys_opt_design -directive Explore
        }
        set w [wns_now]; set h [whs_now]; set fr [fully_routed]
        set dt [expr {[clock seconds] - $t0}]
        set gain [expr {$w - $base}]
        puts "PROBE_${v}_RESULT wns=$w whs=$h routed=$fr dt=${dt}s gain=$gain"
        if {$fr && $h >= 0 && $gain > 0.002 && $gain > $best_gain} {
            set best_gain $gain
            write_checkpoint -force $out_dcp
            puts "PROBE_${v}_SAVED=$out_dcp"
        }
    } err]} {
        puts "PROBE_${v}_ERROR=$err"
    }
}
if {$best_gain > 0.002} {
    puts "PROBE_VERDICT=IMPROVED best_gain=$best_gain"
} else {
    puts "PROBE_VERDICT=NEUTRAL_OR_WORSE best_gain=$best_gain"
}
