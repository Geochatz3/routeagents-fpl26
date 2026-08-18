# route_ruin_probe.tcl — R&D probe (NOT wired into the agent).
#
# Question (2026-06-12): placement-based ILS combos fail on saturated,
# low-spread designs (spam: 3 cycles 0 accepts, dual-seed discarded). Is
# there routing-only headroom? Rip up ONLY the worst-paths' signal nets and
# let the router renegotiate them against the locked (good) placement —
# the route-scope analog of partial ruin. Variants:
#   V1: unroute worst-N paths' nets -> route_design        (timing-driven completion)
#   V2: unroute worst-N paths' nets -> route_design -delay -nets (least-delay per net)
# Each variant restarts from the pristine input checkpoint. Writes out_dcp
# only for the best strictly-improved + fully-routed + hold-clean result.
#
# Usage: vivado -mode batch -source route_ruin_probe.tcl \
#          -tclargs <in.dcp> <out.dcp> [n_paths_small] [n_paths_big]

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
set n_small 200
set n_big   1000
if {$argc >= 3} { set n_small [lindex $argv 2] }
if {$argc >= 4} { set n_big   [lindex $argv 3] }

# (label, n_paths, mode) — mode "complete" = plain route_design after unroute;
# mode "delay" = route_design -delay -nets on the ripped nets.
# Modes: complete = rip critical nets only, timing-driven completion;
# delay = rip critical, least-delay per net; neighborhood = rip critical
# nets PLUS every net touching the critical paths' cells (the met "highway"
# nets holding the contested resources), then timing-driven reroute of the
# freed region — the only variant that actually FREES resources.
set experiments [list \
    [list V1_complete_n$n_small $n_small complete] \
    [list V2_delay_n$n_small    $n_small delay] \
    [list V3_complete_n$n_big   $n_big   complete] \
    [list V4_neighborhood_n$n_small $n_small neighborhood] \
]

set best_gain 0.0
set base ""
foreach exp $experiments {
    lassign $exp label n mode
    if {[catch {
        set t0 [tic]
        open_checkpoint $in_dcp
        if {$base eq ""} {
            set base [wns_now]
            puts "PROBE_BASE_WNS=$base (whs [whs_now])"
        }
        # Worst-N setup paths -> their signal nets (skip clocks/power and
        # anything not currently routed).
        set paths [get_timing_paths -quiet -max_paths $n -nworst 1 -setup]
        if {$mode eq "neighborhood"} {
            set pcells [get_cells -quiet -of_objects [get_pins -quiet -of_objects $paths]]
            set pnets [get_nets -quiet -of_objects $pcells]
        } else {
            set pnets [get_nets -quiet -of_objects $paths]
        }
        set nets [filter -quiet $pnets {TYPE == "SIGNAL" && ROUTE_STATUS == "ROUTED"}]
        puts "PROBE_${label}_NETS=[llength $nets]"
        if {[llength $nets] == 0} { puts "PROBE_${label}_VERDICT=NO_NETS"; continue }
        route_design -unroute -nets $nets
        if {$mode eq "delay"} {
            route_design -delay -nets $nets
        } else {
            route_design
        }
        set w [wns_now]; set h [whs_now]; set fr [fully_routed]
        set dt [expr {[clock seconds] - $t0}]
        set gain [expr {$w - $base}]
        puts "PROBE_${label}_RESULT wns=$w whs=$h routed=$fr dt=${dt}s gain=$gain"
        if {$fr && $h >= 0 && $gain > 0.002 && $gain > $best_gain} {
            set best_gain $gain
            write_checkpoint -force $out_dcp
            puts "PROBE_${label}_SAVED=$out_dcp"
        }
    } err]} {
        puts "PROBE_${label}_ERROR=$err"
    }
}

if {$best_gain > 0.002} {
    puts "PROBE_VERDICT=IMPROVED best_gain=$best_gain"
} else {
    puts "PROBE_VERDICT=NEUTRAL_OR_WORSE best_gain=$best_gain"
}
