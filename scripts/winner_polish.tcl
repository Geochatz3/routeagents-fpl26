# winner_polish.tcl — never-worse phys_opt polish of the multi-restart winner.
#
# Runs in `vivado -mode batch` on wall the wrapper would otherwise strand below
# its attempt floor. Opens the shipped DCP, runs one
# post-route phys_opt pass, and writes the polished DCP ONLY if WNS strictly
# improved AND the design is still fully routed. The wrapper replaces the scored
# file only on the IMPROVED verdict, so the floor is the unpolished winner.
#
# Usage: vivado -mode batch -source winner_polish.tcl -tclargs <in.dcp> <out.dcp> [directive]

# Both timing probes return the empty string when the query came back
# empty. Empty is NOT a slack value: returning 0.0 here would beat any
# negative baseline and publish a fabricated IMPROVED verdict, on which the
# wrapper replaces the SCORED artifact. Unmeasurable means stop, not improved.
proc wns_now {} {
    # Mirror the agent's target-clock-aware WNS: prefer the contest clock.
    set clk [get_clocks -quiet *fpl26contest*]
    if {[llength $clk] > 0} {
        set tp [get_timing_paths -quiet -max_paths 1 -setup -to [lindex $clk 0]]
    } else {
        set tp [get_timing_paths -quiet -max_paths 1 -slack_lesser_than 999]
    }
    if {[llength $tp] == 0} { return "" }
    return [get_property SLACK [lindex $tp 0]]
}

proc whs_now {} {
    set tp [get_timing_paths -quiet -hold -max_paths 1 -slack_lesser_than 999]
    if {[llength $tp] == 0} { return "" }
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

if {$argc < 2} { puts "POLISH_VERDICT=INVALID reason=usage"; exit 1 }
set in_dcp  [lindex $argv 0]
set out_dcp [lindex $argv 1]
set directive "AggressiveExplore"
if {$argc >= 3} { set directive [lindex $argv 2] }
# Optional pass/time budget: ladder iterates phys_opt WHILE improving — the
# standard closure move for near-met designs (each pass converges; keep-best
# preserved by only writing on strict improvement vs the ORIGINAL baseline).
set max_passes 4
if {$argc >= 4} { set max_passes [lindex $argv 3] }
set budget_s 0
if {$argc >= 5} { set budget_s [lindex $argv 4] }
set t0 [clock seconds]

if {[catch {
    open_checkpoint $in_dcp
    set base [wns_now]
    set base_whs [whs_now]
    # No readable baseline means no comparison is possible: refuse rather
    # than polish against an assumed number.
    if {$base eq ""} { error "no_baseline_setup_timing" }
    if {$base_whs eq ""} { error "no_baseline_hold_timing" }
    puts "POLISH_BASE_WNS=$base (whs $base_whs)"
    set best $base
    set wrote 0
    for {set pass 1} {$pass <= $max_passes} {incr pass} {
        if {$budget_s > 0 && [expr {[clock seconds] - $t0}] >= $budget_s} {
            puts "POLISH_PASS=$pass skipped=budget"
            break
        }
        phys_opt_design -directive $directive
        set new [wns_now]
        if {$new eq ""} {
            puts "POLISH_PASS=$pass wns=unmeasurable — stopping ladder"
            break
        }
        puts "POLISH_PASS=$pass wns=$new (best $best)"
        if {![fully_routed]} {
            # A pass broke routing: anything already written is still valid
            # (written only after a fully-routed check); stop here.
            puts "POLISH_PASS=$pass not_fully_routed — stopping ladder"
            break
        }
        if {$new > $best} {
            # Hold-safety: the validator gates hold_passed — never persist a
            # pass that broke hold (write only if WHS clean or not-worse).
            set whs [whs_now]
            if {$whs eq ""} {
                puts "POLISH_PASS=$pass hold_unmeasurable — not persisted"
                break
            }
            if {$whs < 0 && $whs < $base_whs} {
                puts "POLISH_PASS=$pass hold_dirty whs=$whs — not persisted"
                break
            }
            set best $new
            write_checkpoint -force $out_dcp
            set wrote 1
        } else {
            break
        }
    }
    puts "POLISH_NEW_WNS=$best"
    if {$wrote && $best > $base} {
        puts "POLISH_VERDICT=IMPROVED delta=[expr {$best - $base}]"
    } else {
        puts "POLISH_VERDICT=NO_GAIN"
    }
} err]} {
    puts "POLISH_VERDICT=INVALID reason=$err"
}
exit 0
