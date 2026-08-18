# Constraint-ratcheting probe (research, 2026-06-12).
#
# Hypothesis: Vivado's phys_opt/router are satisficing — they idle once WNS>=0
# against the active constraint, so on TIMING-MET (corundum-class) and
# saturated designs the tools have been ALLOWED to stop trying. UG949
# "Overconstraining the Design": tighten with set_clock_uncertainty -setup
# (additive, waveforms unchanged, <=0.5ns), optimize, then reset to 0 and
# measure against the real constraint.
#
# Usage: vivado -mode batch -source ratchet_probe.tcl -tclargs \
#            <input.dcp> <output.dcp> [<extra_margin_ns>] [<passes>]
#
# Emits RATCHET_* key=value lines; verdict IMPROVED iff setup WNS strictly
# better AND hold/pulse-width not degraded AND still fully routed.

if {$argc < 2} { puts "RATCHET_ERROR=usage"; exit 1 }
set in_dcp  [lindex $argv 0]
set out_dcp [lindex $argv 1]
set extra_margin 0.150
if {$argc >= 3} { set extra_margin [lindex $argv 2] }
set passes 2
if {$argc >= 4} { set passes [lindex $argv 3] }

open_checkpoint $in_dcp

# Work with clock NAMES throughout and re-fetch objects with an exact-match
# -filter. Two Vivado Tcl traps otherwise: (a) `get_clocks <name>` globs, so
# names with [N] indexes raise Common 17-161; (b) `foreach c [get_clocks]`
# shimmers first-class objects into strings, so a saved $c stops being usable
# as an object later.
proc clk_obj {name} {
    return [get_clocks -quiet -filter "NAME == \"$name\""]
}
proc wns_of {name} {
    set p [get_timing_paths -setup -max_paths 1 -to [clk_obj $name] -quiet]
    if {[llength $p] == 0} { return "" }
    return [get_property SLACK [lindex $p 0]]
}

set clkname clk_fpl26contest
if {[llength [clk_obj $clkname]] == 0} {
    # OOD designs may carry a different primary clock — pick the clock whose
    # OWN path group has the worst setup WNS (a global worst-path query can
    # land on an unrelated slow clock and mis-aim the whole probe).
    set best ""
    set bestwns 1e9
    foreach cn [get_property NAME [get_clocks]] {
        set s [wns_of $cn]
        if {$s eq ""} { continue }
        puts "RATCHET_CLOCK_SCAN=$cn wns=$s"
        if {$s < $bestwns} { set bestwns $s; set best $cn }
    }
    set clkname $best
    puts "RATCHET_NOTE=clk_fpl26contest not found; using worst-WNS clock $clkname"
}
proc whs_all {} {
    set p [get_timing_paths -hold -max_paths 1 -quiet]
    if {[llength $p] == 0} { return 999 }
    return [get_property SLACK [lindex $p 0]]
}

set base_wns [wns_of $clkname]
if {$base_wns eq ""} { puts "RATCHET_ERROR=no_paths_to_clock_$clkname"; exit 1 }
set base_whs [whs_all]
puts "RATCHET_BASE_WNS=$base_wns"
puts "RATCHET_BASE_WHS=$base_whs"

# Uncertainty: enough that current paths FAIL by ~extra_margin, capped at
# UG949's 0.5ns guidance. If already failing (WNS<0) just add the margin.
set u [expr {($base_wns > 0 ? $base_wns : 0) + $extra_margin}]
if {$u > 0.5} { set u 0.5 }
puts "RATCHET_UNCERTAINTY=$u"
set_clock_uncertainty -setup $u [clk_obj $clkname]

for {set i 1} {$i <= $passes} {incr i} {
    set dir [expr {$i == 1 ? "AggressiveExplore" : "Explore"}]
    puts "RATCHET_PASS=$i directive=$dir"
    if {[catch {phys_opt_design -directive $dir} err]} {
        puts "RATCHET_PASS_ERROR=$err"
        break
    }
    puts "RATCHET_PASS${i}_WNS_OVERCONSTRAINED=[wns_of $clkname]"
}

# Remove the ratchet and measure reality.
set_clock_uncertainty -setup 0 [clk_obj $clkname]

set final_wns [wns_of $clkname]
set final_whs [whs_all]
# 2025.1 quirk (jun05 finding): no "unrouted nets" line when clean — gate on
# the positive "fully routed" statement only.
set route_ok 0
if {![catch {report_route_status -return_string} rrs]} {
    if {[string match "*fully routed*" $rrs]} { set route_ok 1 }
}
puts "RATCHET_FINAL_WNS=$final_wns"
puts "RATCHET_FINAL_WHS=$final_whs"
puts "RATCHET_ROUTE_OK=$route_ok"

set delta [expr {$final_wns - $base_wns}]
puts "RATCHET_DELTA=$delta"
if {$delta > 0.005 && $final_whs >= 0 && $route_ok} {
    write_checkpoint -force $out_dcp
    puts "RATCHET_VERDICT=IMPROVED delta=$delta"
} elseif {$delta >= -0.005} {
    puts "RATCHET_VERDICT=NEUTRAL delta=$delta"
} else {
    puts "RATCHET_VERDICT=REGRESSED delta=$delta"
}
