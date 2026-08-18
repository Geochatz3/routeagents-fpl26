# DRILL D — high-fanout net replication levers (jun14).
# logicnets_jscl (sparse quantized NN) and vexriscv_re-place_v2 are
# high-fanout-dominated. UG949/UG904 prescribe replication levers we have
# NEVER tried as targeted moves (drill C only used AlternateReplication /
# AggressiveExplore / retime / lut_opt):
#   D1  phys_opt_design -directive AggressiveFanoutOpt   (untested directive)
#   D2  phys_opt_design -force_replication_on_nets <top critical HF nets>
#       then re-route + light phys_opt  (the canonical "HF net critical after
#       routing" surgical move)
# From a ship-quality OPTIMIZED (routed) DCP. Each variant is INDEPENDENT
# (re-open from input). Keep-best, never-worse: out_dcp written only if a
# variant strictly beats the input WNS AND is fully routed + hold-clean.
# Emits DRILLD_* lines.
# Usage: vivado -mode batch -source drill_hifanout_repl.tcl -tclargs <opt_in.dcp> <out.dcp>

proc clk_obj {} {
    set c [get_clocks -quiet *fpl26contest*]
    if {[llength $c]} { return [lindex $c 0] }
    return [lindex [get_clocks] 0]
}
proc wns {} {
    set c [clk_obj]
    set p [get_timing_paths -quiet -setup -max_paths 1 -to $c]
    if {[llength $p]==0} { set p [get_timing_paths -quiet -setup -max_paths 1] }
    if {[llength $p]==0} { return 0.0 }
    return [get_property SLACK [lindex $p 0]]
}
proc whs {} {
    set p [get_timing_paths -quiet -hold -max_paths 1 -slack_lesser_than 999]
    if {[llength $p]==0} { return 99.0 }
    return [get_property SLACK [lindex $p 0]]
}
proc routed {} {
    return [expr {[string match "*fully routed*" [report_route_status -return_string]]}]
}
# Collect high-fanout nets along the worst N setup paths to the scored clock.
proc crit_hf_nets {clk topN fanoutMin} {
    set paths [get_timing_paths -quiet -setup -max_paths $topN -to $clk]
    if {[llength $paths]==0} { set paths [get_timing_paths -quiet -setup -max_paths $topN] }
    set pins {}
    foreach p $paths {
        foreach pin [get_pins -quiet -of_objects $p] { lappend pins $pin }
    }
    if {[llength $pins]==0} { return {} }
    set nets [lsort -unique [get_nets -quiet -of_objects $pins]]
    set hf {}
    foreach n $nets {
        set fp [get_property -quiet FLAT_PIN_COUNT $n]
        if {$fp ne "" && $fp >= $fanoutMin} { lappend hf [list $fp $n] }
    }
    # sort by fanout desc, return the net names (cap at 40 to keep phys_opt tractable)
    set hf [lsort -integer -index 0 -decreasing $hf]
    set out {}
    set k 0
    foreach pair $hf {
        lappend out [lindex $pair 1]
        incr k
        if {$k >= 40} { break }
    }
    return $out
}

set in_dcp  [lindex $argv 0]
set out_dcp [lindex $argv 1]

open_checkpoint $in_dcp
set base [wns]
set base_h [whs]
set base_r [routed]
puts "DRILLD_BASE_WNS=$base whs=$base_h routed=$base_r"
close_design

set best $base
set best_tag "baseline"

# ---- D1: AggressiveFanoutOpt directive ----
if {[catch {
    open_checkpoint $in_dcp
    set t0 [clock seconds]
    phys_opt_design -directive AggressiveFanoutOpt
    if {![routed]} { route_design -directive Explore }
    set w [wns]; set h [whs]; set r [routed]
    puts "DRILLD1 dir=AggressiveFanoutOpt wns=$w whs=$h routed=$r dt=[expr {[clock seconds]-$t0}]s"
    if {$r && $h >= 0 && $w > $best + 0.002} {
        set best $w; set best_tag "D1_AggressiveFanoutOpt"
        write_checkpoint -force $out_dcp
        puts "DRILLD_SAVED tag=$best_tag wns=$w"
    }
    close_design
} e]} { puts "DRILLD1_ERR : $e"; catch { close_design } }

# ---- D2: targeted force_replication_on_nets ----
if {[catch {
    open_checkpoint $in_dcp
    set c [clk_obj]
    set hf [crit_hf_nets $c 50 30]
    puts "DRILLD2 found [llength $hf] critical high-fanout nets (fanout>=30 on worst 50 paths)"
    if {[llength $hf] == 0} {
        puts "DRILLD2_SKIP no critical high-fanout nets"
    } else {
        set t0 [clock seconds]
        phys_opt_design -force_replication_on_nets [get_nets $hf]
        if {![routed]} { route_design -directive Explore }
        phys_opt_design -directive Explore
        set w [wns]; set h [whs]; set r [routed]
        puts "DRILLD2 force_repl nets=[llength $hf] wns=$w whs=$h routed=$r dt=[expr {[clock seconds]-$t0}]s"
        if {$r && $h >= 0 && $w > $best + 0.002} {
            set best $w; set best_tag "D2_force_replication"
            write_checkpoint -force $out_dcp
            puts "DRILLD_SAVED tag=$best_tag wns=$w"
        }
    }
    close_design
} e]} { puts "DRILLD2_ERR : $e"; catch { close_design } }

puts "DRILLD_BASE=$base DRILLD_BEST=$best tag=$best_tag"
if {$best > $base + 0.002} {
    puts "DRILLD_VERDICT=IMPROVED gain=[expr {$best-$base}] tag=$best_tag"
} else {
    puts "DRILLD_VERDICT=NEUTRAL_OR_WORSE"
}
