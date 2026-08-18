# DRILL D v2 (jun14) — fix D2 + GT-verify D1.
#  (A) GT-verify the D1 (AggressiveFanoutOpt) saved DCP via report_timing_summary
#      on the scored clock — confirm the +0.029 ns is real, not a get_timing_paths
#      artifact.
#  (B) D2 corrected: force_replication_on_nets is PRE-ROUTE only. Flow =
#      open optimized -> route_design -unroute -> phys_opt -force_replication_on_nets
#      <top global high-fanout SIGNAL nets, selected correctly> -> route -> phys_opt.
#      Keep-best vs the input WNS; routed + hold-clean gated.
# Usage: vivado -mode batch -source drill_hifanout_repl_v2.tcl -tclargs <opt_in.dcp> <d1_saved.dcp> <out.dcp>

proc clk_obj {} {
    set c [get_clocks -quiet *fpl26contest*]
    if {[llength $c]} { return [lindex $c 0] }
    return [lindex [get_clocks] 0]
}
proc gt_wns {} {
    # ground-truth WNS on the scored clock via report_timing_summary
    set rpt [report_timing_summary -no_header -return_string]
    # parse the WNS(ns) value
    if {[regexp {WNS\(ns\)\s*\n?[^\n]*\n[-\s|]*\n\s*(-?\d+\.?\d*)} $rpt -> w]} { return $w }
    # fallback: worst setup path slack
    set p [get_timing_paths -quiet -setup -max_paths 1 -to [clk_obj]]
    if {[llength $p]==0} { return "NA" }
    return [get_property SLACK [lindex $p 0]]
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
set d1_dcp  [lindex $argv 1]
set out_dcp [lindex $argv 2]

# ---- (A) GT-verify input + D1 saved DCP ----
open_checkpoint $in_dcp
puts "GTVERIFY in : report_timing_summary WNS = [gt_wns]  (get_timing_paths [wns])  routed=[routed] whs=[whs]"
close_design
if {[file exists $d1_dcp]} {
    open_checkpoint $d1_dcp
    puts "GTVERIFY D1 : report_timing_summary WNS = [gt_wns]  (get_timing_paths [wns])  routed=[routed] whs=[whs]"
    close_design
} else {
    puts "GTVERIFY D1 : (no D1 dcp at $d1_dcp)"
}

# ---- (B) D2 corrected: pre-route force_replication ----
if {[catch {
    open_checkpoint $in_dcp
    set base [wns]
    puts "DRILLD2v2_BASE_WNS=$base"
    set c [clk_obj]
    # correct HF-net selection: global SIGNAL nets by physical fanout, top 50
    set cand [get_nets -quiet -hierarchical -filter {TYPE == SIGNAL}]
    set scored {}
    foreach n $cand {
        set fp [get_property -quiet FLAT_PIN_COUNT $n]
        if {$fp ne "" && $fp >= 100} { lappend scored [list $fp $n] }
    }
    set scored [lsort -integer -index 0 -decreasing $scored]
    set top {}
    set k 0
    foreach pair $scored { lappend top [lindex $pair 1]; incr k; if {$k >= 50} break }
    puts "DRILLD2v2 selected [llength $top] high-fanout SIGNAL nets (fanout>=100); top fanout = [lindex [lindex $scored 0] 0]"
    if {[llength $top] == 0} {
        puts "DRILLD2v2_SKIP no high-fanout signal nets >=100"
    } else {
        set t0 [clock seconds]
        route_design -unroute
        phys_opt_design -force_replication_on_nets $top
        route_design -directive Explore
        phys_opt_design -directive Explore
        set w [wns]; set h [whs]; set r [routed]
        puts "DRILLD2v2 force_repl nets=[llength $top] wns=$w whs=$h routed=$r dt=[expr {[clock seconds]-$t0}]s base=$base"
        if {$r && $h >= 0 && $w > $base + 0.002} {
            write_checkpoint -force $out_dcp
            puts "DRILLD2v2_SAVED wns=$w gain=[expr {$w-$base}]"
            puts "DRILLD2v2_VERDICT=IMPROVED gain=[expr {$w-$base}]"
        } else {
            puts "DRILLD2v2_VERDICT=NEUTRAL_OR_WORSE (w=$w base=$base routed=$r whs=$h)"
        }
    }
    close_design
} e]} { puts "DRILLD2v2_ERR : $e"; catch { close_design } }
puts "DRILLD2v2_DONE"
