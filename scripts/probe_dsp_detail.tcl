# DSP absorption DETAIL probe (jul04): on mini-isp, characterize the absorbable
# pattern precisely: direction (FF->DSP input vs DSP->FF output), FF primitive
# types (FDRE sync = absorbable; FDCE async = NOT, DSP regs are sync-only),
# distinct FF/DSP pairs covering the worst paths, and current *REG settings.
# Read-only. Also: BRAM sibling probe (DO*_REG=0 with FF on DOUT).
proc clk_obj {} {
    set c [get_clocks -quiet *fpl26contest*]
    if {[llength $c]} { return [lindex $c 0] }
    return [lindex [get_clocks] 0]
}
open_checkpoint [lindex $argv 0]
set paths [get_timing_paths -quiet -setup -max_paths 200 -nworst 1 -to [clk_obj]]
array set dir_count {ff2dsp 0 dsp2ff 0 other 0}
array set fftype {}
set pairs {}
foreach p $paths {
    set scell [get_cells -quiet -of_objects [get_pins -quiet [get_property STARTPOINT_PIN $p]]]
    set ecell [get_cells -quiet -of_objects [get_pins -quiet [get_property ENDPOINT_PIN $p]]]
    set sref [get_property -quiet REF_NAME $scell]; set eref [get_property -quiet REF_NAME $ecell]
    if {[string match "FD*" $sref] && [string match "DSP*" $eref]} {
        incr dir_count(ff2dsp); incr fftype($sref)
        lappend pairs [list $scell $ecell ff2dsp [get_property SLACK $p]]
    } elseif {[string match "DSP*" $sref] && [string match "FD*" $eref]} {
        incr dir_count(dsp2ff); incr fftype($eref)
        lappend pairs [list $ecell $scell dsp2ff [get_property SLACK $p]]
    } else { incr dir_count(other) }
}
puts "DETAIL dirs: ff2dsp=$dir_count(ff2dsp) dsp2ff=$dir_count(dsp2ff) other=$dir_count(other)"
puts "DETAIL ff_types: [array get fftype]"
set uff [lsort -unique [lmap x $pairs {lindex $x 0}]]
set udsp [lsort -unique [lmap x $pairs {lindex $x 1}]]
puts "DETAIL distinct_ffs=[llength $uff] distinct_dsps=[llength $udsp]"
# sample the 3 worst pairs with reg config + which DSP pin the path enters/leaves
set k 0
foreach x [lsort -real -index 3 $pairs] {
    if {$k >= 3} break
    lassign $x ff dsp dir slack
    set dspregs ""
    foreach pr {AREG BREG CREG MREG PREG ADREG} {
        append dspregs "$pr=[get_property -quiet $pr [get_cells $dsp]] "
    }
    puts "DETAIL sample dir=$dir slack=$slack ff=[get_property REF_NAME [get_cells $ff]]:$ff dsp=$dsp regs: $dspregs"
    incr k
}
# BRAM sibling: critical BRAM->FF with DO reg off
set bram_hits 0
foreach p $paths {
    set scell [get_cells -quiet -of_objects [get_pins -quiet [get_property STARTPOINT_PIN $p]]]
    if {[string match "RAMB*" [get_property -quiet REF_NAME $scell]]} { incr bram_hits }
}
puts "DETAIL bram_startpoint_paths=$bram_hits"
close_design
