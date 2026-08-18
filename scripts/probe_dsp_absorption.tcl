# DSP *REG absorption probe (jul03, GitHub #33 — organizer-blessed lever).
# Chris: absorbing an EXISTING adjacent fabric FF into a DSP's *REG is legal
# (latency preserved). This probe only QUANTIFIES the opportunity per design:
#   - how many of the worst-200 setup paths (scored clock) start/end at a DSP
#   - of those, how many have a plain fabric FD* register directly adjacent
#     (the absorbable pattern) and what slack those paths carry
# Read-only; no netlist changes. Usage: -tclargs <optimized.dcp> <design_name>

proc clk_obj {} {
    set c [get_clocks -quiet *fpl26contest*]
    if {[llength $c]} { return [lindex $c 0] }
    return [lindex [get_clocks] 0]
}
set dcp  [lindex $argv 0]
set name [lindex $argv 1]
open_checkpoint $dcp
set c [clk_obj]
set paths [get_timing_paths -quiet -setup -max_paths 200 -nworst 1 -to $c]
set n [llength $paths]
set dsp_touch 0
set absorbable 0
set worst_absorbable_slack ""
foreach p $paths {
    set sp [get_property -quiet STARTPOINT_PIN $p]
    set ep [get_property -quiet ENDPOINT_PIN $p]
    set scell [get_cells -quiet -of_objects [get_pins -quiet $sp]]
    set ecell [get_cells -quiet -of_objects [get_pins -quiet $ep]]
    set sref [get_property -quiet REF_NAME $scell]
    set eref [get_property -quiet REF_NAME $ecell]
    set is_dsp 0
    if {[string match "DSP*" $sref] || [string match "DSP*" $eref]} {
        set is_dsp 1; incr dsp_touch
    }
    # absorbable pattern: fabric FD* register at the OTHER end of a DSP path
    if {$is_dsp} {
        if {([string match "DSP*" $sref] && [string match "FD*" $eref]) ||
            ([string match "DSP*" $eref] && [string match "FD*" $sref])} {
            incr absorbable
            if {$worst_absorbable_slack eq ""} {
                set worst_absorbable_slack [get_property SLACK $p]
            }
        }
    }
}
set total_dsps [llength [get_cells -quiet -hierarchical -filter {REF_NAME =~ DSP48E2*}]]
puts "DSPPROBE design=$name paths_checked=$n dsp_touching=$dsp_touch absorbable_ff_dsp=$absorbable worst_absorbable_slack=$worst_absorbable_slack total_dsps=$total_dsps"
close_design
