# DSP absorption WORK-LIST probe (jul04, read-only): per parent DSP48E2 macro on
# mini-isp's critical wall, verify absorbability: (a) every P-output net's loads
# are ONLY FF D-pins, (b) FF types FDRE (clean) vs FDSE (S must be inert),
# (c) uniform CE net across the bus (maps to CEP), (d) current MREG/PREG.
proc clk_obj {} {
    set c [get_clocks -quiet *fpl26contest*]
    if {[llength $c]} { return [lindex $c 0] }
    return [lindex [get_clocks] 0]
}
open_checkpoint [lindex $argv 0]
set paths [get_timing_paths -quiet -setup -max_paths 200 -nworst 1 -to [clk_obj]]
set parents {}
foreach p $paths {
    set scell [get_cells -quiet -of_objects [get_pins -quiet [get_property STARTPOINT_PIN $p]]]
    if {![string match "DSP*" [get_property -quiet REF_NAME $scell]]} continue
    # parent macro = strip the internal INST suffix
    set full [get_property NAME $scell]
    set parent [join [lrange [split $full /] 0 end-1] /]
    if {[lsearch -exact $parents $parent] < 0} { lappend parents $parent }
}
puts "WL parents=[llength $parents]"
foreach par $parents {
    set pc [get_cells -quiet $par]
    if {$pc eq ""} { puts "WL $par : NOT_A_CELL (flat netlist?)"; continue }
    set mreg [get_property -quiet MREG $pc]; set preg [get_property -quiet PREG $pc]
    # P-output nets and their loads
    set ppins [get_pins -quiet $par/P*]
    set loads_ok 1; set fftypes {}; set cenets {}; set snets {}; set nff 0
    foreach net [get_nets -quiet -of_objects $ppins] {
        foreach lp [get_pins -quiet -of_objects $net -filter {DIRECTION == IN}] {
            set lc [get_cells -quiet -of_objects $lp]
            if {$lc eq ""} continue
            set lref [get_property REF_NAME $lc]
            set lpin [lindex [split [get_property NAME $lp] /] end]
            if {[string match "FD*" $lref] && $lpin eq "D"} {
                incr nff
                if {[lsearch $fftypes $lref] < 0} { lappend fftypes $lref }
                set ce [get_nets -quiet -of_objects [get_pins -quiet $lc/CE]]
                if {$ce ne "" && [lsearch $cenets $ce] < 0} { lappend cenets $ce }
                set sp [get_pins -quiet $lc/S]
                if {$sp ne ""} {
                    set sn [get_nets -quiet -of_objects $sp]
                    if {$sn ne "" && [lsearch $snets $sn] < 0} { lappend snets $sn }
                }
            } elseif {![string match "DSP*" $lref]} {
                set loads_ok 0
            }
        }
    }
    set verdict [expr {$loads_ok && $preg ne "" && int($preg) == 0 ? "CANDIDATE" : "CHECK"}]
    puts "WL $par MREG=$mreg PREG=$preg ff_loads=$nff types={$fftypes} ce_nets=[llength $cenets] s_nets=[llength $snets] pure_ff_loads=$loads_ok => $verdict"
}
close_design
