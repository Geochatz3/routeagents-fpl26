# Diagnostic v3 (self-aiming): take the GLOBAL worst setup path, read its
# actual endpoint clock, and test which uncertainty formulation moves its
# slack. Timing-engine only — no phys_opt. Also dumps the top-10 worst
# paths' clocks so we finally have ground truth on this design.
# Usage: vivado -mode batch -source ratchet_diag.tcl -tclargs <in.dcp>

set in_dcp [lindex $argv 0]
open_checkpoint $in_dcp

proc clk_obj {name} { return [get_clocks -quiet -filter "NAME == \"$name\""] }

puts "DIAG_TOP10_BEGIN"
foreach p [get_timing_paths -setup -max_paths 10] {
    puts "DIAG_PATH slack=[get_property SLACK $p] start_clk=[get_property STARTPOINT_CLOCK $p] end_clk=[get_property ENDPOINT_CLOCK $p]"
}
puts "DIAG_TOP10_END"

set p [lindex [get_timing_paths -setup -max_paths 1] 0]
set sclk [get_property STARTPOINT_CLOCK $p]
set eclk [get_property ENDPOINT_CLOCK $p]
puts "DIAG_BASE_SLACK=[get_property SLACK $p]"
puts "DIAG_BASE_UNCERT=[get_property UNCERTAINTY $p]"
puts "DIAG_BASE_STARTCLK=$sclk"
puts "DIAG_BASE_ENDCLK=$eclk"

proc worst_to {name} {
    set p [lindex [get_timing_paths -setup -max_paths 1 -to [clk_obj $name] -quiet] 0]
    if {$p eq ""} { return "EMPTY EMPTY" }
    return [list [get_property SLACK $p] [get_property UNCERTAINTY $p]]
}

# Variant 1: simple uncertainty on the endpoint clock
set_clock_uncertainty -setup 0.198 [clk_obj $eclk]
puts "DIAG_SIMPLE=[worst_to $eclk]"
set_clock_uncertainty -setup 0 [clk_obj $eclk]

# Variant 2: inter-clock from the path's start clock to its end clock
if {[llength [clk_obj $sclk]] && $sclk ne $eclk} {
    set_clock_uncertainty -setup 0.198 -from [clk_obj $sclk] -to [clk_obj $eclk]
    puts "DIAG_INTER=[worst_to $eclk]"
    set_clock_uncertainty -setup 0 -from [clk_obj $sclk] -to [clk_obj $eclk]
} else {
    set_clock_uncertainty -setup 0.198 -from [clk_obj $eclk] -to [clk_obj $eclk]
    puts "DIAG_INTRA=[worst_to $eclk]"
    set_clock_uncertainty -setup 0 -from [clk_obj $eclk] -to [clk_obj $eclk]
}

puts "DIAG_RESTORED=[worst_to $eclk]"
puts "DIAG_DONE=1"
