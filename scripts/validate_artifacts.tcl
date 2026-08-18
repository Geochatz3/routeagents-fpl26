# Quick DCP integrity + WNS verification harness.
# Invoked as: vivado -mode batch -source _validate_artifacts.tcl -tclargs <dcp_path>
# Prints structured VALIDATE: lines on stdout that the wrapper parses.
#
# Checks:
#   1. open_checkpoint succeeds (DCP not corrupted)
#   2. report_route_status returns clean (no routing errors)
#   3. get_timing_paths -setup returns a valid WNS
#   4. write_edif to a tmp path (then external diff vs the saved EDIF)
#
# Exit codes: 0 = pass-or-data, non-zero = fatal Vivado error (DCP unopenable etc.)

set dcp_path [lindex $argv 0]
# Strip surrounding double-quotes the Vivado-on-WSL2 wrapper sometimes
# embeds in -tclargs values via the cmd.exe quoting bug.  Observed
# 2026-05-18 on the ship-artifact revalidation: $argv arrived as
# "C:\Users\Giorgos\..." with the quotes literally part of the string,
# Vivado then prepended its CWD and failed with [Common 17-69].
set dcp_path [string trim $dcp_path "\""]
puts "VALIDATE:DCP=$dcp_path"

set t0 [clock seconds]
if {[catch {open_checkpoint $dcp_path} err]} {
    puts "VALIDATE:FAIL_OPEN:$err"
    exit 2
}
puts "VALIDATE:OPENED_OK"
puts "VALIDATE:OPEN_ELAPSED_S=[expr {[clock seconds] - $t0}]"

# Route status — clean is "0 nets with routing errors that are routable"
set route_status [report_route_status -return_string]
set route_errors [llength [regexp -all -inline {nets with routing errors that are routable:\s+\d+} $route_status]]
puts "VALIDATE:ROUTE_STATUS_BLOCK_START"
puts $route_status
puts "VALIDATE:ROUTE_STATUS_BLOCK_END"

# Timing — single worst-setup path is enough for cross-check
if {[catch {
    set tps [get_timing_paths -max_paths 1 -setup -sort_by slack]
    if {[llength $tps] > 0} {
        set wns [get_property SLACK [lindex $tps 0]]
        puts "VALIDATE:WNS_NS=$wns"
    } else {
        puts "VALIDATE:WNS_NS=NO_PATHS"
    }
} err]} {
    puts "VALIDATE:FAIL_TIMING:$err"
}

# Fresh EDIF — to a sibling file with .verify.edif suffix.  The wrapper
# size-compares this against the run's saved EDIF; large delta indicates
# the saved EDIF was stale.
set out_edif "${dcp_path}.verify.edif"
if {[catch {write_edif -force $out_edif} err]} {
    puts "VALIDATE:FAIL_EDIF:$err"
} else {
    puts "VALIDATE:FRESH_EDIF=$out_edif"
}

puts "VALIDATE:DONE"
exit 0
