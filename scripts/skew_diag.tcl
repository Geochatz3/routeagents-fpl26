# skew_diag.tcl — diagnostic only (no modification).
#
# Question (2026-06-12): is there recoverable clock-skew on our ship DCPs'
# worst paths? Organizers honor skew-engineered gains (discussion #29). If
# the worst paths show consistent NEGATIVE skew contribution (destination
# clock arrives early) or large imbalance, a USER_CLOCK_ROOT move toward
# the critical region is worth a probe; if skew is ~0/favorable, skip.
#
# Usage: vivado -mode batch -source skew_diag.tcl -tclargs <in.dcp>

set in_dcp [lindex $argv 0]
open_checkpoint $in_dcp

set clk [get_clocks -quiet *fpl26contest*]
if {[llength $clk] > 0} {
    set paths [get_timing_paths -quiet -max_paths 20 -setup -to [lindex $clk 0]]
} else {
    set paths [get_timing_paths -quiet -max_paths 20 -setup]
}
foreach p $paths {
    set sk "" ; set su ""
    catch { set sk [get_property SKEW $p] }
    catch { set su [get_property UNCERTAINTY $p] }
    puts "SKEW_PATH slack=[get_property SLACK $p] skew=$sk uncert=$su endclk=[get_property ENDPOINT_CLOCK $p]"
}
# Clock root location(s) for the scored clock's net, for reference.
if {[llength $clk] > 0} {
    catch {
        set cnets [get_nets -quiet -of_objects $clk -filter {TYPE == GLOBAL_CLOCK}]
        foreach n $cnets {
            catch { puts "SKEW_CLOCK_ROOT net=[get_property NAME $n] root=[get_property CLOCK_ROOT $n]" }
        }
    }
}
puts "SKEW_DONE=1"
