# Retro-fix script: open a DCP and write a readable EDIF beside it.
# Invoked as: vivado -mode batch -source scripts/add_edif.tcl -tclargs <dcp_path>
#
# Reason: Vivado's default write_checkpoint embeds an encrypted EDIF that
# RapidWright cannot read.  validate_dcps.py fails with "Unable to find a
# readable EDIF file" when the optimizer didn't explicitly write_edif.
# This script generates the missing EDIF for already-written DCPs.

if {[llength $argv] != 1} {
    puts "Usage: vivado -mode batch -source scripts/add_edif.tcl -tclargs <dcp_path>"
    exit 2
}
set dcp_path [lindex $argv 0]
set edif_path [file rootname $dcp_path].edf
puts "Opening:   $dcp_path"
puts "EDIF out:  $edif_path"
open_checkpoint $dcp_path
write_edif -force $edif_path
puts "EDIF written: $edif_path"
close_design
exit 0
