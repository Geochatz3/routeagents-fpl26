# DSP *REG absorption transform (jul20 drill, GitHub #33 organizer-blessed lever).
# Legal move: absorb an EXISTING adjacent fabric FF into a DSP's unused *REG
# stage (latency preserved). Pipeline INSERTION (added latency) disqualifies.
#
# Mechanics decision (UG904/UG835 2025.1, checked jul20):
#   phys_opt_design -dsp_register_opt implements exactly the legal moves
#   ("AREG/BREG pull in from fabric", "PREG pull in from fabric", plus
#   latency-neutral rebalances MREG<->PREG etc. — every documented move is
#   register-count-neutral). BUT it is POST-PLACE ONLY (UG904 Table:
#   post-route valid = N). On a routed DCP the flow is therefore:
#       route_design -unroute  ->  phys_opt_design -dsp_register_opt  ->
#       route_design  [-> phys_opt_design -hold_fix if WHS<0]
#   A "control" mode (unroute -> route, no phys_opt) separates route-variance
#   from the transform's real effect.
#
# Usage: vivado -mode batch -source absorb_dsp_regs.tcl -tclargs \
#            <routed.dcp> <outdir> <mode> [modeargs]
#   mode = dryrun  [N=200]           analysis only, nothing changed
#   mode = physopt [npasses=1] [tag=stage1]
#   mode = control [tag=control]
#   mode = manual  [maxgroups=20] [tag=manual]   (hand netlist edit fallback)
# Emits greppable lines prefixed ABSORB / ABSORB-CAND / ABSORB-MEAS.

set dcp    [lindex $argv 0]
set outdir [lindex $argv 1]
set mode   [lindex $argv 2]

proc clk_obj {} {
    set c [get_clocks -quiet *fpl26contest*]
    if {[llength $c]} { return [lindex $c 0] }
    return [lindex [get_clocks] 0]
}
proc net_of {pin} { return [get_nets -quiet -of_objects $pin] }
# root-of-hierarchy net name for cross-segment identity compares
proc rootnet {net} {
    if {$net eq ""} { return "" }
    set p [get_property -quiet PARENT $net]
    if {$p ne ""} { return $p }
    return [get_property NAME $net]
}
proc is_const0 {net} { expr {$net ne "" && [get_property -quiet TYPE $net] eq "GROUND"} }
proc is_const1 {net} { expr {$net ne "" && [get_property -quiet TYPE $net] eq "POWER"} }

# ---------- measurement ----------
proc measure {label} {
    set c [clk_obj]
    set sp [get_timing_paths -quiet -setup -max_paths 1 -nworst 1 -to $c]
    set hp [get_timing_paths -quiet -hold  -max_paths 1 -nworst 1 -to $c]
    set wns [expr {[llength $sp] ? [get_property SLACK [lindex $sp 0]] : "NA"}]
    set whs [expr {[llength $hp] ? [get_property SLACK [lindex $hp 0]] : "NA"}]
    set ncell [llength [get_cells -quiet -hierarchical -filter {IS_PRIMITIVE}]]
    set nff   [llength [get_cells -quiet -hierarchical -filter {REF_NAME =~ FD*}]]
    puts "ABSORB-MEAS label=$label wns=$wns whs=$whs cells=$ncell ffs=$nff"
    return [list $wns $whs $ncell $nff]
}
proc route_errs {} {
    set s [report_route_status -return_string]
    set n "UNKNOWN"
    regexp -nocase {nets with routing errors[^:]*:\s*(\d+)} $s -> n
    # 2025.1 clean designs print "no routing errors" style summaries too
    if {$n eq "UNKNOWN" && [regexp -nocase {fully routed} $s]} { set n 0 }
    return $n
}

# The 2025.1 checkpoint opens DSP48E2 as a macro of primitives (DSP_ALU,
# DSP_A_B_DATA, DSP_OUTPUT, ...). Timing endpoints are SUBCELL pins; the
# AREG/BREG/.../PREG properties live on the parent macro. Map subcell -> macro.
proc dsp_macro_of {cell} {
    set full [get_property NAME $cell]
    set par [join [lrange [split $full /] 0 end-1] /]
    set pc [get_cells -quiet $par]
    if {$pc ne "" && [string match "DSP48E2*" [get_property -quiet REF_NAME $pc]]} {
        return $pc
    }
    return $cell
}

# ---------- per-FF safety check (shared by dryrun/manual) ----------
# Returns "" if the FF can be absorbed into a DSP *REG, else a skip reason.
# DSP internal regs: sync-reset-to-0, init-0, active-high CE/RST, same CLK.
proc ff_eligible {ff dspclk} {
    set ref [get_property REF_NAME $ff]
    if {![regexp {^FD(RE|SE|CE|PE)$} $ref]} { return "FF_TYPE_$ref" }
    set init [string tolower [get_property -quiet INIT $ff]]
    if {$init ne "" && ![string match "*0" $init]} { return "FF_INIT_$init" }
    foreach p {IS_C_INVERTED IS_D_INVERTED IS_R_INVERTED IS_S_INVERTED \
               IS_CLR_INVERTED IS_PRE_INVERTED} {
        set v [get_property -quiet $p $ff]
        if {$v ne "" && $v != 0} { return "FF_INV_$p" }
    }
    set cnet [net_of [get_pins -quiet $ff/C]]
    if {$cnet eq "" || $dspclk eq "" || [rootnet $cnet] ne [rootnet $dspclk]} {
        return "FF_CLK_MISMATCH"
    }
    switch -exact $ref {
        FDSE { if {![is_const0 [net_of [get_pins $ff/S]]]}   { return "FF_SET_ACTIVE" } }
        FDCE { if {![is_const0 [net_of [get_pins $ff/CLR]]]} { return "FF_ASYNC_CLR" } }
        FDPE { if {![is_const0 [net_of [get_pins $ff/PRE]]]} { return "FF_ASYNC_PRE" } }
    }
    return ""
}
# CE / sync-reset keys for group uniformity (map to CEA2/CEB2/CEC/CED/CEP, RSTA/RSTB/RSTC/RSTD/RSTP)
proc ff_ce_key {ff} {
    set n [net_of [get_pins -quiet $ff/CE]]
    if {$n eq "" || [is_const1 $n]} { return "TIE1" }
    return [rootnet $n]
}
proc ff_rst_key {ff} {
    set rp [get_pins -quiet $ff/R]
    if {$rp eq ""} { return "TIE0" }
    set n [net_of $rp]
    if {$n eq "" || [is_const0 $n]} { return "TIE0" }
    return [rootnet $n]
}

# ---------- group checks ----------
# Input side: (dsp, port in {A B C D}) — absorb the fabric FFs driving every
# connected bit of the port into AREG/BREG/CREG/DREG.
proc check_input_group {dsp port} {
    array set regmap {A AREG B BREG C CREG D DREG}
    if {![info exists regmap($port)]} { return [list SKIP "NON_DATA_PORT_$port" {}] }
    set reg $regmap($port)
    set rv [get_property -quiet $reg $dsp]
    if {$rv eq "" || int($rv) != 0} { return [list SKIP "REG_IN_USE:$reg=$rv" {}] }
    if {$port eq "A" || $port eq "B"} {
        set ai [get_property -quiet ${port}_INPUT $dsp]
        if {$ai ne "" && $ai ne "DIRECT"} { return [list SKIP "CASCADE_${port}_INPUT" {}] }
    }
    set clk [net_of [get_pins -quiet $dsp/CLK]]
    set ffs {}; set ces {}; set rsts {}; set gnd 0; set bits 0
    foreach pin [get_pins -quiet -of_objects $dsp -filter {DIRECTION == IN}] {
        set rpn [get_property REF_PIN_NAME $pin]
        if {![regexp "^${port}\\\[\\d+\\\]\$" $rpn]} { continue }
        set n [net_of $pin]
        if {$n eq ""} { continue }
        if {[is_const0 $n]} { incr gnd; continue }
        if {[is_const1 $n]} { return [list SKIP "VCC_TIED_BIT" {}] }
        incr bits
        set drv [get_pins -quiet -of_objects $n -leaf -filter {DIRECTION == OUT}]
        if {[llength $drv] != 1} { return [list SKIP "MULTI_DRIVER" {}] }
        set fc [get_cells -quiet -of_objects [lindex $drv 0]]
        if {![string match "FD*" [get_property -quiet REF_NAME $fc]]} {
            return [list SKIP "NON_FF_DRIVER:[get_property -quiet REF_NAME $fc]" {}]
        }
        set why [ff_eligible $fc $clk]
        if {$why ne ""} { return [list SKIP $why {}] }
        # fanout: every load of FF/Q must be a bit of THIS port of THIS dsp
        # (macro boundary pin matching the port, or a pin INSIDE this macro)
        set dspname [get_property NAME $dsp]
        set qnet [net_of [get_pins $fc/Q]]
        if {[llength [get_ports -quiet -of_objects $qnet]]} { return [list SKIP "FF_DRIVES_PORT" {}] }
        foreach lp [get_pins -quiet -of_objects $qnet -filter {DIRECTION == IN}] {
            set lc [get_cells -quiet -of_objects $lp]
            set lcn [get_property NAME $lc]
            if {$lcn eq $dspname} {
                if {![regexp "^${port}\\\[\\d+\\\]\$" [get_property REF_PIN_NAME $lp]]} {
                    return [list SKIP "FF_EXTRA_FANOUT_PIN:[get_property REF_PIN_NAME $lp]" {}]
                }
            } elseif {[string first "$dspname/" $lcn] != 0} {
                return [list SKIP "FF_EXTRA_FANOUT" {}]
            }
        }
        set fname [get_property NAME $fc]
        if {[lsearch -exact $ffs $fname] < 0} { lappend ffs $fname }
        set ck [ff_ce_key $fc];  if {[lsearch -exact $ces  $ck] < 0} { lappend ces  $ck }
        set rk [ff_rst_key $fc]; if {[lsearch -exact $rsts $rk] < 0} { lappend rsts $rk }
    }
    if {![llength $ffs]}        { return [list SKIP "NO_FF_BITS" {}] }
    if {[llength $ces] > 1}     { return [list SKIP "MIXED_CE" {}] }
    if {[llength $rsts] > 1}    { return [list SKIP "MIXED_RST" {}] }
    return [list PASS "reg=$reg bits=$bits gnd=$gnd nff=[llength $ffs] ce=[lindex $ces 0] rst=[lindex $rsts 0]" $ffs]
}

# Output side: (dsp) — absorb the FF loads of every P bit into PREG.
# PREG also registers PCOUT/CARRYOUT/CARRYCASCOUT/MULTSIGNOUT/PATTERNDETECT:
# those side outputs must be unused, else enabling PREG re-times other logic.
proc check_output_group {dsp} {
    set rv [get_property -quiet PREG $dsp]
    if {$rv eq "" || int($rv) != 0} { return [list SKIP "REG_IN_USE:PREG=$rv" {}] }
    set clk [net_of [get_pins -quiet $dsp/CLK]]
    set ffs {}; set ces {}; set rsts {}; set bits 0
    foreach pin [get_pins -quiet -of_objects $dsp -filter {DIRECTION == OUT}] {
        set rpn [get_property REF_PIN_NAME $pin]
        set n [net_of $pin]
        if {$n eq ""} { continue }
        set loads [get_pins -quiet -of_objects $n -leaf -filter {DIRECTION == IN}]
        if {![llength $loads] && ![llength [get_ports -quiet -of_objects $n]]} { continue }
        if {![regexp {^P\[\d+\]$} $rpn]} {
            # a used non-P output (PCOUT/CARRYOUT/...) would be re-timed by PREG
            return [list SKIP "SIDE_OUTPUT_USED:$rpn" {}]
        }
        if {[llength [get_ports -quiet -of_objects $n]]} { return [list SKIP "P_DRIVES_PORT" {}] }
        incr bits
        foreach lp $loads {
            set lc [get_cells -quiet -of_objects $lp]
            set lref [get_property -quiet REF_NAME $lc]
            if {![string match "FD*" $lref]} { return [list SKIP "NON_FF_LOAD:$lref" {}] }
            if {[get_property REF_PIN_NAME $lp] ne "D"} { return [list SKIP "NON_D_LOAD" {}] }
            set why [ff_eligible $lc $clk]
            if {$why ne ""} { return [list SKIP $why {}] }
            set fname [get_property NAME $lc]
            if {[lsearch -exact $ffs $fname] < 0} { lappend ffs $fname }
            set ck [ff_ce_key $lc];  if {[lsearch -exact $ces  $ck] < 0} { lappend ces  $ck }
            set rk [ff_rst_key $lc]; if {[lsearch -exact $rsts $rk] < 0} { lappend rsts $rk }
        }
    }
    if {![llength $ffs]}     { return [list SKIP "NO_FF_LOADS" {}] }
    if {[llength $ces] > 1}  { return [list SKIP "MIXED_CE" {}] }
    if {[llength $rsts] > 1} { return [list SKIP "MIXED_RST" {}] }
    return [list PASS "reg=PREG bits=$bits nff=[llength $ffs] ce=[lindex $ces 0] rst=[lindex $rsts 0]" $ffs]
}

# Collect candidate groups from the worst-N setup paths.
# Returns list of {key kind dsp port slack} with key unique.
proc collect_groups {N} {
    set paths [get_timing_paths -quiet -setup -max_paths $N -nworst 1 -to [clk_obj]]
    set groups {}
    array set seen {}
    set stats(ff2dsp) 0; set stats(dsp2ff) 0; set stats(other) 0
    foreach p $paths {
        set spin [get_pins -quiet [get_property STARTPOINT_PIN $p]]
        set epin [get_pins -quiet [get_property ENDPOINT_PIN $p]]
        set scell [get_cells -quiet -of_objects $spin]
        set ecell [get_cells -quiet -of_objects $epin]
        set sref [get_property -quiet REF_NAME $scell]
        set eref [get_property -quiet REF_NAME $ecell]
        set slack [get_property SLACK $p]
        if {[string match "FD*" $sref] && [string match "DSP*" $eref]} {
            incr stats(ff2dsp)
            set mac [dsp_macro_of $ecell]
            set macname [get_property NAME $mac]
            # port = the macro BOUNDARY pin this FF actually drives (endpoint
            # pin is an internal subcell pin, e.g. DSP_OUTPUT_INST/ALU_*)
            set port ""
            set qnet [get_nets -quiet -segments -of_objects [get_pins -quiet $scell/Q]]
            foreach bp [get_pins -quiet -of_objects $qnet -filter {DIRECTION == IN}] {
                set oc [get_cells -quiet -of_objects $bp]
                if {$oc ne "" && [get_property NAME $oc] eq $macname} {
                    regexp {^([A-Z0-9]+)} [get_property REF_PIN_NAME $bp] -> port
                    break
                }
            }
            if {$port eq ""} {
                regexp {^([A-Z0-9]+)} [get_property REF_PIN_NAME $epin] -> port
            }
            set key "in:$macname:$port"
            if {![info exists seen($key)]} {
                set seen($key) 1
                lappend groups [list $key in $macname $port $slack]
            }
        } elseif {[string match "DSP*" $sref] && [string match "FD*" $eref]} {
            incr stats(dsp2ff)
            set macname [get_property NAME [dsp_macro_of $scell]]
            set key "out:$macname"
            if {![info exists seen($key)]} {
                set seen($key) 1
                lappend groups [list $key out $macname P $slack]
            }
        } else {
            incr stats(other)
        }
    }
    puts "ABSORB path_mix: npaths=[llength $paths] ff2dsp=$stats(ff2dsp) dsp2ff=$stats(dsp2ff) other=$stats(other)"
    return $groups
}

proc dsp_reg_histogram {} {
    foreach reg {AREG BREG CREG DREG ADREG MREG PREG} {
        array unset h; array set h {}
        foreach c [get_cells -quiet -hierarchical -filter {REF_NAME =~ DSP48E2*}] {
            set v [get_property -quiet $reg $c]
            if {![info exists h($v)]} { set h($v) 0 }
            incr h($v)
        }
        puts "ABSORB dsp_reg_hist $reg: [array get h]"
    }
}

# ================= main =================
open_checkpoint $dcp
set period [get_property PERIOD [clk_obj]]
puts "ABSORB open dcp=$dcp clock=[get_property NAME [clk_obj]] period=$period mode=$mode"
measure baseline

switch -exact $mode {
    dryrun {
        set N 200
        if {[llength $argv] > 3} { set N [lindex $argv 3] }
        dsp_reg_histogram
        set groups [collect_groups $N]
        set pass 0; set skip 0
        array set skips {}
        foreach g $groups {
            lassign $g key kind dspname port slack
            set dsp [get_cells -quiet $dspname]
            if {$kind eq "in"} {
                lassign [check_input_group $dsp $port] st detail ffs
            } else {
                lassign [check_output_group $dsp] st detail ffs
            }
            puts "ABSORB-CAND $st key=$key slack=$slack $detail"
            if {$st eq "PASS"} { incr pass } else {
                incr skip
                set r [lindex [split $detail :] 0]
                if {![info exists skips($r)]} { set skips($r) 0 }
                incr skips($r)
            }
        }
        puts "ABSORB dryrun_summary groups=[llength $groups] pass=$pass skip=$skip skip_reasons=[array get skips]"
    }

    physopt {
        set npasses 1; set tag stage1
        if {[llength $argv] > 3} { set npasses [lindex $argv 3] }
        if {[llength $argv] > 4} { set tag [lindex $argv 4] }
        # snapshot DSP reg attrs for legality audit
        array set before {}
        foreach c [get_cells -quiet -hierarchical -filter {REF_NAME =~ DSP48E2*}] {
            set nm [get_property NAME $c]
            set l {}
            foreach reg {AREG BREG CREG DREG ADREG MREG PREG} {
                lappend l [get_property -quiet $reg $c]
            }
            set before($nm) $l
        }
        route_design -unroute
        puts "ABSORB unrouted"
        set prevmod 0
        for {set i 1} {$i <= $npasses} {incr i} {
            phys_opt_design -dsp_register_opt
            set mod [llength [get_cells -quiet -hierarchical \
                -filter {PHYS_OPT_MODIFIED =~ *DSP_REGISTER_OPT*}]]
            puts "ABSORB physopt_pass $i modified_cells_cum=$mod"
            if {$mod == $prevmod} { break }
            set prevmod $mod
        }
        # per-DSP attr delta (legality: every move must be register-count-neutral)
        set ndelta 0
        foreach c [get_cells -quiet -hierarchical -filter {REF_NAME =~ DSP48E2*}] {
            set nm [get_property NAME $c]
            set l {}
            foreach reg {AREG BREG CREG DREG ADREG MREG PREG} {
                lappend l [get_property -quiet $reg $c]
            }
            if {[info exists before($nm)] && $l ne $before($nm)} {
                incr ndelta
                puts "ABSORB dsp_delta $nm before=$before($nm) after=$l"
            }
        }
        puts "ABSORB dsp_attr_deltas=$ndelta"
        measure post_physopt_preroute
        route_design
        lassign [measure post_route] wns whs
        if {$whs ne "NA" && $whs < 0} {
            puts "ABSORB hold_violation whs=$whs -> phys_opt_design -hold_fix"
            phys_opt_design -hold_fix
            lassign [measure post_holdfix] wns whs
        }
        puts "ABSORB route_errors=[route_errs]"
        report_timing_summary -file $outdir/${tag}_timing.rpt
        report_route_status  -file $outdir/${tag}_route_status.rpt
        write_checkpoint -force $outdir/fir_absorb_${tag}.dcp
        puts "ABSORB wrote $outdir/fir_absorb_${tag}.dcp"
    }

    control {
        set tag control
        if {[llength $argv] > 3} { set tag [lindex $argv 3] }
        route_design -unroute
        puts "ABSORB unrouted"
        route_design
        lassign [measure post_route] wns whs
        puts "ABSORB route_errors=[route_errs]"
        report_timing_summary -file $outdir/${tag}_timing.rpt
        report_route_status  -file $outdir/${tag}_route_status.rpt
        write_checkpoint -force $outdir/fir_absorb_${tag}.dcp
        puts "ABSORB wrote $outdir/fir_absorb_${tag}.dcp"
    }

    manual {
        # Forced ECO PREG pull-in on the ORIGINAL routing (stage1 evidence:
        # the built-in identifies the candidates but refuses them, and its
        # mandatory unroute/reroute alone costs ~0.1ns WNS). Per PASS group:
        #   PREG 0->1 on the macro (+DSP_OUTPUT_INST subcell), CEP<-FF CE net,
        #   RSTP<-FF R net, every FF Q load rewired onto the P-bit net,
        #   FF removed. Latency preserved: 1 FF bank out, 1 PREG stage in.
        # Then route_design completes only the touched (partial) nets.
        set maxg 20; set tag manual
        if {[llength $argv] > 3} { set maxg [lindex $argv 3] }
        if {[llength $argv] > 4} { set tag [lindex $argv 4] }
        set base_nff [llength [get_cells -quiet -hierarchical -filter {REF_NAME =~ FD*}]]
        set groups [collect_groups 200]
        set done 0
        set total_removed 0
        foreach g $groups {
            if {$done >= $maxg} { break }
            lassign $g key kind dspname port slack
            if {$kind ne "out"} { continue }  ;# PREG pull-in covers the paired C-side wall too
            set dsp [get_cells -quiet -hierarchical -filter "NAME == \"$dspname\""]
            lassign [check_output_group $dsp] st detail ffs
            if {$st ne "PASS"} { puts "ABSORB-EDIT SKIP $key $detail"; continue }
            puts "ABSORB-EDIT START $key slack=$slack $detail"
            # gather {ffobj pnet} pairs from the P pins (objects, not names —
            # names contain glob-hostile brackets)
            set jobs {}
            array unset seenff; array set seenff {}
            set cepsrc ""; set rstsrc ""
            foreach pin [get_pins -quiet -of_objects $dsp -filter {DIRECTION == OUT}] {
                if {![regexp {^P\[\d+\]$} [get_property REF_PIN_NAME $pin]]} { continue }
                set pnet [net_of $pin]
                if {$pnet eq ""} { continue }
                foreach lp [get_pins -quiet -of_objects $pnet -leaf -filter {DIRECTION == IN}] {
                    set fc [get_cells -quiet -of_objects $lp]
                    set fn [get_property NAME $fc]
                    if {[info exists seenff($fn)]} { continue }
                    set seenff($fn) 1
                    lappend jobs [list $fc $pnet]
                    if {$cepsrc eq ""} {
                        set cepsrc [net_of [get_pins -quiet -of_objects $fc -filter {REF_PIN_NAME == CE}]]
                        set rstsrc [net_of [get_pins -quiet -of_objects $fc -filter {REF_PIN_NAME == R}]]
                    }
                }
            }
            # 1. enable PREG on macro + output subcell
            if {[catch {set_property PREG 1 $dsp} msg]} {
                puts "ABSORB-EDIT ERR $key set_property PREG on macro: $msg"; continue
            }
            set sub [get_cells -quiet -hierarchical -filter "NAME == \"$dspname/DSP_OUTPUT_INST\""]
            set subpreg "-"
            if {$sub ne ""} {
                catch {set_property PREG 1 $sub}
                set subpreg [get_property -quiet PREG $sub]
            }
            puts "ABSORB-EDIT PREG_SET $key macro=[get_property -quiet PREG $dsp] subcell=$subpreg"
            # 2. wire CEP / RSTP from the FF bank's CE / R nets
            foreach {pn src} [list CEP $cepsrc RSTP $rstsrc] {
                set cpin [lindex [get_pins -quiet -of_objects $dsp -filter "REF_PIN_NAME == $pn"] 0]
                if {$cpin eq "" || $src eq ""} { puts "ABSORB-EDIT WARN $key no wiring for $pn (pin=$cpin src=$src)"; continue }
                set cur [net_of $cpin]
                if {$cur ne ""} { disconnect_net -quiet -objects $cpin }
                connect_net -quiet -hierarchical -net $src -objects $cpin
                puts "ABSORB-EDIT WIRED $key $pn <- [rootnet $src]"
            }
            # 3. rewire each FF's loads onto its P net, then remove the FF
            set nrm 0
            foreach j $jobs {
                lassign $j fc pnet
                set qpin [get_pins -quiet -of_objects $fc -filter {REF_PIN_NAME == Q}]
                set qsegs [get_nets -quiet -segments -of_objects $qpin]
                # rewire at the OUTERMOST connection point: keep macro boundary
                # pins (REF_NAME DSP48E2) and fabric leaf pins; drop pins of
                # DSP-internal primitives (DSP_C_DATA etc. — the boundary pin
                # carries those) so connect_net never punches into the macro.
                set loads {}
                foreach lp [get_pins -quiet -of_objects $qsegs -filter {DIRECTION == IN}] {
                    set lc [get_cells -quiet -of_objects $lp]
                    if {$lc eq ""} { continue }
                    if {[string match "DSP_*" [get_property -quiet REF_NAME $lc]]} { continue }
                    lappend loads $lp
                }
                if {[llength $loads]} {
                    disconnect_net -quiet -objects $loads
                    connect_net -quiet -hierarchical -net $pnet -objects $loads
                }
                disconnect_net -quiet -objects [get_pins -quiet -of_objects $fc]
                remove_cell -quiet $fc
                remove_net -quiet $qsegs
                incr nrm
            }
            puts "ABSORB-EDIT DONE $key ffs_removed=$nrm"
            incr done
            incr total_removed $nrm
        }
        puts "ABSORB manual_groups_transformed=$done ffs_removed_total=$total_removed"
        if {$done == 0} { error "manual mode: no group transformed" }
        lassign [measure post_edit_preroute] _w _h _c nff_now
        if {$nff_now != $base_nff - $total_removed} {
            error "manual mode: FF count mismatch (now=$nff_now expected=[expr {$base_nff - $total_removed}]) — a -quiet edit failed silently"
        }
        route_design
        lassign [measure post_route] wns whs
        if {$whs ne "NA" && $whs < 0} {
            puts "ABSORB hold_violation whs=$whs -> phys_opt_design -hold_fix"
            phys_opt_design -hold_fix
            lassign [measure post_holdfix] wns whs
        }
        puts "ABSORB route_errors=[route_errs]"
        report_timing_summary -file $outdir/${tag}_timing.rpt
        report_route_status  -file $outdir/${tag}_route_status.rpt
        write_checkpoint -force $outdir/fir_absorb_${tag}.dcp
        puts "ABSORB wrote $outdir/fir_absorb_${tag}.dcp"
    }

    default { error "unknown mode $mode" }
}
puts "ABSORB done mode=$mode"
close_design
