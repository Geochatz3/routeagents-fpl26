"""Provides deterministic strategies for optimizing routed FPGA checkpoints.

Recipes call MCP-server tool functions directly, avoiding the protocol layer
used by agent-driven optimization. Each module exposes an application entry
point that writes an output checkpoint and returns timing, modification,
routing-error, and wall-time metrics. Recipes also provide command-line
interfaces for standalone and scheduler-driven execution. Given the same input
and tool version, a recipe produces the same output checkpoint.
"""
