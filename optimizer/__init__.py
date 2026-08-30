"""The optimizer's pipeline stages and mechanisms.

Four modules here are stage mixins composed onto DCPOptimizer
(recipe_passes, tool_dispatch, polish_ladder, finalization); the rest are
pure mechanisms with no instance state. Everything is importable and
testable without spawning Vivado or RapidWright, which is the point of
keeping it out of dcp_optimizer.py. See README.md in this directory.
"""
