# Prompts

| File | Status |
|---|---|
| `system_prompt_scored.txt` | **The system prompt of the scored run.** Loaded by default (`dcp_optimizer.load_system_prompt`). |
| `system_prompt_v2_experimental.txt` | A/B variant, **never active in the scored run**: it loads only when `FPL26_PROMPT_V2=1`, and no ship target sets that flag. Kept because it shipped in the scored archive. |

The V2 file exists as a separate file (rather than an edit of the
first) so prompt changes ride the same default-OFF flag discipline as
every other behavioral change: with the flag unset the scored prompt is
returned byte-for-byte, and a missing V2 file falls back loudly instead
of silently changing behavior.
