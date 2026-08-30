"""Pin the size of the agent's tool surface.

"40 tools" is stated in the README, the slide, the video and the write-up.
It is not a round number someone chose — it is 17 Vivado tools, 17 RapidWright
tools and 6 recipe tools the agent registers itself. Adding or removing one
silently turns a published number into a wrong one, so the count is asserted
here rather than counted by hand at release time.
"""
from __future__ import annotations

import ast
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# server source -> (the README whose table a reader counts, expected count)
MCP_SERVERS = {
    "VivadoMCP/vivado_mcp_server.py": ("VivadoMCP/README.md", 17),
    "RapidWrightMCP/server.py": ("RapidWrightMCP/README.md", 17),
}
SYNTHETIC = 6
TOTAL = sum(n for _, n in MCP_SERVERS.values()) + SYNTHETIC


def mcp_tool_names(path: Path) -> list[str]:
    """Names passed to a `Tool(...)` constructor — the MCP registration."""
    out = []
    for node in ast.walk(ast.parse(path.read_text())):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        label = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", "")
        if label != "Tool":
            continue
        out += [
            kw.value.value for kw in node.keywords
            if kw.arg == "name" and isinstance(kw.value, ast.Constant)
        ]
    return out


def synthetic_tool_names() -> list[str]:
    """Names the agent appends to `self.tools` itself, as OpenAI tool specs."""
    src = REPO / "optimizer" / "tool_dispatch.py"
    out = []
    for node in ast.walk(ast.parse(src.read_text())):
        if not (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "append"
                and node.args and isinstance(node.args[0], ast.Dict)):
            continue
        spec = node.args[0]
        for key, val in zip(spec.keys, spec.values):
            if not (isinstance(key, ast.Constant) and key.value == "function"
                    and isinstance(val, ast.Dict)):
                continue
            out += [
                v.value for k, v in zip(val.keys, val.values)
                if isinstance(k, ast.Constant) and k.value == "name"
                and isinstance(v, ast.Constant)
            ]
    return out


class ToolSurfaceTests(unittest.TestCase):
    def test_each_server_registers_the_expected_number(self):
        for rel, (_, expected) in MCP_SERVERS.items():
            with self.subTest(server=rel):
                names = mcp_tool_names(REPO / rel)
                self.assertEqual(len(names), expected, f"{rel}: {sorted(names)}")
                self.assertEqual(len(set(names)), len(names), "duplicate tool name")

    def test_each_server_readme_lists_every_tool_it_registers(self):
        """The count alone is not enough: the RapidWrightMCP table documented
        11 of its 17 tools for the whole contest, because the six added for
        this project were never added to it. A reader counts the table, not
        the AST."""
        for rel, (doc, _) in MCP_SERVERS.items():
            with self.subTest(server=rel):
                text = (REPO / doc).read_text()
                missing = [n for n in mcp_tool_names(REPO / rel)
                           if f"`{n}`" not in text]
                self.assertEqual(missing, [], f"{doc} does not list: {missing}")

    def test_the_agent_registers_its_own_recipe_tools(self):
        names = synthetic_tool_names()
        self.assertEqual(len(names), SYNTHETIC, sorted(names))
        for n in names:
            self.assertTrue(n.startswith("recipe_"), n)

    def test_the_published_total_is_forty(self):
        total = sum(len(mcp_tool_names(REPO / rel)) for rel in MCP_SERVERS)
        total += len(synthetic_tool_names())
        self.assertEqual(
            total, TOTAL,
            f"the agent now exposes {total} tools; the README, the slide, the "
            f"video and the write-up all say {TOTAL}",
        )


if __name__ == "__main__":
    unittest.main()
