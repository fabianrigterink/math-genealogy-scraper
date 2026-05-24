"""Render a math genealogy as a standalone interactive HTML file via Graphviz.

Renders the ancestor DAG for one mathematician using `dot`. Output inlines the
SVG plus the svg-pan-zoom CDN library, so the file opens directly in a browser
(no web server needed) with pan/zoom, native hover tooltips, and clickable
nodes that link to each mathematician's MGP page.

Author: Fabian Rigterink
"""
import argparse
import html
import pickle
import sys
from pathlib import Path

import graphviz
import networkx as nx


def find_person(G, name):
    target = name.strip().lower()
    exact = [n for n in G.nodes
             if (G.nodes[n].get("name") or "").lower() == target]
    if len(exact) == 1:
        return exact[0]
    if len(exact) > 1:
        _print_matches(G, exact)
        sys.exit(1)
    substring = [n for n in G.nodes
                 if target in (G.nodes[n].get("name") or "").lower()]
    if not substring:
        print(f"no match for {name!r}", file=sys.stderr)
        sys.exit(1)
    if len(substring) == 1:
        return substring[0]
    _print_matches(G, substring)
    sys.exit(1)


def _print_matches(G, ids):
    print("multiple matches; please be more specific:", file=sys.stderr)
    for n in ids[:30]:
        r = G.nodes[n]
        print(f"  id={n}: {r.get('name')} ({r.get('year')}, {r.get('school')})",
              file=sys.stderr)
    if len(ids) > 30:
        print(f"  ... and {len(ids) - 30} more", file=sys.stderr)


def build_dot(G, root):
    ancestor_ids = nx.ancestors(G, root) | {root}
    sub = G.subgraph(ancestor_ids)

    dot = graphviz.Digraph("genealogy")
    dot.attr(rankdir="TB")

    for n in sub.nodes:
        a = G.nodes[n]
        name = a.get("name") or f"#{n}"
        meta_parts = [p for p in (a.get("year"), a.get("school")) if p]
        label = name + ("\n" + ", ".join(str(p) for p in meta_parts) if meta_parts else "")

        tip_lines = [name]
        for field in ("year", "school", "country", "subject", "thesis"):
            v = a.get(field)
            if v:
                tip_lines.append(f"{field.capitalize()}: {v}")
        tooltip = "\n".join(tip_lines)

        url = f"https://genealogy.math.ndsu.nodak.edu/id.php?id={n}"
        dot.node(str(n), label=label, tooltip=tooltip, URL=url, target="_blank")

    for u, v in sub.edges:
        dot.edge(str(u), str(v))

    return dot, len(ancestor_ids)


_BINARY_FORMATS = {".png", ".pdf", ".jpg", ".jpeg", ".gif"}
_TEXT_FORMATS = {".svg", ".dot"}


def render(G, root, output_path):
    dot, n_nodes = build_dot(G, root)
    ext = Path(output_path).suffix.lower()
    focal_name = G.nodes[root].get("name") or f"#{root}"

    if ext == ".html":
        svg = dot.pipe(format="svg", encoding="utf-8")
        if svg.startswith("<?xml"):
            svg = svg.split("?>", 1)[1].lstrip()
        title = html.escape("Math Genealogy: " + focal_name)
        page = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>{title}</title>
<script src="https://cdn.jsdelivr.net/npm/svg-pan-zoom@3.6.2/dist/svg-pan-zoom.min.js"></script>
<style>html,body{{margin:0;height:100%;overflow:hidden}}#g{{width:100vw;height:100vh}}#g svg{{width:100%;height:100%}}</style>
</head>
<body>
<div id="g">{svg}</div>
<script>
window.addEventListener("load", () => {{
  const svg = document.querySelector("#g svg");
  svg.removeAttribute("width");
  svg.removeAttribute("height");
  svgPanZoom(svg, {{controlIconsEnabled: true, fit: true, center: true, minZoom: 0.05, maxZoom: 40}});
}});
</script>
</body>
</html>
"""
        with open(output_path, "w") as f:
            f.write(page)
    elif ext in _TEXT_FORMATS:
        data = dot.pipe(format=ext[1:], encoding="utf-8")
        with open(output_path, "w") as f:
            f.write(data)
    elif ext in _BINARY_FORMATS:
        data = dot.pipe(format=ext[1:])
        with open(output_path, "wb") as f:
            f.write(data)
    else:
        supported = sorted({".html"} | _TEXT_FORMATS | _BINARY_FORMATS)
        raise SystemExit(
            f"unsupported output extension {ext!r}; choose one of {supported}"
        )

    print(f"wrote {output_path}")
    print(f"  focal: {focal_name}")
    print(f"  {n_nodes} mathematicians in the ancestor DAG")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", default="analysis/graph.pkl",
                        help="preprocessed graph from build_graph.py")
    parser.add_argument("--name", required=True,
                        help="mathematician to graph (exact or unique substring)")
    parser.add_argument("--output", default="analysis/genealogy.html",
                        help="output file; format is chosen from extension "
                             "(.html, .svg, .png, .pdf, .jpg, .gif, .dot)")
    args = parser.parse_args()

    with open(args.graph, "rb") as f:
        G = pickle.load(f)
    print(f"loaded {args.graph}: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")

    root = find_person(G, args.name)
    render(G, root, args.output)


if __name__ == "__main__":
    main()
