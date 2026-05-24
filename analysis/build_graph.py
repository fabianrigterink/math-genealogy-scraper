"""Build a clean NetworkX DAG from MGP scraped data.

Reads data.json, drops implausible years, removes cycles via a year-based
heuristic, and writes the resulting graph to graph.pkl.

Author: Fabian Rigterink
"""
import argparse
import json
import pickle
from datetime import datetime

import networkx as nx

MIN_VALID_YEAR = 1500
MAX_VALID_YEAR = datetime.now().year + 1


def build_graph(records):
    G = nx.DiGraph()
    for r in records:
        year = r.get("year")
        if year is not None and not (MIN_VALID_YEAR <= year <= MAX_VALID_YEAR):
            year = None
        G.add_node(
            r["id"],
            name=r.get("name"),
            thesis=r.get("thesis"),
            school=r.get("school"),
            country=r.get("country"),
            year=year,
            subject=r.get("subject"),
        )
    for r in records:
        rid = r["id"]
        for advisor in r.get("advisors") or []:
            G.add_edge(advisor, rid)
        for student in r.get("students") or []:
            G.add_edge(rid, student)
    return G


def _year_violation(G, u, v):
    yu, yv = G.nodes[u].get("year"), G.nodes[v].get("year")
    if yu is None or yv is None:
        return 0
    return yu - yv


def remove_cycles(G):
    self_loops = list(nx.selfloop_edges(G))
    G.remove_edges_from(self_loops)
    removed = []
    for comp in list(nx.strongly_connected_components(G)):
        if len(comp) <= 1:
            continue
        sub = G.subgraph(comp).copy()
        while not nx.is_directed_acyclic_graph(sub):
            cycle = next(iter(nx.simple_cycles(sub, length_bound=len(comp))))
            edges = [(cycle[i], cycle[(i + 1) % len(cycle)]) for i in range(len(cycle))]
            worst = max(edges, key=lambda e: _year_violation(G, *e))
            sub.remove_edge(*worst)
            G.remove_edge(*worst)
            removed.append(worst)
    return len(self_loops), removed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data.json")
    parser.add_argument("--output", default="analysis/graph.pkl")
    args = parser.parse_args()

    print(f"loading {args.input}")
    with open(args.input) as f:
        records = json.load(f)["nodes"]

    G = build_graph(records)
    print(f"raw:   {G.number_of_nodes():>8,} nodes  {G.number_of_edges():>8,} edges")

    n_self, removed = remove_cycles(G)
    print(f"removed {n_self} self-loops and {len(removed)} cycle-breaking edges:")
    for u, v in removed:
        au = G.nodes[u].get("name") or f"#{u}"
        av = G.nodes[v].get("name") or f"#{v}"
        print(f"   {au} -> {av}")

    print(f"clean: {G.number_of_nodes():>8,} nodes  {G.number_of_edges():>8,} edges")
    assert nx.is_directed_acyclic_graph(G), "graph still has cycles after cleanup"

    with open(args.output, "wb") as f:
        pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
