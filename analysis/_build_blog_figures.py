"""Regenerate the 8 blog figures from analysis/graph.pkl.

Mirrors the relevant cells of analysis/_build_eda.py so the figures can be
refreshed without re-executing the full notebook. Writes PNGs to
analysis/blog/figures/.

Author: Fabian Rigterink
"""
import pickle
import re
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import networkx as nx
import numpy as np
import pandas as pd
import powerlaw
import seaborn as sns

# ---------------------------------------------------------------------------
# Paths and styling (match _build_eda.py)
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).parent
GRAPH_PATH = SCRIPT_DIR / "graph.pkl"
OUT_DIR = SCRIPT_DIR / "blog" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="ticks", context="notebook")
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False


def save(name):
    path = OUT_DIR / f"{name}.png"
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  wrote {path.relative_to(SCRIPT_DIR.parent)}")


# ---------------------------------------------------------------------------
# Load graph and precompute the shared structures the notebook uses
# ---------------------------------------------------------------------------

print(f"loading {GRAPH_PATH}")
with open(GRAPH_PATH, "rb") as f:
    G = pickle.load(f)
print(f"  {G.number_of_nodes():,} nodes  /  {G.number_of_edges():,} edges")

df = pd.DataFrame(
    [{"id": n, **a} for n, a in G.nodes(data=True)]
).set_index("id")

# Giant WCC
giant_wcc = max(nx.weakly_connected_components(G), key=len)

# Topological order + transitive descendant sets
topo = list(nx.topological_sort(G))
desc_sets = {}
for v in reversed(topo):
    s = {v}
    for child in G.successors(v):
        s |= desc_sets[child]
    desc_sets[v] = s

# Branch depths (max downward depth from each node)
branch_depths = pd.Series({v: 0 for v in G.nodes()}, dtype=int)
for v in reversed(topo):
    children_depths = [branch_depths[c] for c in G.successors(v)]
    if children_depths:
        branch_depths[v] = 1 + max(children_depths)

out_deg = pd.Series(dict(G.out_degree()))


# ---------------------------------------------------------------------------
# 1. longest_chain.png  (notebook §4.3)
# ---------------------------------------------------------------------------

print("\n[1/8] longest_chain")
longest = nx.dag_longest_path(G)
chain = pd.DataFrame(
    [{"step": i, **G.nodes[nid]} for i, nid in enumerate(longest)]
)[["step", "name", "year", "school", "country"]]

fig, ax = plt.subplots(figsize=(11, 18))
ys = np.arange(len(chain))[::-1]
ax.scatter([0] * len(chain), ys, s=60, color="#2b5d8b", zorder=3)
ax.plot([0] * len(chain), ys, color="#2b5d8b", linewidth=1.5, alpha=0.5, zorder=2)
for i, (_, row) in enumerate(chain.iterrows()):
    y_pos = ys[i]
    year_str = f"{int(row['year'])}" if pd.notna(row["year"]) else "  ?  "
    ax.text(-0.02, y_pos, year_str, va="center", ha="right", fontsize=9,
            color="#666", family="monospace")
    ax.text(0.04, y_pos, row["name"], va="center", fontsize=10)
ax.set_ylim(-1, len(chain))
ax.set_xlim(-0.18, 1.5)
ax.set_title(f"The longest advisor → student chain in MGP ({len(chain)} generations)",
             fontsize=12, pad=15)
ax.axis("off")
save("longest_chain")


# ---------------------------------------------------------------------------
# 2. country_phd_series.png  (notebook §13)
# ---------------------------------------------------------------------------

print("[2/8] country_phd_series")
def country_series(c, y_min=1900, y_max=2014):
    s = df[(df["country"] == c) & df["year"].notna()].copy()
    s = s[(s["year"] >= y_min) & (s["year"] <= y_max)]
    return s["year"].astype(int).value_counts().sort_index().reindex(
        range(y_min, y_max + 1), fill_value=0
    )

countries = ["Germany", "France", "UnitedKingdom", "Russia"]
fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
for ax, c in zip(axes.flat, countries):
    s = country_series(c).rolling(5, center=True).mean()
    ax.plot(s.index, s.values, linewidth=1.8, color="#2b5d8b")
    for start, end, _ in [(1914, 1918, "WWI"), (1939, 1945, "WWII"),
                          (1933, 1939, "Nazi")]:
        ax.axvspan(start, end, color="red", alpha=0.10)
    ax.set_title(c)
    ax.set_ylabel("PhDs / yr (5-yr MA)")
    ax.set_xlim(1900, 2014)
plt.tight_layout()
save("country_phd_series")


# ---------------------------------------------------------------------------
# 3. subject_inheritance.png  (notebook §9.4)
# ---------------------------------------------------------------------------

print("[3/8] subject_inheritance")
def clean_subject(s):
    if not isinstance(s, str):
        return None
    return re.sub(r"^\d{2}\s*[—\-]?\s*", "", s)

def msc_code(s):
    if not isinstance(s, str):
        return None
    m = re.match(r"(\d{2})", s)
    return m.group(1) if m else None

df["msc"] = df["subject"].map(msc_code)
df["subject_clean"] = df["subject"].map(clean_subject)

edges = pd.DataFrame(list(G.edges()), columns=["adv", "stu"])
edges["adv_msc"] = edges["adv"].map(df["msc"])
edges["stu_msc"] = edges["stu"].map(df["msc"])
both = edges.dropna(subset=["adv_msc", "stu_msc"])

top12_msc = df["msc"].value_counts().head(12).index.tolist()
both12 = both[both["adv_msc"].isin(top12_msc) & both["stu_msc"].isin(top12_msc)]
trans = (both12.groupby(["adv_msc", "stu_msc"]).size()
         .unstack(fill_value=0)
         .reindex(index=top12_msc, columns=top12_msc, fill_value=0))
trans_norm = trans.div(trans.sum(axis=1), axis=0)

msc_label = (df.dropna(subset=["msc"])
               .groupby("msc")["subject_clean"]
               .first()
               .reindex(top12_msc))
nice = [f"{m} — {(s or '')[:25]}" for m, s in zip(top12_msc, msc_label)]

fig, ax = plt.subplots(figsize=(11, 8))
sns.heatmap(trans_norm, annot=True, fmt=".0%", cmap="rocket_r", ax=ax,
            xticklabels=nice, yticklabels=nice)
ax.set_xlabel("student MSC")
ax.set_ylabel("advisor MSC")
ax.set_title("P(student subject | advisor subject) — diagonal = inheritance")
plt.tight_layout()
save("subject_inheritance")


# ---------------------------------------------------------------------------
# 4. fertility_lognormal.png  (notebook §10.2)
# ---------------------------------------------------------------------------

print("[4/8] fertility_lognormal  (powerlaw fit may take a few seconds)")
fertile = out_deg[out_deg >= 1].values
fit = powerlaw.Fit(fertile, discrete=True, verbose=False)

fig, ax = plt.subplots(figsize=(9, 5.5))
fit.plot_ccdf(ax=ax, color="black", marker=".", linewidth=0, label="empirical")
fit.power_law.plot_ccdf(ax=ax, color="#c44e52", linestyle="--", linewidth=2,
                        label=f"power-law fit (α = {fit.alpha:.2f}, x_min = {fit.xmin:.0f})")
fit.lognormal.plot_ccdf(ax=ax, color="#55a868", linestyle=":", linewidth=2,
                        label="lognormal fit")
ax.set_xlabel("# students k")
ax.set_ylabel("P(K ≥ k)")
ax.set_title("Advisor fertility: empirical CCDF vs. fitted distributions")
ax.legend()
plt.tight_layout()
save("fertility_lognormal")


# ---------------------------------------------------------------------------
# 5. princeton_vs_gottingen.png  (notebook §23.1)
# ---------------------------------------------------------------------------

print("[5/8] princeton_vs_gottingen")
princeton = df[df["school"] == "Princeton University"].dropna(subset=["year"])
gottingen = df[df["school"] == "Georg-August-Universität Göttingen"].dropna(subset=["year"])

p_by_decade = (princeton["year"].astype(int)
               .pipe(lambda s: (s // 10) * 10)
               .value_counts().sort_index())
g_by_decade = (gottingen["year"].astype(int)
               .pipe(lambda s: (s // 10) * 10)
               .value_counts().sort_index())
years_all = pd.Index(range(1850, 2020, 10))
p_by_decade = p_by_decade.reindex(years_all, fill_value=0)
g_by_decade = g_by_decade.reindex(years_all, fill_value=0)

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(years_all, g_by_decade.values, "o-", linewidth=2,
        label="Göttingen", color="#c44e52")
ax.plot(years_all, p_by_decade.values, "s-", linewidth=2,
        label="Princeton", color="#2b5d8b")
ax.axvspan(1933, 1945, color="red", alpha=0.08)
ax.set_xlabel("decade")
ax.set_ylabel("# PhDs awarded")
ax.set_title("Princeton vs Göttingen — math PhD output by decade")
ax.legend()
plt.tight_layout()
save("princeton_vs_gottingen")


# ---------------------------------------------------------------------------
# 6. mathematical_eve.png  (notebook §25.2)
# ---------------------------------------------------------------------------

print("[6/8] mathematical_eve")
modern = set(df[df["year"].between(2000, 2014)].index)
modern_giant = modern & giant_wcc

candidates = []
threshold = len(modern_giant) * 0.3
for v, ds in desc_sets.items():
    if len(ds) < threshold:
        continue
    n_modern = len(modern_giant & ds)
    if n_modern == 0:
        continue
    candidates.append({
        "id": v,
        "name": G.nodes[v].get("name") or f"#{v}",
        "year": G.nodes[v].get("year"),
        "total_descendants": len(ds),
        "frac_of_moderns": n_modern / len(modern_giant),
    })
cdf = pd.DataFrame(candidates).sort_values("frac_of_moderns", ascending=False)
known_year = cdf.dropna(subset=["year"]).copy()
known_year["year"] = known_year["year"].astype(int)

fig, ax = plt.subplots(figsize=(12, 5.5))
ax.scatter(known_year["year"], known_year["frac_of_moderns"] * 100,
           s=np.clip(known_year["total_descendants"] / 1000, 30, 600),
           alpha=0.55, color="#2b5d8b", edgecolors="none")
top_label = known_year.nlargest(8, "frac_of_moderns")
for _, row in top_label.iterrows():
    ax.annotate(row["name"], (row["year"], row["frac_of_moderns"] * 100),
                xytext=(8, 4), textcoords="offset points", fontsize=9, alpha=0.8)
ax.set_xlabel("year of ancestor (known)")
ax.set_ylabel("% of modern PhDs descending from this ancestor")
ax.set_title("Mathematical 'broadness' over time — coverage of moderns by single ancestor")
ax.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))
ax.axhline(50, color="red", linestyle=":", alpha=0.5, label="50% of moderns")
ax.legend()
plt.tight_layout()
save("mathematical_eve")


# ---------------------------------------------------------------------------
# 7. wwii_branch_geography.png  (notebook §26.2)
# ---------------------------------------------------------------------------

print("[7/8] wwii_branch_geography")
german_1933 = df[(df["country"] == "Germany") & df["year"].between(1900, 1933)].index

records = []
for nid in german_1933:
    students = list(G.successors(nid))
    students_post33 = [s for s in students
                       if pd.notna(df.at[s, "year"]) and df.at[s, "year"] > 1933]
    if len(students_post33) < 3:
        continue
    s_countries = [df.at[s, "country"] for s in students_post33]
    non_german = sum(1 for c in s_countries if isinstance(c, str) and c != "Germany")
    records.append({
        "id": nid,
        "frac_non_german_post33": non_german / len(students_post33),
        "total_descendants": len(desc_sets[nid]),
    })
records = pd.DataFrame(records)
records["group"] = np.where(records["frac_non_german_post33"] >= 0.5,
                            "Emigrated (≥50% non-DE)", "Stayed (<50% non-DE)")

emig_advisors = records[records["group"].str.startswith("Emigrated")]["id"].tolist()
stay_advisors = records[records["group"].str.startswith("Stayed")]["id"].tolist()

emig_desc = set()
for a in emig_advisors:
    emig_desc |= desc_sets[a]
stay_desc = set()
for a in stay_advisors:
    stay_desc |= desc_sets[a]

emig_c = df.loc[list(emig_desc), "country"].value_counts(normalize=True).head(6)
stay_c = df.loc[list(stay_desc), "country"].value_counts(normalize=True).head(6)

fig, ax = plt.subplots(figsize=(11, 5))
top_countries_combined = pd.Index(set(emig_c.index) | set(stay_c.index))
emig_aligned = emig_c.reindex(top_countries_combined, fill_value=0)
stay_aligned = stay_c.reindex(top_countries_combined, fill_value=0)
order = (emig_aligned + stay_aligned).sort_values(ascending=False).index
x = np.arange(len(order))
w = 0.4
ax.bar(x - w/2, emig_aligned.reindex(order) * 100, w, label="Emigrant branch", color="#c44e52")
ax.bar(x + w/2, stay_aligned.reindex(order) * 100, w, label="Stayer branch", color="#2b5d8b")
ax.set_xticks(x)
ax.set_xticklabels(order, rotation=30, ha="right")
ax.set_ylabel("% of branch's descendants (PhD country)")
ax.set_title("Where the descendants of each branch ended up")
ax.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))
ax.legend()
plt.tight_layout()
save("wwii_branch_geography")


# ---------------------------------------------------------------------------
# 8. galton_watson_vs_mgp.png  (notebook §29)
# ---------------------------------------------------------------------------

print("[8/8] galton_watson_vs_mgp")
cohort = df[df["year"].between(1900, 1950)].index
empirical_off = pd.Series([G.out_degree(n) for n in cohort])

off_values = empirical_off.values
off_probs = np.bincount(off_values) / len(off_values)
off_support = np.arange(len(off_probs))

RNG_SIM = np.random.default_rng(42)
def simulate_one(max_gens=20):
    alive = 1
    max_depth = 0
    for g in range(1, max_gens + 1):
        if alive == 0:
            break
        sampled = RNG_SIM.choice(off_support, size=alive, p=off_probs)
        new_alive = int(sampled.sum())
        if new_alive > 0:
            max_depth = g
        alive = new_alive
        if alive > 100_000:
            alive = 100_000
    return max_depth

N_SIM = 10_000
sim_depths = np.array([simulate_one() for _ in range(N_SIM)])

sim = pd.Series(sim_depths).value_counts(normalize=True).sort_index()
emp = pd.Series(branch_depths.loc[cohort]).value_counts(normalize=True).sort_index()
max_depth = int(max(sim.index.max(), emp.index.max()))
sim = sim.reindex(range(max_depth + 1), fill_value=0)
emp = emp.reindex(range(max_depth + 1), fill_value=0)

fig, ax = plt.subplots(figsize=(11, 5))
x = np.arange(max_depth + 1)
w = 0.4
ax.bar(x - w/2, emp.values * 100, w, label="MGP empirical (cohort 1900-1950)", color="#c44e52")
ax.bar(x + w/2, sim.values * 100, w, label="Galton-Watson simulated", color="#2b5d8b")
ax.set_xlabel("lineage depth")
ax.set_ylabel("% of starting mathematicians reaching this depth")
ax.set_title("Lineage depth: MGP vs. Galton-Watson branching process")
ax.legend()
ax.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))
plt.tight_layout()
save("galton_watson_vs_mgp")

print("\ndone — 8 figures written to", OUT_DIR.relative_to(SCRIPT_DIR.parent))
