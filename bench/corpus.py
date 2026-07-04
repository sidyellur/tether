"""Corpus data model + the MINI hermetic fixture.

A corpus is authored with string `key`s (stable, human-readable); the loader
maps keys -> db ids at load time. Two query classes:
  - graph_only: gold is relevant but semantically FAR from the query
    (reachable only via behavioral edges) -> measures UPSIDE.
  - control:    gold IS the directly-matched target that v0.2 ranks well
    -> measures NO-REGRESSION (assoc must not demote it)."""
from dataclasses import dataclass, field


@dataclass
class Memory:
    key: str
    type: str
    title: str
    body: str


@dataclass
class Task:
    key: str
    member_keys: list


@dataclass
class Query:
    query: str
    target_key: str
    gold_keys: list
    kind: str  # "graph_only" | "control"


@dataclass
class Corpus:
    name: str
    memories: list
    tasks: list
    links: list = field(default_factory=list)   # list of (key_a, key_b)
    queries: list = field(default_factory=list)

    def by_kind(self, kind):
        return [q for q in self.queries if q.kind == kind]

    def member_of_task(self, mem_key):
        out = set()
        for t in self.tasks:
            if mem_key in t.member_keys:
                out.update(t.member_keys)
        out.discard(mem_key)
        return out


# --- MINI: hermetic fixture. Keep tiny; the FakeEmbedder makes cosine trivial,
# so "far" here just means "distinct key vectors", not real semantics. ---
MINI = Corpus(
    name="mini",
    memories=[
        Memory("bug", "note", "auth 500s", "login returns 500 under load"),
        Memory("pref", "note", "dave orm stance", "dave distrusts the ORM layer"),
        Memory("editor", "note", "editor choice", "team standardized on neovim"),
        Memory("car", "note", "commute", "i drive my car to the office"),
    ],
    tasks=[Task("auth", ["bug", "pref"])],
    links=[],
    queries=[
        Query("why did login break", "bug", ["pref"], "graph_only"),
        Query("login returns 500 under load", "bug", ["bug"], "control"),
    ],
)
