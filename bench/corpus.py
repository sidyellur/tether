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
        Memory("bug", "project", "auth 500s", "login returns 500 under load"),
        Memory("pref", "user", "dave orm stance", "dave distrusts the ORM layer"),
        Memory("editor", "project", "editor choice", "team standardized on neovim"),
        Memory("car", "user", "commute", "i drive my car to the office"),
    ],
    tasks=[Task("auth", ["bug", "pref"])],
    links=[],
    queries=[
        Query("why did login break", "bug", ["pref"], "graph_only"),
        Query("login returns 500 under load", "bug", ["bug"], "control"),
    ],
)


# --- SCENARIO: the real hand-authored corpus (existence proof). ---
# A fictional logistics-SaaS team ("Meridian"). Memories span incidents,
# people's preferences, decisions, and infra. TASKS group memories that were
# *used together* in one work episode; a task deliberately mixes topically
# DIFFERENT members (a bug + a person's stance + a decision) so that a
# graph_only query about one member reaches a gold that shares little
# vocabulary — reachable only through behavioral (co-recall) edges, not
# semantic similarity. Every graph_only (query, gold) pair is verified
# cos < 0.35 under the real embedder by assert_golds_far; control queries
# have gold == target (a memory v0.2 already surfaces) to prove assoc does
# not demote a good direct hit. Authored from the scenario, not tuned to the
# result.
SCENARIO = Corpus(
    name="meridian",
    memories=[
        # -- people (preferences/stances; kept trait-focused so they stay
        #    far from the technical symptom queries) --
        Memory("dave_orm", "user", "dave on ORMs",
               "Dave, the senior backend engineer, distrusts ORMs; he argues "
               "they hide the real cost of a query and insists on hand-written "
               "SQL for anything on the hot path."),
        Memory("priya_oncall", "user", "priya on launches",
               "Priya runs reliability and is fierce about on-call hygiene; "
               "she will veto any launch that ships without a rollback plan "
               "and a dashboard someone actually watches."),
        Memory("sam_a11y", "user", "sam on accessibility",
               "Sam, the frontend lead, cares most about accessibility and "
               "refuses to ship a component that a keyboard user cannot "
               "operate end to end."),
        Memory("lena_utc", "user", "lena on time",
               "Lena owns analytics and has been burned before, so she stores "
               "every timestamp in UTC and converts only at the presentation "
               "layer, never in the pipeline."),
        Memory("marco_release", "user", "marco on releases",
               "Marco, the engineering lead, is the sole approver for a "
               "production release and prefers a written checklist over "
               "tribal knowledge in someone's head."),
        Memory("nadia_trust", "user", "nadia on charges",
               "Nadia, the head of product, owns the billing roadmap and "
               "treats any double charge as a sev-one breach of customer "
               "trust, not a mere accounting slip."),
        # -- auth outage task --
        Memory("auth_bug", "project", "login 500s",
               "After the March deploy the login endpoint began returning "
               "HTTP 500 whenever traffic spiked; the database connection "
               "pool was being exhausted."),
        Memory("pool_fix", "project", "pool ceiling raised",
               "Fixed the outage by raising the connection pool ceiling and "
               "adding a short checkout timeout so callers fail fast instead "
               "of queueing forever."),
        Memory("rollback_policy", "project", "auto rollback rule",
               "Agreed to automatically undo any deploy whose error rate "
               "stays above two percent for five minutes."),
        # -- search latency task --
        Memory("search_slow", "project", "search page crawls",
               "Customers reported the shipment search page taking eight "
               "seconds; the query was doing a full scan of the orders "
               "table."),
        Memory("redis_cache", "project", "cache in front of search",
               "Placed a Redis cache ahead of shipment search with a sixty "
               "second expiry, cutting the ninety-fifth percentile from eight "
               "seconds to three hundred milliseconds."),
        Memory("index_decision", "project", "composite index added",
               "Added a composite index on customer and creation date and "
               "agreed to review slow query logs every Friday."),
        # -- onboarding task --
        Memory("new_hire_setup", "reference", "day one setup",
               "New engineers receive a laptop, repository access, and a "
               "staging account on their first day; the setup guide lives in "
               "the wiki."),
        Memory("neovim_std", "project", "shared editor config",
               "The team standardized its editor on Neovim with one shared "
               "configuration so that pair debugging sessions look identical "
               "on every machine."),
        Memory("vpn_gotcha", "reference", "staging needs vpn",
               "The staging environment is only reachable through the "
               "corporate VPN; without it every request simply times out with "
               "no useful error."),
        # -- billing migration task --
        Memory("stripe_migration", "project", "moved to stripe",
               "Migrated payments off the legacy processor onto Stripe across "
               "two weekends, keeping both paths live behind a flag."),
        Memory("idempotency_bug", "project", "double charges",
               "A retry storm charged forty customers twice because the "
               "payment call carried no idempotency key; all were refunded "
               "within the hour."),
        Memory("flag_rollout", "project", "gradual checkout rollout",
               "Rolled the new checkout to five percent of traffic first, "
               "then twenty five, then everyone, watching the refund rate at "
               "each step."),
        # -- release process task --
        Memory("ci_permissions", "project", "repository not found",
               "A release job failed with repository not found because the "
               "workflow declared a single permission, which silently set "
               "every other permission to none."),
        Memory("release_checklist", "reference", "release checklist",
               "Every release runs through a written checklist: database "
               "migrations, feature flags, a rollback plan, and a named owner "
               "who is awake."),
        Memory("staging_gate", "reference", "staging is the last gate",
               "Staging mirrors the production data shape with scrubbed "
               "personal data and is the final gate a change passes before it "
               "reaches customers."),
        # -- data pipeline task --
        Memory("etl_nightly", "project", "nightly aggregation",
               "The nightly job rolls the day's shipments up into the "
               "analytics warehouse at two in the morning."),
        Memory("timezone_bug", "project", "revenue split across days",
               "Daily revenue looked wrong because the aggregation bucketed "
               "moments near midnight by local clock, splitting a single day "
               "across two."),
        Memory("dashboard_owner", "reference", "who owns the dashboard",
               "The revenue dashboard belongs to analytics; any change to how "
               "a metric is defined needs Lena's sign off."),
        # -- distractors / density (not in any task; the noise the graph must
        #    NOT surface, and targets for extra control queries) --
        Memory("deploy_freeze", "project", "go-live freeze",
               "Deploys are frozen during the week of a major customer go "
               "live so nothing surprising ships."),
        Memory("db_backups", "reference", "backup drills",
               "Postgres backups run every six hours and are proven with a "
               "monthly restore drill."),
        Memory("api_ratelimit", "project", "public api limit",
               "The public API is limited to one hundred requests per minute "
               "per key to shield the database."),
        Memory("error_budget", "project", "availability target",
               "The service targets three nines of availability; once the "
               "error budget is spent, feature work pauses until it recovers."),
        Memory("oncall_rotation", "reference", "weekly rotation",
               "On call rotates weekly and the handoff lists open incidents "
               "and any deploy that is known to be fragile."),
        Memory("customer_northwind", "project", "biggest customer",
               "Northwind is the largest account and moves roughly a third of "
               "all shipment volume on the platform."),
        Memory("feature_flags_tool", "reference", "flags in launchdarkly",
               "Feature flags live in LaunchDarkly and stale ones are swept "
               "out every quarter."),
        Memory("secrets_vault", "reference", "secrets in the vault",
               "Secrets belong in the vault and never in an env file that "
               "gets committed to the repository."),
        Memory("mobile_offline", "project", "driver app offline",
               "The driver mobile app records shipment status while offline "
               "and reconciles once it reconnects."),
        Memory("slo_dashboard", "reference", "slo dashboard",
               "The service dashboard tracks latency, error rate, and "
               "saturation for each service in one place."),
    ],
    tasks=[
        Task("auth", ["auth_bug", "pool_fix", "rollback_policy", "dave_orm"]),
        Task("search", ["search_slow", "redis_cache", "index_decision",
                        "priya_oncall"]),
        Task("onboarding", ["new_hire_setup", "neovim_std", "vpn_gotcha",
                            "sam_a11y"]),
        Task("billing", ["stripe_migration", "idempotency_bug", "flag_rollout",
                        "nadia_trust"]),
        Task("release", ["ci_permissions", "release_checklist", "staging_gate",
                        "marco_release"]),
        Task("data", ["etl_nightly", "timezone_bug", "dashboard_owner",
                     "lena_utc"]),
    ],
    links=[
        ("auth_bug", "pool_fix"),
        ("search_slow", "redis_cache"),
        ("stripe_migration", "idempotency_bug"),
        ("etl_nightly", "timezone_bug"),
    ],
    queries=[
        # -- graph_only: technical-symptom query -> behaviorally-linked but
        #    topically-distant gold (verified cos < 0.35). ~14 candidates;
        #    pruning to the passing set happens against the real embedder. --
        Query("why did the login endpoint start returning 500 errors under "
              "heavy traffic", "auth_bug", ["dave_orm"], "graph_only"),
        Query("the shipment search page is painfully slow for our customers",
              "search_slow", ["priya_oncall"], "graph_only"),
        Query("we put a cache ahead of the slow page to speed it up",
              "redis_cache", ["priya_oncall"], "graph_only"),
        Query("duplicate charges happened while payments were being retried",
              "idempotency_bug", ["flag_rollout"], "graph_only"),
        Query("a new engineer's requests to staging all time out with no "
              "error", "vpn_gotcha", ["neovim_std"], "graph_only"),
        Query("the release job failed saying repository not found",
              "ci_permissions", ["marco_release"], "graph_only"),
        Query("a workflow permissions change quietly broke the deploy job",
              "ci_permissions", ["staging_gate"], "graph_only"),
        Query("the daily revenue numbers look wrong around midnight",
              "timezone_bug", ["dashboard_owner"], "graph_only"),
        Query("the nightly job that builds the analytics warehouse",
              "etl_nightly", ["lena_utc"], "graph_only"),
        Query("search is doing a full table scan and it is slow",
              "search_slow", ["index_decision"], "graph_only"),
        Query("we moved payments onto stripe behind a flag",
              "stripe_migration", ["nadia_trust"], "graph_only"),
        # -- control: direct-match query, gold == target (assert no demote) --
        Query("login endpoint returning 500 when the connection pool is "
              "exhausted", "auth_bug", ["auth_bug"], "control"),
        Query("shipment search page doing a full table scan taking eight "
              "seconds", "search_slow", ["search_slow"], "control"),
        Query("added a redis cache with a sixty second expiry in front of "
              "shipment search", "redis_cache", ["redis_cache"], "control"),
        Query("migrated payments to stripe behind a feature flag over two "
              "weekends", "stripe_migration", ["stripe_migration"], "control"),
        Query("a retry storm double charged customers with no idempotency "
              "key", "idempotency_bug", ["idempotency_bug"], "control"),
        Query("the team standardized its editor on neovim with a shared "
              "config", "neovim_std", ["neovim_std"], "control"),
        Query("staging is only reachable over the corporate vpn or requests "
              "time out", "vpn_gotcha", ["vpn_gotcha"], "control"),
        Query("a workflow set one permission and zeroed the rest so it failed "
              "with repository not found", "ci_permissions",
              ["ci_permissions"], "control"),
        Query("the aggregation bucketed timestamps by local clock near "
              "midnight splitting a day", "timezone_bug", ["timezone_bug"],
              "control"),
        Query("every release follows a written checklist of migrations flags "
              "and rollback", "release_checklist", ["release_checklist"],
              "control"),
        Query("postgres backups run every six hours with a monthly restore "
              "drill", "db_backups", ["db_backups"], "control"),
        Query("the public api is rate limited to one hundred requests per "
              "minute per key", "api_ratelimit", ["api_ratelimit"], "control"),
    ],
)
