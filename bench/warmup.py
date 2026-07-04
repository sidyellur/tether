"""Warm the usage graph through the REAL recall path: replay each task as a
session of recalls so its members co-occur and get Hebbian-wired by
touch_session. No hand-wiring — this is the mechanism under test."""


def warm(corpus, store, id_of, repeats=3):
    title_of = {m.key: m.title for m in corpus.memories}
    for _ in range(repeats):
        for task in corpus.tasks:
            sid = f"warm:{task.key}"
            for mkey in task.member_keys:
                # a query that directly surfaces this member, in the task
                # session, so all members land in the session working set.
                store.recall(title_of[mkey], session=sid)
