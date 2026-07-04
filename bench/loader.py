"""Load a Corpus into a Store, returning a corpus-key -> db-id map."""


def load(corpus, store):
    id_of = {}
    for m in corpus.memories:
        rec = store.remember(m.type, m.title, m.body)
        id_of[m.key] = rec["id"]
    for (a, b) in corpus.links:
        store.link(id_of[a], id_of[b])
    return id_of
