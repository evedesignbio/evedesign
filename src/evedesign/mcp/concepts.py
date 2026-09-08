"""
Curated explanations of evedesign's core concepts, for the MCP server's explain_concept tool.
"""
from __future__ import annotations

TOPICS: dict[str, dict[str, str]] = {
    "system_and_entities": {
        "title": "System and Entity",
        "source": "evedesign/system.py",
        "content": """
A System is an ordered list of Entity objects: the fixed, problem-level description of what
is being designed (one entity per protein/DNA/RNA chain or ligand), independent of any
particular realized sequence. Prefer the Protein, DNA, RNA and Ligand subclasses over the
generic Entity class directly.

Key Entity fields:
type
    "protein", "dna", "rna" or "ligand"
rep
    Representative/reference sequence of the entity (ex. wild-type), or None if not fixed
first_index
    One-based numbering offset of the first residue, required for biopolymers. All position
    arguments throughout the API (Mutation.pos, fixed_pos, single_mutation_scan, ...) are in
    this numbering, *not* zero-based array indices; subtract entity.first_index before
    indexing into a raw array
sequences
    MSA set attached to an entity (used by ex. EVmutation2, EVE, EVcouplings, ...)
structures
    StructureChainMap of known structures for the entity
min_length, max_length, copies, symmetry, residue_bias, cyclic
    Additional design constraints imposed on an entity

System and Entity both provide serialize()/deserialize() to and from a JSON-compatible dict.
""",
    },
    "instances_and_representations": {
        "title": "SystemInstance and EntityInstance",
        "source": "evedesign/system.py",
        "content": """
Where System/Entity describe the design problem, SystemInstance (a list of EntityInstance)
describes one concrete realization (a specific sequence, its embedding, and any structural
models, in the same entity order as its parent System).

Key EntityInstance fields:
rep
    Realized sequence, as a numpy array of dtype 'U1' (or a plain str).
    Encodes insertions and deletions relative to the Entity's rep, see
    mutations_and_indels.
embedding
    Per-residue (2D) or per-entity (1D) numpy array produced by a Transformer
models
    StructureChainMap of structural models for this specific instance
score, confidence
    Set by Scorer/Transformer calls, not part of the constructor

Note: System.rep_to_instance() builds the trivial SystemInstance matching each entity's own
rep (the "wild-type" instance); use it instead of hand-building an instance when the entity
already carries a full representative sequence, as shown in
examples/scoring_and_transforming_instances.

Both classes support copy() (shallow: rep/embedding/models point to the same underlying
arrays) and serialize()/deserialize(). Every model interface method that returns instances
(score(), transform(), generate()) must return copies rather than mutate its input.
""",
    },
    "mutations_and_indels": {
        "title": "Mutation, Mutant and insertion/deletion coding",
        "source": "evedesign/system.py",
        "content": """
A Mutation is a NamedTuple (entity, pos, ref, to); a Mutant is Sequence[Mutation] (one or
more mutations applied together, e.g. for a double mutant). All mutations in a Mutant are
specified relative to the sequence before any of them are applied. Positions DO NOT shift to
account for earlier insertions/deletions in the same mutant.

Encoding conventions, shared between Mutation.to and EntityInstance.rep:
1. Deletion: to = GAP (the GAP symbol from evedesign.constants).
2. Insertion: ref = "", to = the lowercase insert symbols returned by
   Entity.alphabet(include_inserts=True), positioned directly after the referenced pos. An
   insertion before the first residue uses pos = entity.first_index - 1.

This follows the same convention as the A3M alignment format. EntityInstance.normalized_rep() 
strips this coding back to a plain uppercase, gap-free sequence.

A model's handles_deletions and handles_insertions flags declare whether it can interpret
this coding at all; requires_fixed_length=True implies handles_insertions=False (a model
like LigandMPNN or ESM2 that requires fixed-length input cannot model indels).
""",
    },
    "scoring_workflow": {
        "title": "score() vs score_conditional() vs score_mutants() vs single_mutation_scan()",
        "source": "evedesign/model.py",
        "content": """
Four related but distinct ways to get a number out of a Scorer-family model. Pick the
narrowest that fits:

score(instances) (Scorer)
    Most general. Raw, unnormalized logits/likelihoods, one per whole instance. Comparable
    within one call, not necessarily across separate calls; include a reference/wild-type
    instance in the batch if normalization is needed.
score_mutants(instance, mutants) (MutationScorer)
    Scores for a list of (possibly higher-order) mutants of one instance, relative to that
    instance (self-mutation scores 0, beneficial > 0, damaging < 0). Falls back to calling
    score() on mutant and reference and subtracting, if a model has no more specialized
    implementation.
single_mutation_scan(instance, entity=None, positions=None) (MutationScorer)
    All single substitutions across one instance in one call, returned as a dataframe
    indexed by entity/pos/ref triplets by symbol columns, the standard "mutation matrix"
    shape. Prefer this over looping score_mutants() per position.
score_conditional(instances, entities, positions) (ConditionalMutationScorer)
    Batches one position's substitutions across many different instances (e.g. one Gibbs
    sampling sweep step across a whole population), the transpose of
    single_mutation_scan's batching axis. Not normalized to any reference-values are
    comparable only across symbols at the same instance/entity/position row.

All four return or accept Mutation/Mutant objects and dataframes whose column order follows
Entity.alphabet() (or Entity.merge_alphabet_symbols() when entities have different
alphabets). Transformer.transform() is a separate operation for mapping instances to a
different representation (ex. sequence to embedding, or to predicted structure); it may
also set the score attribute as a byproduct when a model can compute both in one pass.
""",
    },
    "adding_a_new_model": {
        "title": "Implementing a new model",
        "source": "evedesign/model.py, README.md",
        "content": """
A new model is a class inheriting BaseModel plus whichever of Generator, Scorer,
Transformer, MutationScorer and ConditionalMutationScorer it implements (most concrete
classes implement several; EVmutation2 implements all five). Call list_interfaces() for the
exact abstract members each of those requires.

Conventions to follow, all visible in the existing reference implementations (EVmutation2,
ESM-2, LigandMPNN and the Gibbs sampler are a decently representative set):
1. name, citations, and the requires_*/handles_*/supports_* flags from _Core are almost
   always plain class attributes (requires_gpu: bool = False, ...), not computed
   properties. This is what lets list_models() describe a class without instantiating it.
2. If the model depends on an optional package (torch, a specific extra, an external CLI),
   guard the import in a module-level try/except ImportError block and expose
   available = IMPORT_AVAILABLE on the class, then check self.available/self.ready before
   doing real work. See evedesign.models.esm2 or evedesign.codons.dnachisel.
3. build(system, data, status_callback=None) must call self.can_model_or_raise(system, data),
   assign self.system, and return self for chaining.
4. score()/transform()/generate() must return copies of the input instances (a shallow copy
   via .copy() is sufficient; do not mutate the caller's instances), with score and
   confidence set.
5. If the model can plausibly speed up score_conditional() or score_mutants() beyond the
   default mixin implementation in ConditionalMutationScorer/MutationScorer (e.g. by
   batching all substitutions for a position into one forward pass), override it. The
   docstrings on those default implementations explain exactly what a specialized version
   needs to preserve.

Once implemented, add the class to the appropriate table in the top-level README.md, and to
evedesign.mcp.registry.EXTRAS_BY_CLASS if it needs an optional dependency, so this MCP
server's discovery tools stay accurate.
""",
    },
}


def list_topics() -> list[dict[str, str]]:
    """
    List available explain_concept topics

    Returns
    -------
    Topic, title and source file for each entry
    """
    return [{"topic": key, "title": value["title"], "source": value["source"]} for key, value in TOPICS.items()]


def get(topic: str) -> dict[str, str]:
    """
    Look up one topic by name

    Parameters
    ----------
    topic
        Topic key, as returned by list_topics()

    Returns
    -------
    Topic, title, source file and content

    Raises
    ------
    KeyError
        If topic is not a known topic
    """
    try:
        entry = TOPICS[topic]
    except KeyError:
        raise KeyError(topic) from None
    return {"topic": topic, **entry, "content": entry["content"].strip()}
