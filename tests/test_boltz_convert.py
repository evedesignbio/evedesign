from pathlib import Path

import pytest

from evedesign.models.boltz.convert import _write_a3m, _write_csv
from evedesign.sequence import Sequence, Sequences
from evedesign.system import Entity, EntityInstance

# convert.py imports pyyaml, which only ships in the boltz2fold extras
pytestmark = pytest.mark.boltz2fold


def _read_a3m(path: Path) -> list[tuple[str, str]]:
    """Return [(header, seq), ...] from an A3M file (query first, then hits)."""
    records = []
    header = None
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith(">"):
                header = line[1:]
            elif header is not None:
                records.append((header, line))
                header = None
    return records


def _entity_with_msa(rep: str, hits: list[Sequence]) -> Entity:
    return Entity(
        type="protein",
        rep=rep,
        id="query",
        first_index=1,
        sequences=Sequences(seqs=hits, aligned=True, format="a3m"),
    )


def test_write_a3m_old_query_none_writes_verbatim(tmp_path):
    # Back-compat: no old_query -> hits written exactly as stored
    entity = _entity_with_msa("ALCD", [Sequence("ALSD", id="seq1"), Sequence("ALCE", id="seq2")])
    instance = EntityInstance(rep="VICD")
    out = _write_a3m(entity, instance, tmp_path / "msa" / "A.a3m", old_query=None)

    records = _read_a3m(out)
    assert records[0] == ("query", "VICD")          # query line is the instance rep
    assert [r[1] for r in records[1:]] == ["ALSD", "ALCE"]  # hits unchanged


def test_write_a3m_old_query_equals_instance_no_remap(tmp_path):
    # old_query == instance rep -> remap skipped, hits verbatim
    entity = _entity_with_msa("ALCD", [Sequence("ALSD", id="seq1"), Sequence("ALCE", id="seq2")])
    instance = EntityInstance(rep="ALCD")
    out = _write_a3m(entity, instance, tmp_path / "msa" / "A.a3m", old_query=entity.rep)

    records = _read_a3m(out)
    assert records[0] == ("query", "ALCD")
    assert [r[1] for r in records[1:]] == ["ALSD", "ALCE"]


def test_write_a3m_remap_applied_on_deletion(tmp_path):
    # old_query differs (deletion at column 1) -> hits remapped, column dropped
    entity = _entity_with_msa("ALCD", [Sequence("ALSD", id="seq1"), Sequence("ALCE", id="seq2")])
    instance = EntityInstance(rep="V-CD")
    out = _write_a3m(entity, instance, tmp_path / "msa" / "A.a3m", old_query=entity.rep)

    records = _read_a3m(out)
    # query line is the normalized designed sequence: gap stripped
    assert records[0] == ("query", "VCD")
    # hits remapped: column 1 removed from each
    assert [r[1] for r in records[1:]] == ["ASD", "ACE"]


def test_write_a3m_remap_applied_on_insertion(tmp_path):
    # insertion (lowercase 't') -> gap column added to each hit
    entity = _entity_with_msa("ALCD", [Sequence("ALSD", id="seq1"), Sequence("ALCE", id="seq2")])
    instance = EntityInstance(rep="VICtD")
    out = _write_a3m(entity, instance, tmp_path / "msa" / "A.a3m", old_query=entity.rep)

    records = _read_a3m(out)
    # query line normalized: lowercase insertion uppercased
    assert records[0] == ("query", "VICTD")
    assert [r[1] for r in records[1:]] == ["ALS-D", "ALC-E"]


def test_write_a3m_remap_failure_falls_back_verbatim(tmp_path):
    # Malformed input: hit length (3) != len(old_query) (4) -> remap_query raises
    # ValueError; _write_a3m must fall back to verbatim and log a warning.
    entity = _entity_with_msa("ALCD", [Sequence("ALS", id="seq1")])
    instance = EntityInstance(rep="VICD")

    import loguru

    messages: list[str] = []
    handler_id = loguru.logger.add(lambda m: messages.append(str(m)), level="WARNING")
    try:
        out = _write_a3m(entity, instance, tmp_path / "msa" / "A.a3m", old_query=entity.rep)
    finally:
        loguru.logger.remove(handler_id)

    records = _read_a3m(out)
    assert records[0] == ("query", "VICD")
    assert [r[1] for r in records[1:]] == ["ALS"]  # unmodified hit
    assert any("could not remap MSA" in m for m in messages)


def test_write_csv_real_mmseqs_pair_keys(tmp_path):
    # Thomas's producer emits "pair-0", "pair-1", ... in order. First-appearance
    # mapping must reproduce the old strip-"pair-" behavior: taxonomy_ids 0, 1.
    hits = [
        Sequence("MAST", id="p0", key="pair-0"),
        Sequence("VAST", id="p1", key="pair-1"),
        Sequence("GAST", id="u0", key=None),      # unpaired
    ]
    entity = _entity_with_msa("MAST", hits)
    instance = EntityInstance(rep="MAST")
    out = _write_csv(entity, instance, tmp_path / "msa" / "A.csv")

    rows = out.read_text().splitlines()
    assert rows == [
        "key,sequence",
        "0,MAST",      # query first
        "0,MAST",      # hit p0: pair-0 -> taxid 0
        "1,VAST",      # hit p1: pair-1 -> taxid 1
        "-1,GAST",     # unpaired -> -1
    ]


def test_write_csv_is_key_format_agnostic(tmp_path):
    # Arbitrary (non-"pair-") key values: two sequences from the same organism
    # share a key -> same taxonomy_id; a different organism -> different id;
    # key=None -> -1. The consumer never inspects the key's format.
    hits = [
        Sequence("MAST", id="a1", key="orgA"),
        Sequence("VAST", id="a2", key="orgA"),   # same key as a1 -> same taxid
        Sequence("GAST", id="b1", key="orgB"),   # different key -> different taxid
        Sequence("KAST", id="u0", key=None),     # unpaired
    ]
    entity = _entity_with_msa("MAST", hits)
    instance = EntityInstance(rep="MAST")
    out = _write_csv(entity, instance, tmp_path / "msa" / "A.csv")

    rows = out.read_text().splitlines()
    assert rows[0] == "key,sequence"
    assert rows[1] == "0,MAST"                   # query first
    a1_tax, a1_seq = rows[2].split(",")
    a2_tax, _ = rows[3].split(",")
    b1_tax, _ = rows[4].split(",")
    assert a1_tax == a2_tax                       # shared key -> shared taxid
    assert b1_tax != a1_tax                        # distinct key -> distinct taxid
    assert rows[5] == "-1,KAST"                   # unpaired -> -1
    # taxonomy_ids are stable integers by first appearance
    assert {a1_tax, b1_tax} == {"0", "1"}


def test_write_csv_remaps_and_preserves_pairing(tmp_path):
    hits = [
        Sequence("MAST", id="p0", key="org"),
        Sequence("VAST", id="p1", key="org"),
        Sequence("GKST", id="u", key=None),
    ]
    entity = _entity_with_msa("MAST", hits)
    instance = EntityInstance(rep="M-ST")
    out = _write_csv(entity, instance, tmp_path / "msa" / "A.csv", old_query=entity.rep)

    rows = out.read_text().splitlines()
    assert rows == [
        "key,sequence",
        "0,MST",
        "0,MST",
        "0,VST",
        "-1,GST",
    ]
