import pytest

from evedesign.sequence import Sequence, Sequences


def _base_sequences():
    hits = [Sequence("ALSD", id="seq1"), Sequence("ALCE", id="seq2")]
    return Sequences(seqs=hits, aligned=True, format="a3m")


def test_remap_query_substitution_only():
    # new_query differs only by substitutions -> hits unchanged
    sequences = _base_sequences()
    result = sequences.remap_query("ALCD", "VICD", prepend_new_query=False)
    assert [s.seq for s in result.seqs] == ["ALSD", "ALCE"]


def test_remap_query_deletion():
    # deletion at column 1 -> that column dropped from every hit
    sequences = _base_sequences()
    result = sequences.remap_query("ALCD", "V-CD", prepend_new_query=False)
    assert [s.seq for s in result.seqs] == ["ASD", "ACE"]


def test_remap_query_insertion():
    # insertion (lowercase 't') between cols 2 and 3 -> gap column added to every hit
    sequences = _base_sequences()
    result = sequences.remap_query("ALCD", "VICtD", prepend_new_query=False)
    assert [s.seq for s in result.seqs] == ["ALS-D", "ALC-E"]


def test_remap_query_preserves_format_and_ids():
    sequences = _base_sequences()
    result = sequences.remap_query("ALCD", "VICtD", prepend_new_query=False)
    assert result.format_ == "a3m"
    assert [s.id_ for s in result.seqs] == ["seq1", "seq2"]


def test_remap_query_unsupported_format():
    hits = [Sequence("ALSD", id="seq1")]
    sequences = Sequences(seqs=hits, aligned=False, format="fasta")
    with pytest.raises(NotImplementedError):
        sequences.remap_query("ALCD", "VICD", prepend_new_query=False)


def test_remap_query_wrong_column_count():
    # new_query consumes 5 alignment columns but old_query has 4
    sequences = _base_sequences()
    with pytest.raises(ValueError):
        sequences.remap_query("ALCD", "VICDE", prepend_new_query=False)


def test_remap_query_hit_length_mismatch():
    # hit length (3) doesn't match len(old_query) (4)
    hits = [Sequence("ALS", id="seq1")]
    sequences = Sequences(seqs=hits, aligned=True, format="a3m")
    with pytest.raises(ValueError):
        sequences.remap_query("ALCD", "VICD", prepend_new_query=False)


# lowercase (A3M insert-state) hits; old_query = "ALCD" (4 match columns)

def _lc(hit):
    return Sequences(seqs=[Sequence(hit, id="h")], aligned=True, format="a3m")


def test_remap_query_lc_substitution():
    # hit "ALsCD" (insert s between cols 1,2), new "VICD" (all substitutions)
    result = _lc("ALsCD").remap_query("ALCD", "VICD", prepend_new_query=False)
    assert [s.seq for s in result.seqs] == ["ALsCD"]


def test_remap_query_lc_deletion():
    # hit "ALsCD", new "A-CD" (delete col 1 = L)
    result = _lc("ALsCD").remap_query("ALCD", "A-CD", prepend_new_query=False)
    assert [s.seq for s in result.seqs] == ["AsCD"]


def test_remap_query_lc_deletion_at_insert_boundary():
    # hit "ALsCD", new "AL-D" (delete col 2 = C, which has insert s before it)
    result = _lc("ALsCD").remap_query("ALCD", "AL-D", prepend_new_query=False)
    assert [s.seq for s in result.seqs] == ["ALsD"]


def test_remap_query_lc_insertion_in_query():
    # hit "ALsCD", new "ALCtD" (insert t between cols 2,3)
    result = _lc("ALsCD").remap_query("ALCD", "ALCtD", prepend_new_query=False)
    assert [s.seq for s in result.seqs] == ["ALsC-D"]


def test_remap_query_lc_trailing_insertion():
    # hit "ALCDy" (trailing insert y), new "ALCD"
    result = _lc("ALCDy").remap_query("ALCD", "ALCD", prepend_new_query=False)
    assert [s.seq for s in result.seqs] == ["ALCDy"]


def test_remap_query_lc_leading_insertion():
    # hit "xALCD" (leading insert x), new "VLCD"
    result = _lc("xALCD").remap_query("ALCD", "VLCD", prepend_new_query=False)
    assert [s.seq for s in result.seqs] == ["xALCD"]


def test_remap_query_lc_too_many_match_columns_raises():
    # hit "ALCDE" has 5 match columns vs old_query's 4
    with pytest.raises(ValueError):
        _lc("ALCDE").remap_query("ALCD", "VICD", prepend_new_query=False)


def test_remap_query_lc_too_few_match_columns_raises():
    # hit "ALC" has 3 match columns vs old_query's 4
    with pytest.raises(ValueError):
        _lc("ALC").remap_query("ALCD", "VICD", prepend_new_query=False)


def test_remap_query_mixed_clean_and_lowercase_hits():
    hits = [Sequence("ALSD", id="clean"), Sequence("ALsCD", id="lc")]
    sequences = Sequences(seqs=hits, aligned=True, format="a3m")
    result = sequences.remap_query("ALCD", "ALCtD", prepend_new_query=False)
    assert [s.seq for s in result.seqs] == ["ALS-D", "ALsC-D"]


def test_remap_query_preserves_key():
    hits = [Sequence("ALSD", id="h", key="pair-0")]
    result = Sequences(seqs=hits, aligned=True, format="a3m").remap_query("ALCD", "V-CD", prepend_new_query=False)
    assert result.seqs[0].seq == "ASD"
    assert result.seqs[0].key == "pair-0"


def test_remap_query_prepends_new_query_by_default():
    # default prepend_new_query=True inserts the new query as the first record,
    # ungapped and uppercased, ahead of the remapped hits
    hits = [Sequence("VAST", id="p0")]
    result = Sequences(seqs=hits, aligned=True, format="a3m").remap_query("MAST", "M-ST")
    assert [s.seq for s in result.seqs] == ["MST", "VST"]
    assert result.seqs[0].id_ is None
    assert result.seqs[0].key is None


def test_remap_query_prepended_query_uppercases_insertions():
    # lowercase insertions in the new query become uppercase query residues
    hits = [Sequence("ALSD", id="h")]
    result = Sequences(seqs=hits, aligned=True, format="a3m").remap_query("ALCD", "ALCtD")
    assert [s.seq for s in result.seqs] == ["ALCTD", "ALS-D"]
