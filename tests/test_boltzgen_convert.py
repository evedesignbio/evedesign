"""
Tests for the evedesign System -> BoltzGen YAML conversion.

Two tiers:

1. Emission tests assert the YAML we produce for each input case
   BoltzGen accepts (design/context entities, ligands, oligomers,
   conditioning blocks, rejections).
2. Round-trip tests feed that YAML through BoltzGen's own
   YamlDesignParser and assert the internal representation the model
   consumes, so the expectations are checked against upstream rather
   than against our reading of it. They skip when the parser or its
   molecule cache is unavailable.
"""
from pathlib import Path

import pytest
import yaml

from evedesign.models.boltz.convert_design import (
    _atom_bond_constraints,
    _entity_to_sequence_spec,
    _parse_single_design,
    _positions_to_range_spec,
    _secondary_structure_spec,
    _to_spec_pos,
    parse_design_output,
    system_to_boltzgen_yaml,
)
from evedesign.models.boltz.chains import _get_chain_ids
from evedesign.system import (
    AtomBond,
    Insertion,
    Interaction,
    Ligand,
    Protein,
    ResidueBias,
    SecondaryStructure,
    System,
)

# convert_design imports pyyaml, which arrives with the boltzgen extra
pytestmark = pytest.mark.boltzgen


# Sequence spec: how a chain declares what is designed


def test_range_when_min_and_max_set():
    e = Protein(rep=None, min_length=60, max_length=80, id="binder")
    assert _entity_to_sequence_spec(e) == "60..80"


def test_single_length_when_only_min_or_only_max():
    assert _entity_to_sequence_spec(Protein(rep=None, min_length=60, id="binder")) == "60"
    assert _entity_to_sequence_spec(Protein(rep=None, max_length=80, id="binder")) == "80"


def test_length_from_rep_when_no_range():
    e = Protein(rep="ACDEFGHIKL", min_length=10, max_length=10, id="binder")
    assert _entity_to_sequence_spec(e) == "10..10"


def test_falls_back_to_vanilla_range():
    assert _entity_to_sequence_spec(Protein(rep=None, id="binder")) == "80..140"


# Position numbering: evedesign first_index -> BoltzGen's 1-based spec


@pytest.mark.parametrize("first_index", [1, 20])
def test_spec_pos_shifts_first_index_to_one(first_index):
    e = Protein(rep="ACDEFGHIKL", first_index=first_index, id="binder")
    positions = [first_index, first_index + 4, first_index + 9]
    assert _to_spec_pos(e, positions, "test") == [1, 5, 10]


@pytest.mark.parametrize("first_index", [1, 20])
def test_spec_pos_agrees_with_boltzgen_parse_range(first_index):
    # Check the index math against upstream rather than against our
    # reading of it: every position must land on the same residue
    parse_range = pytest.importorskip(
        "boltzgen.data.parse.schema"
    ).parse_range
    rep = "ACDEFGHIKL"
    e = Protein(rep=rep, first_index=first_index, id="binder")
    for pos in range(first_index, first_index + len(rep)):
        (spec_pos,) = _to_spec_pos(e, [pos], "test")
        (offset,) = parse_range(str(spec_pos))  # 0-based into the chain
        assert rep[offset] == rep[pos - first_index]


@pytest.mark.parametrize("bad_pos", [19, 30])
def test_spec_pos_rejects_positions_outside_the_entity(bad_pos):
    e = Protein(rep="ACDEFGHIKL", first_index=20, id="binder")
    with pytest.raises(ValueError, match=r"outside 20\.\.29"):
        _to_spec_pos(e, [bad_pos], "test")


def test_spec_pos_range_form_agrees_with_boltzgen():
    # Consumers emit collapsed ranges, so check that form too
    parse_range = pytest.importorskip(
        "boltzgen.data.parse.schema"
    ).parse_range
    e = Protein(rep="ACDEFGHIKL", first_index=20, id="binder")
    spec = _positions_to_range_spec(
        _to_spec_pos(e, [24, 25, 26, 29], "test")
    )
    assert spec == "5..7,10"
    assert parse_range(spec) == [4, 5, 6, 9]


# MASK in rep: positions that must be designed


@pytest.mark.parametrize(
    "rep,expected",
    [
        ("ACD***GHI", "ACD3GHI"),   # masked run in the middle
        ("***DEFGHI", "3DEFGHI"),   # at the start
        ("ACDEFG***", "ACDEFG3"),   # at the end
        ("*CD***GH*", "1CD3GH1"),   # several runs
        ("*********", "9"),         # fully masked
        ("ACDEFGHIK", "9"),         # no mask -> whole rep designable
    ],
)
def test_mask_marks_designed_positions(rep, expected):
    assert _entity_to_sequence_spec(Protein(rep=rep, id="binder")) == expected


def test_masked_positions_are_fixed_length_and_numbered():
    # 3 stars are 3 real positions, numbered from first_index
    e = Protein(rep="ACD***GHI", first_index=20, id="binder")
    assert e.positions() == list(range(20, 29))
    assert _entity_to_sequence_spec(e) == "ACD3GHI"


def test_fixed_pos_cannot_name_a_masked_position():
    e = Protein(rep="ACD***GHI", first_index=20, id="binder")
    with pytest.raises(ValueError, match=r"fixed_pos \[23\] names masked"):
        _entity_to_sequence_spec(e, fixed_pos=[23])


def test_fixed_pos_alongside_mask_is_allowed_on_defined_residues():
    e = Protein(rep="ACD***GHI", first_index=20, id="binder")
    assert _entity_to_sequence_spec(e, fixed_pos=[20, 21, 22]) == "ACD3GHI"


def test_mask_routes_an_unselected_entity_to_the_design_emitter(tmp_path):
    # A masked position has no residue to hold fixed, so the entity is
    # designed even though entities did not name it, same as fixed_pos.
    # Emitting it as context would drop the '*' and shorten the chain.
    system = System([
        Protein(rep="ACD***GHI", id="motif"),
        Protein(rep=None, min_length=60, max_length=80, id="binder"),
    ])
    spec = yaml.safe_load(system_to_boltzgen_yaml(
        system, tmp_path / "spec.yaml", entities=[1]
    ).read_text())
    assert spec["entities"][0]["protein"]["sequence"] == "ACD3GHI"


# Motif scaffolding: letters kept, numbers designed


@pytest.mark.parametrize(
    "fixed_pos,expected",
    [
        ([4, 5, 6], "3EFG4"),   # ACDEFGHIKL -> keep E,F,G in the middle
        ([1, 2], "AC8"),        # motif at the start
        ([9, 10], "8KL"),       # motif at the end
        (list(range(1, 11)), "ACDEFGHIKL"),  # nothing designed
        (None, "10..10"),       # no motif -> plain length spec
    ],
)
def test_motif_spec_interleaves_letters_and_run_lengths(fixed_pos, expected):
    e = Protein(rep="ACDEFGHIKL", min_length=10, max_length=10, id="binder")
    assert _entity_to_sequence_spec(e, fixed_pos=fixed_pos) == expected


def test_motif_requires_rep():
    with pytest.raises(ValueError, match="fixed_pos needs rep"):
        _entity_to_sequence_spec(Protein(rep=None, min_length=5, id="binder"), fixed_pos=[1])


def test_motif_position_must_be_in_range():
    e = Protein(rep="ACDEFGHIKL", min_length=10, max_length=10, id="binder")
    with pytest.raises(ValueError, match=r"outside 1\.\.10"):
        _entity_to_sequence_spec(e, fixed_pos=[99])


def test_fixed_pos_routes_entity_to_the_design_emitter(tmp_path):
    # rep set and no min/max is normally a context entity, but a motif
    # spec contains digits, which BoltzGen counts as designed
    system = System([
        Protein(rep="MKTAYIAKQR", id="target"),
        Protein(rep="ACDEFGHIKL", id="motif"),
    ])
    spec = yaml.safe_load(system_to_boltzgen_yaml(
        system, tmp_path / "spec.yaml", fixed_pos={1: [4, 5, 6]}
    ).read_text())
    assert spec["entities"][1]["protein"]["sequence"] == "3EFG4"


# Insertions: inline range tokens spliced into the spec


@pytest.mark.parametrize(
    "entity,expected",
    [
        # insert between kept residues
        (Protein(rep="ACDEFGHI", id="b",
                 insertions=[Insertion(pos=4, min_length=5, max_length=10)]),
         "ACDE5..10FGHI"),
        # fixed length insert (min == max) emits a bare count
        (Protein(rep="ACDEFGHI", id="b",
                 insertions=[Insertion(pos=4, min_length=3, max_length=3)]),
         "ACDE3FGHI"),
        # first_index - 1 is the N-terminal extension
        (Protein(rep="ACDEFGHI", id="b",
                 insertions=[Insertion(pos=0, min_length=2, max_length=4)]),
         "2..4ACDEFGHI"),
        # after the last position is a C-terminal extension
        (Protein(rep="ACDEFGHI", id="b",
                 insertions=[Insertion(pos=8, min_length=2, max_length=4)]),
         "ACDEFGHI2..4"),
        # several inserts, each after the position it names
        (Protein(rep="ACDEFGHI", id="b",
                 insertions=[Insertion(pos=2, min_length=2, max_length=2),
                             Insertion(pos=6, min_length=3, max_length=3)]),
         "AC2DEFG3HI"),
        # MASK decides the split, the insert still lands after position 8
        (Protein(rep="ACD***GHI", id="b",
                 insertions=[Insertion(pos=8, min_length=4, max_length=6)]),
         "ACD3GH4..6I"),
    ],
)
def test_insertions_emit_inline_range_tokens(entity, expected):
    assert _entity_to_sequence_spec(entity, designed=False) == expected


def test_adjacent_numeric_tokens_are_comma_separated():
    # without the comma "4" + "5..10" + "4" would tokenize as the single
    # range 45..104 (schema.py:936), a 45 to 104 residue chain
    e = Protein(rep="ACDEFGHI", id="b",
                insertions=[Insertion(pos=4, min_length=5, max_length=10)])
    assert _entity_to_sequence_spec(e, designed=True) == "4,5..10,4"


def test_insertions_promote_an_unselected_entity_but_keep_its_rep(tmp_path):
    # insertions imply design, so the entity is not emitted as context,
    # but only the inserted residues are designed
    system = System([
        Protein(rep="ACDEFGHI", id="motif",
                insertions=[Insertion(pos=4, min_length=5, max_length=10)]),
        Protein(rep=None, min_length=60, max_length=80, id="binder"),
    ])
    spec = yaml.safe_load(system_to_boltzgen_yaml(
        system, tmp_path / "spec.yaml", entities=[1]
    ).read_text())
    assert spec["entities"][0]["protein"]["sequence"] == "ACDE5..10FGHI"


@pytest.mark.parametrize(
    "insertions,match",
    [
        ([Insertion(pos=None, min_length=2, max_length=4)], "needs pos"),
        ([Insertion(pos=4, min_length=2)], "needs both"),
        ([Insertion(pos=4, min_length=9, max_length=4)], "below min_length"),
        ([Insertion(pos=4, min_length=2, max_length=4,
                    secondary_structure="H")], "secondary_structure"),
        ([Insertion(pos=4, min_length=2, max_length=4,
                    interactions=[Interaction(id="x")])], "interactions"),
        ([Insertion(pos=2, min_length=1, max_length=4),
          Insertion(pos=6, min_length=1, max_length=4)], "one variable-length"),
    ],
)
def test_insertions_reject_what_boltzgen_cannot_express(insertions, match):
    e = Protein(rep="ACDEFGHI", id="b", insertions=insertions)
    with pytest.raises(ValueError, match=match):
        _entity_to_sequence_spec(e)


def test_insertions_require_a_rep():
    e = Protein(rep=None, min_length=10, id="b",
                insertions=[Insertion(pos=4, min_length=2, max_length=4)])
    with pytest.raises(ValueError, match="insertions need rep"):
        _entity_to_sequence_spec(e)


def test_roundtrip_insertion_yields_the_intended_design_mask(tmp_path):
    system = System([
        Protein(rep="MKTAYIAKQR", id="target"),
        Protein(rep="ACDEFGHI", id="binder",
                insertions=[Insertion(pos=4, min_length=3, max_length=3)]),
    ])
    _, target = _parse_with_boltzgen(system, tmp_path, entities=[])

    start, end = _residue_slice(target, 1)
    mask = target.design_info.res_design_mask[start:end].tolist()
    # ACDE kept, 3 inserted residues designed, FGHI kept
    assert mask == [False] * 4 + [True] * 3 + [False] * 4


# Entity kinds


def test_design_and_context_entities(tmp_path):
    system = System([
        Protein(rep="MKTAYIAKQR", id="target"),
        Protein(rep=None, min_length=60, max_length=80, id="binder"),
    ])
    entities = yaml.safe_load(system_to_boltzgen_yaml(
        system, tmp_path / "spec.yaml", entities=[1]
    ).read_text())["entities"]
    assert entities[0]["protein"] == {"id": "A", "sequence": "MKTAYIAKQR"}
    assert entities[1]["protein"] == {"id": "B", "sequence": "60..80"}


@pytest.mark.parametrize(
    "rep_type,rep,key",
    [("smiles", "CCO", "smiles"), ("ccd", "WHL", "ccd")],
)
def test_ligand_with_a_rep_is_emitted_as_a_ligand(tmp_path, rep_type, rep, key):
    system = System([
        Protein(rep=None, min_length=10, id="binder"),
        Ligand(rep=rep, ligand_rep_type=rep_type, id="lig"),
    ])
    entities = yaml.safe_load(system_to_boltzgen_yaml(
        system, tmp_path / "spec.yaml"
    ).read_text())["entities"]
    assert "protein" not in entities[1]
    assert entities[1]["ligand"] == {"id": "B", key: rep}


def test_designable_ligand_without_rep_defaults_to_unk(tmp_path):
    system = System([
        Protein(rep=None, min_length=10, id="binder"),
        Ligand(rep=None, ligand_rep_type="ccd", id="lig"),
    ])
    entities = yaml.safe_load(system_to_boltzgen_yaml(
        system, tmp_path / "spec.yaml"
    ).read_text())["entities"]
    assert entities[1]["ligand"] == {"id": "B", "ccd": "UNK"}


def test_homo_oligomer_id_becomes_a_list(tmp_path):
    system = System([
        Protein(rep="MKTAYIAKQR", id="target", copies=2),
        Protein(rep=None, min_length=10, id="binder"),
    ])
    entities = yaml.safe_load(system_to_boltzgen_yaml(
        system, tmp_path / "spec.yaml"
    ).read_text())["entities"]
    assert entities[0]["protein"]["id"] == ["A", "B"]
    assert entities[1]["protein"]["id"] == "C"


def test_context_entity_with_structure_emits_a_file_entry(tmp_path):
    cif = Path("examples/structure_validated_design/target_cache/1g13.cif")
    if not cif.exists():
        pytest.skip("cached 1G13 structure not available")

    from evedesign.structure import StructureFile

    model = StructureFile(str(cif), format="cif").get_model()
    system = System([
        Protein(rep="MKTAYIAKQR", id="target", structures={"x": model}),
        Protein(rep=None, min_length=10, id="binder"),
    ])
    entities = yaml.safe_load(system_to_boltzgen_yaml(
        system, tmp_path / "spec.yaml", entities=[1]
    ).read_text())["entities"]

    file_entry = entities[0]["file"]
    assert file_entry["include"] == [{"chain": {"id": "A"}}]
    assert Path(file_entry["path"]).exists()
    # BoltzGen's mmCIF reader needs _entity_poly_seq, which an
    # AtomArray cannot supply
    assert file_entry["path"].endswith(".pdb")


# Conditioning blocks


def test_secondary_structure_is_emitted_as_ranges():
    e = Protein(
        rep=None, min_length=20, max_length=20,
        secondary_structure=[
            SecondaryStructure(pos=14, type="C"),
            SecondaryStructure(pos=15, type="H"),
            SecondaryStructure(pos=16, type="H"),
            SecondaryStructure(pos=17, type="H"),
            SecondaryStructure(pos=19, type="E"),
        ],
        id="binder",
    )
    # evedesign H/E/C -> boltzgen helix/sheet/loop, positions collapsed
    assert _secondary_structure_spec(e) == {
        "loop": "14", "helix": "15..17", "sheet": "19",
    }


def test_secondary_structure_without_a_known_length():
    # ranges carry no length, so an entity with no min/max is fine
    e = Protein(rep=None, secondary_structure=[SecondaryStructure(pos=1, type="H")], id="binder")
    assert _secondary_structure_spec(e) == {"helix": "1"}


def test_secondary_structure_beyond_the_shortest_length():
    # positions past min_length stay expressible, unlike the string form
    e = Protein(
        rep=None, min_length=5, max_length=50,
        secondary_structure=[SecondaryStructure(pos=40, type="H")],
        id="binder",
    )
    assert _secondary_structure_spec(e) == {"helix": "40"}


def test_secondary_structure_positions_shift_by_first_index(tmp_path):
    e = Protein(
        rep="ACDEFGHIKL", first_index=20,
        secondary_structure=[
            SecondaryStructure(pos=24, type="H"),
            SecondaryStructure(pos=25, type="H"),
            SecondaryStructure(pos=29, type="E"),
        ],
        id="binder",
    )
    assert _secondary_structure_spec(e) == {"helix": "5..6", "sheet": "10"}


def test_binding_types_collapse_positions_into_ranges(tmp_path):
    system = System([
        Protein(
            rep="MKTAYIAKQR", id="target",
            interactions=[
                Interaction(id="avoid", pos=[2, 3, 4], avoid=True),
                Interaction(id="want", pos=[8]),
            ],
        ),
        Protein(rep=None, min_length=10, id="binder"),
    ])
    entities = yaml.safe_load(system_to_boltzgen_yaml(
        system, tmp_path / "spec.yaml"
    ).read_text())["entities"]
    assert entities[0]["protein"]["binding_types"] == {
        "binding": "8", "not_binding": "2..4",
    }


def test_interaction_without_positions_covers_the_whole_entity(tmp_path):
    system = System([
        Protein(rep="MKTA", id="target",
                interactions=[Interaction(id="none", pos=None, avoid=True)]),
        Protein(rep=None, min_length=10, id="binder"),
    ])
    entities = yaml.safe_load(system_to_boltzgen_yaml(
        system, tmp_path / "spec.yaml"
    ).read_text())["entities"]
    assert entities[0]["protein"]["binding_types"] == {"not_binding": "1..4"}


def test_binding_positions_shift_by_first_index(tmp_path):
    # Same chain residues as the test above, named from first_index=20
    # instead of 1, so the emitted spec must come out identical:
    #   evedesign 21,22,23 -> spec 2,3,4 ; evedesign 27 -> spec 8
    system = System([
        Protein(
            rep="MKTAYIAKQR", id="target", first_index=20,
            interactions=[
                Interaction(id="avoid", pos=[21, 22, 23], avoid=True),
                Interaction(id="want", pos=[27]),
            ],
        ),
        Protein(rep=None, min_length=10, id="binder"),
    ])
    entities = yaml.safe_load(system_to_boltzgen_yaml(
        system, tmp_path / "spec.yaml"
    ).read_text())["entities"]
    assert entities[0]["protein"]["binding_types"] == {
        "binding": "8", "not_binding": "2..4",
    }


def test_whole_entity_interaction_spans_from_first_index(tmp_path):
    # pos=None covers the chain, which runs 20..23 here, not 1..4.
    # Collecting 1..4 instead would send position 1 to spec -18.
    system = System([
        Protein(rep="MKTA", id="target", first_index=20,
                interactions=[Interaction(id="none", pos=None, avoid=True)]),
        Protein(rep=None, min_length=10, id="binder"),
    ])
    entities = yaml.safe_load(system_to_boltzgen_yaml(
        system, tmp_path / "spec.yaml"
    ).read_text())["entities"]
    assert entities[0]["protein"]["binding_types"] == {"not_binding": "1..4"}


def test_cyclic_flag(tmp_path):
    system = System([Protein(rep=None, min_length=10, cyclic=True, id="binder")])
    assert yaml.safe_load(system_to_boltzgen_yaml(
        system, tmp_path / "spec.yaml"
    ).read_text())["entities"][0]["protein"]["cyclic"] is True


def test_atom_bonds_become_top_level_constraints(tmp_path):
    system = System([
        Protein(rep="MCTAYIAKQR", id="target"),
        Protein(
            rep="AAACAAAAAAAA", min_length=12, max_length=12,
            atom_bonds=[AtomBond(
                type="covalent", source_pos=4, source_atom="SG",
                target_entity_id="target", target_pos=2, target_atom="SG",
            )],
        ),
    ])
    spec = yaml.safe_load(system_to_boltzgen_yaml(
        system, tmp_path / "spec.yaml"
    ).read_text())
    assert spec["constraints"] == [
        {"bond": {"atom1": ["B", 4, "SG"], "atom2": ["A", 2, "SG"]}}
    ]


def test_bond_ends_shift_by_their_own_entity(tmp_path):
    # target starts at 20, binder at 1, so a single global offset would
    # get one end wrong: source 4 stays 4, target 21 becomes 2
    system = System([
        Protein(rep="MCTAYIAKQR", id="target", first_index=20),
        Protein(
            rep="AAACAAAAAAAA", min_length=12, max_length=12, first_index=1,
            atom_bonds=[AtomBond(
                type="covalent", source_pos=4, source_atom="SG",
                target_entity_id="target", target_pos=21, target_atom="SG",
            )],
        ),
    ])
    spec = yaml.safe_load(system_to_boltzgen_yaml(
        system, tmp_path / "spec.yaml"
    ).read_text())
    assert spec["constraints"] == [
        {"bond": {"atom1": ["B", 4, "SG"], "atom2": ["A", 2, "SG"]}}
    ]


def test_no_constraints_key_when_there_are_no_bonds(tmp_path):
    system = System([Protein(rep=None, min_length=10, id="binder")])
    assert "constraints" not in yaml.safe_load(system_to_boltzgen_yaml(
        system, tmp_path / "spec.yaml"
    ).read_text())


def test_ignores_residue_bias(tmp_path):
    # not in optional_entity_attributes, so it is simply unused
    system = System([
        Protein(rep="ACDE", min_length=4, max_length=4,
               residue_bias=[ResidueBias(pos=1, bias={"A": 1.0})],
               id="binder",
           ),
    ])
    spec = yaml.safe_load(system_to_boltzgen_yaml(
        system, tmp_path / "spec.yaml"
    ).read_text())
    assert spec["entities"][0]["protein"] == {"id": "A", "sequence": "4..4"}


def test_ignores_symmetry(tmp_path):
    # symmetric_group is an int token label, not a point group
    system = System([
        Protein(rep=None, min_length=10, symmetry="C", copies=2, id="binder"),
    ])
    spec = yaml.safe_load(system_to_boltzgen_yaml(
        system, tmp_path / "spec.yaml"
    ).read_text())
    assert spec["entities"][0]["protein"] == {"id": ["A", "B"], "sequence": "10"}


def test_rejects_interaction_partner_ids(tmp_path):
    system = System([
        Protein(rep=None, min_length=10,
               interactions=[Interaction(id="s", pos=[1], partner_ids=["x"])]),
    ])
    with pytest.raises(ValueError, match="partner_ids"):
        yaml.safe_load(system_to_boltzgen_yaml(
            system, tmp_path / "spec.yaml"
        ).read_text())


def test_rejects_contradictory_binding_positions(tmp_path):
    system = System([
        Protein(rep=None, min_length=10, interactions=[
            Interaction(id="a", pos=[2]),
            Interaction(id="b", pos=[2], avoid=True),
        ]),
    ])
    with pytest.raises(ValueError, match="both"):
        yaml.safe_load(system_to_boltzgen_yaml(
            system, tmp_path / "spec.yaml"
        ).read_text())


def test_rejects_non_covalent_bonds():
    system = System([
        Protein(rep="ACDE", id="x", atom_bonds=[AtomBond(
            type="hydrogen", source_pos=1, source_atom="N",
            target_entity_id="x", target_pos=2, target_atom="O",
        )]),
    ])
    with pytest.raises(ValueError, match="covalent"):
        _atom_bond_constraints(system, _get_chain_ids(system))


def test_rejects_bond_to_unknown_entity():
    system = System([
        Protein(rep="ACDE", id="x", atom_bonds=[AtomBond(
            type="covalent", source_pos=1, source_atom="SG",
            target_entity_id="nope", target_pos=2, target_atom="SG",
        )]),
    ])
    with pytest.raises(ValueError, match="unknown entity"):
        _atom_bond_constraints(system, _get_chain_ids(system))


# Output parsing


def test_parse_returns_empty_when_the_run_did_not_reach_filtering(tmp_path):
    # no final_ranked_designs/ means filtering never ran
    system = System([Protein(rep=None, min_length=10, id="binder")])
    assert parse_design_output(tmp_path, system) == []


def test_design_output_is_renumbered_from_first_index(tmp_path):
    # BoltzGen numbers output residues from 1; EntityInstance.models must
    # match the entity's numbering, as Boltz-2 output already does
    import biotite.structure as struc

    from evedesign.structure import Structure

    atoms = [
        struc.Atom([i * 3.8, 0.0, 0.0], chain_id="A", res_id=i,
                   res_name="ALA", atom_name=name, element=el)
        for i in range(1, 6)
        for name, el in [("N", "N"), ("CA", "C"), ("C", "C"), ("O", "O")]
    ]
    cif = tmp_path / "design_spec_0.cif"
    Structure(struc.array(atoms)).to_file(str(cif), format="cif")

    system = System([Protein(rep="AAAAA", first_index=20, id="binder")])
    instance = _parse_single_design(cif, system, {"A": 0})

    res_ids = sorted(set(instance[0].models["model_0"].res_df()["res_id"]))
    assert res_ids == list(range(20, 25))


_AA3 = {"A": "ALA", "C": "CYS", "D": "ASP", "E": "GLU", "F": "PHE",
        "G": "GLY", "H": "HIS", "I": "ILE", "K": "LYS"}


def _write_design_cif(path, n_res=5, chain="A", seq=None):
    import biotite.structure as struc
    from evedesign.structure import Structure

    names = [_AA3[c] for c in seq] if seq else ["ALA"] * n_res
    atoms = [
        struc.Atom([i * 3.8, 0.0, 0.0], chain_id=chain, res_id=i,
                   res_name=names[i - 1], atom_name=name, element=el)
        for i in range(1, len(names) + 1)
        for name, el in [("N", "N"), ("CA", "C"), ("C", "C"), ("O", "O")]
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    Structure(struc.array(atoms)).to_file(str(path), format="cif")


@pytest.mark.parametrize(
    "cif_name,design_id",
    [
        ("rank01_design_spec_5.cif", "design_spec_5"),  # batch of designs
        ("rank1_design_spec.cif", "design_spec"),       # num_designs=1
    ],
)
def test_parses_both_ranked_filename_forms(tmp_path, cif_name, design_id):
    # BoltzGen prepends rank<N>_ to whatever it named the design
    # (filter.py:567), and drops the _<M> index for a single design
    final = tmp_path / "final_ranked_designs" / "final_10_designs"
    _write_design_cif(final / cif_name)
    (tmp_path / "final_ranked_designs" / "final_designs_metrics_10.csv").write_text(
        f"id,iptm,complex_plddt\n{design_id},0.8,0.9\n"
    )

    system = System([Protein(rep="AAAAA", id="binder")])
    instances = parse_design_output(tmp_path, system)

    assert len(instances) == 1
    assert instances[0].metadata["boltzgen_design_id"] == design_id
    assert instances[0].score == 0.8


@pytest.mark.parametrize(
    "insertions,design,expected",
    [
        # fixed insert after position 4
        ([Insertion(pos=4, min_length=3, max_length=3)],
         "ACDEKKKFGHI", "ACDEkkkFGHI"),
        # a variable insert takes whatever length is left over
        ([Insertion(pos=4, min_length=2, max_length=5)],
         "ACDEKKKKFGHI", "ACDEkkkkFGHI"),
        # N-terminal extension
        ([Insertion(pos=0, min_length=2, max_length=2)],
         "KKACDEFGHI", "kkACDEFGHI"),
        # C-terminal extension
        ([Insertion(pos=8, min_length=2, max_length=2)],
         "ACDEFGHIKK", "ACDEFGHIkk"),
        # two fixed inserts, each after the position it names
        ([Insertion(pos=2, min_length=2, max_length=2),
          Insertion(pos=6, min_length=3, max_length=3)],
         "ACKKDEFGKKKHI", "ACkkDEFGkkkHI"),
    ],
)
def test_inserted_residues_are_lowercased(tmp_path, insertions, design, expected):
    # BoltzGen's output has no record of where an insert landed, so the
    # spans come from the length difference
    cif = tmp_path / "d.cif"
    _write_design_cif(cif, seq=design)
    system = System([Protein(rep="ACDEFGHI", id="b", insertions=insertions)])

    instance = _parse_single_design(cif, system, {"A": 0})

    assert "".join(instance[0].rep) == expected


def test_unexplained_length_leaves_the_rep_uncoded(tmp_path):
    # 2 extra residues, but the only insertion adds exactly 3
    cif = tmp_path / "d.cif"
    _write_design_cif(cif, seq="ACDEKKFGHI")
    system = System([Protein(
        rep="ACDEFGHI", id="b",
        insertions=[Insertion(pos=4, min_length=3, max_length=3)])])

    instance = _parse_single_design(cif, system, {"A": 0})

    assert "".join(instance[0].rep) == "ACDEKKFGHI"


def test_insert_coded_design_validates_with_its_structure(tmp_path):
    # Structure.represents compares rep symbols against residue names, so
    # the lowercase coding must not read as a mismatch
    cif = tmp_path / "d.cif"
    _write_design_cif(cif, seq="ACDEKKKFGHI")
    system = System([Protein(
        rep="ACDEFGHI", id="b",
        insertions=[Insertion(pos=4, min_length=3, max_length=3)])])

    instance = _parse_single_design(cif, system, {"A": 0})

    assert instance[0].models is not None
    assert system.valid_instance(
        instance, fixed_length=False, require_reps=True
    )


# Round-trip: our YAML through BoltzGen's own parser

MOLDIR = Path(
    "~/.cache/huggingface/hub/datasets--boltzgen--inference-data/snapshots/"
    "c3d36fd276e9caf098c75d4113c6d5eb320b1a4c/mols.zip"
).expanduser()


def _parse_with_boltzgen(system, tmp_path, **kwargs):
    """Emit our YAML, then parse it with BoltzGen's YamlDesignParser."""
    parser = pytest.importorskip(
        "boltzgen.data.parse.schema", reason="boltzgen not installed"
    ).YamlDesignParser
    if not MOLDIR.exists():
        pytest.skip("boltzgen molecule cache not downloaded")

    path = system_to_boltzgen_yaml(system, tmp_path / "spec.yaml", **kwargs)
    schema = yaml.safe_load(path.read_text())
    target = parser(mol_dir=MOLDIR).parse_boltzgen_schema(
        name="test", schema=schema, mols={}, mol_dir=MOLDIR,
        base_file_path=tmp_path,
    )
    return schema, target


def _residue_slice(target, chain_idx):
    chain = target.structure.chains[chain_idx]
    start = int(chain["res_idx"])
    return start, start + int(chain["res_num"])


def test_roundtrip_motif_yields_the_intended_design_mask(tmp_path):
    system = System([
        Protein(rep="MKTAYIAKQR", id="target"),
        Protein(rep="ACDEFGHIKL", min_length=10, max_length=10, id="motif"),
    ])
    _, target = _parse_with_boltzgen(system, tmp_path, fixed_pos={1: [4, 5, 6]})

    start, end = _residue_slice(target, 1)
    mask = target.design_info.res_design_mask[start:end].tolist()
    # 3 designed, E/F/G kept, 4 designed
    assert mask == [True] * 3 + [False] * 3 + [True] * 4


def test_roundtrip_mask_yields_the_intended_design_mask(tmp_path):
    system = System([
        Protein(rep="MKTAYIAKQR", id="target"),
        Protein(rep="ACD***GHI", id="binder"),
    ])
    _, target = _parse_with_boltzgen(system, tmp_path, entities=[1])

    start, end = _residue_slice(target, 1)
    mask = target.design_info.res_design_mask[start:end].tolist()
    # ACD kept, 3 designed, GHI kept -- and 9 residues, not 6
    assert mask == [False] * 3 + [True] * 3 + [False] * 3


def test_roundtrip_secondary_structure_types(tmp_path):
    from boltzgen.data.const import ss_type_ids

    system = System([
        Protein(rep="MKTAYIAKQR", id="target"),
        Protein(rep=None, min_length=8, max_length=8, id="binder",
                secondary_structure=[
                    SecondaryStructure(pos=2, type="H"),
                    SecondaryStructure(pos=3, type="E"),
                    SecondaryStructure(pos=5, type="C"),
                ]),
    ])
    _, target = _parse_with_boltzgen(system, tmp_path)

    start, end = _residue_slice(target, 1)
    by_id = {v: k for k, v in ss_type_ids.items()}
    types = [by_id[int(v)] for v in target.design_info.res_ss_types[start:end]]
    assert types == [
        "UNSPECIFIED", "HELIX", "SHEET", "UNSPECIFIED", "LOOP",
        "UNSPECIFIED", "UNSPECIFIED", "UNSPECIFIED",
    ]


def test_roundtrip_binding_types(tmp_path):
    from boltzgen.data.const import binding_type_ids

    system = System([
        Protein(rep="MKTAYIAKQR", id="target", interactions=[
            Interaction(id="avoid", pos=[2, 3, 4], avoid=True),
            Interaction(id="want", pos=[8]),
        ]),
        Protein(rep=None, min_length=6, max_length=6, id="binder"),
    ])
    _, target = _parse_with_boltzgen(system, tmp_path, entities=[1])

    start, end = _residue_slice(target, 0)
    by_id = {v: k for k, v in binding_type_ids.items()}
    types = [by_id[int(v)] for v in target.design_info.res_binding_type[start:end]]
    assert types == [
        "UNSPECIFIED", "NOT_BINDING", "NOT_BINDING", "NOT_BINDING",
        "UNSPECIFIED", "UNSPECIFIED", "UNSPECIFIED", "BINDING",
        "UNSPECIFIED", "UNSPECIFIED",
    ]


def test_roundtrip_covalent_bond_resolves_to_real_atoms(tmp_path):
    # both cysteines must be fixed residues: designed positions are
    # placeholder glycines and have no SG atom
    system = System([
        Protein(rep="MCTAYIAKQR", id="target"),
        Protein(rep="AAACAAAAAAAA", min_length=12, max_length=12, id="binder",
                atom_bonds=[AtomBond(
                    type="covalent", source_pos=4, source_atom="SG",
                    target_entity_id="target", target_pos=2, target_atom="SG",
                )]),
    ])
    _, target = _parse_with_boltzgen(system, tmp_path, fixed_pos={1: [4]}, entities=[1])
    assert len(target.structure.bonds) >= 1


def test_roundtrip_structural_target(tmp_path):
    cif = Path("examples/structure_validated_design/target_cache/1g13.cif")
    if not cif.exists():
        pytest.skip("cached 1G13 structure not available")

    import biotite.structure as struc
    from evedesign.structure import Structure, StructureFile

    chain = StructureFile(str(cif), format="cif").get_model().get_chain("A")
    aa = chain.atom_array[struc.filter_amino_acids(chain.atom_array)]
    system = System([
        Protein(rep="MKTAYIAKQR", id="target", structures={"x": Structure(aa)}),
        Protein(rep=None, min_length=10, max_length=10, id="binder"),
    ])
    _, target = _parse_with_boltzgen(system, tmp_path, entities=[1])
    assert len(target.structure.chains) == 2
