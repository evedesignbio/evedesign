from protdesign.samplers.gibbs import GibbsSampler
from protdesign.restraints.seq_dist import LinearSeqDistRestraint
from protdesign.entity import System, Protein, Entity


def test_seq_dist_restraint_and_gibbs_sampler():
    """
    Use LinearSeqDistRestraint to test itself and Gibbs sampler
    """
    system = System([
        Protein(
            id="prot1", rep="AAAAAAAAA", first_index=100,
        ),
        Protein(
            id="prot2", rep="CCCCC", first_index=1,
        ),
        Protein(
            id="prot3", rep="FFFFY", first_index=200,
        ),
        Entity(
            type="rna", id="rna1", rep="AAAAAA", first_index=300,
        ),
    ])

    c = LinearSeqDistRestraint().build(
        system, data={
            0: ["KAAACAAAA", "RAAACAAAA"],
            2: ["DEQDA", "DEEDA"],
            3: ["UUUUUU"],
        }
    )

    # minimize distance, so use negative weight
    g = GibbsSampler(
        [c], weights=[-1], num_sweeps=1,
    )

    designs = g.generate(
        num_designs=2,
        entities=[0, 2, 3],
        fixed_pos={2: [204]},
        temperature=0.00000001
    )

    for design in designs:
        assert "".join(design[0].rep) == "KAAACAAAA" or "".join(design[0].rep) == "RAAACAAAA"
        assert "".join(design[1].rep) == "CCCCC"
        assert "".join(design[2].rep) == "DEQDY" or "".join(design[2].rep) == "DEEDY"
        assert "".join(design[3].rep) == "UUUUUU"
