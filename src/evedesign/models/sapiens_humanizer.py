"""Antibody humanization with Sapiens."""

from collections.abc import Mapping, Sequence
from contextlib import redirect_stdout
from copy import deepcopy
from io import StringIO
from numbers import Real
from typing import Any, ClassVar, Self

# Imported before the evedesign modules below: abnumber's numbering backend and
# scikit-learn (pulled in by evedesign.model) each load their own OpenMP runtime,
# and initialising them in the reverse order aborts the interpreter on macOS.
try:
    from abnumber import Chain, ChainParseError  # type: ignore[import-untyped]
    from sapiens import predict_scores  # type: ignore[import-untyped]

    IMPORT_AVAILABLE = True
except ImportError:
    Chain = None
    ChainParseError = ValueError
    predict_scores = None
    IMPORT_AVAILABLE = False

from evedesign.model import BaseModel, Generator
from evedesign.system import Entity, EntityInstance, System, SystemInstance
from evedesign.types import (
    EntityPosList,
    SapiensGeneration,
    SapiensMutation,
    StatusCallback,
)
from evedesign.utils import status_done, status_start


AMINO_ACIDS = frozenset("ACDEFGHIKLMNPQRSTVWY")


class SapiensHumanizer(BaseModel, Generator):
    """Humanize paired antibody variable regions with Sapiens."""

    available = IMPORT_AVAILABLE
    name: str = "SapiensHumanizer"
    citations: ClassVar[list[str]] = ["doi:10.1080/19420862.2021.2020203"]

    requires_target: bool = True
    requires_fixed_length: bool = True
    handles_deletions: bool = False
    handles_insertions: bool = False
    requires_gpu: bool = False
    supports_gpu: bool = False
    supports_cpu_parallel: bool = False
    supports_gpu_parallel: bool = False

    required_entity_attributes: ClassVar[list[str] | None] = []
    optional_entity_attributes: ClassVar[list[str] | None] = []

    def __init__(
        self,
        iterations: int = 1,
        scheme: str = "kabat",
        cdr_definition: str = "kabat",
        humanize_cdrs: bool = False,
        backmutate_vernier: bool = False,
    ):
        """
        Parameters
        ----------
        iterations
            Number of successive Sapiens argmax prediction rounds.
        scheme
            AbNumber residue-numbering scheme used to parse both variable regions.
        cdr_definition
            AbNumber CDR definition. CDR residues are restored after each round
            unless ``humanize_cdrs`` is enabled.
        humanize_cdrs
            Allow Sapiens substitutions within CDRs.
        backmutate_vernier
            Restore parental Vernier residues when parental CDRs are retained.
        """
        if not self.available:
            raise ImportError(
                "Sapiens dependencies could not be imported. Install evedesign "
                "with the 'sapiens' optional dependency."
            )
        if not isinstance(iterations, int) or isinstance(iterations, bool):
            raise TypeError("iterations must be an integer")
        if iterations < 1:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(scheme, str):
            raise TypeError("scheme must be a string")
        if not scheme.strip():
            raise ValueError("scheme must be a non-empty string")
        if not isinstance(cdr_definition, str):
            raise TypeError("cdr_definition must be a string")
        if not cdr_definition.strip():
            raise ValueError("cdr_definition must be a non-empty string")
        if not isinstance(humanize_cdrs, bool):
            raise TypeError("humanize_cdrs must be a boolean")
        if not isinstance(backmutate_vernier, bool):
            raise TypeError("backmutate_vernier must be a boolean")
        if humanize_cdrs and backmutate_vernier:
            raise ValueError("Cannot backmutate Vernier positions when humanizing CDRs")

        self.iterations = iterations
        self.scheme = scheme
        self.cdr_definition = cdr_definition
        self.humanize_cdrs = humanize_cdrs
        self.backmutate_vernier = backmutate_vernier
        self._system: System | None = None

    @property
    def ready(self):
        return self._system is not None

    @property
    def system(self) -> System | None:
        return self._system

    @staticmethod
    def _entity_sequence(entity: Entity) -> str:
        if entity.rep is None:
            return ""
        return "".join(entity.rep)

    @classmethod
    def _check_system(cls, system: System, data: None) -> tuple[bool, str]:
        if data is not None:
            return False, "Model does not support data parameter (must be None)"
        if len(system) != 2:
            return False, "System must contain exactly two antibody protein entities"
        if any(entity.type != "protein" for entity in system):
            return False, "Can only handle protein entities"
        if any(not entity.defined_sequence() for entity in system):
            return False, "Both antibody entities must have defined rep sequences"
        if any(
            set(cls._entity_sequence(entity)).difference(AMINO_ACIDS)
            for entity in system
        ):
            return False, "Antibody sequences must contain only canonical amino acids"
        if not cls.available:
            return False, "Sapiens dependencies are not installed"
        return True, ""

    @staticmethod
    def _parse_chains(
        system: System,
        scheme: str,
        cdr_definition: str,
    ) -> tuple[Any, Any]:
        assert Chain is not None
        chains = []
        with redirect_stdout(StringIO()):
            for entity in system:
                chains.append(
                    Chain(
                        SapiensHumanizer._entity_sequence(entity),
                        scheme=scheme,
                        cdr_definition=cdr_definition,
                        use_anarcii=True,
                    )
                )
        return chains[0], chains[1]

    @staticmethod
    def _check_chain_types(chains: Sequence[Any]) -> tuple[bool, str]:
        chain_types = [chain.chain_type for chain in chains]
        if (
            chain_types.count("H") != 1
            or sum(t in {"K", "L"} for t in chain_types) != 1
        ):
            return False, "System must contain one heavy and one light antibody chain"
        return True, ""

    @classmethod
    def can_model(cls, system: System, data: None = None) -> tuple[bool, str]:
        valid, reason = cls._check_system(system, data)
        if not valid:
            return valid, reason

        try:
            chains = cls._parse_chains(system, "kabat", "kabat")
        except (ChainParseError, ValueError):
            return False, "Both entities must be valid antibody variable regions"

        return cls._check_chain_types(chains)

    def _chains_for_system(self) -> tuple[Any, Any]:
        assert self._system is not None
        try:
            chains = self._parse_chains(
                self._system,
                self.scheme,
                self.cdr_definition,
            )
        except (ChainParseError, ValueError) as error:
            raise ValueError(
                "Both entities must be valid antibody variable regions for the configured "
                "scheme and CDR definition"
            ) from error

        valid, reason = self._check_chain_types(chains)
        if not valid:
            raise ValueError(reason)

        for chain, entity in zip(chains, self._system, strict=True):
            if len(chain.seq) != len(self._entity_sequence(entity)):
                raise ValueError(
                    "Antibody entities must contain only variable region residues"
                )
        return chains

    def build(
        self,
        system: System,
        data: None = None,
        status_callback: StatusCallback | None = None,
    ) -> Self:
        valid, reason = self._check_system(system, data)
        if not valid:
            raise ValueError(reason)

        self._system = system
        try:
            self._chains_for_system()
        except Exception:
            self._system = None
            raise
        return self

    def _validate_generation_options(
        self,
        num_designs: int,
        entities: Sequence[int] | None,
        fixed_pos: EntityPosList | None,
        temperature: float,
    ) -> tuple[tuple[int, ...], dict[int, tuple[int, ...]]]:
        if not isinstance(num_designs, int) or isinstance(num_designs, bool):
            raise TypeError("num_designs must be an integer")
        if num_designs < 1:
            raise ValueError("num_designs must be a positive integer")
        if not isinstance(temperature, Real) or isinstance(temperature, bool):
            raise TypeError("temperature must be a real number")
        if temperature != 1.0:
            raise ValueError("SapiensHumanizer only supports temperature=1.0")

        if entities is None:
            selected = (0, 1)
        else:
            if isinstance(entities, (str, bytes)):
                raise TypeError("entities must be a sequence of entity indices")
            try:
                selected = tuple(entities)
            except TypeError as error:
                raise TypeError(
                    "entities must be a sequence of entity indices"
                ) from error

        if not selected:
            raise ValueError("entities must select at least one entity")
        if any(
            not isinstance(entity, int) or isinstance(entity, bool)
            for entity in selected
        ):
            raise TypeError("entities must contain integer entity indices")
        if len(set(selected)) != len(selected):
            raise ValueError("entities must not contain duplicate indices")
        if any(entity not in {0, 1} for entity in selected):
            raise ValueError("entities must contain valid system entity indices")

        if fixed_pos is None:
            return selected, {}
        if not isinstance(fixed_pos, Mapping):
            raise TypeError("fixed_pos must map entity indices to positions")
        if any(
            not isinstance(entity, int) or isinstance(entity, bool)
            for entity in fixed_pos
        ):
            raise TypeError("fixed_pos keys must be integer entity indices")
        if set(fixed_pos).difference(selected):
            raise ValueError("fixed_pos can only reference selected entities")

        fixed: dict[int, tuple[int, ...]] = {}
        for entity, positions in fixed_pos.items():
            if isinstance(positions, (str, bytes)):
                raise TypeError("fixed_pos values must be sequences of positions")
            try:
                normalized = tuple(positions)
            except TypeError as error:
                raise TypeError(
                    "fixed_pos values must be sequences of positions"
                ) from error
            if any(
                not isinstance(position, int) or isinstance(position, bool)
                for position in normalized
            ):
                raise TypeError("fixed positions must be integers")
            if len(set(normalized)) != len(normalized):
                raise ValueError("fixed positions must not contain duplicates")
            self.valid_positions(normalized, entities=entity, raise_invalid=True)
            fixed[entity] = normalized

        return selected, fixed

    @staticmethod
    def _predicted_sequence(scores: Any, expected_length: int) -> str:
        try:
            score_count = len(scores)
        except TypeError as error:
            raise RuntimeError(
                "Sapiens returned scores in an unexpected format"
            ) from error
        if score_count != expected_length:
            raise RuntimeError("Sapiens returned scores with an unexpected length")
        try:
            residues = scores.idxmax(axis=1).tolist()
        except (AttributeError, TypeError, ValueError) as error:
            raise RuntimeError(
                "Sapiens returned scores in an unexpected format"
            ) from error
        if len(residues) != expected_length or any(
            not isinstance(residue, str) or residue not in AMINO_ACIDS
            for residue in residues
        ):
            raise RuntimeError("Sapiens returned invalid predicted residues")
        return "".join(residues)

    def _humanize_chain(
        self,
        entity: int,
        parental: Any,
        fixed_positions: Sequence[int],
    ) -> str:
        assert self._system is not None
        assert predict_scores is not None

        source = self._entity_sequence(self._system[entity])
        first_index = self._system[entity].first_index
        assert first_index is not None
        fixed_offsets = [position - first_index for position in fixed_positions]

        humanized = parental.clone()
        for _ in range(self.iterations):
            scores = predict_scores(seq=humanized.seq, chain_type=humanized.chain_type)
            predicted = self._predicted_sequence(scores, len(parental.seq))
            humanized = parental.clone(predicted)

            if not self.humanize_cdrs:
                humanized = parental.graft_cdrs_onto(
                    humanized,
                    backmutate_vernier=self.backmutate_vernier,
                )

            if fixed_offsets:
                sequence = list(humanized.seq)
                for offset in fixed_offsets:
                    sequence[offset] = source[offset]
                humanized = parental.clone("".join(sequence))

        if len(humanized.seq) != len(source):
            raise RuntimeError("Sapiens returned a sequence with an unexpected length")
        return humanized.seq

    def _metadata(
        self,
        selected: Sequence[int],
        fixed: Mapping[int, Sequence[int]],
        sequences: Sequence[str],
    ) -> SapiensGeneration:
        assert self._system is not None
        mutations: list[SapiensMutation] = []
        for entity in selected:
            source = self._entity_sequence(self._system[entity])
            first_index = self._system[entity].first_index
            assert first_index is not None
            for offset, (ref, to) in enumerate(
                zip(source, sequences[entity], strict=True)
            ):
                if ref != to:
                    mutations.append(
                        {
                            "entity": entity,
                            "pos": first_index + offset,
                            "ref": ref,
                            "to": to,
                        }
                    )

        return {
            "iterations": self.iterations,
            "scheme": self.scheme,
            "cdr_definition": self.cdr_definition,
            "humanize_cdrs": self.humanize_cdrs,
            "backmutate_vernier": self.backmutate_vernier,
            "entities": list(selected),
            "fixed_positions": {
                entity: list(positions) for entity, positions in fixed.items()
            },
            "mutations": mutations,
        }

    def generate(
        self,
        num_designs: int,
        entities: Sequence[int] | None = None,
        fixed_pos: EntityPosList | None = None,
        temperature: float = 1.0,
        status_callback: StatusCallback | None = None,
    ) -> list[SystemInstance]:
        """Return independent copies of the deterministic Sapiens design."""
        self.ready_or_raise()
        selected, fixed = self._validate_generation_options(
            num_designs, entities, fixed_pos, temperature
        )
        chains = self._chains_for_system()

        status_start(status_callback, "Humanizing antibody sequences")

        assert self._system is not None
        sequences = [self._entity_sequence(entity) for entity in self._system]
        for entity in selected:
            sequences[entity] = self._humanize_chain(
                entity,
                chains[entity],
                fixed.get(entity, ()),
            )

        metadata = self._metadata(selected, fixed, sequences)
        designs = []
        for _ in range(num_designs):
            instance = self._system.rep_to_instance()
            for entity in selected:
                instance[entity] = EntityInstance(rep=sequences[entity])
            instance.metadata = {"sapiens": deepcopy(metadata)}
            designs.append(instance)

        status_done(status_callback, "Humanization complete")
        return designs
