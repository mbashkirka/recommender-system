from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Paths:
    root: Path
    raw: Path
    processed: Path
    splits: Path
    reports: Path
    artifacts: Path


@dataclass(frozen=True)
class DataConfig:
    dataset: str
    url: str
    expected_md5: str
    archive_name: str
    extracted_dir: str
    implicit_positive_threshold: float


@dataclass(frozen=True)
class SplitConfig:
    train_fraction: float
    validation_fraction: float
    test_fraction: float
    primary_eval_segment: str

    def validate(self) -> None:
        fractions = (self.train_fraction, self.validation_fraction, self.test_fraction)
        if any(value <= 0.0 for value in fractions):
            raise ValueError("All split fractions must be > 0")
        if abs(sum(fractions) - 1.0) > 1e-9:
            raise ValueError("train/validation/test split fractions must sum to 1.0")


@dataclass(frozen=True)
class EvaluationConfig:
    ks: tuple[int, ...]
    primary_k: int

    def validate(self) -> None:
        if not self.ks or any(k <= 0 for k in self.ks):
            raise ValueError("evaluation.ks must contain positive integers")
        if len(set(self.ks)) != len(self.ks):
            raise ValueError("evaluation.ks must not contain duplicates")
        if self.primary_k not in self.ks:
            raise ValueError("evaluation.primary_k must be included in evaluation.ks")


@dataclass(frozen=True)
class ExperimentConfig:
    default_name: str
    default_salt: str
    control_weight: float
    treatment_weight: float


@dataclass(frozen=True)
class CollaborativeConfig:
    ranks: tuple[int, ...]
    popularity_blends: tuple[float, ...]
    tuning_k: int
    candidate_ks: tuple[int, ...]

    def validate(self) -> None:
        if not self.ranks or any(rank <= 0 for rank in self.ranks):
            raise ValueError("collaborative.ranks must contain positive integers")
        if len(set(self.ranks)) != len(self.ranks):
            raise ValueError("collaborative.ranks must not contain duplicates")
        if not self.popularity_blends or any(not 0.0 <= b < 1.0 for b in self.popularity_blends):
            raise ValueError("collaborative.popularity_blends must be in [0, 1)")
        if self.tuning_k <= 0:
            raise ValueError("collaborative.tuning_k must be positive")
        if not self.candidate_ks or any(k <= 0 for k in self.candidate_ks):
            raise ValueError("collaborative.candidate_ks must contain positive integers")


@dataclass(frozen=True)
class HybridConfig:
    metadata_blends: tuple[float, ...]
    ndcg_guardrail_ratio: float
    sparse_positive_threshold: int
    sparse_metadata_multiplier: float
    cold_item_penalty: float
    cold_start_k: int
    cold_item_quota: int

    def validate(self) -> None:
        if not self.metadata_blends or any(not 0.0 <= x < 1.0 for x in self.metadata_blends):
            raise ValueError("hybrid.metadata_blends must be in [0, 1)")
        if 0.0 not in self.metadata_blends:
            raise ValueError("hybrid.metadata_blends must include 0.0 as the M3 no-regression anchor")
        if not 0.0 < self.ndcg_guardrail_ratio <= 1.0:
            raise ValueError("hybrid.ndcg_guardrail_ratio must be in (0, 1]")
        if self.sparse_positive_threshold < 0:
            raise ValueError("hybrid.sparse_positive_threshold must be >= 0")
        if self.sparse_metadata_multiplier <= 0:
            raise ValueError("hybrid.sparse_metadata_multiplier must be > 0")
        if self.cold_item_penalty < 0:
            raise ValueError("hybrid.cold_item_penalty must be >= 0")
        if self.cold_start_k <= 0:
            raise ValueError("hybrid.cold_start_k must be > 0")
        if self.cold_item_quota <= 0 or self.cold_item_quota > self.cold_start_k:
            raise ValueError("hybrid.cold_item_quota must be in [1, cold_start_k]")


@dataclass(frozen=True)
class RankingConfig:
    candidate_k: int
    rank_train_fraction: float
    rank_tune_fraction: float
    negatives_per_positive: int
    c_values: tuple[float, ...]
    ltr_blends: tuple[float, ...]
    primary_k: int

    def validate(self) -> None:
        if self.candidate_k <= 0:
            raise ValueError("ranking.candidate_k must be > 0")
        if not 0.0 < self.rank_train_fraction < 1.0:
            raise ValueError("ranking.rank_train_fraction must be in (0, 1)")
        if not 0.0 < self.rank_tune_fraction < 1.0:
            raise ValueError("ranking.rank_tune_fraction must be in (0, 1)")
        if self.rank_train_fraction + self.rank_tune_fraction >= 1.0:
            raise ValueError("ranking train+tune fractions must be < 1")
        if self.negatives_per_positive <= 0:
            raise ValueError("ranking.negatives_per_positive must be > 0")
        if not self.c_values or any(x <= 0 for x in self.c_values):
            raise ValueError("ranking.c_values must contain positive values")
        if not self.ltr_blends or any(not 0.0 <= x <= 1.0 for x in self.ltr_blends):
            raise ValueError("ranking.ltr_blends must be in [0, 1]")
        if 0.0 not in self.ltr_blends:
            raise ValueError("ranking.ltr_blends must include 0.0 as the M4 no-regression anchor")
        if self.primary_k <= 0 or self.primary_k > self.candidate_k:
            raise ValueError("ranking.primary_k must be in [1, candidate_k]")


@dataclass(frozen=True)
class RerankingConfig:
    diversity_weights: tuple[float, ...]
    novelty_weights: tuple[float, ...]
    ndcg_guardrail_ratio: float
    recall_guardrail_ratio: float
    tune_fraction: float
    pool_k: int
    primary_k: int
    max_per_genre: int

    def validate(self) -> None:
        if not self.diversity_weights or any(x < 0.0 for x in self.diversity_weights):
            raise ValueError("reranking.diversity_weights must be non-negative")
        if not self.novelty_weights or any(x < 0.0 for x in self.novelty_weights):
            raise ValueError("reranking.novelty_weights must be non-negative")
        if 0.0 not in self.diversity_weights or 0.0 not in self.novelty_weights:
            raise ValueError("reranking grids must include the zero/zero no-regression anchor")
        if not 0.0 < self.ndcg_guardrail_ratio <= 1.0:
            raise ValueError("reranking.ndcg_guardrail_ratio must be in (0, 1]")
        if not 0.0 < self.recall_guardrail_ratio <= 1.0:
            raise ValueError("reranking.recall_guardrail_ratio must be in (0, 1]")
        if not 0.0 < self.tune_fraction < 1.0:
            raise ValueError("reranking.tune_fraction must be in (0, 1)")
        if self.pool_k <= 0 or self.primary_k <= 0 or self.primary_k > self.pool_k:
            raise ValueError("reranking requires 0 < primary_k <= pool_k")
        if self.max_per_genre < 0:
            raise ValueError("reranking.max_per_genre must be >= 0")


@dataclass(frozen=True)
class ABTestingConfig:
    demo_experiment_name: str
    demo_units: int
    demo_exposures_per_unit: int
    demo_treatment_relative_click_lift: float
    alpha: float
    power: float
    target_relative_mde: float
    min_practical_relative_lift: float
    srm_alpha: float
    max_latency_increase_ms: float
    max_error_rate_increase: float

    def validate(self) -> None:
        if not self.demo_experiment_name:
            raise ValueError("ab_testing.demo_experiment_name must be non-empty")
        if self.demo_units < 100:
            raise ValueError("ab_testing.demo_units must be >= 100")
        if self.demo_exposures_per_unit <= 0:
            raise ValueError("ab_testing.demo_exposures_per_unit must be > 0")
        if self.demo_treatment_relative_click_lift <= -1.0:
            raise ValueError("ab_testing.demo_treatment_relative_click_lift must be > -1")
        if not 0.0 < self.alpha < 1.0:
            raise ValueError("ab_testing.alpha must be in (0, 1)")
        if not 0.0 < self.power < 1.0:
            raise ValueError("ab_testing.power must be in (0, 1)")
        if self.target_relative_mde <= 0.0:
            raise ValueError("ab_testing.target_relative_mde must be > 0")
        if self.min_practical_relative_lift < 0.0:
            raise ValueError("ab_testing.min_practical_relative_lift must be >= 0")
        if not 0.0 < self.srm_alpha < 1.0:
            raise ValueError("ab_testing.srm_alpha must be in (0, 1)")
        if self.max_latency_increase_ms < 0.0:
            raise ValueError("ab_testing.max_latency_increase_ms must be >= 0")
        if self.max_error_rate_increase < 0.0:
            raise ValueError("ab_testing.max_error_rate_increase must be >= 0")


@dataclass(frozen=True)
class OperationsConfig:
    benchmark_requests: int
    benchmark_cold_share: float
    benchmark_top_k: int
    max_p95_latency_ms: float
    max_p99_latency_ms: float
    max_error_rate: float
    min_throughput_rps: float
    metrics_window: int

    def validate(self) -> None:
        if self.benchmark_requests <= 0:
            raise ValueError("operations.benchmark_requests must be > 0")
        if not 0.0 <= self.benchmark_cold_share <= 1.0:
            raise ValueError("operations.benchmark_cold_share must be in [0, 1]")
        if self.benchmark_top_k <= 0:
            raise ValueError("operations.benchmark_top_k must be > 0")
        if self.max_p95_latency_ms <= 0 or self.max_p99_latency_ms <= 0:
            raise ValueError("operations latency thresholds must be > 0")
        if self.max_p99_latency_ms < self.max_p95_latency_ms:
            raise ValueError("operations.max_p99_latency_ms must be >= max_p95_latency_ms")
        if not 0.0 <= self.max_error_rate <= 1.0:
            raise ValueError("operations.max_error_rate must be in [0, 1]")
        if self.min_throughput_rps <= 0:
            raise ValueError("operations.min_throughput_rps must be > 0")
        if self.metrics_window <= 0:
            raise ValueError("operations.metrics_window must be > 0")


@dataclass(frozen=True)
class Settings:
    name: str
    seed: int
    paths: Paths
    data: DataConfig
    split: SplitConfig
    experiment: ExperimentConfig
    evaluation: EvaluationConfig | None = None
    collaborative: CollaborativeConfig | None = None
    hybrid: HybridConfig | None = None
    ranking: RankingConfig | None = None
    reranking: RerankingConfig | None = None
    ab_testing: ABTestingConfig | None = None
    operations: OperationsConfig | None = None


def _source_project_root() -> Path:
    """Best-effort project root for an editable/source checkout."""
    return Path(__file__).resolve().parents[2]


def resolve_config_path() -> Path:
    """Resolve project.toml for source, editable, and installed/container layouts.

    Resolution order is intentionally explicit:
    1. RECSYS_CONFIG, when set. This is the production/container contract.
    2. ./config/project.toml from the current working directory.
    3. The historical source-checkout location relative to this module.

    The second rule is what makes an installed wheel work when the process is
    launched from /app, while the first rule removes any dependence on cwd in
    production images.
    """
    configured = os.environ.get("RECSYS_CONFIG")
    if configured:
        path = Path(configured).expanduser()
        if not path.is_absolute():
            path = Path.cwd() / path
        path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError(f"RECSYS_CONFIG does not point to a file: {path}")
        return path

    cwd_candidate = (Path.cwd() / "config" / "project.toml").resolve()
    if cwd_candidate.is_file():
        return cwd_candidate

    source_candidate = (_source_project_root() / "config" / "project.toml").resolve()
    if source_candidate.is_file():
        return source_candidate

    raise FileNotFoundError(
        "Could not locate config/project.toml. Set RECSYS_CONFIG to an explicit path."
    )


def project_root() -> Path:
    return resolve_config_path().parent.parent


def load_settings() -> Settings:
    cfg_path = resolve_config_path()
    root = cfg_path.parent.parent
    with cfg_path.open("rb") as fh:
        raw = tomllib.load(fh)

    paths = Paths(
        root=root,
        raw=root / "data" / "raw",
        processed=root / "data" / "processed",
        splits=root / "data" / "splits",
        reports=root / "reports",
        artifacts=root / "artifacts",
    )
    split = SplitConfig(**raw["split"])
    split.validate()
    evaluation_raw = raw.get("evaluation", {"ks": [5, 10, 20], "primary_k": 10})
    evaluation = EvaluationConfig(
        ks=tuple(int(k) for k in evaluation_raw["ks"]),
        primary_k=int(evaluation_raw["primary_k"]),
    )
    evaluation.validate()
    collaborative = None
    if "collaborative" in raw:
        c = raw["collaborative"]
        collaborative = CollaborativeConfig(
            ranks=tuple(int(x) for x in c["ranks"]),
            popularity_blends=tuple(float(x) for x in c["popularity_blends"]),
            tuning_k=int(c["tuning_k"]),
            candidate_ks=tuple(int(x) for x in c["candidate_ks"]),
        )
        collaborative.validate()
    hybrid = None
    if "hybrid" in raw:
        h = raw["hybrid"]
        hybrid = HybridConfig(
            metadata_blends=tuple(float(x) for x in h["metadata_blends"]),
            ndcg_guardrail_ratio=float(h["ndcg_guardrail_ratio"]),
            sparse_positive_threshold=int(h["sparse_positive_threshold"]),
            sparse_metadata_multiplier=float(h["sparse_metadata_multiplier"]),
            cold_item_penalty=float(h["cold_item_penalty"]),
            cold_start_k=int(h["cold_start_k"]),
            cold_item_quota=int(h["cold_item_quota"]),
        )
        hybrid.validate()
    ranking = None
    if "ranking" in raw:
        r = raw["ranking"]
        ranking = RankingConfig(
            candidate_k=int(r["candidate_k"]),
            rank_train_fraction=float(r["rank_train_fraction"]),
            rank_tune_fraction=float(r["rank_tune_fraction"]),
            negatives_per_positive=int(r["negatives_per_positive"]),
            c_values=tuple(float(x) for x in r["c_values"]),
            ltr_blends=tuple(float(x) for x in r["ltr_blends"]),
            primary_k=int(r["primary_k"]),
        )
        ranking.validate()
    reranking = None
    if "reranking" in raw:
        rr = raw["reranking"]
        reranking = RerankingConfig(
            diversity_weights=tuple(float(x) for x in rr["diversity_weights"]),
            novelty_weights=tuple(float(x) for x in rr["novelty_weights"]),
            ndcg_guardrail_ratio=float(rr["ndcg_guardrail_ratio"]),
            recall_guardrail_ratio=float(rr["recall_guardrail_ratio"]),
            tune_fraction=float(rr["tune_fraction"]),
            pool_k=int(rr["pool_k"]),
            primary_k=int(rr["primary_k"]),
            max_per_genre=int(rr["max_per_genre"]),
        )
        reranking.validate()
    ab_testing = None
    if "ab_testing" in raw:
        ab = raw["ab_testing"]
        ab_testing = ABTestingConfig(
            demo_experiment_name=str(ab["demo_experiment_name"]),
            demo_units=int(ab["demo_units"]),
            demo_exposures_per_unit=int(ab["demo_exposures_per_unit"]),
            demo_treatment_relative_click_lift=float(ab["demo_treatment_relative_click_lift"]),
            alpha=float(ab["alpha"]),
            power=float(ab["power"]),
            target_relative_mde=float(ab["target_relative_mde"]),
            min_practical_relative_lift=float(ab["min_practical_relative_lift"]),
            srm_alpha=float(ab["srm_alpha"]),
            max_latency_increase_ms=float(ab["max_latency_increase_ms"]),
            max_error_rate_increase=float(ab["max_error_rate_increase"]),
        )
        ab_testing.validate()
    operations = None
    if "operations" in raw:
        op = raw["operations"]
        operations = OperationsConfig(
            benchmark_requests=int(op["benchmark_requests"]),
            benchmark_cold_share=float(op["benchmark_cold_share"]),
            benchmark_top_k=int(op["benchmark_top_k"]),
            max_p95_latency_ms=float(op["max_p95_latency_ms"]),
            max_p99_latency_ms=float(op["max_p99_latency_ms"]),
            max_error_rate=float(op["max_error_rate"]),
            min_throughput_rps=float(op["min_throughput_rps"]),
            metrics_window=int(op["metrics_window"]),
        )
        operations.validate()

    return Settings(
        name=raw["project"]["name"],
        seed=int(raw["project"]["seed"]),
        paths=paths,
        data=DataConfig(**raw["data"]),
        split=split,
        experiment=ExperimentConfig(**raw["experiment"]),
        evaluation=evaluation,
        collaborative=collaborative,
        hybrid=hybrid,
        ranking=ranking,
        reranking=reranking,
        ab_testing=ab_testing,
        operations=operations,
    )
