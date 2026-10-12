"""Generate practical-review and QSVC-audit files from the notebook ranking.

Run the ranking/export cell in ``smiles_label_baseline.ipynb`` first, then run:
    python data/generate_qsvc_audit.py

This script does not train the ranking model or rewrite ranking/top-50 files.
Its outputs are retrospective review artifacts, not challenge submissions.
QSVC is a four-feature Qiskit ZZ fidelity kernel simulated with exact
statevectors; it is not evidence of quantum advantage.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from qiskit.circuit.library import zz_feature_map
from qiskit.quantum_info import Statevector
from rdkit import Chem, DataStructs
from rdkit.Chem import (
    Descriptors,
    FilterCatalog,
    Lipinski,
    rdFingerprintGenerator,
    rdMolDescriptors,
)
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


DATA_DIR = Path(__file__).resolve().parent
SEED = 42
N_FEATURES = 4
FEATURE_MAP_REPS = 2
SVC_C = 1.0
NEGATIVE_TO_POSITIVE_RATIOS = (5, 10)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_and_deduplicate_labels(path: Path) -> tuple[pd.DataFrame, np.ndarray, set[str]]:
    raw = pd.read_csv(path)
    required = {"Label", "RDKit_SMILES"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"Label file is missing columns: {sorted(missing)}")
    if raw.Label.isna().any() or set(raw.Label.unique()) != {0, 1}:
        raise ValueError("Label column must contain both 0 and 1 with no missing values")

    molecules = [
        Chem.MolFromSmiles(value) if isinstance(value, str) else None
        for value in raw.RDKit_SMILES
    ]
    valid = np.asarray([molecule is not None for molecule in molecules])
    clean = raw.loc[valid].copy()
    clean["canonical"] = [
        Chem.MolToSmiles(molecule, isomericSmiles=True)
        for molecule in molecules
        if molecule is not None
    ]

    label_counts = clean.groupby("canonical").Label.nunique()
    conflicting = set(label_counts[label_counts > 1].index)
    clean = clean.loc[~clean.canonical.isin(conflicting)]
    clean = clean.drop_duplicates("canonical").reset_index(drop=True)
    if clean.Label.nunique() != 2:
        raise ValueError("Both classes must remain after SMILES validation and deduplication")
    return clean, clean.Label.to_numpy(dtype=np.int8), set(clean.canonical)


def physicochemical_features(molecule: Chem.Mol) -> list[float]:
    return [
        Descriptors.MolWt(molecule),
        Descriptors.MolLogP(molecule),
        Descriptors.TPSA(molecule),
        Lipinski.NumHDonors(molecule),
        Lipinski.NumHAcceptors(molecule),
        Lipinski.NumRotatableBonds(molecule),
        Lipinski.HeavyAtomCount(molecule),
        rdMolDescriptors.CalcNumRings(molecule),
        rdMolDescriptors.CalcNumAromaticRings(molecule),
        rdMolDescriptors.CalcFractionCSP3(molecule),
        sum(atom.GetFormalCharge() for atom in molecule.GetAtoms()),
    ]


def feature_matrix(smiles: list[str]) -> tuple[np.ndarray, np.ndarray]:
    molecules = [
        Chem.MolFromSmiles(value) if isinstance(value, str) else None
        for value in smiles
    ]
    invalid = [index for index, molecule in enumerate(molecules) if molecule is None]
    if invalid:
        raise ValueError(f"Invalid SMILES at input rows: {invalid[:10]}")

    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=2, fpSize=2048, includeChirality=True
    )
    fingerprints = np.stack([
        generator.GetFingerprintAsNumPy(molecule) for molecule in molecules
    ]).astype(np.uint8, copy=False)
    descriptors = np.asarray(
        [physicochemical_features(molecule) for molecule in molecules],
        dtype=np.float64,
    )
    return fingerprints, descriptors


def prepare_angles(
    train_fingerprints: np.ndarray,
    train_descriptors: np.ndarray,
    candidate_fingerprints: np.ndarray,
    candidate_descriptors: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    svd = TruncatedSVD(n_components=N_FEATURES, n_iter=5, random_state=SEED)
    train_fp = svd.fit_transform(train_fingerprints)
    candidate_fp = svd.transform(candidate_fingerprints)

    descriptor_scaler = StandardScaler().fit(train_descriptors)
    train_combined = np.column_stack((
        train_fp,
        descriptor_scaler.transform(train_descriptors),
    ))
    candidate_combined = np.column_stack((
        candidate_fp,
        descriptor_scaler.transform(candidate_descriptors),
    ))

    projection = PCA(n_components=N_FEATURES, random_state=SEED).fit(train_combined)
    train_projected = projection.transform(train_combined)
    candidate_projected = projection.transform(candidate_combined)
    angle_scaler = StandardScaler().fit(train_projected)
    train_angles = np.pi * np.tanh(angle_scaler.transform(train_projected) / 2.0)
    candidate_angles = np.pi * np.tanh(angle_scaler.transform(candidate_projected) / 2.0)
    return train_angles.astype(np.float64), candidate_angles.astype(np.float64)


def quantum_states(angles: np.ndarray) -> np.ndarray:
    feature_map = zz_feature_map(
        feature_dimension=N_FEATURES,
        reps=FEATURE_MAP_REPS,
        entanglement="linear",
    )
    parameters = tuple(feature_map.parameters)
    states = []
    for row in angles:
        circuit = feature_map.assign_parameters(
            dict(zip(parameters, row, strict=True)),
            inplace=False,
        )
        states.append(Statevector.from_instruction(circuit).data)
    return np.asarray(states, dtype=np.complex128)


def fidelity_kernel(left_states: np.ndarray, right_states: np.ndarray) -> np.ndarray:
    kernel = np.abs(left_states @ right_states.conj().T) ** 2
    return np.clip(kernel.real, 0.0, 1.0)


def load_notebook_ranking(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(
            f"Notebook ranking not found: {path}\n"
            "This script intentionally does not train or create the ranking. "
            "Open data/smiles_label_baseline.ipynb, run the cell titled "
            "'Rank unlabeled candidates for review' (cell id "
            "'rank-unlabeled-candidates'), and confirm it writes "
            "'retrospective_candidates_ranked.csv' to the data folder. "
            "Then rerun this command."
        )
    ranked = pd.read_csv(path)
    required = {"BaselineRank", "CatalogID", "SMILES", "ModelScore", "RankingModel"}
    missing = required - set(ranked.columns)
    if missing:
        raise ValueError(f"Notebook ranking is missing columns: {sorted(missing)}")
    if ranked.empty or ranked.CatalogID.isna().any() or ranked.CatalogID.duplicated().any():
        raise ValueError("Notebook ranking must have non-empty, unique CatalogIDs")
    ranks = pd.to_numeric(ranked.BaselineRank, errors="coerce")
    if ranks.isna().any() or not np.array_equal(
        ranks.to_numpy(dtype=np.int64),
        np.arange(1, len(ranked) + 1),
    ):
        raise ValueError("BaselineRank must be contiguous from 1 through the final row")
    if ranked.SMILES.isna().any() or ranked.ModelScore.isna().any():
        raise ValueError("Notebook ranking contains missing SMILES or ModelScore values")
    if ranked.RankingModel.nunique() != 1:
        raise ValueError("Notebook ranking must contain exactly one selected RankingModel")
    ranked["CatalogID"] = ranked.CatalogID.astype(str)
    ranked["BaselineRank"] = ranks.astype(int)
    return ranked


def build_practical_screen(
    top500: pd.DataFrame,
    known_structures: set[str],
    labels_frame: pd.DataFrame,
) -> pd.DataFrame:
    molecules = [
        Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
        for smiles in top500.SMILES
    ]
    invalid = [index for index, molecule in enumerate(molecules) if molecule is None]
    if invalid:
        raise ValueError(f"Invalid candidate SMILES in ranked top 500 at rows: {invalid[:10]}")

    canonical = [
        Chem.MolToSmiles(molecule, isomericSmiles=True)
        for molecule in molecules
    ]
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=2, fpSize=2048, includeChirality=True
    )
    candidate_fps = [generator.GetFingerprint(molecule) for molecule in molecules]
    positive_smiles = labels_frame.loc[labels_frame.Label.eq(1), "canonical"].tolist()
    positive_molecules = [Chem.MolFromSmiles(smiles) for smiles in positive_smiles]
    known_hit_fps = [generator.GetFingerprint(molecule) for molecule in positive_molecules]
    if not known_hit_fps:
        raise ValueError("No positive labeled compounds available for similarity review")

    top500 = top500.copy().reset_index(drop=True)
    top500["CanonicalSMILES"] = canonical
    top500["MaxTanimotoToKnownHit"] = [
        max(DataStructs.BulkTanimotoSimilarity(fp, known_hit_fps))
        for fp in candidate_fps
    ]

    selected = [0]
    max_similarity_to_selected = np.zeros(len(top500), dtype=np.float64)
    for _ in range(49):
        last_fp = candidate_fps[selected[-1]]
        similarities = np.asarray(
            DataStructs.BulkTanimotoSimilarity(last_fp, candidate_fps)
        )
        max_similarity_to_selected = np.maximum(max_similarity_to_selected, similarities)
        available = np.ones(len(top500), dtype=bool)
        available[selected] = False
        remaining = np.flatnonzero(available)
        next_index = remaining[np.lexsort((
            -top500.ModelScore.to_numpy(dtype=np.float64)[remaining],
            max_similarity_to_selected[remaining],
        ))[0]]
        selected.append(int(next_index))

    diversity_order = {
        str(top500.iloc[index].CatalogID): order
        for order, index in enumerate(selected, start=1)
    }
    model_catalog_ids = set(top500.head(50).CatalogID)
    diverse_catalog_ids = set(diversity_order)
    practical = top500.loc[
        top500.CatalogID.isin(model_catalog_ids | diverse_catalog_ids)
    ].copy()
    practical["DiversityPickOrder"] = practical.CatalogID.map(diversity_order)
    practical["ReviewSets"] = [
        (
            "diverse_top50;model_top50"
            if catalog_id in model_catalog_ids and catalog_id in diverse_catalog_ids
            else "model_top50"
            if catalog_id in model_catalog_ids
            else "diverse_top50"
        )
        for catalog_id in practical.CatalogID
    ]
    practical["ExactMatchToRevealedDataset"] = practical.CanonicalSMILES.isin(
        known_structures
    )
    practical["NoveltyStatus"] = (
        "No exact match to local revealed dataset; challenge novelty not verified"
    )
    practical["AvailabilityStatus"] = (
        "Unverified; Enamine catalog and stock data not present"
    )
    practical["SolubilityStatus"] = (
        "Not measured; RDKit properties are not a solubility result"
    )

    filter_params = FilterCatalog.FilterCatalogParams()
    for filter_set in (
        FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS_A,
        FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS_B,
        FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS_C,
        FilterCatalog.FilterCatalogParams.FilterCatalogs.BRENK,
    ):
        filter_params.AddCatalog(filter_set)
    alert_catalog = FilterCatalog.FilterCatalog(filter_params)
    descriptors = [physicochemical_features(molecule) for molecule in molecules]
    descriptor_frame = pd.DataFrame({
        "CatalogID": top500.CatalogID,
        "MolecularWeight": [row[0] for row in descriptors],
        "RDKit_cLogP": [row[1] for row in descriptors],
        "TPSA_A2": [row[2] for row in descriptors],
        "HBD": [row[3] for row in descriptors],
        "HBA": [row[4] for row in descriptors],
        "RotatableBonds": [row[5] for row in descriptors],
        "RuleOfFiveViolations": [
            int(weight > 500) + int(logp > 5) + int(hbd > 5) + int(hba > 10)
            for weight, logp, _, hbd, hba, *_ in descriptors
        ],
        "PAINS_Brenk_Alerts": [
            ";".join(sorted({
                match.GetDescription()
                for match in alert_catalog.GetMatches(molecule)
            }))
            for molecule in molecules
        ],
    })
    practical = practical.merge(descriptor_frame, on="CatalogID", validate="one_to_one")
    practical = practical.sort_values("BaselineRank", kind="stable").reset_index(drop=True)
    columns = [
        "CatalogID", "ReviewSets", "BaselineRank", "DiversityPickOrder",
        "ModelScore", "MaxTanimotoToKnownHit", "SMILES", "MolecularWeight",
        "RDKit_cLogP", "TPSA_A2", "HBD", "HBA", "RotatableBonds",
        "RuleOfFiveViolations", "PAINS_Brenk_Alerts",
        "ExactMatchToRevealedDataset", "NoveltyStatus", "AvailabilityStatus",
        "SolubilityStatus",
    ]
    return practical[columns]


def append_ledger_rows(
    ledger_path: Path,
    output_records: list[dict[str, str]],
) -> None:
    columns = [
        "RunID", "RecordType", "RecordID", "Status", "LabelDataSHA256",
        "CandidateDataSHA256", "ModelOrSelection", "ParametersOrMetrics",
        "ArtifactPath", "ArtifactSHA256", "SubmissionReference", "Notes",
    ]
    if ledger_path.is_file():
        ledger = pd.read_csv(ledger_path, dtype=str, keep_default_na=False)
        missing = set(columns) - set(ledger.columns)
        if missing:
            raise ValueError(f"Experiment ledger is missing columns: {sorted(missing)}")
    else:
        ledger = pd.DataFrame(columns=columns)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    for row in output_records:
        row["RunID"] = run_id
    pd.concat([ledger, pd.DataFrame(output_records)], ignore_index=True).to_csv(
        ledger_path,
        index=False,
    )


def main() -> None:
    label_path = DATA_DIR / "retrospective_revealed_labels.csv"
    ranking_path = DATA_DIR / "retrospective_candidates_ranked.csv"
    practical_path = DATA_DIR / "retrospective_practical_screen.csv"
    output_path = DATA_DIR / "retrospective_top500_qsvc_audit.csv"
    ledger_path = DATA_DIR / "experiment_submission_log.csv"
    if not label_path.is_file():
        raise FileNotFoundError(f"Label file not found: {label_path}")

    labels_frame, y, known_structures = parse_and_deduplicate_labels(label_path)
    ranked = load_notebook_ranking(ranking_path)
    if len(ranked) < 500:
        raise ValueError(f"At least 500 ranked candidates are required; found {len(ranked)}")
    top500 = ranked.head(500).copy()
    practical = build_practical_screen(top500, known_structures, labels_frame)

    train_fingerprints, train_descriptors = feature_matrix(
        labels_frame.canonical.tolist()
    )
    candidate_fingerprints, candidate_descriptors = feature_matrix(
        top500.SMILES.astype(str).tolist()
    )
    train_angles, candidate_angles = prepare_angles(
        train_fingerprints,
        train_descriptors,
        candidate_fingerprints,
        candidate_descriptors,
    )
    train_states = quantum_states(train_angles)
    candidate_states = quantum_states(candidate_angles)
    negative_indices = np.flatnonzero(y == 0)
    positive_indices = np.flatnonzero(y == 1)
    if not len(positive_indices) or not len(negative_indices):
        raise ValueError("Both positive and negative labeled examples are required")

    result = top500[["BaselineRank", "CatalogID", "SMILES", "ModelScore"]].copy()
    run_config: dict[str, object] = {
        "seed": SEED,
        "ranking_source": ranking_path.name,
        "ranking_sha256": sha256_file(ranking_path),
        "ranking_model": str(ranked.RankingModel.iloc[0]),
        "feature_pipeline": (
            "2048-bit chirality-aware Morgan radius-2 fingerprints + 11 RDKit "
            "descriptors; TruncatedSVD(4), scaled descriptors, PCA(4), "
            "standardized then pi*tanh(x/2) angle scaling"
        ),
        "kernel": (
            f"Exact statevector ZZ fidelity kernel; {N_FEATURES} features, "
            f"{FEATURE_MAP_REPS} reps, linear entanglement"
        ),
        "estimator": f"sklearn SVC(kernel='precomputed', C={SVC_C}, class_weight='balanced')",
        "label_rows_after_cleaning": int(len(y)),
        "positive_rows": int(y.sum()),
        "candidate_rows": int(len(top500)),
    }

    for ratio in NEGATIVE_TO_POSITIVE_RATIOS:
        sampling_seed = SEED + ratio
        rng = np.random.default_rng(sampling_seed)
        sample_count = min(len(negative_indices), ratio * len(positive_indices))
        sampled_negatives = rng.choice(
            negative_indices,
            size=sample_count,
            replace=False,
        )
        sampled_train = np.concatenate((positive_indices, sampled_negatives))
        sampled_train = sampled_train[rng.permutation(len(sampled_train))]

        sampled_states = train_states[sampled_train]
        training_kernel = fidelity_kernel(sampled_states, sampled_states)
        training_kernel = (training_kernel + training_kernel.T) / 2.0
        np.fill_diagonal(training_kernel, 1.0)
        candidate_kernel = fidelity_kernel(candidate_states, sampled_states)
        classifier = SVC(
            kernel="precomputed",
            C=SVC_C,
            class_weight="balanced",
        )
        classifier.fit(training_kernel, y[sampled_train])
        decisions = classifier.decision_function(candidate_kernel)
        result[f"QSVC_1to{ratio}_Decision"] = decisions
        result[f"QSVC_1to{ratio}_Rank"] = (
            pd.Series(decisions)
            .rank(method="first", ascending=False)
            .astype(int)
            .to_numpy()
        )
        run_config[f"1_to_{ratio}_sample"] = {
            "seed": sampling_seed,
            "positive_count": int(len(positive_indices)),
            "negative_count": int(sample_count),
            "total_training_count": int(len(sampled_train)),
        }

    practical.to_csv(practical_path, index=False)
    temp_path = output_path.with_suffix(".csv.tmp")
    result.to_csv(temp_path, index=False)
    temp_path.replace(output_path)

    label_hash = sha256_file(label_path)
    ranking_hash = sha256_file(ranking_path)
    run_config["label_sha256"] = label_hash
    run_config["ranking_sha256"] = ranking_hash
    print(f"Wrote {len(practical)} practical-review candidates to {practical_path}")
    print(f"Wrote {len(result)} QSVC audit candidates to {output_path}")
    print("Run configuration:")
    print(json.dumps(run_config, indent=2, sort_keys=True))

    common = {
        "Status": "generated_for_review",
        "LabelDataSHA256": label_hash,
        "CandidateDataSHA256": ranking_hash,
        "SubmissionReference": "",
    }
    records = [
        {
            **common,
            "RecordType": "ranking_artifact",
            "RecordID": "practical_screen",
            "ModelOrSelection": f"Notebook ranking: {ranked.RankingModel.iloc[0]}",
            "ParametersOrMetrics": json.dumps({
                "rows": int(len(practical)),
                "selection": "union of model top 50 and diverse top 50 from ranked top 500",
                "availability_and_solubility": "not verified or measured",
            }, sort_keys=True),
            "ArtifactPath": practical_path.name,
            "ArtifactSHA256": sha256_file(practical_path),
            "Notes": "Retrospective review artifact only; not submitted.",
        },
        {
            **common,
            "RecordType": "quantum_audit",
            "RecordID": "retrospective_top500_qsvc_audit",
            "ModelOrSelection": "Four-feature ZZ fidelity-kernel SVC",
            "ParametersOrMetrics": json.dumps(run_config, sort_keys=True),
            "ArtifactPath": output_path.name,
            "ArtifactSHA256": sha256_file(output_path),
            "Notes": "Exploratory simulated-kernel audit only; not a submission.",
        },
    ]
    append_ledger_rows(ledger_path, records)
    print(f"Appended practical-screen and QSVC provenance to {ledger_path}")


if __name__ == "__main__":
    main()
