# Code, Derived Artifact, and Upstream EEG Archive Licensing

This repository maintains a strict separation between **executable source code**, **derived tabular benchmark artifacts**, and **upstream public EEG recordings**:

## 1. Software Source Code (`MIT` License)
All original Python modules and pipeline scripts (`src/`, `scripts/`, `README.md`, `requirements_lock.txt`) are released under the **MIT License** (see top-level `LICENSE`).

## 2. Derived Tabular Artifacts (`release_artifacts/` — `CC BY 4.0`)
Because the derived cohort tables consolidate harmonized features computed across archives that include a CC BY 4.0 upstream release (Figshare `4244171`), the derived tabular benchmark artifacts (`release_artifacts/derived_session_features_733.csv`, `release_artifacts/cv_fold_indices_split0.csv`, `release_artifacts/oof_predictions_split0.csv`, and Zenodo/Online Resource 2 machine-readable tables) are released under **Creative Commons Attribution 4.0 International (CC BY 4.0)** rather than MIT. Users must cite the corresponding primary datasets and this benchmark archive.

## 3. Upstream Raw EEG Datasets (External Hosts)
Raw EEG recordings are **not redistributed** inside this code repository; users retrieve them directly from their canonical public repositories under each dataset's respective license:

| Study | Repository & Accession | DOI | Upstream License |
| :--- | :--- | :--- | :---: |
| Study 1 (Sleep Deprivation) | OpenNeuro `ds004902` | `10.18112/openneuro.ds004902` (Xiang et al., 2024: `10.1038/s41597-024-03268-2`) | CC0 |
| Study 2 (Outpatient MDD) | Figshare `4244171` | `10.6084/m9.figshare.4244171` | CC BY 4.0 |
| Study 3 (Univ. Depression Screening) | OpenNeuro `ds003478` | `10.18112/openneuro.ds003478.v1.1.0` | CC0 |
| Study 4 (UCSD Parkinson's Disease) | OpenNeuro `ds002778` | `10.18112/openneuro.ds002778.v1.0.5` | CC0 |
| Study 5 (AHEPA Clinical Dementia) | OpenNeuro `ds004504` | `10.18112/openneuro.ds004504.v1.0.9` | CC0 |
| Study 6 (UNM PD Washout) | OpenNeuro `ds003490` | `10.18112/openneuro.ds003490.v1.1.0` | CC0 |
| Study 7 (Iowa PD Multi-Scale) | OpenNeuro `ds004584` | `10.18112/openneuro.ds004584.v1.0.0` | CC0 |
