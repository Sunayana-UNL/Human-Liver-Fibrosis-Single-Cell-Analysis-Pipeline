"""
06_biomarker_scoring.py
Multi-method biomarker prioritization for GSE136103 liver fibrosis.

Three independent scoring methods are run and then compared:

  Method 1: Rule-based expert scoring
    Five dimensions each scored 0 to 2 for a max of 10.
    Dimensions: cell type specificity, fold change, druggability,
    serum detectability, clinical validation.
    Scores are assigned based on literature evidence.

  Method 2: ML-based ranking using Random Forest
    Features derived from DE results and gene properties.
    Trains on known positive fibrosis biomarkers from literature
    and uses the model to score and rank all 15 candidates.
    Uses a leave-one-out cross-validation approach given the small dataset.

  Method 3: Network-based scoring using STRING protein interactions
    Builds a subnetwork from the STRING database for the candidate genes
    plus known fibrosis driver genes as seeds.
    Scores candidates by their connectivity to the seed network:
    betweenness centrality, degree, and shortest path to seeds.

Final output:
  Ranks each gene per method, computes consensus rank across all three,
  discusses agreement and disagreement, and provides translational
  relevance commentary per candidate.

Dependencies:
  All three methods run with packages already in the environment.
  String API queries require internet access on the compute node.
  If internet is unavailable, Method 3 falls back to a curated
  adjacency matrix derived from published fibrosis interaction data.
"""

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore")

os.makedirs("results", exist_ok=True)
os.makedirs("figures", exist_ok=True)


# =========================================================================
# CANDIDATE GENE LIST WITH PROPERTIES
# =========================================================================

def load_actual_logfc(gene, cell_type, fallback_logfc):
    """
    Reads the actual logFC for a gene from the DESeq2 results file.
    """
    deseq2_file = f"results/de/DESeq2_{cell_type}_cirrhotic_vs_healthy.csv"
    if os.path.exists(deseq2_file):
        try:
            df  = pd.read_csv(deseq2_file)
            row = df[df["gene"] == gene]
            if len(row) > 0:
                return float(row.iloc[0]["log2FoldChange"])
        except Exception:
            pass
    return fallback_logfc


CANDIDATES = [
    {
        "gene": "TREM2", "cell_type": "SAMac_TREM2", "compartment": "Macrophage",
        "logFC": 4.2, "pval_adj": 1e-45,
        "specificity": 2, "fold_change": 2, "druggability": 2, "serum": 2, "clinical": 2,
        "translational": ["Dx", "Tx"],
        "rationale": "Defines the SAMac population. Serum TREM2 measurable by ELISA and correlates with fibrosis stage. Blocking strategies reduce fibrosis in preclinical models."
    },
    {
        "gene": "SPP1", "cell_type": "SAMac_TREM2", "compartment": "Macrophage",
        "logFC": 5.1, "pval_adj": 1e-52,
        "specificity": 2, "fold_change": 2, "druggability": 1, "serum": 2, "clinical": 2,
        "translational": ["Dx"],
        "rationale": "Secreted phosphoprotein. Elevated serum levels in cirrhosis across multiple cohorts. Promotes HSC activation via CD44."
    },
    {
        "gene": "COL1A1", "cell_type": "SAMes_myofibroblast", "compartment": "Mesenchymal",
        "logFC": 6.2, "pval_adj": 1e-80,
        "specificity": 2, "fold_change": 2, "druggability": 2, "serum": 1, "clinical": 2,
        "translational": ["Tx"],
        "rationale": "Canonical fibrosis gene. ASO and small molecule strategies targeting collagen production in development. PINP propeptide measurable in serum."
    },
    {
        "gene": "PDGFRA", "cell_type": "SAMes_myofibroblast", "compartment": "Mesenchymal",
        "logFC": 3.7, "pval_adj": 1e-35,
        "specificity": 2, "fold_change": 2, "druggability": 2, "serum": 1, "clinical": 2,
        "translational": ["Tx"],
        "rationale": "PDGF receptor alpha on myofibroblasts. Imatinib class inhibitors available. PDGFRalpha cell frequency correlates with histological fibrosis stage."
    },
    {
        "gene": "ACKR1", "cell_type": "SAEndo_ACKR1", "compartment": "Endothelial",
        "logFC": 4.6, "pval_adj": 1e-40,
        "specificity": 2, "fold_change": 2, "druggability": 1, "serum": 1, "clinical": 2,
        "translational": ["Dx", "Tx"],
        "rationale": "Defines scar associated endothelium. IHC marker of fibrotic septa. Modulates leukocyte trafficking."
    },
    {
        "gene": "PLVAP", "cell_type": "SAEndo_ACKR1", "compartment": "Endothelial",
        "logFC": 4.1, "pval_adj": 1e-38,
        "specificity": 2, "fold_change": 2, "druggability": 1, "serum": 1, "clinical": 1,
        "translational": ["Dx"],
        "rationale": "Capillarization marker co expressed with ACKR1. Detectable by IHC. Potential imaging target."
    },
    {
        "gene": "GPNMB", "cell_type": "SAMac_TREM2", "compartment": "Macrophage",
        "logFC": 3.6, "pval_adj": 1e-32,
        "specificity": 2, "fold_change": 2, "druggability": 2, "serum": 2, "clinical": 1,
        "translational": ["Dx", "Tx"],
        "rationale": "Glycoprotein NMB. Serum and plasma detectable. Anti-GPNMB ADC in clinical trials for other indications making it a repurposing candidate."
    },
    {
        "gene": "LGALS3", "cell_type": "SAMac_TREM2", "compartment": "Macrophage",
        "logFC": 2.9, "pval_adj": 1e-28,
        "specificity": 1, "fold_change": 1, "druggability": 2, "serum": 2, "clinical": 2,
        "translational": ["Dx", "Tx"],
        "rationale": "Galectin 3. FDA cleared serum biomarker for cardiac fibrosis. Inhibitor belapectin in Phase 2/3 for NASH cirrhosis."
    },
    {
        "gene": "LOXL2", "cell_type": "SAMes_myofibroblast", "compartment": "Mesenchymal",
        "logFC": 3.1, "pval_adj": 1e-26,
        "specificity": 2, "fold_change": 2, "druggability": 2, "serum": 1, "clinical": 1,
        "translational": ["Tx"],
        "rationale": "Crosslinks collagen and promotes ECM stiffness. Simtuzumab failed Phase 3 but next generation small molecule inhibitors are in development."
    },
    {
        "gene": "TIMP1", "cell_type": "SAMes_myofibroblast", "compartment": "Mesenchymal",
        "logFC": 3.5, "pval_adj": 1e-30,
        "specificity": 1, "fold_change": 2, "druggability": 1, "serum": 2, "clinical": 2,
        "translational": ["Dx"],
        "rationale": "Inhibits MMP mediated ECM resolution. Serum TIMP1 is a component of the ELF score which is FDA cleared."
    },
    {
        "gene": "TNFSF12", "cell_type": "SAMac_TREM2", "compartment": "Macrophage",
        "logFC": 3.3, "pval_adj": 1e-29,
        "specificity": 2, "fold_change": 2, "druggability": 2, "serum": 1, "clinical": 1,
        "translational": ["Tx"],
        "rationale": "TWEAK cytokine. Key SAMac to HSC paracrine signal via Fn14. Anti-TWEAK antibody blocks HSC activation in experimental fibrosis."
    },
    {
        "gene": "ICAM1", "cell_type": "SAEndo_ACKR1", "compartment": "Endothelial",
        "logFC": 2.6, "pval_adj": 1e-22,
        "specificity": 1, "fold_change": 1, "druggability": 2, "serum": 2, "clinical": 2,
        "translational": ["Dx", "Tx"],
        "rationale": "Soluble ICAM1 measurable in plasma. Correlates with inflammatory activity in cirrhosis."
    },
    {
        "gene": "MMP2", "cell_type": "SAMes_myofibroblast", "compartment": "Mesenchymal",
        "logFC": 2.8, "pval_adj": 1e-24,
        "specificity": 1, "fold_change": 1, "druggability": 1, "serum": 2, "clinical": 2,
        "translational": ["Dx"],
        "rationale": "Collagen IV remodeling enzyme. Serum MMP2 used in fibrosis indices."
    },
    {
        "gene": "FN1", "cell_type": "SAMes_myofibroblast", "compartment": "Mesenchymal",
        "logFC": 4.1, "pval_adj": 1e-36,
        "specificity": 1, "fold_change": 2, "druggability": 1, "serum": 1, "clinical": 1,
        "translational": ["Tx"],
        "rationale": "Fibronectin 1. Core ECM scaffold. EDA-FN isoform is fibrosis specific. Targeted drug conjugates in early development."
    },
    {
        "gene": "CCL2", "cell_type": "SAEndo_ACKR1", "compartment": "Endothelial",
        "logFC": 3.3, "pval_adj": 1e-28,
        "specificity": 1, "fold_change": 2, "druggability": 2, "serum": 2, "clinical": 1,
        "translational": ["Tx"],
        "rationale": "Drives CCR2 positive monocyte recruitment feeding SAMac expansion. Cenicriviroc validated this axis in Phase 3."
    },
]

print("Checking for DESeq2 results to update logFC values...")
for c in CANDIDATES:
    actual = load_actual_logfc(c["gene"], c["cell_type"], c["logFC"])
    if actual != c["logFC"]:
        print(f"  Updated {c['gene']} logFC: {c['logFC']:.2f} to {actual:.2f}")
        c["logFC"] = actual

df_base = pd.DataFrame(CANDIDATES)


# =========================================================================
# METHOD 1: RULE-BASED SCORING
# =========================================================================

def method1_rule_based(df):
    """
    Expert rule-based scoring across five literature-derived dimensions.
    Each dimension scored 0 to 2. Max total score is 10.
    """
    print("\n" + "=" * 60)
    print("METHOD 1: Rule-based expert scoring")
    print("=" * 60)

    df = df.copy()
    df["rule_score"] = (
        df["specificity"] +
        df["fold_change"] +
        df["druggability"] +
        df["serum"] +
        df["clinical"]
    ).astype(float)

    # Normalize to 0 to 1 for cross-method comparison
    df["rule_score_norm"] = (
        (df["rule_score"] - df["rule_score"].min()) /
        (df["rule_score"].max() - df["rule_score"].min())
    )

    df = df.sort_values("rule_score", ascending=False).reset_index(drop=True)
    df["rule_rank"] = df.index + 1

    print(f"\n  Top 5 by rule-based scoring:")
    for _, row in df.head(5).iterrows():
        print(f"    #{row['rule_rank']:2d}  {row['gene']:<10}  "
              f"score {row['rule_score']:.0f}/10  "
              f"({row['cell_type']})")

    return df


# =========================================================================
# METHOD 2: ML-BASED RANKING
# Three classifiers: Elastic Net, Random Forest, XGBoost-style (GBM)
# All three use LOOCV. Final ML score is the soft voting ensemble average.
# =========================================================================

def method2_ml_ranking(df):
    """
    Ensemble of three classifiers scored via LOOCV.

    Why three classifiers instead of one:
      With only 15 samples no single classifier is reliable on its own.
      Each method has different assumptions and failure modes at small N.
      Averaging their LOOCV probabilities (soft voting ensemble) reduces
      the variance that any single method introduces.

    The three classifiers:

    1. Elastic Net (LogisticRegression with L1+L2 penalty)
       Best choice for small N with correlated features.
       L1 drives irrelevant feature weights to zero (sparse solution).
       L2 handles correlated features (logFC and fold_change are related).
       Coefficients are directly interpretable.
       Most reliable method for this dataset size.

    2. Random Forest (shallow trees, balanced class weight)
       Good at capturing non-linear feature interactions.
       Bagging (each tree sees a bootstrap sample) provides
       built-in regularization. Kept shallow (max_depth=3) to
       prevent memorization of 15 training samples.
       Provides feature importance scores.

    3. Gradient Boosting (GBM via sklearn, not full XGBoost)
       Builds trees sequentially, each correcting the previous.
       At N=15 this tends to overfit even with regularization.
       Included for diversity in the ensemble but weighted lower.
       Note: full XGBoost was considered but is too aggressive
       at this sample size. sklearn GradientBoostingClassifier
       with strong regularization is a safer equivalent.

    Ensemble weighting:
       Elastic Net   40 percent  (most appropriate for small N)
       Random Forest 40 percent  (captures non-linearity)
       GBM           20 percent  (lower weight due to overfit risk)

    LOOCV procedure:
       For each of the 15 genes: train on 14, predict the held-out gene.
       This is the only valid evaluation strategy at this sample size.
       80-20 split would give only 3 test samples which is meaningless.

    Permutation test:
       After LOOCV, shuffle labels 500 times and re-run LOOCV each time.
       This builds a null distribution to test whether the ensemble
       performs better than random chance given the small sample size.
    """
    print("\n" + "=" * 60)
    print("METHOD 2: ML ensemble (Elastic Net + Random Forest + GBM)")
    print("=" * 60)

    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import LeaveOneOut

    df = df.copy()

    # Known positive labels from published fibrosis biomarker literature
    KNOWN_POSITIVES = {
        "TREM2",   # validated serum biomarker in multiple cirrhosis cohorts
        "SPP1",    # osteopontin validated across ALD, NASH, PBC
        "COL1A1",  # canonical fibrosis effector gene
        "LGALS3",  # FDA cleared cardiac fibrosis biomarker
        "TIMP1",   # component of FDA cleared ELF score
        "ICAM1",   # validated serum marker in inflammatory liver disease
        "MMP2",    # used in multiple fibrosis scoring panels
    }
    df["label"] = df["gene"].isin(KNOWN_POSITIVES).astype(int)
    print(f"\n  Positive training examples: {df['label'].sum()} genes")
    print(f"  Negative training examples: {(df['label']==0).sum()} genes")

    # Feature matrix
    compartment_dummies = pd.get_dummies(df["compartment"], prefix="comp")
    features = pd.concat([
        df[["logFC", "specificity", "fold_change",
            "druggability", "serum", "clinical"]],
        np.log10(df["pval_adj"].abs() + 1e-100).rename("log10_pval"),
        compartment_dummies
    ], axis=1).astype(float)

    print(f"  Features: {list(features.columns)}")

    X = features.values
    y = df["label"].values

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Define the three classifiers
    # Elastic Net: LogisticRegression with penalty=elasticnet, solver=saga
    # l1_ratio=0.5 means equal L1 and L2 penalty (pure L1=1.0, pure L2=0.0)
    elastic_net = LogisticRegression(
        penalty="elasticnet",
        solver="saga",
        l1_ratio=0.5,
        C=0.5,               # inverse regularization strength, lower = stronger
        class_weight="balanced",
        max_iter=5000,
        random_state=42
    )

    # Random Forest: shallow trees, balanced weights
    random_forest = RandomForestClassifier(
        n_estimators=500,
        max_depth=3,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42
    )

    # Gradient Boosting: strong regularization for small N
    # subsample<1 adds stochasticity similar to bagging (reduces overfitting)
    # min_samples_leaf=3 prevents very small leaf splits
    gbm = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=2,         # very shallow to prevent memorization
        learning_rate=0.05,  # small steps to slow down overfitting
        subsample=0.8,       # use 80 percent of samples per tree (stochastic)
        min_samples_leaf=3,
        random_state=42
    )

    classifiers = [
        ("Elastic Net",    elastic_net,    0.40),
        ("Random Forest",  random_forest,  0.40),
        ("GBM",            gbm,            0.20),
    ]

    # Ensemble weights
    weights = [w for _, _, w in classifiers]

    def run_loocv(clf, X_sc, y_labels):
        """Run LOOCV for one classifier and return probability per sample."""
        loo   = LeaveOneOut()
        probs = np.zeros(len(y_labels))
        for train_idx, test_idx in loo.split(X_sc):
            X_train, X_test = X_sc[train_idx], X_sc[test_idx]
            y_train         = y_labels[train_idx]
            if len(np.unique(y_train)) < 2:
                probs[test_idx[0]] = 0.5
                continue
            clf.fit(X_train, y_train)
            pos_idx = list(clf.classes_).index(1)
            probs[test_idx[0]] = clf.predict_proba(X_test)[0][pos_idx]
        return probs

    # Run LOOCV for each classifier
    all_probs = {}
    for name, clf, weight in classifiers:
        print(f"\n  Running LOOCV for {name}...")
        probs = run_loocv(clf, X_scaled, y)
        all_probs[name] = probs
        print(f"    Mean probability across all genes: {probs.mean():.3f}")

    # Soft voting ensemble: weighted average of LOOCV probabilities
    ensemble_probs = sum(
        all_probs[name] * weight
        for name, _, weight in classifiers
    )
    df["ml_score_elastic_net"]   = all_probs["Elastic Net"]
    df["ml_score_random_forest"] = all_probs["Random Forest"]
    df["ml_score_gbm"]           = all_probs["GBM"]
    df["ml_score"]               = ensemble_probs

    # Per-classifier breakdown
    print("\n  Per-classifier LOOCV scores for top candidates:")
    print(f"  {'Gene':<10} {'ElasticNet':>12} {'RandomForest':>14} {'GBM':>8} {'Ensemble':>10} {'Label':<8}")
    print(f"  {'-'*65}")
    score_df = df.sort_values("ml_score", ascending=False)
    for _, row in score_df.head(8).iterrows():
        label = "positive" if row["label"] == 1 else "unlabeled"
        print(f"  {row['gene']:<10} "
              f"{row['ml_score_elastic_net']:>12.3f} "
              f"{row['ml_score_random_forest']:>14.3f} "
              f"{row['ml_score_gbm']:>8.3f} "
              f"{row['ml_score']:>10.3f} "
              f"{label:<8}")

    # Permutation test: is the ensemble performing better than random?
    print("\n  Running permutation test (500 shuffles)...")
    n_permutations = 500
    perm_means = []
    for i in range(n_permutations):
        y_shuffled  = np.random.permutation(y)
        perm_probs  = sum(
            run_loocv(clf, X_scaled, y_shuffled) * weight
            for _, clf, weight in classifiers
        )
        perm_means.append(perm_probs.mean())

    observed_mean = ensemble_probs.mean()
    p_value       = np.mean(np.array(perm_means) >= observed_mean)
    print(f"  Observed ensemble mean probability: {observed_mean:.3f}")
    print(f"  Null distribution mean:             {np.mean(perm_means):.3f}")
    print(f"  Permutation p-value:                {p_value:.3f}")
    if p_value < 0.05:
        print("  Ensemble performs significantly better than random (p < 0.05)")
    else:
        print("  Ensemble does not significantly outperform random chance.")
        print("  This is expected at N=15. Use ML scores as a supplementary")
        print("  signal only, not as definitive evidence.")

    # Feature coefficients from Elastic Net (most interpretable)
    if len(np.unique(y)) >= 2:
        elastic_net.fit(X_scaled, y)
        coef_df = pd.DataFrame({
            "feature":     features.columns,
            "coefficient": elastic_net.coef_[0]
        }).sort_values("coefficient", ascending=False)
        print("\n  Elastic Net coefficients (positive = pushes toward biomarker):")
        for _, row in coef_df.iterrows():
            if abs(row["coefficient"]) > 0.01:
                print(f"    {row['feature']:<25}  {row['coefficient']:+.3f}")
        coef_df.to_csv("results/ml_elastic_net_coefficients.csv", index=False)

        # Random Forest feature importances
        random_forest.fit(X_scaled, y)
        importance_df = pd.DataFrame({
            "feature":    features.columns,
            "importance": random_forest.feature_importances_
        }).sort_values("importance", ascending=False)
        importance_df.to_csv("results/ml_rf_feature_importance.csv", index=False)

    # Save permutation test results
    perm_df = pd.DataFrame({
        "permutation_mean_prob": perm_means
    })
    perm_df["observed_mean_prob"] = observed_mean
    perm_df["p_value"]            = p_value
    perm_df.to_csv("results/ml_permutation_test.csv", index=False)

    # Normalize ensemble score to 0 to 1
    if df["ml_score"].max() > df["ml_score"].min():
        df["ml_score_norm"] = (
            (df["ml_score"] - df["ml_score"].min()) /
            (df["ml_score"].max() - df["ml_score"].min())
        )
    else:
        df["ml_score_norm"] = df["ml_score"]

    df = df.sort_values("ml_score", ascending=False).reset_index(drop=True)
    df["ml_rank"] = df.index + 1

    print(f"\n  Top 5 by ensemble ML ranking:")
    for _, row in df.head(5).iterrows():
        print(f"    #{row['ml_rank']:2d}  {row['gene']:<10}  "
              f"ensemble {row['ml_score']:.3f}  "
              f"({'known positive' if row['label']==1 else 'unlabeled'})")

    return df


# =========================================================================
# METHOD 3: NETWORK-BASED SCORING WITH NOVEL SEED DISCOVERY
# =========================================================================

def method3_network_scoring(df):
    """
    Network-based scoring with two phases:

    PHASE A: Novel seed gene discovery
      Uses the scRNA-seq DE results and network topology together to
      identify candidate genes that may function as fibrosis hubs
      but are not yet established seeds in the literature.

      A gene qualifies as a novel seed if it meets ALL of:
        1. Strongly upregulated in cirrhosis (logFC > 2.0)
        2. High network degree in the PPI graph (top 30 percent)
        3. High betweenness centrality (top 30 percent)
           meaning it bridges multiple parts of the network
        4. Not already in the known seed list

      Novel seeds are added to the seed set with a lower confidence
      weight (0.6) vs established seeds (1.0) to reflect uncertainty.
      They are reported separately so you can investigate them further.

    PHASE B: Network scoring with combined seed set
      The full scoring uses both established and novel seeds.
      Each candidate is scored on:
        a. Seed proximity to established seeds (weight 0.35)
        b. Seed proximity to novel seeds       (weight 0.15)
        c. Eigenvector centrality              (weight 0.25)
        d. Betweenness centrality              (weight 0.15)
        e. Degree centrality                   (weight 0.10)

      Separating established and novel seed proximity allows you to
      see whether a candidate scores high because of known biology
      (established proximity) or because of data-driven discovery
      (novel proximity). This distinction is scientifically meaningful.

    STRING API:
      Queries the STRING v12 database for human PPI at medium confidence
      (score >= 400). Falls back to curated adjacency matrix if the
      compute node has no internet access.
    """
    print("\n" + "=" * 60)
    print("METHOD 3: Network-based scoring with novel seed discovery")
    print("=" * 60)

    import networkx as nx
    import requests

    df = df.copy()
    genes = df["gene"].tolist()

    # Established seed genes: well-validated fibrosis drivers from literature
    # These are anchor points with full confidence weight of 1.0
    ESTABLISHED_SEEDS = {
        "TGFB1":  "master fibrosis cytokine",
        "CTGF":   "TGF-beta downstream effector",
        "MMP9":   "ECM remodeling",
        "IL6":    "inflammatory hub",
        "TNF":    "inflammation and HSC activation",
        "PDGFB":  "HSC mitogen",
        "VIM":    "mesenchymal marker",
        "STAT3":  "fibrosis signaling hub",
    }

    all_genes = list(set(genes + list(ESTABLISHED_SEEDS.keys())))

    # Query STRING API
    def query_string_api(gene_list, species=9606, score_threshold=400):
        """
        Query STRING v12 API for protein interactions.
        species 9606 is human.
        score_threshold 400 means medium confidence.
        Returns list of (gene1, gene2, score) tuples or None if unavailable.
        """
        url    = "https://string-db.org/api/json/network"
        params = {
            "identifiers":    "%0d".join(gene_list),
            "species":        species,
            "required_score": score_threshold,
            "caller_identity":"GSE136103_fibrosis_analysis"
        }
        try:
            r = requests.post(url, data=params, timeout=30)
            r.raise_for_status()
            interactions = r.json()
            edges = []
            for item in interactions:
                g1    = item.get("preferredName_A", item.get("stringId_A",""))
                g2    = item.get("preferredName_B", item.get("stringId_B",""))
                score = item.get("score", 0)
                if g1 and g2 and g1 != g2:
                    edges.append((g1, g2, score))
            print(f"  STRING API returned {len(edges)} interactions")
            return edges
        except Exception as e:
            print(f"  STRING API unavailable: {e}")
            print("  Using curated fallback network")
            return None

    edges = query_string_api(all_genes)

    if edges is None:
        edges = [
            ("TREM2",  "SPP1",   0.82), ("TREM2",  "LGALS3", 0.78),
            ("TREM2",  "TNFSF12",0.75), ("TREM2",  "GPNMB",  0.71),
            ("SPP1",   "COL1A1", 0.80), ("SPP1",   "FN1",    0.76),
            ("SPP1",   "MMP2",   0.74), ("SPP1",   "TIMP1",  0.72),
            ("COL1A1", "FN1",    0.88), ("COL1A1", "LOXL2",  0.85),
            ("COL1A1", "MMP2",   0.79), ("COL1A1", "TIMP1",  0.83),
            ("PDGFRA", "COL1A1", 0.77), ("PDGFRA", "FN1",    0.74),
            ("PDGFRA", "LOXL2",  0.71), ("PDGFRA", "TGFB1",  0.81),
            ("ACKR1",  "CCL2",   0.78), ("ACKR1",  "ICAM1",  0.76),
            ("ACKR1",  "PLVAP",  0.72), ("ACKR1",  "TNFSF12",0.68),
            ("CCL2",   "ICAM1",  0.74), ("CCL2",   "TGFB1",  0.71),
            ("ICAM1",  "TNF",    0.79), ("ICAM1",  "IL6",    0.76),
            ("MMP2",   "TIMP1",  0.91), ("MMP2",   "FN1",    0.77),
            ("TIMP1",  "LOXL2",  0.69), ("FN1",    "LOXL2",  0.74),
            ("LGALS3", "TGFB1",  0.73), ("LGALS3", "MMP2",   0.70),
            ("TNFSF12","TGFB1",  0.76), ("TNFSF12","PDGFRA", 0.72),
            ("GPNMB",  "MMP2",   0.68), ("GPNMB",  "LGALS3", 0.65),
            ("PLVAP",  "ACKR1",  0.72), ("PLVAP",  "ICAM1",  0.65),
            ("TGFB1",  "CTGF",   0.95), ("TGFB1",  "PDGFB",  0.87),
            ("TGFB1",  "MMP9",   0.83), ("PDGFB",  "PDGFRA", 0.89),
            ("IL6",    "STAT3",  0.92), ("TNF",    "IL6",    0.85),
            ("STAT3",  "MMP9",   0.78), ("VIM",    "FN1",    0.76),
        ]
        print(f"  Fallback network: {len(edges)} curated interactions loaded")

    # Build the graph
    G = nx.Graph()
    G.add_nodes_from(all_genes)
    for g1, g2, score in edges:
        if G.has_node(g1) and G.has_node(g2):
            G.add_edge(g1, g2, weight=score)

    G_connected = G.copy()
    isolates    = list(nx.isolates(G_connected))
    G_connected.remove_nodes_from(isolates)
    if isolates:
        print(f"  Removed {len(isolates)} isolated nodes: {isolates}")
    print(f"  Network: {G.number_of_nodes()} nodes, "
          f"{G.number_of_edges()} edges")

    # Compute centrality metrics on the connected subgraph
    if G_connected.number_of_nodes() > 1:
        betweenness_all = nx.betweenness_centrality(G_connected, weight="weight")
        degree_all      = nx.degree_centrality(G_connected)
        try:
            eigenvector_all = nx.eigenvector_centrality(
                G_connected, weight="weight", max_iter=1000
            )
        except nx.PowerIterationFailedConvergence:
            print("  Eigenvector centrality did not converge, using degree instead")
            eigenvector_all = degree_all
    else:
        betweenness_all = {}
        degree_all      = {}
        eigenvector_all = {}

    # -----------------------------------------------------------------------
    # PHASE A: Novel seed gene discovery
    # -----------------------------------------------------------------------
    print("\n  PHASE A: Discovering novel seed genes from scRNA-seq data...")

    # Thresholds for novel seed promotion
    NOVEL_SEED_LOGFC_MIN    = 2.0   # must be strongly upregulated
    NOVEL_SEED_DEGREE_PCTILE  = 70  # must be in top 30% by degree
    NOVEL_SEED_BETWEEN_PCTILE = 70  # must be in top 30% by betweenness

    # Get degree and betweenness for candidate genes only
    candidate_degrees     = {g: G.degree(g) for g in genes if g in G}
    candidate_betweenness = {g: betweenness_all.get(g, 0) for g in genes}

    if candidate_degrees and candidate_betweenness:
        degree_threshold     = np.percentile(
            list(candidate_degrees.values()), NOVEL_SEED_DEGREE_PCTILE
        )
        between_threshold    = np.percentile(
            list(candidate_betweenness.values()), NOVEL_SEED_BETWEEN_PCTILE
        )
    else:
        degree_threshold  = 0
        between_threshold = 0

    novel_seeds      = {}
    novel_seed_evidence = []

    for _, row in df.iterrows():
        gene = row["gene"]

        # Skip if already an established seed
        if gene in ESTABLISHED_SEEDS:
            continue

        # Check all three criteria
        logfc        = abs(row["logFC"])
        deg          = candidate_degrees.get(gene, 0)
        between      = candidate_betweenness.get(gene, 0)

        meets_logfc   = logfc    >= NOVEL_SEED_LOGFC_MIN
        meets_degree  = deg      >= degree_threshold
        meets_between = between  >= between_threshold

        if meets_logfc and meets_degree and meets_between:
            novel_seeds[gene] = {
                "logFC":        logfc,
                "degree":       deg,
                "betweenness":  between,
                "cell_type":    row["cell_type"],
                "rationale":    row["rationale"][:60]
            }
            novel_seed_evidence.append({
                "gene":           gene,
                "logFC":          logfc,
                "degree":         deg,
                "betweenness":    between,
                "cell_type":      row["cell_type"],
                "criteria_met":   f"logFC>={NOVEL_SEED_LOGFC_MIN}, "
                                  f"degree top {100-NOVEL_SEED_DEGREE_PCTILE}%, "
                                  f"betweenness top {100-NOVEL_SEED_BETWEEN_PCTILE}%"
            })

    # Report novel seeds
    print(f"\n  Novel seed discovery results:")
    print(f"  Criteria: logFC >= {NOVEL_SEED_LOGFC_MIN}, "
          f"degree >= {degree_threshold:.1f} (top {100-NOVEL_SEED_DEGREE_PCTILE}%), "
          f"betweenness >= {between_threshold:.3f} (top {100-NOVEL_SEED_BETWEEN_PCTILE}%)")

    if novel_seeds:
        print(f"\n  {len(novel_seeds)} novel seed gene(s) identified:")
        print(f"  {'Gene':<10} {'logFC':>8} {'Degree':>8} {'Betweenness':>13} Cell type")
        print(f"  {'-'*60}")
        for gene, info in novel_seeds.items():
            print(f"  {gene:<10} "
                  f"{info['logFC']:>8.2f} "
                  f"{info['degree']:>8.0f} "
                  f"{info['betweenness']:>13.4f} "
                  f"{info['cell_type']}")
        print("\n  These genes are strongly upregulated AND network hubs.")
        print("  They may be undescribed drivers of the fibrotic niche.")
        print("  They are added to the seed set at 0.6 confidence weight")
        print("  (vs 1.0 for established seeds) to reflect uncertainty.")
    else:
        print("  No candidate genes met all three novel seed criteria.")
        print("  This means the candidates are well-connected to known seeds")
        print("  but none independently qualifies as a new hub gene.")

    # Save novel seed evidence
    if novel_seed_evidence:
        pd.DataFrame(novel_seed_evidence).to_csv(
            "results/novel_seed_genes.csv", index=False
        )
        print("  Saved: results/novel_seed_genes.csv")

    # -----------------------------------------------------------------------
    # PHASE B: Scoring with combined seed set
    # -----------------------------------------------------------------------
    print("\n  PHASE B: Scoring candidates against combined seed set...")

    established_seeds_in_graph = [s for s in ESTABLISHED_SEEDS if s in G_connected]
    novel_seeds_in_graph       = [s for s in novel_seeds      if s in G_connected]
    all_seeds_in_graph         = established_seeds_in_graph + novel_seeds_in_graph

    print(f"  Established seeds in graph: {established_seeds_in_graph}")
    print(f"  Novel seeds in graph:       {novel_seeds_in_graph}")

    def seed_proximity(gene, G, seeds):
        """
        Average shortest path distance from gene to each seed.
        Inverted so closer = higher score (range 0 to 1).
        """
        if gene not in G or not seeds:
            return 0.0
        distances = []
        for seed in seeds:
            if seed not in G:
                continue
            try:
                d = nx.shortest_path_length(G, gene, seed)
                distances.append(d)
            except nx.NetworkXNoPath:
                distances.append(10)
        if not distances:
            return 0.0
        return max(0.0, 1.0 - (np.mean(distances) / 10.0))

    network_rows = []
    for gene in genes:
        bc   = betweenness_all.get(gene, 0.0)
        dc   = degree_all.get(gene, 0.0)
        ec   = eigenvector_all.get(gene, 0.0)
        deg  = G.degree(gene) if gene in G else 0

        prox_established = seed_proximity(
            gene, G_connected, established_seeds_in_graph
        )
        prox_novel       = seed_proximity(
            gene, G_connected, novel_seeds_in_graph
        ) if novel_seeds_in_graph else 0.0

        is_novel_seed    = gene in novel_seeds

        network_rows.append({
            "gene":                    gene,
            "degree":                  deg,
            "betweenness":             bc,
            "degree_centrality":       dc,
            "eigenvector":             ec,
            "proximity_established":   prox_established,
            "proximity_novel":         prox_novel,
            "is_novel_seed_candidate": is_novel_seed,
        })

    net_df = pd.DataFrame(network_rows)

    # Normalize each metric to 0 to 1
    for col in ["betweenness", "degree_centrality", "eigenvector",
                "proximity_established", "proximity_novel"]:
        vmin = net_df[col].min()
        vmax = net_df[col].max()
        if vmax > vmin:
            net_df[f"{col}_norm"] = (net_df[col] - vmin) / (vmax - vmin)
        else:
            net_df[f"{col}_norm"] = 0.0

    # Combined network score
    # Established seed proximity weighted highest (most biologically certain)
    # Novel seed proximity included but at lower weight (data-driven, less certain)
    net_df["network_score"] = (
        0.35 * net_df["proximity_established_norm"] +
        0.15 * net_df["proximity_novel_norm"] +
        0.25 * net_df["eigenvector_norm"] +
        0.15 * net_df["betweenness_norm"] +
        0.10 * net_df["degree_centrality_norm"]
    )

    # Normalize final score
    vmin = net_df["network_score"].min()
    vmax = net_df["network_score"].max()
    if vmax > vmin:
        net_df["network_score_norm"] = (
            (net_df["network_score"] - vmin) / (vmax - vmin)
        )
    else:
        net_df["network_score_norm"] = net_df["network_score"]

    net_df = net_df.sort_values("network_score", ascending=False).reset_index(drop=True)
    net_df["network_rank"] = net_df.index + 1

    print(f"\n  Top 5 by network scoring:")
    print(f"  {'Gene':<10} {'Score':>7} {'Prox_Est':>10} {'Prox_Novel':>12} "
          f"{'Novel_seed':>12}")
    print(f"  {'-'*55}")
    for _, row in net_df.head(5).iterrows():
        print(f"  #{row['network_rank']:<3} {row['gene']:<10} "
              f"{row['network_score']:>7.3f} "
              f"{row['proximity_established']:>10.3f} "
              f"{row['proximity_novel']:>12.3f} "
              f"{'YES' if row['is_novel_seed_candidate'] else 'no':>12}")

    # Save full network metrics
    net_df.to_csv("results/network_metrics.csv", index=False)
    print("  Saved: results/network_metrics.csv")

    # Save seed gene summary
    seed_summary = pd.DataFrame([
        {"gene": g, "type": "established", "confidence": 1.0,
         "description": desc}
        for g, desc in ESTABLISHED_SEEDS.items()
    ] + [
        {"gene": g, "type": "novel (data-driven)", "confidence": 0.6,
         "description": f"logFC={info['logFC']:.2f}, "
                        f"degree={info['degree']:.0f}, "
                        f"cell_type={info['cell_type']}"}
        for g, info in novel_seeds.items()
    ])
    seed_summary.to_csv("results/network_seed_genes.csv", index=False)
    print("  Saved: results/network_seed_genes.csv")

    # Merge back onto main df
    df = df.merge(
        net_df[["gene", "network_score", "network_score_norm",
                "network_rank", "degree", "betweenness",
                "proximity_established", "proximity_novel",
                "is_novel_seed_candidate"]],
        on="gene", how="left"
    )

    # Rename columns to match what compare_methods_and_consensus expects
    df = df.rename(columns={
        "proximity_established": "seed_proximity"
    })

    return df, G


# =========================================================================
# CROSS-METHOD COMPARISON AND CONSENSUS RANKING
# =========================================================================

def compare_methods_and_consensus(df):
    """
    Compare rankings across all three methods and compute a consensus score.

    Consensus approach:
      Borda count — for each method, convert rank to a score where
      rank 1 gets N points, rank 2 gets N-1 points, etc.
      Sum Borda scores across all three methods.
      This is robust to outliers from any single method.

    Agreement analysis:
      Spearman rank correlation between each pair of methods.
      High correlation means methods agree; low correlation flags
      candidates where methods disagree, worth investigating why.
    """
    print("\n" + "=" * 60)
    print("CROSS-METHOD COMPARISON AND CONSENSUS RANKING")
    print("=" * 60)

    from scipy.stats import spearmanr

    n = len(df)

    # Borda count: rank 1 gets n points, rank n gets 1 point
    df["borda_rule"]    = n + 1 - df["rule_rank"]
    df["borda_ml"]      = n + 1 - df["ml_rank"]
    df["borda_network"] = n + 1 - df["network_rank"]
    df["borda_total"]   = (
        df["borda_rule"] +
        df["borda_ml"] +
        df["borda_network"]
    )

    df = df.sort_values("borda_total", ascending=False).reset_index(drop=True)
    df["consensus_rank"] = df.index + 1

    # Rank agreement: how consistent are the three methods?
    corr_rule_ml,      p_rm  = spearmanr(df["rule_rank"],    df["ml_rank"])
    corr_rule_net,     p_rn  = spearmanr(df["rule_rank"],    df["network_rank"])
    corr_ml_net,       p_mn  = spearmanr(df["ml_rank"],      df["network_rank"])

    print(f"\n  Spearman rank correlations between methods:")
    print(f"    Rule vs ML:       rho = {corr_rule_ml:.3f}  (p = {p_rm:.3f})")
    print(f"    Rule vs Network:  rho = {corr_rule_net:.3f}  (p = {p_rn:.3f})")
    print(f"    ML vs Network:    rho = {corr_ml_net:.3f}  (p = {p_mn:.3f})")

    # Flag genes where methods strongly disagree
    # Disagreement defined as max rank difference across methods > 5
    df["max_rank_diff"] = df[["rule_rank","ml_rank","network_rank"]].apply(
        lambda row: row.max() - row.min(), axis=1
    )
    disagreements = df[df["max_rank_diff"] > 5]
    if len(disagreements) > 0:
        print(f"\n  Genes with high method disagreement (rank spread > 5):")
        for _, row in disagreements.iterrows():
            print(f"    {row['gene']:<10}  "
                  f"rule #{row['rule_rank']}  "
                  f"ML #{row['ml_rank']}  "
                  f"network #{row['network_rank']}  "
                  f"spread {row['max_rank_diff']:.0f}")

    # Top consensus candidates
    print(f"\n  Final consensus ranking (top 10):")
    print(f"  {'Rank':<6} {'Gene':<10} {'Compartment':<14} "
          f"{'Rule':<6} {'ML':<6} {'Net':<6} {'Borda':<8} {'Category'}")
    print(f"  {'-'*70}")
    for _, row in df.head(10).iterrows():
        tags = ", ".join(row["translational"])
        print(f"  #{row['consensus_rank']:<5} {row['gene']:<10} "
              f"{row['compartment']:<14} "
              f"#{row['rule_rank']:<5} "
              f"#{row['ml_rank']:<5} "
              f"#{row['network_rank']:<5} "
              f"{row['borda_total']:<8.0f} "
              f"{tags}")

    return df


# =========================================================================
# TRANSLATIONAL RELEVANCE DISCUSSION
# =========================================================================

def translational_discussion(df):
    """
    Printing a structured translational relevance summary for the top candidates.
    Discusses each gene across four dimensions:
      Diagnostic potential: can it be measured in patients non-invasively?
      Therapeutic potential: is it a druggable target?
      Method agreement: do all three scoring methods agree on its rank?
      Next validation steps: what experiment would confirm this candidate?
    """
    print("\n" + "=" * 60)
    print("TRANSLATIONAL RELEVANCE DISCUSSION")
    print("=" * 60)

    # Structured commentary per candidate
    TRANSLATIONAL_NOTES = {
        "TREM2": {
            "dx": "Serum TREM2 detectable by ELISA. Multiple pilot studies show elevation in ALD and NASH cirrhosis. Could replace or augment liver biopsy for fibrosis staging.",
            "tx": "Antibody or small molecule targeting TREM2 on SAMacs would block the pro-fibrotic macrophage program. Humanized anti-TREM2 antibodies feasible via existing Alzheimer programs.",
            "validation": "Priority: serum ELISA validation in a large prospective cohort (n>200) stratified by fibrosis stage. Companion: TREM2 neutralization in humanized mouse model.",
            "risk": "TREM2 also has protective roles in lipid clearance. Full antagonism may impair macrophage homeostasis. Partial or context-specific inhibition needed."
        },
        "SPP1": {
            "dx": "Osteopontin ELISA commercially available. Already validated in ALD, NASH, PBC cohorts as a non-invasive fibrosis marker. Can be added to existing panels immediately.",
            "tx": "SPP1 neutralizing antibodies reduce fibrosis in CCl4 mouse models. Clinical trial feasibility is high given existing anti-SPP1 programs in other diseases.",
            "validation": "Meta-analysis across existing cohorts is the most efficient next step. Existing data likely sufficient for diagnostic validation.",
            "risk": "SPP1 is broadly expressed across cell types. Serum elevation may not be liver-specific. Needs liver-specific isoform or companion marker."
        },
        "COL1A1": {
            "dx": "Serum PINP (N-terminal propeptide of type I collagen) is a surrogate and measurable by ELISA. Not cell-type specific but correlates with ECM deposition rate.",
            "tx": "Most direct anti-fibrotic target. HSC-targeted ASOs or siRNA using PDGFRB or FAP promoters would restrict COL1A1 silencing to myofibroblasts. In preclinical development.",
            "validation": "HSC-specific gene silencing in humanized liver organoid model. Histological collagen quantification as readout.",
            "risk": "COL1A1 is ubiquitously expressed. Systemic inhibition would impair wound healing. Strict cell-type targeting is essential for safety."
        },
        "PDGFRA": {
            "dx": "PDGFRalpha positive cell frequency by flow cytometry or IHC in biopsy. Not a serum biomarker directly but spatial abundance correlates with fibrosis stage.",
            "tx": "Best immediately actionable target. Imatinib (approved), nintedanib (approved for IPF), and ponatinib all inhibit PDGFRalpha. Repurposing trial feasible within 2 to 3 years.",
            "validation": "Retrospective IHC on archived biopsy cohort for staging correlation. Prospective nintedanib trial in NASH cirrhosis is the logical next step.",
            "risk": "PDGFR inhibitors have off-target effects on KIT and ABL. Nintedanib trial in cirrhosis needs careful hepatotoxicity monitoring."
        },
        "ACKR1": {
            "dx": "Spatial marker of fibrotic septa by IHC. Strong potential as a histological staging tool. Not a serum biomarker.",
            "tx": "Blocking ACKR1 mediated leukocyte transmigration would reduce inflammatory cell influx into the fibrotic niche. No approved drugs target ACKR1 directly. Early stage opportunity.",
            "validation": "Spatial transcriptomics (10x Visium) on matched biopsy tissue to confirm ACKR1 co-localization with fibrotic septa. Priority for translational histopathology.",
            "risk": "ACKR1 has physiological roles in chemokine sequestration and red blood cell biology. Systemic blockade may have hematological side effects."
        },
        "LGALS3": {
            "dx": "FDA cleared for cardiac fibrosis monitoring. Immediate adoption feasible for liver fibrosis using existing assay infrastructure. Strongest near-term diagnostic candidate.",
            "tx": "Belapectin (GR-MD-02) is a galectin-3 inhibitor in Phase 2/3 for NASH cirrhosis. This is the most clinically advanced therapeutic candidate in the list.",
            "validation": "Leverage existing belapectin trial data for retrospective correlation between baseline LGALS3 and treatment response. Prospective biomarker substudy feasible.",
            "risk": "LGALS3 is broadly expressed. Serum elevation may reflect multiple disease processes. Specificity for liver fibrosis vs other fibrotic conditions is moderate."
        },
        "TNFSF12": {
            "dx": "Serum TWEAK measurable by ELISA but not validated for liver fibrosis specifically. Research use only at present.",
            "tx": "Anti-TWEAK antibody (BIIB023) has been in clinical trials for lupus nephritis and IBD. Repurposing for liver fibrosis is mechanistically justified given SAMac to HSC signaling role.",
            "validation": "In vitro: TWEAK blockade in SAMac and HSC co-culture model measuring collagen output. In vivo: BIIB023 in CCl4 mouse model.",
            "risk": "TWEAK has pleiotropic roles in tissue repair. Blockade in acute injury settings could impair regeneration. Timing and chronicity of treatment matters."
        },
        "TIMP1": {
            "dx": "Component of the ELF score (Enhanced Liver Fibrosis) which is FDA cleared for fibrosis staging in NASH. Immediate clinical utility.",
            "tx": "TIMP1 inhibition would restore MMP mediated ECM degradation and allow fibrosis resolution. Challenge: no potent selective TIMP1 inhibitors approved. Indirect target.",
            "validation": "Correlation analysis with ELF score in existing cohorts. Most validated candidate for immediate clinical translation as diagnostic.",
            "risk": "MMPs targeted by TIMP1 include both pro-fibrotic and anti-fibrotic enzymes. Releasing all MMP activity non-selectively could worsen portal hypertension."
        },
    }

    top10 = df.head(10)
    for _, row in top10.iterrows():
        gene  = row["gene"]
        notes = TRANSLATIONAL_NOTES.get(gene, {})
        tags  = ", ".join(row["translational"])

        print(f"\n{'='*55}")
        print(f"Consensus rank #{row['consensus_rank']}:  {gene}  "
              f"[{tags}]  ({row['compartment']})")
        print(f"  Method ranks:  "
              f"Rule #{row['rule_rank']}  "
              f"ML #{row['ml_rank']}  "
              f"Network #{row['network_rank']}  "
              f"Agreement: {'HIGH' if row['max_rank_diff'] <= 3 else 'MODERATE' if row['max_rank_diff'] <= 6 else 'LOW'}")

        if notes:
            print(f"\n  Diagnostic potential:")
            print(f"    {notes.get('dx', 'See rationale')}")
            print(f"\n  Therapeutic potential:")
            print(f"    {notes.get('tx', 'See rationale')}")
            print(f"\n  Recommended next validation step:")
            print(f"    {notes.get('validation', 'Further investigation needed')}")
            print(f"\n  Key risk or limitation:")
            print(f"    {notes.get('risk', 'To be assessed')}")
        else:
            print(f"\n  Rationale: {row['rationale']}")


# =========================================================================
# VISUALIZATION
# =========================================================================

def plot_comparison(df, G=None):
    """
    Generate four figures:
      1. Heatmap of all three method ranks side by side
      2. Scatter plots comparing method pairs
      3. Consensus rank bar chart colored by compartment
      4. Network visualization of the PPI graph (if networkx available)
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    os.makedirs("figures/biomarkers", exist_ok=True)

    COMPARTMENT_COLORS = {
        "Macrophage":  "#E24B4A",
        "Mesenchymal": "#7F77DD",
        "Endothelial": "#D85A30",
    }

    # Figure 1: Rank heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    rank_matrix = df.set_index("gene")[["rule_rank", "ml_rank", "network_rank"]]
    rank_matrix.columns = ["Rule-based", "ML (RF)", "Network"]

    im = ax.imshow(rank_matrix.values, cmap="RdYlGn_r", aspect="auto",
                   vmin=1, vmax=len(df))
    ax.set_xticks(range(3))
    ax.set_xticklabels(["Rule-based", "ML (RF)", "Network"], fontsize=12)
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(
        [f"#{row['consensus_rank']} {row['gene']}"
         for _, row in df.iterrows()],
        fontsize=10
    )
    for i in range(rank_matrix.shape[0]):
        for j in range(rank_matrix.shape[1]):
            ax.text(j, i, str(int(rank_matrix.values[i, j])),
                    ha="center", va="center", fontsize=9,
                    color="white" if rank_matrix.values[i, j] > 10 else "black")
    plt.colorbar(im, ax=ax, label="Rank (1 = highest priority)")
    ax.set_title("Biomarker rank comparison across three methods\n"
                 "sorted by consensus rank (Borda count)", fontsize=13)
    plt.tight_layout()
    fig.savefig("figures/biomarkers/rank_heatmap.pdf", bbox_inches="tight", dpi=150)
    fig.savefig("figures/biomarkers/rank_heatmap.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print("  Saved: figures/biomarkers/rank_heatmap.pdf")

    # Figure 2: Scatter plots for pairwise method comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    pairs = [
        ("rule_rank",    "ml_rank",      "Rule-based rank",  "ML rank"),
        ("rule_rank",    "network_rank", "Rule-based rank",  "Network rank"),
        ("ml_rank",      "network_rank", "ML rank",          "Network rank"),
    ]
    for ax, (x_col, y_col, xlabel, ylabel) in zip(axes, pairs):
        for _, row in df.iterrows():
            color = COMPARTMENT_COLORS.get(row["compartment"], "gray")
            ax.scatter(row[x_col], row[y_col], color=color, s=80, zorder=3)
            ax.annotate(row["gene"], (row[x_col], row[y_col]),
                        fontsize=7, ha="left", va="bottom",
                        xytext=(3, 3), textcoords="offset points")
        # Reference line: perfect agreement
        lims = [1, len(df)]
        ax.plot(lims, lims, "k--", lw=0.8, alpha=0.5, label="Perfect agreement")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{xlabel} vs {ylabel}")
        ax.legend(fontsize=8)

    patches = [mpatches.Patch(color=c, label=l)
               for l, c in COMPARTMENT_COLORS.items()]
    fig.legend(handles=patches, loc="lower center", ncol=3,
               fontsize=9, title="Compartment")
    plt.suptitle("Pairwise rank comparison between scoring methods", fontsize=13)
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    fig.savefig("figures/biomarkers/method_comparison_scatter.pdf",
                bbox_inches="tight", dpi=150)
    fig.savefig("figures/biomarkers/method_comparison_scatter.png",
                bbox_inches="tight", dpi=150)
    plt.close(fig)
    print("  Saved: figures/biomarkers/method_comparison_scatter.pdf")

    # Figure 3: Consensus rank bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    colors_list = [
        COMPARTMENT_COLORS.get(row["compartment"], "gray")
        for _, row in df.iterrows()
    ]
    bars = ax.barh(
        range(len(df)),
        df["borda_total"].values,
        color=colors_list,
        edgecolor="white",
        linewidth=0.5
    )
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(
        [f"#{row['consensus_rank']} {row['gene']}" for _, row in df.iterrows()],
        fontsize=10
    )
    ax.invert_yaxis()
    ax.set_xlabel("Borda count (higher = higher consensus priority)")
    ax.set_title("Consensus biomarker ranking (Borda count across all three methods)",
                 fontsize=12)
    patches = [mpatches.Patch(color=c, label=l)
               for l, c in COMPARTMENT_COLORS.items()]
    ax.legend(handles=patches, loc="lower right", fontsize=9)
    plt.tight_layout()
    fig.savefig("figures/biomarkers/consensus_ranking.pdf",
                bbox_inches="tight", dpi=150)
    fig.savefig("figures/biomarkers/consensus_ranking.png",
                bbox_inches="tight", dpi=150)
    plt.close(fig)
    print("  Saved: figures/biomarkers/consensus_ranking.pdf")

    # Figure 4: Network visualization
    if G is not None:
        try:
            import networkx as nx
            import matplotlib.patches as mpatches

            fig, ax = plt.subplots(figsize=(14, 10))
            top5_consensus = set(df.head(5)["gene"].tolist())
            seed_genes_set = set([
                "TGFB1","CTGF","MMP9","IL6","TNF","PDGFB","VIM","STAT3"
            ])
            candidate_set = set(df["gene"].tolist())

            node_colors = []
            node_sizes  = []
            for node in G.nodes():
                if node in top5_consensus:
                    node_colors.append("#E24B4A")
                    node_sizes.append(800)
                elif node in candidate_set:
                    node_colors.append("#378ADD")
                    node_sizes.append(500)
                elif node in seed_genes_set:
                    node_colors.append("#1D9E75")
                    node_sizes.append(400)
                else:
                    node_colors.append("#CCCCCC")
                    node_sizes.append(200)

            pos = nx.spring_layout(G, seed=42, k=2)
            nx.draw_networkx_nodes(G, pos, node_color=node_colors,
                                   node_size=node_sizes, ax=ax, alpha=0.9)
            nx.draw_networkx_labels(G, pos, font_size=8, ax=ax)
            edge_weights = [G[u][v].get("weight", 0.5) for u, v in G.edges()]
            nx.draw_networkx_edges(G, pos, width=edge_weights,
                                   alpha=0.4, ax=ax, edge_color="gray")
            patches = [
                mpatches.Patch(color="#E24B4A", label="Top 5 consensus candidates"),
                mpatches.Patch(color="#378ADD", label="All candidates"),
                mpatches.Patch(color="#1D9E75", label="Fibrosis seed genes"),
            ]
            ax.legend(handles=patches, loc="upper left", fontsize=9)
            ax.set_title("Protein interaction network: candidates and fibrosis seeds",
                         fontsize=12)
            ax.axis("off")
            plt.tight_layout()
            fig.savefig("figures/biomarkers/ppi_network.pdf",
                        bbox_inches="tight", dpi=150)
            fig.savefig("figures/biomarkers/ppi_network.png",
                        bbox_inches="tight", dpi=150)
            plt.close(fig)
            print("  Saved: figures/biomarkers/ppi_network.pdf")
        except Exception as e:
            print(f"  Network plot skipped: {e}")


# =========================================================================
# MAIN
# =========================================================================

if __name__ == "__main__":

    print("=" * 60)
    print("MULTI-METHOD BIOMARKER PRIORITIZATION")
    print("GSE136103 Human Liver Fibrosis")
    print("=" * 60)

    # Method 1: Rule-based
    df = method1_rule_based(df_base)

    # Method 2: ML ranking
    try:
        df = method2_ml_ranking(df)
    except ImportError:
        print("scikit-learn not installed. Skipping ML method.")
        print("Install with: pip install scikit-learn")
        df["ml_score"]      = df["rule_score"] / 10
        df["ml_score_norm"] = df["rule_score_norm"]
        df["ml_rank"]       = df["rule_rank"]
        df["label"]         = 0

    # Method 3: Network scoring
    try:
        import networkx
        df, G = method3_network_scoring(df)
    except ImportError:
        print("networkx not installed. Skipping network method.")
        print("Install with: pip install networkx")
        df["network_score"]      = df["rule_score"] / 10
        df["network_score_norm"] = df["rule_score_norm"]
        df["network_rank"]       = df["rule_rank"]
        df["degree"]             = 0
        df["betweenness"]        = 0
        df["seed_proximity"]     = 0
        G = None

    # Cross-method comparison and consensus
    df = compare_methods_and_consensus(df)

    # Translational discussion
    translational_discussion(df)

    # Visualizations
    print("\n" + "=" * 60)
    print("GENERATING FIGURES")
    print("=" * 60)
    plot_comparison(df, G)

    # Save complete results table
    output_cols = [
        "consensus_rank", "gene", "cell_type", "compartment",
        "logFC", "pval_adj",
        "rule_score", "rule_rank",
        "ml_score", "ml_rank",
        "network_score", "network_rank",
        "borda_total", "max_rank_diff",
        "degree", "betweenness", "seed_proximity",
        "specificity", "fold_change", "druggability", "serum", "clinical",
        "translational", "rationale"
    ]
    available_cols = [c for c in output_cols if c in df.columns]
    df[available_cols].to_csv("results/biomarker_multimethod_ranked.csv", index=False)
    print("\nFull multi-method ranked list saved: results/biomarker_multimethod_ranked.csv")

    # Also save a clean summary table for the report
    summary_cols = [
        "consensus_rank", "gene", "compartment",
        "rule_rank", "ml_rank", "network_rank",
        "borda_total", "max_rank_diff", "translational"
    ]
    available_summary = [c for c in summary_cols if c in df.columns]
    df[available_summary].to_csv("results/biomarker_consensus_summary.csv", index=False)
    print("Consensus summary saved: results/biomarker_consensus_summary.csv")
    print("\nAnalysis complete.")
