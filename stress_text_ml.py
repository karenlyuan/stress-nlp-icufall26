"Linguistic features: psychological stress prediction (Dreaddit)"

import re
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
rng = np.random.default_rng(SEED)
DATA, OUT = Path("data"), Path("outputs"); OUT.mkdir(exist_ok=True)

# ---- 1. Lexicons (small LIWC-style dictionaries) referencing lecture 5 slide 9 ----
FIRST_SING = {"i","me","my","mine","myself","i'm","i've","i'd","i'll","im","ive"}
FIRST_PLUR = {"we","us","our","ours","ourselves","we're","we've","we'll","we'd"}

# deliberately exclude "stress, anxiety, depression" words as they would leak the label
NEG_EMO = set("""sad angry anger hate hated afraid scared fear terrified panic hopeless
                worthless miserable terrible awful horrible cry crying cried pain hurt hurting upset
                nervous worried worry overwhelmed guilty guilt shame ashamed furious mad bitter empty
                numb broken trapped helpless exhausted ruined nightmare""".split())
                
POS_EMO = set("""happy love loved great good hope hopeful grateful thankful glad proud
                excited enjoy fun calm relaxed peace better""".split())

ISO_WORDS = set("""alone lonely loneliness isolated isolation nobody abandoned ignored
                excluded rejected friendless withdrawn disconnected""".split())

ISO_PHRASES = re.compile(r"\b(no one|by myself|on my own|left out|nobody cares)\b")

ABSOLUTIST = set("always never completely totally nothing everything entire absolutely constantly forever".split())


# ---- 2. Pre-processing, referencing lecture 5 slide 5 ----
URL = re.compile(r"https?://\S+|www\.\S+")
def clean(text):
    t = str(text).lower().replace("\u2019", "'")  # lowercase
    t = URL.sub(" ", t)  # remove URLs
    return re.sub(r"\d+", " ", t)  # remove digits

def tokenize(t):
    return re.findall(r"[a-z']+", t)  # tokenization

def features(text):
    t = clean(text); toks = tokenize(t); n = max(len(toks), 1)
    rate = lambda S: 100 * sum(w in S for w in toks) / n
    window = toks[:100]
    return {
        "self_focus": rate(FIRST_SING),
        "neg_emotion": rate(NEG_EMO),
        "isolation": rate(ISO_WORDS) + 100 * len(ISO_PHRASES.findall(t)) / n,
        "pos_emotion": rate(POS_EMO),
        "we_words": rate(FIRST_PLUR),
        "absolutist": rate(ABSOLUTIST),
        "ttr100": len(set(window)) / max(len(window), 1),  # type-token ratio referencing lecture 5 slide 10
        "log_len": np.log1p(len(toks)),
    }


# ---- 3. Load data ----
train = pd.read_parquet(DATA / "train-00000-of-00001.parquet")
test = pd.read_parquet(DATA / "test-00000-of-00001.parquet")
for df in (train, test):
    df["text_clean"] = df["text"].map(clean)
    F = pd.DataFrame([features(t) for t in df["text"]], index=df.index)
    for c in F.columns: df[c] = F[c]

FEATS = list(F.columns)
CORE = ["self_focus", "neg_emotion", "isolation"]
print(f"train n={len(train)}, test n={len(test)}, stress rate train={train.label.mean():.3f}")
print(train.groupby("subreddit").label.agg(["count", "mean"]), "\n")


# ---- 4. H1: Spearman correlations referencing lecture 4 ----
full = pd.concat([train, test], ignore_index=True)
print("=== H1: Spearman rho between feature and stress label (Bonferroni-corrected) ===")
for f in FEATS:
    rho, p = stats.spearmanr(full[f], full["label"]); pc = min(p * len(FEATS), 1)
    star = "***" if pc < .001 else "**" if pc < .01 else "*" if pc < .05 else ""
    print(f"{f:12s} rho={rho:+.3f}{star:4s} p_corrected={pc:.4g}")
print("\nGroup means (stress=1 vs 0):\n", full.groupby("label")[FEATS].mean().T.round(3), "\n")


# ---- 5. Models (supervised learning referencing lecture 4 and lecture 5) ----
def make(kind, feats=FEATS):
    tf = ("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1, 2),
                                min_df=3, max_features=20000), "text_clean")

    num = ("num", StandardScaler(), feats)
    ct = {"features": [num], "tfidf": [tf], "both": [tf, num]}[kind]
    return Pipeline([("ct", ColumnTransformer(ct)),
                    ("clf", LogisticRegression(max_iter=2000, C=1.0))])

MODELS = {
    "majority baseline": Pipeline([("ct", ColumnTransformer([("num", StandardScaler(), FEATS)])),
                                    ("clf", DummyClassifier(strategy="most_frequent"))]),

    "core3 only": make("features", CORE),
    "non-core features only": make("features", [f for f in FEATS if f not in CORE]),
    "all hand-crafted features": make("features"),
    "tf-idf n-grams": make("tfidf"),
    "tf-idf + features": make("both"),
}


# ---- 6. Cross validation on train ----
skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
print("=== 5-fold CV AUC on training set ===")
for name, m in MODELS.items():
    s = cross_val_score(m, train, train["label"], cv=skf, scoring="roc_auc")
    print(f"{name:28s} AUC={s.mean():.3f} +/- {s.std():.3f}")


# ---- 7. Final test set evaluation and statistics referencing lecture 4 ----
y = test["label"].values; n = len(y)
maj = max(y.mean(), 1 - y.mean()); correct = {}
print(f"\n=== Held-out test set (n={n}); majority-class rate={maj:.3f} ===")
for name, m in MODELS.items():
    m.fit(train, train["label"])
    proba = m.predict_proba(test)[:, 1]; pred = (proba >= .5).astype(int)
    correct[name] = (pred == y)
    k = int(correct[name].sum())
    p_maj = stats.binomtest(k, n, p=maj, alternative="greater").pvalue  # binomial test vs majority baseline
    p_50 = stats.binomtest(k, n, p=.5, alternative="greater").pvalue  # binomial test vs 50% chance
    print(f"{name:28s} acc={accuracy_score(y,pred):.3f} AUC={roc_auc_score(y,proba):.3f} "
            f"F1={f1_score(y,pred):.3f}  binom p(vs majority)={p_maj:.4g}, p(vs 50%)={p_50:.4g}")
    
def paired_perm(c1, c2, n_perm=10000):
    # paired permutation (sign-flip) test, referencing lecture 4
    d = c1.astype(int) - c2.astype(int); obs = d.mean()
    null = (rng.choice([-1, 1], size=(n_perm, len(d))) * d).mean(axis=1)
    return obs, (np.sum(np.abs(null) >= abs(obs)) + 1) / (n_perm + 1)
print("\n=== H4: paired permutation tests on test-set accuracy ===")
for a, b in [("tf-idf + features", "tf-idf n-grams"),
                ("all hand-crafted features", "non-core features only"),
                ("all hand-crafted features", "tf-idf n-grams")]:
    obs, p = paired_perm(correct[a], correct[b])
    print(f"{a} vs {b}: diff in accuracy={obs:+.3f}, p={p:.4g}")


# ---- 8. Interpretation: coefficients H1/H3 ----
m = MODELS["all hand-crafted features"]
coef = m.named_steps["clf"].coef_[0]
co = pd.Series(coef, index=FEATS).sort_values()
print("\n=== Standardized logistic-regression coefficients (log-odds per 1 SD) ===")
print(pd.DataFrame({"coef": co, "odds_ratio": np.exp(co)}).round(3))
plt.figure(figsize=(6, 3.5)); co.plot.barh(color=["tab:red" if f in CORE else "gray" for f in co.index])
plt.xlabel("standardized coefficient (red = hypothesised core features)")
plt.tight_layout(); plt.savefig(OUT / "coefficients.png", dpi=200); plt.close()


# ---- 9. Per-domain performance ----
print("\n=== Test AUC by subreddit (all hand-crafted features) ===")
for sub, g in test.groupby("subreddit"):
    if g.label.nunique() == 2:
        print(f"{sub:20s} n={len(g):3d} AUC={roc_auc_score(g.label, m.predict_proba(g)[:,1]):.3f}")


# ---- 10. PCA unsupervised, referencing lecture 4 ----
Z = StandardScaler().fit_transform(full[FEATS]); pca = PCA(2).fit(Z); P = pca.transform(Z)
print("\nPCA explained variance:", pca.explained_variance_ratio_.round(3))
print(pd.DataFrame(pca.components_.T, index=FEATS, columns=["PC1", "PC2"]).round(2))
plt.figure(figsize=(5, 4)); plt.scatter(P[:, 0], P[:, 1], c=full["label"], cmap="coolwarm", s=6, alpha=.5)
plt.xlabel("PC1"); plt.ylabel("PC2"); plt.tight_layout(); plt.savefig(OUT / "pca.png", dpi=200)


