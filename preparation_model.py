"""
ÉTAPE 3 — Entraînement des modèles
Deux modèles :
  • clf   : GradientBoostingClassifier  → prédit W / D / L (proba)
  • reg_h : GradientBoostingRegressor   → prédit buts marqués par l'équipe
  • reg_a : GradientBoostingRegressor   → prédit buts encaissés

⚡ Pour passer à CatBoost (recommandé en production) :
    pip install catboost
    Décommenter le bloc CatBoost ci-dessous et commenter le bloc sklearn.

Sortie : models/clf.pkl | models/reg_goals_for.pkl | models/reg_goals_against.pkl
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.model_selection import StratifiedKFold, cross_val_score, KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURES_CSV  = Path("data\\features_train.csv") ### Put the right path to your features_train.csv file here
MODELS_DIR    = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

# Features numériques utilisées pour l'entraînement
FEATURE_COLS = [
    "is_home",
    "win_rate",          "world_cup_win_rate", "avg_goals_scored",
    "avg_goals_conceded","avg_goal_diff",       "players_value",
    "fifa_ranking",
    "opp_win_rate",      "opp_avg_gs",          "opp_avg_gc",
    "opp_players_value", "opp_fifa_ranking",    "ranking_diff",
    "value_ratio",
    "form_wins_5",       "form_draws_5",        "form_losses_5",
    "form_pts_5",        "form_gf_5",           "form_ga_5",
    "form_gd_5",         "form_home_wr_5",      "form_away_wr_5",
]


def load_data():
    df = pd.read_csv(FEATURES_CSV)
    df = df.dropna(subset=["result", "goals_for", "goals_against"])
    # Garder seulement les colonnes features disponibles
    available = [c for c in FEATURE_COLS if c in df.columns]
    missing   = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        print(f"  ⚠ Colonnes absentes (seront ignorées) : {missing}")
    X = df[available].fillna(0).astype(float)
    y_result  = df["result"]
    y_gf      = df["goals_for"].clip(0, 10)
    y_ga      = df["goals_against"].clip(0, 10)
    return X, y_result, y_gf, y_ga, available

def build_classifier(params=None):
    if params is None:
        params = {
            'n_estimators': 287,
            'learning_rate': 0.016361173567871284,
            'max_depth': 3,
            'subsample': 0.8,
            'min_samples_leaf': 14
        }
        
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", GradientBoostingClassifier(
            **params,
            random_state=42
        ))
    ])

def build_regressor(params=None):
    if params is None:
        params = {
            'n_estimators': 100,
            'learning_rate': 0.01622229446106638,
            'max_depth': 3,
            'subsample': 0.7308364665121777,
            'min_samples_leaf': 5
        }
        
    return Pipeline([
        ("scaler", StandardScaler()),
        ("reg", GradientBoostingRegressor(
            **params,
            random_state=42
        ))
    ])
import optuna
########## début Optuna ##########
def optimize_classifier(X, y):
    def objective(trial):
        # 1. Définir l'espace de recherche
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 8),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 5, 20)
        }
        
        # 2. Créer le Pipeline avec ces paramètres
        pipeline = build_classifier(params)
        
        # 3. Évaluer avec 3 splits (pour aller plus vite pendant l'optimisation)
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        score = cross_val_score(pipeline, X, y, cv=cv, scoring="accuracy", n_jobs=-1)
        
        return score.mean()

    print("   Lancement d'Optuna (Classifieur)...")
    optuna.logging.set_verbosity(optuna.logging.WARNING) # Masque les logs à chaque essai
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=20) # Teste 20 combinaisons
    return study.best_params

#meme chose que précédemment mais pour le régresseur
def optimize_regressor(X, y):
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 8),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 5, 20)
        }
        
        pipeline = build_regressor(params)
        
        cv = KFold(n_splits=3, shuffle=True, random_state=42)
        # On cherche à minimiser l'erreur (MAE), on utilise donc neg_mean_absolute_error
        score = -cross_val_score(pipeline, X, y, cv=cv, scoring="neg_mean_absolute_error", n_jobs=-1)
        
        return score.mean()

    print("   Lancement d'Optuna (Régresseur)...")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=20)
    return study.best_params

#--------------- Fin Optuna ----------------
from catboost import CatBoostClassifier, CatBoostRegressor


def train_and_evaluate():
    print("📂 Chargement des features...")
    X, y_result, y_gf, y_ga, used_cols = load_data()
    print(f"   {len(X)} matchs | {len(used_cols)} features")

    le = LabelEncoder()
    y_enc = le.fit_transform(y_result)
    print(f"   Classes : {list(le.classes_)}")

#-------------------------changement
# ── Classifieur W/D/L ──────────────────────────────────────────────────
    print("\n🔍 Recherche des meilleurs paramètres (Classifieur)...")
    best_clf_params = optimize_classifier(X, y_enc)
    print(f"   Meilleurs paramètres trouvés : {best_clf_params}")

    print("🎯 Entraînement classifieur final (W/D/L)...")
    clf = build_classifier(best_clf_params)
    cv_clf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores_clf = cross_val_score(clf, X, y_enc, cv=cv_clf, scoring="accuracy")
    print(f"   CV Accuracy : {scores_clf.mean():.3f} ± {scores_clf.std():.3f}")
    clf.fit(X, y_enc)

    # ... [Le rapport de classification reste le même] ...

    # ── Régresseur buts marqués ────────────────────────────────────────────
    print("\n🔍 Recherche des meilleurs paramètres (Buts marqués)...")
    best_gf_params = optimize_regressor(X, y_gf)
    print(f"   Meilleurs paramètres trouvés : {best_gf_params}")

    print("⚽ Entraînement régresseur final (buts marqués)...")
    reg_gf = build_regressor(best_gf_params)
    cv_reg = KFold(n_splits=5, shuffle=True, random_state=42)
    mae_gf = -cross_val_score(reg_gf, X, y_gf, cv=cv_reg, scoring="neg_mean_absolute_error")
    print(f"   CV MAE buts marqués : {mae_gf.mean():.3f} ± {mae_gf.std():.3f}")
    reg_gf.fit(X, y_gf)

    # ── Régresseur buts encaissés ──────────────────────────────────────────
    print("\n🔍 Recherche des meilleurs paramètres (Buts encaissés)...")
    best_ga_params = optimize_regressor(X, y_ga)
    print(f"   Meilleurs paramètres trouvés : {best_ga_params}")

    print("🥅 Entraînement régresseur final (buts encaissés)...")
    reg_ga = build_regressor(best_ga_params)
    mae_ga = -cross_val_score(reg_ga, X, y_ga, cv=cv_reg, scoring="neg_mean_absolute_error")
    print(f"   CV MAE buts encaissés : {mae_ga.mean():.3f} ± {mae_ga.std():.3f}")
    reg_ga.fit(X, y_ga)
    #fin changement ----------------
    # ── Feature importance (classifieur) ───────────────────────────────────
    try:
        importances = clf.named_steps["clf"].feature_importances_
        fi = pd.Series(importances, index=used_cols).sort_values(ascending=False)
        print("\n── Top 10 features (classifieur) ──")
        print(fi.head(10).to_string())
    except Exception:
        pass

    # ── Sauvegarde ────────────────────────────────────────────────────────
    bundle = {
        "clf":        clf,
        "reg_gf":     reg_gf,
        "reg_ga":     reg_ga,
        "label_enc":  le,
        "feature_cols": used_cols,
    }
    model_path = MODELS_DIR / "model_bundle.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(bundle, f)
    print(f"\n✅ Modèles sauvegardés → {model_path}")
    return bundle


if __name__ == "__main__":
    train_and_evaluate()