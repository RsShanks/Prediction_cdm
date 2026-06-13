"""
ÉTAPE 4 — Prédiction de match
Usage :
    python 4_predict.py --home "France" --away "Argentine"
    python 4_predict.py --home "Japon"  --away "Maroc" --neutral

Sortie :
    • Probabilités V / N / D (en %)
    • Score estimé (arrondi)
    • Cote implicite (format décimal)
    • Analyse narrative
"""

import argparse
import pickle
import re
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from math import exp

warnings.filterwarnings("ignore")

# ── Config ───────────────────────────────────────────────────────────────────
TEAMS_CSV   = Path("prediction_cdm\data\data\stats_equipe.csv")###put the right path to your stats_equipe.csv file here
MODELS_DIR  = Path("models")
FORM_WINDOW = 5

# ── Helpers ───────────────────────────────────────────────────────────────────

def parse_value(raw) -> float:
    if not raw or (isinstance(raw, float) and np.isnan(raw)):
        return 0.0
    raw = str(raw).strip().lower()
    raw = re.sub(r'[€\s]', '', raw)
    if 'mio' in raw:
        num = re.sub(r'mio\.?', '', raw).replace(',', '.')
        return float(num) * 1_000_000
    if 'k' in raw:
        num = re.sub(r'k', '', raw).replace(',', '.')
        return float(num) * 1_000
    try:
        return float(raw.replace(',', '.'))
    except ValueError:
        return 0.0

def load_teams() -> pd.DataFrame:
    df = pd.read_csv(TEAMS_CSV)
    df["players_value_eur"] = df["players_value"].apply(parse_value)
    df["win_rate"]           = pd.to_numeric(df["win_rate"],           errors="coerce").fillna(0) / 100
    df["world_cup_win_rate"] = pd.to_numeric(df["world_cup_win_rate"], errors="coerce").fillna(0) / 100
    df["Fifa_ranking"]       = pd.to_numeric(df["Fifa_ranking"],       errors="coerce").fillna(99)
    return df.set_index("Team_name")

def load_bundle() -> dict:
    path = MODELS_DIR / "model_bundle.pkl"
    if not path.exists():
        raise FileNotFoundError(
            "❌ Modèle introuvable. Lance d'abord : python 3_train_model.py"
        )
    with open(path, "rb") as f:
        return pickle.load(f)

def poisson_score(lambda_h: float, lambda_a: float, max_goals: int = 8):
    """
    Distribue les scores possibles selon Poisson et retourne :
    - prob_home_win, prob_draw, prob_away_win
    - score le plus probable (mode)
    """
    from math import factorial
    def pois(lam, k):
        return (lam ** k) * exp(-lam) / factorial(k)

    matrix = np.zeros((max_goals + 1, max_goals + 1))
    for h in range(max_goals + 1):
        for a in range(max_goals + 1):
            matrix[h, a] = pois(lambda_h, h) * pois(lambda_a, a)

    p_hw = float(np.tril(matrix, -1).sum())
    p_dr = float(np.trace(matrix))
    p_aw = float(np.triu(matrix, 1).sum())

    idx_h, idx_a = np.unravel_index(matrix.argmax(), matrix.shape)
    return p_hw, p_dr, p_aw, int(idx_h), int(idx_a)


# ── Vecteur de features pour un duel ─────────────────────────────────────────

def build_feature_vector(
    home_name: str,
    away_name: str,
    teams: pd.DataFrame,
    feature_cols: list,
    is_neutral: bool = False,
) -> pd.DataFrame:
    """
    Construit le vecteur de features pour home_team.
    (On appellera la fonction deux fois : une fois pour chaque équipe.)
    """
    if home_name not in teams.index:
        raise ValueError(f"Équipe inconnue : '{home_name}'. Vérifie l'orthographe.")
    if away_name not in teams.index:
        raise ValueError(f"Équipe inconnue : '{away_name}'. Vérifie l'orthographe.")

    ht = teams.loc[home_name]
    at = teams.loc[away_name]

    # Avantage domicile : désactivé si terrain neutre
    home_advantage = 0 if is_neutral else 1

    rec = {
        "is_home":            home_advantage,
        "win_rate":           ht["win_rate"],
        "world_cup_win_rate": ht["world_cup_win_rate"],
        "avg_goals_scored":   ht["avg_goals_scored"],
        "avg_goals_conceded": ht["avg_goals_conceded"],
        "avg_goal_diff":      ht["avg_goal_difference"],
        "players_value":      ht["players_value_eur"],
        "fifa_ranking":       ht["Fifa_ranking"],
        "opp_win_rate":       at["win_rate"],
        "opp_avg_gs":         at["avg_goals_scored"],
        "opp_avg_gc":         at["avg_goals_conceded"],
        "opp_players_value":  at["players_value_eur"],
        "opp_fifa_ranking":   at["Fifa_ranking"],
        "ranking_diff":       ht["Fifa_ranking"] - at["Fifa_ranking"],
        "value_ratio":        (
            ht["players_value_eur"] / at["players_value_eur"]
            if at["players_value_eur"] > 0 else 1.0
        ),
        # Forme récente inconnue → on utilise les stats globales comme proxy
        "form_wins_5":     round(ht["win_rate"] * FORM_WINDOW),
        "form_draws_5":    round((1 - ht["win_rate"]) * FORM_WINDOW * 0.3),
        "form_losses_5":   round((1 - ht["win_rate"]) * FORM_WINDOW * 0.7),
        "form_pts_5":      round(ht["win_rate"] * FORM_WINDOW * 2.5),
        "form_gf_5":       ht["avg_goals_scored"] * FORM_WINDOW,
        "form_ga_5":       ht["avg_goals_conceded"] * FORM_WINDOW,
        "form_gd_5":       ht["avg_goal_difference"] * FORM_WINDOW,
        "form_home_wr_5":  ht["win_rate"],
        "form_away_wr_5":  ht["win_rate"] * 0.85,
    }

    # Construire le DataFrame dans l'ordre exact des colonnes du modèle
    row = {col: rec.get(col, 0.0) for col in feature_cols}
    return pd.DataFrame([row])


# ── Prédiction principale ─────────────────────────────────────────────────────

def predict_match(home: str, away: str, neutral: bool = False):
    teams  = load_teams()
    bundle = load_bundle()
    clf, reg_gf, reg_ga = bundle["clf"], bundle["reg_gf"], bundle["reg_ga"]
    le      = bundle["label_enc"]
    f_cols  = bundle["feature_cols"]

    # ── Features pour les deux équipes ──────────────────────────────────────
    X_home = build_feature_vector(home, away, teams, f_cols, is_neutral=neutral)
    X_away = build_feature_vector(away, home, teams, f_cols, is_neutral=neutral)

    # ── Probabilités W/D/L ──────────────────────────────────────────────────
    proba_home = clf.predict_proba(X_home)[0]  # shape (3,)
    proba_away = clf.predict_proba(X_away)[0]

    # Mapping label → index
    classes = list(le.classes_)  # ex: ['D','L','W']
    w_idx = classes.index("W")
    d_idx = classes.index("D")
    l_idx = classes.index("L")

    p_home_win  = proba_home[w_idx]
    p_draw      = (proba_home[d_idx] + proba_away[d_idx]) / 2
    p_away_win  = proba_away[w_idx]

    # Normaliser
    total = p_home_win + p_draw + p_away_win
    p_home_win /= total
    p_draw     /= total
    p_away_win /= total

    # ── Buts estimés (régresseur) ────────────────────────────────────────────
    lambda_h = max(0.1, float(reg_gf.predict(X_home)[0]))
    lambda_a = max(0.1, float(reg_gf.predict(X_away)[0]))

    # Affiner avec Poisson pour cohérence proba / score
    ph_poisson, pd_poisson, pa_poisson, score_h, score_a = poisson_score(lambda_h, lambda_a)

    # Moyenne pondérée (ML + Poisson)
    alpha = 0.6  # poids du modèle ML
    p_home_win_final = alpha * p_home_win + (1 - alpha) * ph_poisson
    p_draw_final     = alpha * p_draw     + (1 - alpha) * pd_poisson
    p_away_win_final = alpha * p_away_win + (1 - alpha) * pa_poisson

    # ── Affichage ───────────────────────────────────────────────────────────
    venue = "⬛ Terrain neutre" if neutral else f"🏠 {home} (domicile)"

    print("\n" + "═" * 58)
    print(f"  ⚽  {home}  VS  {away}")
    print(f"  📍  {venue}")
    print("═" * 58)

    print(f"\n{'PROBABILITÉS':─<40}")
    print(f"  🔵 Victoire  {home:<20} {p_home_win_final*100:5.1f}%  (cote: {1/p_home_win_final:.2f})")
    print(f"  ⚪ Match nul {'':<20} {p_draw_final*100:5.1f}%  (cote: {1/p_draw_final:.2f})")
    print(f"  🔴 Victoire  {away:<20} {p_away_win_final*100:5.1f}%  (cote: {1/p_away_win_final:.2f})")

    print(f"\n{'SCORE ESTIMÉ':─<40}")
    print(f"  λ buts attendus : {home} {lambda_h:.2f}  –  {away} {lambda_a:.2f}")
    print(f"  Score le + probable : {home} {score_h} – {score_a} {away}")

    print(f"\n{'ANALYSE':─<40}")
    home_stats = teams.loc[home]
    away_stats = teams.loc[away]
    ranking_adv = "meilleur" if home_stats["Fifa_ranking"] < away_stats["Fifa_ranking"] else "moins bien"
    print(f"  • {home} est {ranking_adv} classé FIFA "
          f"(#{int(home_stats['Fifa_ranking'])} vs #{int(away_stats['Fifa_ranking'])})")
    val_h = home_stats["players_value_eur"] / 1e6
    val_a = away_stats["players_value_eur"] / 1e6
    print(f"  • Valeur des effectifs : {home} {val_h:.1f}M€  vs  {away} {val_a:.1f}M€")
    print(f"  • Forme récente (moy buts) : {home} {home_stats['avg_goals_scored']:.2f}/match"
          f"  vs  {away} {away_stats['avg_goals_scored']:.2f}/match")

    # Verdict
    print(f"\n{'VERDICT':─<40}")
    if p_home_win_final > 0.45:
        print(f"  ✅  Favori : {home}")
    elif p_away_win_final > 0.45:
        print(f"  ✅  Favori : {away}")
    else:
        print(f"  ⚠️  Match très ouvert — nul possible")
    print("═" * 58 + "\n")

    return {
        "home":             home,
        "away":             away,
        "p_home_win":       round(p_home_win_final, 4),
        "p_draw":           round(p_draw_final, 4),
        "p_away_win":       round(p_away_win_final, 4),
        "expected_goals_home": round(lambda_h, 2),
        "expected_goals_away": round(lambda_a, 2),
        "most_likely_score": f"{score_h}-{score_a}",
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prédiction de match Coupe du Monde")
    parser.add_argument("--home",    required=True,  help="Nom de l'équipe à domicile (ou équipe 1)")
    parser.add_argument("--away",    required=True,  help="Nom de l'équipe à l'extérieur (ou équipe 2)")
    parser.add_argument("--neutral", action="store_true", help="Terrain neutre (Coupe du Monde)")
    args = parser.parse_args()

    predict_match(args.home, args.away, neutral=args.neutral)
