# Plan d'expériences ATD + Smart Tagging

Ce protocole complète la baseline officielle sans la modifier. Il utilise le même scorer et le même ground truth public que `evaluation/run_evaluation.py`.

## Objectif

Mesurer si les signaux disponibles dans l'ATD et le Smart Tagging améliorent la reconstruction temporelle des possessions, en particulier `spellF1`, sans utiliser la vérité terrain privée.

## Données et règles

- Flux tactique uniquement pour les prédictions : `data/tactical/`.
- Ground truth public uniquement pour contrôler le pipeline : `evaluation/gt/BAR-ATH_possession_gt_public.csv`.
- Aucun réglage manuel frame par frame.
- Les résultats publics ne servent pas à sélectionner un hyperparamètre final.
- Les coordonnées ATD sont converties en mètres avec `105.3 x 68.0`.
- Les slots sont regroupés par équipe ; ils ne sont pas traités comme des identités de joueurs.

## Expériences

| Nom | Entrées | Description |
|---|---|---|
| `baseline_raw` | baseline officielle | Joueur le plus proche, sans lissage |
| `baseline_smooth25` | baseline officielle | Vote centré `±25` frames |
| `baseline_smooth50` | baseline officielle | Référence officielle |
| `baseline_smooth125` | baseline officielle | Vote centré `±125` frames |
| `margin_raw` | ATD | Distance H/A avec ambiguïtés supprimées, sans lissage |
| `collective_raw` | ATD | Score distance + joueurs proches, sans lissage |
| `tags_raw` | ATD + Smart Tagging | Prior faible du champ `Team`, sans lissage |
| `control_raw` | ATD | Temps d'arrivée estimé à partir de la distance et de la vitesse du joueur |
| `ball_interp5_raw` | ATD | Interpolation uniquement des gaps ballon de cinq frames maximum |
| `margin_smooth50` | ATD | Ignore les observations ambiguës quand l'écart H/A est inférieur à `0.75 m`, puis lisse |
| `collective_smooth50` | ATD | Combine distance minimale et nombre de joueurs proches du ballon |
| `tags_smooth50` | ATD + Smart Tagging | Ajoute un prior faible du champ `Team` aux scores ATD |
| `control_smooth50` | ATD | Lissage du score de temps d'arrivée |
| `ball_interp5_smooth50` | ATD | Lissage après interpolation courte du ballon |
| `bridge_short_gaps50` | ATD | Reconnecte un gap `D` court lorsque les labels avant/après sont identiques |
| `stateful_baseline` | ATD | Changement d'équipe confirmé sur cinq observations, maintien 50 frames sans ballon |
| `stateful_control` | ATD | Même machine à états avec score de temps d'arrivée |
| `video_motion_smooth50` | Vidéo tactique + homographie + ATD | Mouvement global et mouvement local autour du ballon projeté, appliqués aux `D` suspects |

Les seuils de ces variantes sont fixés à l'avance pour comparer des hypothèses. Ils ne doivent pas être optimisés sur les 17 possessions publiques.

## Métriques produites

Pour chaque variante, le script produit les mêmes métriques que le scorer :

- `macro_f1` ;
- `spell_f1_mean` ;
- `spell_f1_at_headline` ;
- `alive_dead_f1` ;
- précision d'équipe sur les frames vivantes ;
- nombre de possessions prédites et erreur de comptage ;
- durée moyenne des possessions ;
- taux de possession de Barcelone.

## Sorties locales

Le script écrit dans `reports/atd_experiments/` :

- `metrics.csv` : tableau comparatif ;
- `public_window_metrics.csv` : performances sur les 7 500 frames publiques des 17 possessions ;
- `metrics.json` : résultats complets du scorer ;
- `submissions/<experiment>.csv` : prédiction frame-level de chaque variante ;
- `run_config.json` : paramètres et chemins utilisés.

L'expérience de récupération visuelle détaillée est exécutée par `scripts/video_ball_recovery_experiment.py` et écrit dans `reports/video_ball_recovery/`.

Ces fichiers sont dérivés du match et doivent rester locaux.

La vidéo tactique est analysée par échantillonnage déterministe des frames candidates et de référence afin d'éviter un parcours séquentiel impraticable du fichier de plusieurs Go. Le nombre d'échantillons est contrôlé par `--video-samples`.

## Commande

```powershell
uv run python scripts/run_atd_experiments.py
```

Le meilleur modèle ne sera retenu qu'après validation externe indépendante, conformément au plan de modélisation.
