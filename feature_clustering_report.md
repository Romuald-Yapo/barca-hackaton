# Feature store et clustering sans entraînement

## Objectif

Construire une table de caractéristiques par frame à partir des données déjà
présentes dans le repo, puis vérifier si un regroupement non supervisé révèle
des régimes utiles pour la prédiction de la possession. Les labels publics ne
sont utilisés qu'après le clustering, comme diagnostic d'évaluation ; ils ne
servent pas à ajuster les clusters.

## Données produites

- `features/BAR-ATH_frame_features.parquet` : 199 319 frames et 54 colonnes.
- `features/feature_store_qa.json` : contrôle qualité du feature store.
- `reports/clustering/clustered_features.parquet` : labels KMeans et GMM pour
  chaque frame.
- `reports/clustering/cluster_summary.csv` : taille et taux de visibilité du
  ballon / activité des tags par cluster.
- `reports/clustering/cluster_quality.csv` : qualité des clusters sur la plage
  publique, uniquement pour le diagnostic.

Les 54 colonnes couvrent :

- position, distance et vitesse du joueur le plus proche ;
- présence, trous, interpolation courte, vitesse et accélération du ballon ;
- nombre de joueurs visibles, centroïdes, dispersion et vitesses moyennes des
  deux équipes ;
- distances ballon-centroïde et différence entre équipes ;
- tags actifs, set pieces, transitions et score issu du tagging ;
- validité et erreur de reprojection de l'homographie ;
- qualité minimale du tracking ;
- indicateurs des détecteurs vidéo bright-blob, template et optical-flow.

## Méthodes

Le script `scripts/run_feature_clustering.py` remplace les valeurs manquantes
par la médiane, standardise les variables, projette dans 12 composantes PCA,
puis ajuste :

- MiniBatch KMeans avec 8, 12 et 16 clusters ;
- GMM diagonal avec 8 et 12 composantes.

Les prédictions sont calculées par lots sur toutes les frames. Le GMM est
ajusté sur 8 000 frames afin de rester exécutable localement.

## Résultats

Le meilleur regroupement est `gmm_8` :

| méthode | pureté publique | ARI | NMI |
|---|---:|---:|---:|
| `gmm_8` | 75,39 % | 0,260 | 0,314 |
| `gmm_12` | 73,95 % | 0,210 | 0,300 |
| `kmeans_16` | 67,03 % | 0,169 | 0,206 |
| `kmeans_12` | 63,76 % | 0,129 | 0,220 |
| `kmeans_8` | 61,83 % | 0,176 | 0,175 |

La pureté est une borne de diagnostic obtenue en associant chaque cluster à
sa classe publique majoritaire. Cette association utilise les labels publics
et ne constitue donc pas un modèle utilisable en production.

Les clusters séparent surtout des états de qualité et de contexte : certains
ont presque 0 % de ballon observé, d'autres presque 100 %, et les taux
d'activité des Smart Tags varient fortement. Cela confirme que le clustering
capture une information réelle, mais ne montre pas encore une séparation
directe et stable entre les deux équipes.

## Conclusion opérationnelle

Le clustering mérite d'être conservé comme source de variables de contexte :
`cluster_id`, confiance GMM et changement de cluster entre frames peuvent
servir à pondérer les règles de décision ou à détecter les frames ambiguës.
En revanche, il ne faut pas remplacer directement la baseline par la classe
majoritaire d'un cluster : cette règle serait contaminée par le diagnostic
public et ne résout pas encore le problème équipe A / équipe B.

## Reproduction

```powershell
uv run python scripts/build_feature_store.py
uv run python scripts/run_feature_clustering.py
uv run python scripts/evaluate_cluster_quality.py
```

## Heuristiques clusterisées testées

Le script `scripts/run_cluster_heuristics.py` ajoute les clusters à la
baseline sans apprentissage supervisé :

- confiance GMM `gmm_8_confidence` ;
- changement de cluster et stabilité locale ;
- score de qualité basé sur ballon observé, tracking suffisamment complet et
  marge du joueur le plus proche ;
- gating des frames peu fiables ;
- combinaison avec la distance du ballon aux centroïdes des équipes ;
- majority voting final avec fenêtres 25, 50 et 125 frames.

La meilleure variante pour le macro-F1 est `cluster_gate70_smooth50` :

| méthode | macro-F1 public | Spell-F1 public | team accuracy GT alive | accuracy |
|---|---:|---:|---:|---:|
| baseline smooth50 | 0,621 | 0,326 | 0,835 | 0,713 |
| cluster gate55 smooth50 | 0,701 | 0,318 | 0,836 | 0,751 |
| cluster gate70 smooth50 | 0,702 | 0,307 | 0,837 | 0,752 |
| cluster centroid blend025 smooth50 | 0,599 | 0,315 | 0,801 | 0,686 |

Les clusters réduisent donc les erreurs de classe D et améliorent la vue
frame-level, mais ils retirent trop de frames vivantes ou fragmentent encore
les possessions. Ils ne constituent pas une amélioration du critère principal
Spell-F1. La baseline `smooth50` reste la meilleure solution équilibrée parmi
ces expériences ; `smooth125` reste meilleure sur la seule fenêtre publique,
ce qui est un effet de la faible quantité de possessions annotées.

Les métriques et soumissions détaillées sont dans
`reports/clustering/cluster_heuristics_metrics.csv` et
`reports/clustering/submissions/`.

## Balayage du lissage

Un balayage de 25 à 125 frames par pas de 5 a été effectué avec
`scripts/sweep_smoothing_windows.py`.

| méthode | meilleure fenêtre Spell-F1 | meilleur Spell-F1 | meilleure fenêtre macro-F1 | meilleur macro-F1 |
|---|---:|---:|---:|---:|
| baseline | 120 | 0,465 | 45 | 0,623 |
| cluster gate55 | 110 | 0,390 | 45 | 0,706 |
| cluster gate70 | 100 | 0,423 | 35 | 0,708 |
| cluster centroid blend025 | 100 | 0,436 | 40 | 0,599 |
| cluster centroid blend05 | 115 | 0,400 | 50 | 0,584 |
| cluster centroid blend10 | 125 | 0,413 | 40 | 0,579 |

Sur la fenêtre publique seule, le meilleur réglage Spell-F1 est donc la
baseline avec une fenêtre de 120 frames. Ce choix doit rester provisoire : la
fenêtre publique ne contient que 17 possessions et peut favoriser un réglage
qui ne se généralise pas. Les résultats complets sont dans
`reports/clustering/smoothing_sweep_metrics.csv` et le résumé dans
`reports/clustering/smoothing_sweep_summary.json`.

## Modèle temporel à score continu

Le script `scripts/run_temporal_control_model.py` utilise un score continu de
contrôle pour chaque équipe, un score de ballon mort, puis un décodage Viterbi
sur les états `H/A/D`. Il ajoute une pénalité de changement d'équipe, une
pénalité de changement vivant/mort, une durée minimale de possession et un
majority voting final.

La meilleure configuration publique est :

```text
switch_penalty = 0.50
dead_penalty = 0.30
minimum_alive_frames = 50
majority_window = 120
```

Résultats :

| méthode | macro-F1 | Spell-F1 | team accuracy | accuracy | possessions prédites |
|---|---:|---:|---:|---:|---:|
| baseline smooth50 | 0,621 | 0,326 | 0,835 | 0,713 | 26 |
| baseline smooth120 | 0,586 | 0,465 | 0,885 | 0,729 | 14 |
| Viterbi + contrôle + smooth120 | 0,597 | **0,520** | **0,897** | **0,740** | 13 |

Cette expérience est la première amélioration sur le critère principal, mais le
réglage a été sélectionné sur les 17 possessions publiques. Il est donc
prometteur, mais pas encore démontré comme généralisable. Les 360
configurations et les soumissions sont dans
`reports/temporal_control/metrics.csv` et
`reports/temporal_control/submissions/`.

## Lissage adaptatif autour des frontières

Le script `scripts/run_adaptive_temporal_model.py` a testé 384 variantes avec
une fenêtre courte autour des transitions, une fenêtre intermédiaire dans les
phases stables et une fenêtre longue lorsque l'observation est incertaine.

Le meilleur compromis simultané par rapport à `baseline_smooth50` est :

```text
switch_penalty = 1.00
dead_penalty = 0.75
minimum_alive_frames = 50
stable_window = 35
boundary_window = 15
uncertain_window = 50
```

| méthode | macro-F1 | Spell-F1 | accuracy |
|---|---:|---:|---:|
| baseline smooth50 | 0,621 | 0,326 | 0,713 |
| adaptatif équilibré | **0,636** | **0,409** | **0,731** |
| Viterbi global + smooth120 | 0,597 | **0,520** | 0,740 |

Le lissage adaptatif améliore donc bien les deux métriques par rapport à la
baseline smooth50, mais il ne dépasse pas le meilleur réglage Spell-F1 global.
Le résultat est plus équilibré et probablement plus proche de l'objectif de
représenter correctement les frames et les intervalles de possession.

Les résultats complets sont dans `reports/adaptive_temporal/metrics.csv` et
la frontière de compromis dans `reports/adaptive_temporal/pareto_frontier.csv`.

