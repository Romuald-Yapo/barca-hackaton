# Benchmark final des expériences

## Périmètre

Ce document récapitule toutes les expériences exécutées dans ce dépôt avant entraînement supervisé. Les scores proviennent uniquement du ground truth public : 7 500 frames et 17 possessions, dans la fenêtre vidéo `11185:18685`.

Le fichier public complet contient 199 319 lignes, mais les frames `X` hors de cette fenêtre ne sont pas évaluées. Ces résultats servent donc au diagnostic du pipeline et non à la sélection définitive d'un modèle.

Commande principale :

```powershell
uv run python scripts/run_atd_experiments.py --video-samples 200
```

Expérience vidéo détaillée :

```powershell
uv run python scripts/video_ball_recovery_experiment.py --samples 300
```

## Expériences de données et QA

1. Vérification de la présence des assets tactiques et broadcast.
2. Vérification ATD tactique : 199 319 lignes, `frame_id` 1 à 199 319.
3. Vérification ATD broadcast : 164 488 lignes, timeline séparée.
4. Vérification homographie tactique : 199 319 matrices, toutes valides.
5. Vérification Smart Tagging : 638 intervalles.
6. Vérification de l'alignement ATD : `video_frame = frame_id - 1`.
7. Vérification de l'alignement Smart Tagging : `round(timestamp * 25)`.
8. Mesure de visibilité du ballon : 56,9 % des frames tactiques.
9. Mesure du nombre de joueurs suivis et des gaps du ballon.
10. Projection de positions ATD vers les pixels avec les matrices du CSV d'homographie.

## Baselines officielles

Le script existant `starter/baseline.py` est appelé directement, sans recopier son implémentation.

11. `baseline_raw` — joueur le plus proche, aucune fenêtre.
12. `baseline_smooth25` — vote majoritaire centré `±25` frames.
13. `baseline_smooth50` — vote majoritaire centré `±50` frames.
14. `baseline_smooth125` — vote majoritaire centré `±125` frames.

## Expériences ATD brutes

15. `margin_raw` — suppression des évidences ambiguës sous `0,75 m`.
16. `collective_raw` — distance minimale + nombre de joueurs dans un rayon de `5 m`.
17. `tags_raw` — prior faible de l'équipe issue du Smart Tagging.
18. `control_raw` — temps d'arrivée estimé avec distance et vitesse ATD.
19. `ball_interp5_raw` — interpolation du ballon sur des gaps de cinq frames maximum.

## Expériences ATD avec majority voting

Chaque méthode précédente a été testée avec les trois fenêtres suivantes :

20. `margin_smooth25`, `margin_smooth50`, `margin_smooth125`.
21. `collective_smooth25`, `collective_smooth50`, `collective_smooth125`.
22. `tags_smooth25`, `tags_smooth50`, `tags_smooth125`.
23. `control_smooth25`, `control_smooth50`, `control_smooth125`.
24. `ball_interp5_smooth25`, `ball_interp5_smooth50`, `ball_interp5_smooth125`.

## Expériences temporelles supplémentaires

25. `bridge_short_gaps50` — reconnecter un gap `D` de maximum 50 frames lorsque l'équipe avant et après est identique.
26. `stateful_baseline` — changement d'équipe confirmé sur cinq observations et maintien de 50 frames sans observation.
27. `stateful_control` — même machine à états avec le score de temps d'arrivée.

## Expériences vidéo

28. `video_motion_smooth50` — mouvement global et mouvement local autour du ballon projeté par homographie, sur 200 échantillons.
29. `video_ball_recovery_raw` — détection locale de blobs lumineux dans les crops projetés.
30. `video_ball_recovery_smooth25`.
31. `video_ball_recovery_smooth50`.
32. `video_ball_recovery_smooth125`.
33. `video_template_raw` — template matching entre frames voisines.
34. `video_template_smooth25`, `video_template_smooth50`, `video_template_smooth125`.
35. `video_flow_raw` — suivi de points locaux avec Lucas-Kanade optical flow.
36. `video_flow_smooth25`, `video_flow_smooth50`, `video_flow_smooth125`.

La récupération vidéo détaillée a été testée sur 300 frames où l'ATD voit le ballon et 300 gaps. La détection a retrouvé 96,3 % des ballons connus, avec une erreur médiane d'environ 19 pixels, et a produit une récupération valide sur 74 % des gaps échantillonnés.

## Résultats principaux

Scores sur les 17 possessions publiques :

| Expérience | macroF1 | spellF1 | Team accuracy |
|---|---:|---:|---:|
| `baseline_raw` | 0,5972 | 0,0378 | 72,79 % |
| `tags_raw` | 0,5983 | 0,0378 | 72,96 % |
| `ball_interp5_raw` | 0,5978 | 0,0376 | 72,99 % |
| `collective_raw` | 0,5968 | 0,0395 | 72,72 % |
| `control_raw` | 0,5217 | 0,0188 | 61,17 % |
| `baseline_smooth25` | 0,6200 | 0,2868 | 79,09 % |
| `baseline_smooth50` | 0,6207 | 0,3256 | 83,49 % |
| `ball_interp5_smooth50` | 0,6199 | 0,3256 | 83,80 % |
| `tags_smooth50` | 0,6197 | 0,3256 | 83,32 % |
| `collective_smooth50` | 0,6208 | 0,3163 | 83,47 % |
| `bridge_short_gaps50` | 0,6091 | **0,3415** | 84,02 % |
| `baseline_smooth125` | 0,5812 | **0,4242** | 88,54 % |
| `video_motion_smooth50` | 0,6207 | 0,3256 | 83,49 % |
| `video_ball_recovery_smooth50` | 0,6144 | 0,3182 | 83,62 % |
| `video_ball_recovery_smooth125` | 0,5705 | **0,4242** | 88,54 % |
| `video_flow_smooth50` | 0,6153 | **0,3349** | 84,55 % |
| `video_template_smooth50` | 0,6136 | 0,3111 | 84,28 % |

## Conclusion actuelle

- Le meilleur signal frame-level brut reste la baseline de proximité.
- Le gain principal vient de la persistance temporelle, pas des features ajoutées.
- `baseline_smooth50` reste le candidat équilibré.
- `smooth125` gagne sur la petite fenêtre publique, mais ne doit pas être sélectionné uniquement avec ces 17 possessions.
- La vidéo actuelle détecte des candidats, mais son erreur spatiale est trop élevée pour améliorer les possessions.
- L'optical flow réduit l'erreur vidéo médiane à environ `1,37 px` sur les frames observées et récupère environ `85 %` des gaps échantillonnés ; son gain de `spellF1` public reste faible et doit être confirmé sur davantage de frames.
- Aucune expérience ne justifie encore de remplacer la baseline officielle sur le match complet.

## V1 sans entraînement supplémentaire

La V1 combine la prédiction ATD officielle avec une interpolation limitée aux gaps du ballon de cinq frames maximum. Elle applique ensuite le majority voting à la fin.

| Sortie V1 | macroF1 public | spellF1 public | Team accuracy |
|---|---:|---:|---:|
| `v1_raw` | 0,5975 | 0,0376 | 72,82 % |
| `v1_smooth25` | 0,6200 | 0,2868 | 79,09 % |
| `v1_smooth50` | 0,6206 | 0,3256 | 83,47 % |
| `v1_smooth125` | 0,5811 | 0,4242 | 88,53 % |

La V1 confirme que l'interpolation courte ne produit pas encore de gain mesurable par rapport à la baseline `smooth50`. Elle fournit toutefois une première version complète, non entraînée, reproductible et évaluée dans les deux étapes demandées : frame-level brut puis majority voting.

La commande est :

```powershell
uv run python scripts/run_v1_no_training.py
```

Les sorties sont dans `reports/v1_no_training/`.

## Verdict d'impact

À ce stade, aucune méthode ne fournit un gain robuste par rapport à `baseline_smooth50`.

- La meilleure sortie publique en `spellF1` est `smooth125 = 0,4242`, mais elle est obtenue par la baseline et plusieurs variantes qui ont exactement le même découpage public ; elle ne permet pas de conclure à une amélioration.
- La V1 `smooth50` donne `0,3256`, soit le même `spellF1` que la baseline `smooth50`, avec un `macroF1` légèrement inférieur.
- La vidéo avec récupération visuelle donne `0,3182` après `smooth50`, donc moins que la baseline.
- Les features de contrôle, les machines à états et les scores de temps d'arrivée dégradent les résultats.
- L'interpolation courte améliore légèrement l'identification brute dans certains runs, mais pas le `spellF1` après majority voting.

La meilleure soumission provisoire reste donc la baseline officielle `baseline_smooth50`. Les expériences ont produit un impact diagnostique important — elles montrent où l'ATD échoue — mais pas encore un gain de performance suffisant pour remplacer la baseline.

Les tableaux complets sont générés dans `reports/atd_experiments/metrics.csv` et `reports/atd_experiments/public_window_metrics.csv`. Les sorties vidéo détaillées sont dans `reports/video_ball_recovery/`.
## Mise à jour : feature store et clustering

### Feature store

Le script `scripts/build_feature_store.py` a produit une table frame-level de
199 319 frames et 54 colonnes. Elle regroupe :

- distances et ETA des joueurs les plus proches ;
- présence, gaps, interpolation courte, vitesse et accélération du ballon ;
- centroïdes, dispersion, visibilité et vitesse moyenne des équipes ;
- distances ballon-centroïde ;
- Smart Tags, set pieces et transitions ;
- qualité de l'homographie et du tracking ;
- sorties des détecteurs vidéo bright-blob, template matching et optical flow.

QA : ballon observé sur 56,94 % des frames, 59,31 % sur les frames évaluées,
638 événements Smart Tagging, homographie tactique valide sur 100 % des frames.

### Clustering non supervisé

Le script `scripts/run_feature_clustering.py` standardise les variables,
applique une PCA à 12 composantes, puis ajuste MiniBatch KMeans et GMM. Les
labels sont calculés sur toutes les frames ; les annotations publiques ne sont
utilisées qu'après coup pour le diagnostic.

| méthode | pureté publique | ARI | NMI | interprétation |
|---|---:|---:|---:|---|
| GMM 8 | 0,754 | 0,260 | 0,314 | meilleur signal de contexte |
| GMM 12 | 0,739 | 0,210 | 0,300 | signal similaire, plus fragmenté |
| KMeans 16 | 0,670 | 0,169 | 0,206 | séparation partielle |
| KMeans 12 | 0,638 | 0,129 | 0,220 | signal faible |
| KMeans 8 | 0,618 | 0,176 | 0,175 | surtout qualité/visibilité |

Interprétation : les clusters séparent principalement la visibilité du ballon,
la qualité du tracking et l'activité des tags. Ils ne constituent pas une
identité stable de l'équipe en possession.

## Expériences clusterisées

`scripts/run_cluster_heuristics.py` a testé la confiance GMM, la stabilité des
clusters, le gating des frames peu fiables, les distances aux centroïdes et le
majority voting.

| méthode | macro-F1 | Spell-F1 | accuracy | interprétation |
|---|---:|---:|---:|---|
| baseline smooth50 | 0,621 | 0,326 | 0,713 | référence équilibrée |
| cluster gate55 smooth50 | 0,701 | 0,318 | 0,751 | meilleure vue frame-level, spells moins bons |
| cluster gate70 smooth50 | 0,702 | 0,307 | 0,752 | améliore vivant/mort, trop d'abstentions |
| cluster centroid blend025 smooth50 | 0,599 | 0,315 | 0,686 | centroïde trop imprécis |

Conclusion : les clusters améliorent les métriques frame-level mais ne
résolvent pas la segmentation des possessions.

## Balayage des fenêtres de majority voting

`scripts/sweep_smoothing_windows.py` a testé toutes les fenêtres de 25 à 125
frames par pas de 5.

| méthode | meilleure fenêtre Spell-F1 | meilleur Spell-F1 | meilleure fenêtre macro-F1 | meilleur macro-F1 |
|---|---:|---:|---:|---:|
| baseline | 120 | 0,465 | 45 | 0,623 |
| cluster gate55 | 110 | 0,390 | 45 | 0,706 |
| cluster gate70 | 100 | 0,423 | 35 | 0,708 |
| centroid blend025 | 100 | 0,436 | 40 | 0,599 |

Interprétation : une grande fenêtre réduit les faux spells et améliore le
Spell-F1, mais absorbe les transitions et dégrade le macro-F1. Une fenêtre
globale ne peut donc pas optimiser les deux objectifs simultanément.

## Modèle temporel à score continu et Viterbi

`scripts/run_temporal_control_model.py` construit des émissions continues
Home/Away/Dead à partir de la distance, de l'ETA, de la densité de joueurs,
des centroïdes, des tags et de la qualité du tracking. Une optimisation Viterbi
impose ensuite une cohérence temporelle, une pénalité de changement et une
durée minimale de possession. Le majority voting est appliqué en sortie.

360 configurations ont été testées. Meilleure configuration Spell-F1 :

```text
switch_penalty = 0.50
dead_penalty = 0.30
minimum_alive_frames = 50
majority_window = 120
```

| méthode | macro-F1 | Spell-F1 | team accuracy | accuracy | possessions prédites |
|---|---:|---:|---:|---:|---:|
| baseline smooth50 | 0,621 | 0,326 | 0,835 | 0,713 | 26 |
| baseline smooth120 | 0,586 | 0,465 | 0,885 | 0,729 | 14 |
| Viterbi + contrôle + smooth120 | 0,597 | **0,520** | **0,897** | **0,740** | 13 |

Interprétation : le gain vient surtout de la régularisation temporelle et de la
durée minimale, pas directement du clustering. Le Spell-F1 progresse, mais au
prix d'une baisse du macro-F1.

## Lissage adaptatif autour des frontières

`scripts/run_adaptive_temporal_model.py` a testé 384 configurations. La fenêtre
est courte autour des transitions, intermédiaire dans les phases stables et
plus longue lorsque le ballon est absent ou l'évidence faible.

Meilleur compromis simultané par rapport à `baseline_smooth50` :

```text
switch_penalty = 1.00
dead_penalty = 0.75
minimum_alive_frames = 50
stable_window = 35
boundary_window = 15
uncertain_window = 50
```

Les fenêtres sont des demi-largeurs centrées, à 25 FPS.

| méthode | macro-F1 | Spell-F1 | accuracy | interprétation |
|---|---:|---:|---:|---|
| baseline smooth50 | 0,621 | 0,326 | 0,713 | référence |
| adaptatif équilibré | **0,636** | **0,409** | **0,731** | améliore les deux métriques |
| Viterbi global smooth120 | 0,597 | **0,520** | 0,740 | optimise surtout les spells |

Le lissage adaptatif est donc la meilleure solution équilibrée actuellement,
tandis que le Viterbi global reste le meilleur choix si le classement ne
retient que le Spell-F1.

## Broadcast et audio : état

Les fichiers ATD et homographie broadcast sont présents, mais la vidéo
`BAR-ATH_broadcast.mp4` n'était pas encore disponible dans le repo au moment de
ces expériences. L'audio n'a donc pas encore été testé.

L'audio devrait être utilisé prioritairement pour les frontières et les phases
mortes : sifflets, reprises de jeu, ruptures d'ambiance et détection des
replays. Il est peu probable qu'il identifie directement l'équipe en
possession. La vidéo broadcast devra d'abord être synchronisée avec la timeline
tactique, car elle contient des cuts, des replays et une homographie valide sur
seulement 65,8 % de ses frames.

## Verdict mis à jour

Les premières heuristiques locales n'apportaient pas de gain robuste. Le
modèle temporel change toutefois ce constat :

- le Viterbi améliore le Spell-F1 public jusqu'à 0,520 ;
- le lissage adaptatif améliore simultanément macro-F1 et Spell-F1 par rapport
  à `baseline_smooth50` ;
- aucune méthode n'atteint encore 90 % et la séparation Home/Away reste
  imparfaite ;
- la sélection repose toujours sur 17 possessions publiques, donc les scores
  doivent être considérés comme exploratoires ;
- la prochaine source de gain prioritaire est l'audio broadcast pour améliorer
  les frontières, puis une extraction complète et synchronisée de la vidéo.

Les sorties sont disponibles dans :

- `reports/atd_experiments/` ;
- `reports/video_heuristics/` ;
- `reports/clustering/` ;
- `reports/temporal_control/` ;
- `reports/adaptive_temporal/`.
