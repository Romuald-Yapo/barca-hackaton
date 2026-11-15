# Plan de modélisation — Estimation de la possession de balle

**Challenge :** More Than a Hack 2027 — Phase 1: Ball Possession  
**Match cible :** FC Barcelona – Athletic Club (BAR-ATH)  
**Périmètre :** flux tactique uniquement — **aucune donnée Broadcast utilisée**  
**Version :** 1.0 — 9 novembre 2026

---

## 0. Objectif et principes de travail

L'objectif est de produire une prédiction pour chacune des **199 319 frames** de la vidéo tactique, à 25 images par seconde, avec l'un des labels suivants :

- `H` : Barcelone possède le ballon et le jeu est en cours.
- `A` : Athletic Club possède le ballon et le jeu est en cours.
- `D` : ballon arrêté / jeu non en cours.
- `X` : abstention du modèle ou frame sans vérité terrain.

La métrique de classement principale est le **`spellF1`**. Elle évalue la reconstruction des possessions comme des séquences temporelles complètes, avec un appariement un-à-un entre possessions réelles et prédites à plusieurs seuils d'IoU temporel. Le `macroF1` frame par frame est rapporté comme métrique secondaire et sert de premier critère de départage.

Le projet doit donc chercher à :

1. retrouver la bonne équipe en possession ;
2. détecter correctement les vrais changements de possession ;
3. éviter de fragmenter une possession en plusieurs segments artificiels ;
4. éviter de fusionner deux possessions distinctes ;
5. reconnaître les véritables arrêts de jeu (`D`) ;
6. rester robuste lorsque le ballon ou certains joueurs ne sont pas détectés.

**Principe directeur :** nous ne cherchons pas seulement à classifier chaque frame. Nous cherchons à reconstruire une séquence d'états cohérente sur tout le match.

### 0.1. Périmètre des données

Le plan utilise uniquement les ressources du dossier `data/tactical/` :

- vidéo tactique originale ;
- Automatic Tracking Data (ATD) tactique ;
- Smart Tagging tactique ;
- homographie tactique.

Le dossier `data/broadcast/` et toutes les données issues du flux Broadcast sont exclus du projet décrit ici.

### 0.2. Distinction entre acquis et propositions

Ce document distingue explicitement :

- **Déjà testé par le collègue :** résultats rapportés dans `xp.md`. Ils constituent un point de départ à reproduire, mais le code et les sorties détaillées doivent encore être audités.
- **Proposé pour la suite :** étapes, features et architectures à développer et à évaluer. Elles ne sont pas considérées comme validées avant expérimentation.

---

## 1. Comprendre les données et leurs limites

### 1.1. Inventaire du flux tactique

| Ressource | Taille / granularité | Rôle dans le projet |
|---|---:|---|
| Vidéo tactique | 199 319 frames, 25 FPS, 1920 × 1080 | Observation visuelle du match et éventuelle extraction de features vidéo |
| ATD tactique | 199 319 enregistrements, un par frame | Positions et vitesses des joueurs suivis et du ballon |
| Smart Tagging | 638 annotations sous forme d'intervalles | Contexte tactique : phases de jeu, transitions et coups de pied arrêtés |
| Homographie tactique | 199 319 matrices 3 × 3 | Projection entre coordonnées normalisées du terrain et pixels de la vidéo |

La vidéo tactique définit la timeline de référence : la ligne `i` de la soumission correspond à la frame vidéo `i`, avec des indices allant de `0` à `199318`.

### 1.2. ATD — Automatic Tracking Data

L'ATD est généré automatiquement par le pipeline de vision par ordinateur de Metrica. Il fournit, à chaque frame, la position et la vitesse des objets détectés sur le terrain, dont le ballon.

Les colonnes typiques exposées après chargement avec `kloppy` sont :

- `frame_id` ;
- `ball_x`, `ball_y` ;
- pour chaque slot de tracking : position `x`, position `y` et vitesse `s`.

Les coordonnées sont normalisées entre `0` et `1`, et non exprimées directement en mètres. Les métadonnées indiquent environ 105 × 68 m ; la configuration `games.py` utilisée par la baseline porte une dimension corrigée de 105,3 × 68,0 m. Pour les distances physiques, reprendre la convention du dépôt officiel et vérifier qu'elle est identique entre toutes les expériences. La conversion des deux axes doit tenir compte du rapport longueur/largeur du terrain.

**Alignement :** `video_frame = frame_id - 1`. Le décalage d'une unité doit être testé automatiquement dès le chargement.

#### Limites à prendre en compte

- Il existe **46 slots de tracking pour 22 joueurs**. Un slot n'est pas une identité permanente de joueur : les slots peuvent disparaître, réapparaître ou changer d'identité.
- L'XML de métadonnées associe les slots à une équipe (`teamId`) ; cette association doit être chargée depuis le fichier de métadonnées et contrôlée. L'indice du slot ou le champ `ShirtNumber` ne doit pas être interprété comme un vrai numéro de maillot ou une identité de joueur.
- L'ATD ne fournit donc pas une identité individuelle fiable sur tout le match. Il faut privilégier des représentations collectives et permutation-invariantes.
- Le nombre moyen d'objets suivis sur les frames évaluées est d'environ 20,5 et peut ponctuellement dépasser 22 lorsque le tracker duplique un objet.
- Les positions et vitesses sont automatiques, non corrigées manuellement ; elles peuvent être bruitées, fragmentées ou incohérentes.
- Les valeurs manquantes sont représentées par `NaN`.
- Le ballon est visible sur 56,9 % de toutes les frames, et sur 59,3 % des frames évaluées. Il est donc absent sur **40,7 % des frames évaluées** (64 643 sur 158 857). Les absences sont notamment liées aux occultations, aux regroupements de joueurs et aux ballons aériens ; elles ne doivent pas être considérées comme nécessairement aléatoires.

**Conséquence de modélisation :** toute feature doit être capable de fonctionner avec le ballon absent. La présence/absence du ballon doit elle-même être une information explicite du modèle.

### 1.3. Smart Tagging

Le Smart Tagging tactique contient 638 annotations automatiques de phases de jeu et de situations arrêtées. Le fichier expose notamment :

- `timestamp` et `end_timestamp` ;
- un `code` décrivant le type d'événement/phase ;
- des groupes de tags tels que `Team`, `Half`, `Type`, `Side` ou d'autres attributs selon le code.

Les catégories mentionnées dans la documentation incluent notamment `BUILD UP`, `PROGRESSION`, `ATTACKING TRANSITION`, `DEFENSIVE TRANSITION`, `SET PIECES`, `BALL IN FINAL THIRD`, `BALL IN THE BOX`, `PLAYERS IN THE BOX`, `CREATING CHANCES` et `LONG BALL`.

**Alignement :** les timestamps sont exprimés en secondes depuis le début de la vidéo tactique. La conversion documentée est `video_frame = round(timestamp_seconds * 25)`. Les intervalles doivent être projetés sur la timeline des frames, en conservant leur début et leur fin.

#### Limites à prendre en compte

- Le Smart Tagging est lui aussi **automatique**, et non une vérité terrain annotée manuellement.
- Il peut comporter des faux positifs et faux négatifs.
- Une phase tactique n'est pas nécessairement équivalente à une possession : un intervalle de tag peut couvrir une partie seulement d'une possession, commencer avant ou finir après, voire chevaucher plusieurs séquences.
- Le champ `Team` décrit l'équipe associée à l'événement tagué ; il ne faut pas en déduire automatiquement que cette équipe possède le ballon sur l'intégralité de l'intervalle.
- Les valeurs et noms de colonnes doivent être normalisés sans supposer une casse homogène.

**Conséquence de modélisation :** utiliser les tags comme contexte ou prior probabiliste, jamais comme labels de possession à recopier directement.

### 1.4. Homographie tactique

L'homographie est une matrice `3 × 3` par frame. Elle relie les coordonnées normalisées du terrain aux pixels de l'image vidéo. Les données tactiques disposent d'une homographie marquée valide sur toutes les frames.

Champs importants :

- les neuf coefficients de la matrice ;
- `valid` : indique si une transformation est disponible ;
- `reprojection_error` : renseigne sur la qualité de l'ajustement ; plus l'erreur est faible, plus la transformation est fiable.

Sur le flux tactique, l'erreur médiane rapportée est d'environ 0,24 pendant le jeu, et augmente autour de la mi-temps lorsque la caméra n'est pas dirigée vers le terrain.

**Usages envisagés :**

- projeter les positions ATD dans l'image pour visualiser/contrôler l'alignement ;
- extraire des régions d'intérêt autour du ballon ou des joueurs ;
- construire des features spatiales cohérentes ;
- vérifier automatiquement les cas d'incohérence entre tracking et image.

L'homographie ne constitue pas une observation indépendante de la possession : c'est une transformation géométrique qui facilite la combinaison du tracking et de la vidéo.

### 1.5. Périmètre évalué et frames non évaluées

Les frames évaluées sont les deux périodes de jeu suivantes :

| Segment | Première frame | Dernière frame | Nombre | Évaluation |
|---|---:|---:|---:|---|
| Avant match | 0 | 11 184 | 11 185 | Non |
| Première mi-temps | 11 185 | 90 651 | 79 467 | Oui |
| Mi-temps | 90 652 | 110 807 | 20 156 | Non |
| Deuxième mi-temps | 110 808 | 190 197 | 79 390 | Oui |
| Après le coup de sifflet final | 190 198 | 199 318 | 9 121 | Non |
| **Total** | **0** | **199 318** | **199 319** | **158 857 évaluées** |

Les 40 462 frames non évaluées peuvent recevoir `X` sans impact sur le score. Il ne faut cependant pas confondre ces zones avec les périodes de jeu évaluées : les frames des mi-temps sont délimitées précisément par le dépôt.

Sur les frames évaluées, `X` n'est pas une classe neutre : `macroF1` le compte comme une erreur. Pour le `spellF1`, le scorer remplace une abstention par le dernier label fourni dans la mi-temps avant l'appariement des possessions. `X` doit donc être utilisé avec discernement, et non comme valeur générique lorsque le modèle est incertain.

### 1.6. Labels et règles métier de possession

Les quatre labels de sortie sont `H`, `A`, `D`, `X`, mais la vérité terrain des frames évaluées contient les trois états `H`, `A`, `D`.

Points essentiels :

- Une possession change lorsque l'équipe adverse obtient un **contrôle effectif** du ballon.
- Un simple contact, une déviation non contrôlée ou un duel aérien gagné sans contrôle ultérieur ne suffisent pas à établir un changement de possession.
- Lorsque le ballon est en jeu mais disputé ou libre, la possession reste attribuée à la dernière équipe qui la détenait.
- `D` signifie que le ballon n'est pas en jeu ; ce n'est pas une classe de doute ou de ballon simplement non détecté.
- Les états « transition », « contesté » ou « inconnu » ne sont pas des classes distinctes de la vérité terrain.

Ces règles doivent guider la segmentation temporelle et l'interprétation des changements de labels.

---

## 2. Règles de conformité et prévention des fuites

Cette section doit être respectée dans toutes les étapes du projet.

1. **Aucune annotation manuelle du match BAR-ATH.** Il est interdit de créer à la main des labels de possession ou de corriger manuellement les positions du tracking, les identités ou les segments de sortie.
2. **Aucun usage de la vérité terrain privée.** Elle ne doit jamais être utilisée pour entraîner, ajuster, choisir des features, calibrer des seuils ou comparer des variantes.
3. **Vérité terrain publique limitée aux tests de pipeline.** Les cinq premières minutes de jeu (17 possessions) peuvent servir à vérifier l'alignement, le format et la bonne interprétation des labels. Elles ne doivent pas servir de jeu d'entraînement ni de jeu de tuning.
4. **Données annotées d'autres matchs :** elles peuvent être utilisées si leur provenance, leur licence, leur volume et leur méthode d'annotation sont documentés et si les organisateurs peuvent les inspecter.
5. **Pseudo-labels :** les méthodes non supervisées, auto-supervisées ou les pseudo-labels produits automatiquement sur BAR-ATH sont possibles, à condition qu'aucun humain ne décide ou ne corrige ces labels.
6. **Reproductibilité :** le dépôt public doit contenir le code, les dépendances fixées et la commande permettant de reproduire exactement la soumission.
7. **Confidentialité des données :** ne pas publier ni redistribuer les vidéos, les fichiers ATD, les Smart Tags, l'homographie ou des données dérivées permettant de reconstruire le match. Ne pas versionner le dossier `data/`.
8. **Usage de LLM :** les règles permettent l'usage de LLM, mais interdisent de transmettre les vidéos, datasets, informations confidentielles ou extraits des ressources du concours à un LLM tiers hébergé/cloud. Toute extraction vidéo ou inférence visuelle sur les assets du challenge doit donc être exécutée dans un environnement local ou explicitement autorisé.

---

## 3. Architecture générale proposée

Nous recommandons une architecture progressive : partir d'un pipeline ATD robuste, reprendre et auditer le HMM du collègue, enrichir les features, puis comparer des modèles temporels plus expressifs. La fusion vidéo ne sera introduite qu'après validation des premières briques.

```text
                 FLUX TACTIQUE (timeline de référence)
                               |
          +--------------------+---------------------+
          |                    |                     |
       ATD tactique       Smart Tagging        Vidéo tactique
          |                    |                     |
          |                Alignement             Homographie
          |                    |                     |
          +--------------------+---------------------+
                               |
                      Contrôles qualité / QA
                               |
                       Feature engineering
                               |
       +-----------------------+------------------------+
       |                       |                        |
   Features ATD          Features tactiques       Features visuelles
       |                       |                        |
       +-----------------------+------------------------+
                               |
                      Fusion multimodale
                         (par étapes)
                               |
                Modèle temporel : HMM / TCN / CRF
                               |
                    Décodage / post-traitement
                               |
                   Segmentation des possessions
                               |
                   frame,label (199 319 lignes)
                               |
             validation officielle du fichier CSV
```

### Stratégie de développement

- **Version 1 :** ATD seul + baseline officielle reproduite.
- **Version 2 :** HMM du collègue audité et reproduit.
- **Version 3 :** HMM enrichi avec features collectives et Smart Tagging.
- **Version 4 :** modèle supervisé temporel (TCN) entraîné uniquement sur des matchs externes annotés ou une source de labels autorisée.
- **Version 5 :** fusion TCN/features/Smart Tagging et décodeur temporel HMM ou CRF.
- **Version 6 :** ajout expérimental de features extraites de la vidéo tactique, exécutées localement.

Le modèle final ne sera pas choisi sur la complexité de l'architecture mais sur les performances mesurées sur des matchs de validation indépendants, avec la métrique `spellF1`.

---

## 4. Étape 1 — Chargement, alignement et contrôle qualité des données

### 4.1. Outils et dépendances

Socle recommandé :

- Python ;
- `numpy` et `pandas` pour les tableaux et calculs ;
- `kloppy` pour charger le tracking Metrica/EPTS et les annotations Sportscode ;
- `opencv-python` pour la lecture vidéo, les projections et les crops ;
- `matplotlib` ou `plotly` pour les visualisations ;
- `scipy` pour les filtres et calculs géométriques ;
- `scikit-learn` pour les métriques et modèles classiques ;
- `PyTorch` pour les modèles temporels/visuels ;
- `pyarrow` pour stocker les tables intermédiaires en Parquet.

Les versions devront être épinglées dans `requirements.txt` ou `environment.yml`.

### 4.2. Chargement

Utiliser en priorité les chargeurs fournis dans le dépôt officiel :

- `starter/load_atd.py` ou le chargement `kloppy.metrica.load_tracking_epts(...)` ;
- `starter/load_smart_tagging.py` ou `kloppy.sportscode.load(...)` ;
- `starter/load_homography.py` pour lire les matrices et projeter les positions.

Ne pas réécrire immédiatement les parsers : commencer par reproduire les notebooks fournis, puis ajouter des tests autour des données chargées.

### 4.3. Tests d'alignement obligatoires

- Vérifier que l'ATD contient 199 319 lignes.
- Vérifier que les `frame_id` vont de 1 à 199 319 et convertir par `frame_id - 1`.
- Vérifier que l'homographie a 199 319 lignes et des indices 0 à 199 318.
- Convertir les timestamps Smart Tagging en frames avec `round(timestamp_seconds * 25)` et contrôler quelques intervalles.
- Vérifier que toutes les tables sont associées à la même timeline tactique.
- Vérifier l'absence de décalage d'une frame, de doublons ou de frames manquantes.
- Projeter plusieurs positions ATD sur la vidéo avec l'homographie et contrôler visuellement l'alignement.

### 4.4. Contrôles qualité à produire

Créer un rapport automatique contenant :

- taux de présence du ballon sur toutes les frames et sur les frames évaluées ;
- longueur et distribution des séquences consécutives où le ballon est absent ;
- nombre de joueurs/slots visibles par frame et par équipe (en utilisant le mapping d'équipe de l'XML ATD) ;
- fréquence des disparitions/réapparitions de slots ;
- distributions des positions, vitesses et accélérations ;
- détections de valeurs impossibles ou aberrantes ;
- taux de valeurs manquantes par variable ;
- nombre, durée et catégories des Smart Tags ;
- chevauchements et lacunes entre annotations ;
- distribution de `reprojection_error` et vérification de `valid`.

**Livrable :** un rapport QA reproductible et des visualisations de séquences représentatives, sans publier les données dérivées du match dans le dépôt public.

---

## 5. Étape 2 — Feature engineering à partir de l'ATD

Les features doivent être calculées frame par frame, puis éventuellement agrégées dans des fenêtres temporelles. Les slots de tracking étant instables, les features collectives doivent être privilégiées.

### 5.1. Features du ballon

- `ball_visible` : indicateur de présence du ballon.
- `ball_x`, `ball_y` : position normalisée lorsque disponible.
- vitesse du ballon, si elle est disponible ou calculable de manière fiable.
- accélération du ballon, uniquement après contrôle du bruit.
- temps depuis la dernière détection.
- temps jusqu'à la prochaine détection (utilisable en analyse hors ligne).
- longueur de la séquence courante d'absence.
- indicateurs signalant si les coordonnées sont observées, interpolées ou manquantes.

**Règle de traitement des gaps :**

- Ne pas interpoler sans distinction des absences courtes et longues.
- Une interpolation courte peut être testée comme feature dérivée, mais elle doit rester accompagnée d'un masque indiquant qu'elle n'est pas observée.
- Ne pas extrapoler le ballon sur de longues périodes comme s'il s'agissait d'une observation réelle.
- Comparer systématiquement les modèles avec et sans interpolation.

### 5.2. Features joueur-ballon

Lorsque le ballon est visible :

- distance du ballon au joueur le plus proche de chaque équipe ;
- distance au deuxième et troisième joueur le plus proche ;
- différence entre la meilleure distance H et la meilleure distance A ;
- nombre de joueurs de chaque équipe dans des rayons autour du ballon ;
- vitesse du joueur le plus proche ;
- vitesse relative entre joueur et ballon ;
- estimation du temps d'arrivée au ballon ;
- indicateur d'ambiguïté lorsque plusieurs joueurs sont à des distances proches.

Ces variables améliorent la baseline « joueur le plus proche », mais ne doivent pas être assimilées directement à une preuve de contrôle.

### 5.3. Features collectives par équipe

Pour chaque frame et chaque équipe, calculer des agrégats robustes :

- nombre de joueurs détectés ;
- centroïde de l'équipe ;
- distance moyenne/médiane au centroïde ;
- dispersion spatiale en X et Y ;
- aire de l'enveloppe convexe, si le nombre de points est suffisant ;
- distances entre les lignes ou groupes de joueurs ;
- vitesse moyenne, médiane et quantiles des joueurs ;
- dispersion des vitesses ;
- direction moyenne des déplacements ;
- compacité et orientation du bloc ;
- distance entre les centroïdes des deux équipes.

Calculer également des variables comparatives H vs A : différence de compacité, différence de vitesse moyenne, différence de progression vers le but adverse, etc.

### 5.4. Features cinématiques

À partir de séries de positions :

- vitesse lissée ;
- accélération ;
- changement de direction ;
- déplacement net sur plusieurs horizons ;
- variation de distance entre les deux blocs.

Les dérivées amplifient le bruit. Utiliser des filtres robustes ou des différences sur plusieurs frames, puis comparer les résultats avec les variables brutes. Ne pas supposer qu'un slot correspond au même joueur tout au long du match.

### 5.5. Features de contrôle potentiel / Pitch Control

Explorer une représentation plus riche que la simple proximité :

- temps d'arrivée estimé au ballon pour les joueurs des deux équipes ;
- capacité relative d'une équipe à atteindre le ballon avant l'autre ;
- vitesse et orientation du joueur ;
- position des adversaires proches ;
- probabilité approximative de contrôle d'une zone.

Le Pitch Control est une piste expérimentale, pas une vérité terrain. Il doit être calculé sans identité permanente des joueurs et évalué sur des matchs externes annotés.

### 5.6. Gestion des slots instables

- Ne pas encoder `Track 12` comme une identité permanente.
- Éviter les statistiques longitudinales par slot sans mécanisme de ré-identification validé.
- Construire des ensembles de joueurs par équipe à chaque frame.
- Utiliser des agrégations invariantes à l'ordre des joueurs (mean, max, quantiles, attention/Deep Sets).
- Conserver, si utile, des indicateurs de continuité locale de trajectoire, sans prétendre qu'ils représentent une identité stable.

**Livrable :** une table de features ATD, une documentation de chaque variable et un rapport montrant la distribution et les valeurs manquantes.

---

## 6. Étape 3 — Transformer le Smart Tagging en variables temporelles

### 6.1. Construction des features

À partir des intervalles de tags, créer pour chaque frame :

- indicateur actif/inactif pour chaque catégorie ;
- catégorie de tag active ;
- équipe associée au tag, lorsqu'elle existe ;
- période (première ou deuxième mi-temps) ;
- type de coup de pied arrêté, lorsqu'il existe ;
- temps écoulé depuis le début du tag ;
- temps restant avant la fin du tag ;
- nombre de tags actifs simultanément ;
- indicateur signalant l'absence de tag.

Les intervalles qui se chevauchent doivent être conservés sous forme multi-label plutôt que forcés dans une unique catégorie si le format le permet.

### 6.2. Utilisation correcte

- Le tag `Team` peut devenir un indice en faveur de l'équipe qui réalise l'action.
- Il ne doit pas être copié comme label de possession sur tout l'intervalle.
- Les tags de phase offensive/transition peuvent constituer un prior, mais doivent être pondérés par l'ATD et le modèle temporel.
- Les tags de set pieces peuvent aider à reconnaître le contexte de reprise, mais la prédiction doit toujours respecter la définition `D` lorsque le ballon n'est pas en jeu.

### 6.3. Ablations obligatoires

Comparer au minimum :

1. ATD seul ;
2. ATD + Smart Tagging complet ;
3. ATD + prior d'équipe du Smart Tagging uniquement ;
4. ATD + catégories de phase sans champ équipe ;
5. Smart Tagging seul comme diagnostic, pas nécessairement comme modèle final.

**Livrable :** table frame-level de features Smart Tagging et tableau d'ablation avec `spellF1`, `macroF1` et métriques spécifiques aux périodes où le ballon est absent.

---

## 7. Étape 4 — Utiliser la vidéo tactique et l'homographie

Cette étape est optionnelle dans la première version, mais elle doit être préparée dans l'architecture.

### 7.1. Usages visuels ciblés

Ne pas commencer par entraîner un gros modèle vidéo de bout en bout. Commencer par des extractions ciblées :

- frames ou courts clips autour des périodes où le ballon est visible ;
- crops autour du ballon projeté par l'homographie ;
- crops autour des joueurs proches du ballon ;
- zones denses où plusieurs joueurs sont en interaction ;
- séquences contenant un possible changement de possession.

### 7.2. Features visuelles possibles

- embeddings d'image ou de courts clips produits par un modèle préentraîné accessible et reproductible ;
- indicateurs visuels de proximité/contact pied-ballon, sous réserve de pouvoir les estimer automatiquement ;
- mouvement visuel local/optical flow ;
- cohérence de l'image avec le tracking ;
- représentation visuelle des interactions entre joueurs.

Les modèles doivent être publics ou accessibles aux organisateurs, versionnés et exécutables dans l'environnement du projet. Toute inférence sur les assets du challenge doit rester dans un environnement local ou explicitement autorisé, conformément aux règles de confidentialité du concours.

### 7.3. Contrôles d'homographie

- vérifier `valid` avant toute projection ;
- conserver `reprojection_error` comme indicateur de confiance ;
- ne pas traiter une projection comme fiable uniquement parce qu'une matrice existe ;
- vérifier les projections sur un échantillon de frames et automatiser les tests de cohérence.

### 7.4. Évaluer la valeur ajoutée de la vidéo

Ajouter la modalité vidéo uniquement après avoir construit une référence ATD + Smart Tagging stable. Comparer les modèles sur les mêmes données de validation externe. Le coût de calcul, la latence de préparation et les gains réels de `spellF1` doivent être documentés.

**Livrable :** pipeline local d'extraction de features visuelles, cache de features et expérience d'ablation ATD+tags vs ATD+tags+vidéo.

---

## 8. Étape 5 — Reproduire les baselines officielles

Avant toute nouvelle architecture, reconstruire les quatre baselines officielles pour vérifier le chargement des données, la timeline et le calcul des métriques.

Les quatre configurations appliquent le même algorithme : joueur suivi le plus proche du ballon, puis vote majoritaire temporel. Dans la baseline brute, si le ballon n'est pas détecté, aucun joueur n'est choisi et le label direct devient `D`. Dans les variantes lissées, les observations sans ballon ne votent ni pour H ni pour A ; une frame reçoit `D` lorsque la fenêtre ne contient aucune observation H/A. C'est un proxy très imparfait, car une absence de détection du ballon ne signifie pas que le ballon est réellement arrêté. Seule la demi-largeur de la fenêtre change :

| Baseline | Fenêtre | Taille totale | Durée à 25 FPS |
|---|---:|---:|---:|
| Raw | 0 | 1 frame | 0,04 s |
| Smooth 25 | ±25 | 51 frames | 2,04 s |
| Smooth 50 | ±50 | 101 frames | 4,04 s |
| Smooth 125 | ±125 | 251 frames | 10,04 s |

Résultats officiels sur le match complet :

| Baseline | macroF1 | spellF1 |
|---|---:|---:|
| Raw | 0,6532 | 0,0486 |
| Smooth 25 | 0,6752 | 0,1996 |
| Smooth 50 | 0,6724 | **0,2433** |
| Smooth 125 | 0,6356 | 0,2242 |

`baseline_smooth50` est la référence officielle à dépasser (`spellF1 = 0,2433`).

Attention : sur les cinq minutes de vérité terrain publique, `smooth125` peut sembler meilleur que `smooth50`. Ce résultat illustre le risque de choisir les hyperparamètres sur une fenêtre de seulement 17 possessions. La vérité publique sert à vérifier le pipeline, pas à sélectionner le modèle.

**Livrable :** reproductibilité exacte des quatre fichiers de baseline et un rapport des métriques.

---

## 9. Étape 6 — Auditer et consolider le HMM déjà développé

Le HMM est le premier modèle avancé à reprendre, car il a déjà été expérimenté par un membre de l'équipe.

### 9.1. Modèle existant rapporté

Le rapport d'expérience indique :

- états cachés : `H`, `A`, `D` ;
- canaux d'observation : joueur le plus proche, vitesse du ballon, vitesse moyenne des joueurs ;
- estimation des paramètres avec EM ;
- décodage de la séquence avec Viterbi ;
- paramètre `alpha` ;
- durée minimale testée de 3 secondes.

Le meilleur réglage rapporté est `alpha = 0,15` avec une durée minimale de 3 secondes.

Sur deux matchs Metrica artificiellement dégradés, le collègue rapporte `spellF1 = 0,387` contre `0,252` pour la baseline, avec un gain observé sur 12 copies dégradées. Sur BAR-ATH, il rapporte un `macroF1` de 0,75 contre 0,62 pour la baseline, mais le `spellF1` public reste non concluant compte tenu des 17 possessions disponibles.

Il rapporte également :

- avec le ballon absent, le tracking seul atteint environ 49 % de bonne identification d'équipe, contre 68 % avec le Smart Tagging ;
- l'usage de `X` comme label d'incertitude n'a pas amélioré le `spellF1` ;
- le modèle semi-Markov testé n'apporte qu'environ +0,01, considéré comme non concluant ;
- une sur-prédiction de `D` a été observée dans certaines expériences.

Ces résultats sont des résultats **rapportés dans `xp.md`**, à reproduire. Les 12 copies d'un petit nombre de matchs ne constituent pas 12 matchs indépendants.

### 9.2. Audit technique avant reprise

Vérifier dans le code :

- définition mathématique exacte d'`alpha` ;
- distributions d'émission de chaque canal ;
- matrice de transition et contraintes éventuelles ;
- initialisation des paramètres EM ;
- critère d'arrêt et stabilité d'EM ;
- association des états latents aux labels `H`, `A`, `D` (risque de permutation des états) ;
- traitement des `NaN` et des longues séquences sans ballon ;
- normalisation des features ;
- calcul de la durée minimale et endroit où elle est appliquée ;
- gestion des deux mi-temps et réinitialisation des états ;
- reproductibilité avec graines fixes.

### 9.3. Expériences de consolidation

- reproduire le résultat du collègue à l'identique ;
- mesurer le résultat par match et non seulement en agrégeant toutes les frames ;
- tester la stabilité selon plusieurs graines ;
- tester les canaux un par un puis en combinaison ;
- vérifier si les gains de vitesse des joueurs se maintiennent sur des matchs non vus ;
- analyser le nombre de possessions prédites, leurs durées et les changements trop fréquents ;
- examiner les erreurs de la classe `D`.

**Livrable :** HMM reproductible, configuration documentée, rapport d'ablation et tests unitaires.

---

## 10. Étape 7 — Construire un protocole d'entraînement et de validation valide

C'est une étape critique : nous ne disposons pas de vérité terrain complète sur BAR-ATH.

### 10.1. Données d'entraînement

Pour entraîner des modèles supervisés (TCN, CRF neuronal, modèle de vision), utiliser des matchs externes pour lesquels les labels de possession frame-level sont disponibles ou peuvent être annotés conformément aux règles.

Pour chaque jeu de données externe, documenter :

- source et licence ;
- nombre de matchs ;
- format et fréquence ;
- labels disponibles et leur définition ;
- méthode d'annotation ;
- volume de frames et de possessions ;
- règles de conversion vers `H`, `A`, `D` ;
- limites et éventuels décalages temporels.

Ne pas supposer que les jeux de données Metrica publics comprennent automatiquement des labels frame-level équivalents à ceux du challenge : vérifier les événements, les intervalles et leur sémantique avant de les utiliser.

### 10.2. Validation par match

- Séparer les données par match, jamais par frames aléatoires.
- Garder un ou plusieurs matchs entiers comme validation.
- Éviter qu'une séquence ou une version dégradée du même match apparaisse à la fois en train et en validation.
- Les copies artificiellement dégradées d'un match ne comptent pas comme des matchs indépendants.
- Si le nombre de matchs est faible, le signaler et publier les résultats par match avec moyenne, dispersion et intervalles d'incertitude.

### 10.3. Dégradation synthétique des données externes

Pour rapprocher les matchs externes de la qualité du tracking BAR-ATH, construire un générateur de dégradation contrôlé :

- pertes du ballon avec un taux global proche de celui documenté pour BAR-ATH ;
- séquences d'absence courtes et longues, avec des distributions de durées réalistes ;
- disparitions ponctuelles de joueurs ;
- bruit sur les coordonnées et vitesses ;
- fragmentation et réapparition de trajectoires ;
- éventuelles duplications de tracks.

Ne pas se limiter à supprimer des frames au hasard. La durée et la structure temporelle des gaps comptent. Tester plusieurs profils de dégradation et consigner les graines.

### 10.4. Public ground truth : usage autorisé et limité

La vérité terrain publique couvre les cinq premières minutes de jeu, frames 11 185 à 18 684, avec 17 possessions. Elle sert uniquement à :

- vérifier l'alignement frame/ATD/vidéo/tags ;
- contrôler le format de sortie ;
- vérifier la correspondance des labels ;
- tester l'exécution du scorer.

Elle ne doit pas servir à entraîner, choisir un seuil, optimiser `alpha`, régler une fenêtre ou sélectionner une architecture.

---

## 11. Étape 8 — Modèles à comparer après le HMM

Nous recommandons de développer les modèles dans l'ordre suivant.

### 11.1. Modèle A — HMM enrichi

**Entrées :** features ATD + indicateurs de qualité + Smart Tagging.  
**Rôle :** modèle de référence avancé.

Axes à tester :

- nouvelles features collectives ;
- features de disponibilité du ballon ;
- prior de Smart Tagging complet vs prior d'équipe ;
- calibration de `D` ;
- transitions plus ou moins pénalisées ;
- traitement séparé des périodes d'absence longue du ballon.

### 11.2. Modèle B — TCN temporel

Un Temporal Convolutional Network apprend les dépendances temporelles à partir de fenêtres de features.

**Entrées possibles :**

- séquences de features ATD ;
- masques de présence/missingness ;
- features collectives par équipe ;
- features Smart Tagging ;
- contexte passé et futur (modèle hors ligne bidirectionnel ou fenêtres centrées).

**Sortie :** probabilités par frame pour `H`, `A`, `D`.

Le TCN doit être entraîné sur des matchs externes annotés ou sur un protocole de labels algorithmiques documenté. La vérité terrain de BAR-ATH ne peut pas servir à l'entraînement.

### 11.3. Modèle C — TCN + HMM ou CRF

Le TCN estime les scores d'émission par frame ; un HMM ou CRF impose ensuite une cohérence entre états successifs.

Cette architecture sépare :

- l'apprentissage des indices locaux et temporels ;
- les contraintes de séquence et la reconstruction des possessions.

Elle est particulièrement pertinente vis-à-vis du `spellF1`.

### 11.4. Modèle D — Détection en deux étapes

Tester une architecture hiérarchique :

1. détecter si le ballon est en jeu (`alive`) ou arrêté (`D`) ;
2. lorsque le jeu est en cours, identifier l'équipe en possession (`H` ou `A`).

Cette décomposition est une hypothèse à tester. Elle ne doit pas être imposée sans comparaison avec un modèle trois classes direct. La détection de `D` devra exploiter les joueurs, leur disposition collective, le contexte tactique et éventuellement la vidéo.

### 11.5. Modèle E — Graphes / chemin de possession

Piste avancée à explorer si les modèles précédents plafonnent :

- représenter les joueurs comme des nœuds ;
- représenter les relations spatiales ou de possession potentielle comme des arêtes ;
- construire un graphe dynamique au fil des frames ;
- rechercher une séquence cohérente de joueurs impliqués dans la possession ;
- décoder un chemin de possession avec CRF/Viterbi.

Cette piste s'inspire des travaux de type PathCRF. Elle est prometteuse pour les périodes sans ballon, mais plus coûteuse et plus complexe. Elle ne doit être lancée qu'après une baseline de validation solide.

---

## 12. Étape 9 — Fusion multimodale adaptative

La fusion doit être progressive et justifiée par des expériences d'ablation.

### 12.1. Early fusion

Concaténer les features ATD, Smart Tagging et éventuellement vidéo dans un vecteur frame-level, puis entraîner un modèle commun.

Avantage : simple à mettre en place.  
Limite : les sources bruitées ou manquantes peuvent contaminer les autres variables ; les échelles et granularités doivent être contrôlées.

### 12.2. Late fusion

Entraîner des modèles séparés puis combiner leurs probabilités :

- modèle ATD ;
- modèle basé sur Smart Tagging ;
- modèle visuel tactique.

Avantage : contribution de chaque source facilement mesurable.  
Limite : une partie des interactions entre modalités peut être perdue.

### 12.3. Mid-level fusion — option cible

- encodeur ATD : MLP/TCN ou agrégateur de joueurs permutation-invariant ;
- encodeur Smart Tagging : embeddings de catégories et d'équipe ;
- encodeur vidéo : CNN ou modèle vidéo accessible, exécuté localement ;
- variables de qualité : présence du ballon, nombre de tracks, missingness, reprojection error ;
- mécanisme de fusion : attention ou gated fusion ;
- décodeur temporel : HMM ou CRF.

Le mécanisme de fusion doit pouvoir diminuer le poids d'une modalité lorsqu'elle est indisponible ou peu fiable. Par exemple :

- ballon visible + tracking fiable : poids fort des features de proximité/trajectoire ;
- ballon absent : plus de poids aux features collectives et au contexte tactique ;
- homographie de qualité faible ou image ambiguë : limiter le poids des features visuelles projetées.

### 12.4. Plan d'ablation

| Expérience | Données | Question |
|---|---|---|
| E1 | ATD seul | Référence tabulaire |
| E2 | ATD + features collectives | Les dynamiques d'équipe aident-elles ? |
| E3 | ATD + Smart Tagging | Le contexte tactique apporte-t-il un gain ? |
| E4 | ATD + vidéo tactique | La vision ajoute-t-elle un signal de contrôle ? |
| E5 | ATD + tags + vidéo | La fusion complète améliore-t-elle le score ? |
| E6 | Fusion avec gating/attention | La pondération adaptative améliore-t-elle la robustesse ? |

Chaque ablation doit utiliser le même split par match, les mêmes métriques et le même protocole de réglage.

---

## 13. Étape 10 — Décodage temporel et reconstruction des possessions

Le modèle frame-level ne doit pas être considéré comme le résultat final. Il faut convertir ses probabilités en une séquence de labels dont les segments respectent les règles métier.

### 13.1. Décodage

Comparer :

- argmax frame par frame ;
- lissage majoritaire centré ;
- HMM/Viterbi ;
- CRF/Viterbi ;
- transitions contraintes ou pénalisées ;
- règles de durée minimales testées sur des validations externes.

Utiliser le contexte passé et futur puisque le match complet est disponible hors ligne. Les frontières des mi-temps doivent être traitées explicitement : ne pas propager un état d'une mi-temps à l'autre.

### 13.2. Gestion des transitions

Les transitions `H ↔ A` ne doivent pas être déclenchées par un simple contact ou une déviation. Chercher des indices de contrôle effectif : maintien du ballon, séquence de passes, comportement des joueurs, trajectoire et contexte.

Les périodes de ballon libre pendant le jeu ne doivent pas automatiquement devenir `D` : `D` signifie que le ballon n'est pas en jeu selon les règles du challenge.

### 13.3. Contrôle de fragmentation et fusion

Pour chaque version du modèle, analyser :

- nombre total de possessions prédites ;
- distribution des durées ;
- possessions très courtes ;
- changements H/A rapprochés ;
- périodes D artificiellement insérées dans une possession ;
- possessions réelles fusionnées en un seul segment ;
- délais de détection aux frontières.

Ne pas imposer une durée minimale arbitraire sans mesure : une durée de 3 secondes est un réglage testé par le collègue, pas une règle officielle de possession.

---

## 14. Étape 11 — Évaluation et sélection des modèles

### 14.1. Métriques obligatoires

- `spellF1` : métrique principale de sélection.
- `macroF1` : métrique secondaire et tie-break.
- F1 par classe `H`, `A`, `D` sur les jeux de validation externes.
- précision/rappel de `D`.
- taux de possession prédite par équipe.
- nombre de possessions et distribution des durées.
- délai et erreur de localisation des transitions.
- résultats séparés selon que le ballon est visible ou absent.

### 14.2. Analyse par sous-cas

Évaluer au minimum :

1. ballon visible ;
2. ballon absent moins d'une seconde ;
3. ballon absent entre 1 et 3 secondes ;
4. ballon absent plus de 3 secondes ;
5. présence de Smart Tagging ;
6. transitions offensives/défensives ;
7. coups de pied arrêtés ;
8. situations de tracking joueur incomplet ;
9. périodes de jeu et frontières de mi-temps.

Les intervalles exacts peuvent être ajustés selon les distributions observées sur les matchs de développement externes.

### 14.3. Principes de sélection

- Choisir les modèles sur des matchs de validation externes, pas sur les 5 minutes publiques de BAR-ATH.
- Rapporter les résultats par match, moyenne et dispersion.
- Utiliser bootstrap ou intervalles d'incertitude lorsque le nombre de matchs le permet.
- Comparer les modèles avec la même vérité terrain, le même périmètre et le même scorer.
- Ne pas considérer 12 copies d'un même match dégradé comme 12 matchs indépendants.

### 14.4. Baseline à dépasser

Le seuil de référence officiel sur le match complet est `spellF1 = 0,2433` pour `baseline_smooth50`. Les résultats privés ne sont pas disponibles pour l'équipe ; aucune conclusion de généralisation sur BAR-ATH ne doit être tirée d'une validation externe seule.

---

## 15. Étape 12 — Génération et validation de la soumission

Le fichier final doit respecter strictement le contrat officiel :

- nom de colonnes exact : `frame,label` ;
- exactement 199 319 lignes de données ;
- `frame` dans l'ordre `0` à `199318` ;
- labels parmi `H`, `A`, `D`, `X` ;
- aucune cellule vide ;
- UTF-8 ;
- aucun index pandas supplémentaire.

Les frames non évaluées peuvent être remplies avec `X`. Sur les frames évaluées, l'usage de `X` doit être explicitement justifié et testé selon le comportement du scorer.

Avant chaque soumission :

```bash
python evaluation/validate_submission.py path/to/submission.csv
```

Pour contrôler localement les résultats sur la vérité terrain publique :

```bash
python evaluation/run_evaluation.py
```

Le scorer public sert à vérifier l'exécution et la compréhension de la métrique. Il ne doit pas guider le réglage des hyperparamètres ou le choix final de l'architecture.

---

## 16. Organisation du code et des artefacts

Structure indicative du dépôt de travail (sans données du match) :

```text
project/
├── README.md
├── requirements.txt
├── configs/
│   ├── baseline.yaml
│   ├── hmm.yaml
│   ├── tcn.yaml
│   └── fusion.yaml
├── src/
│   ├── data/
│   │   ├── load_atd.py
│   │   ├── load_smart_tagging.py
│   │   ├── load_homography.py
│   │   ├── align_frames.py
│   │   └── quality_report.py
│   ├── features/
│   │   ├── ball_features.py
│   │   ├── player_features.py
│   │   ├── team_features.py
│   │   ├── tagging_features.py
│   │   ├── visual_features.py
│   │   └── feature_pipeline.py
│   ├── models/
│   │   ├── baseline.py
│   │   ├── hmm.py
│   │   ├── tcn.py
│   │   ├── crf_decoder.py
│   │   └── fusion.py
│   ├── postprocessing/
│   │   ├── sequence_decode.py
│   │   └── possession_segments.py
│   ├── evaluation/
│   │   ├── metrics.py
│   │   ├── diagnostics.py
│   │   └── validate_external.py
│   └── submission/
│       └── build_submission.py
├── notebooks/
│   ├── 01_data_audit.ipynb
│   ├── 02_tracking_visualisation.ipynb
│   ├── 03_feature_analysis.ipynb
│   ├── 04_baseline_reproduction.ipynb
│   └── 05_error_analysis.ipynb
├── tests/
│   ├── test_frame_alignment.py
│   ├── test_feature_shapes.py
│   ├── test_missing_data.py
│   └── test_submission_format.py
├── reports/
│   ├── data_quality.md
│   ├── experiments.md
│   └── model_card.md
└── data/                 # local uniquement, exclu de Git
```

Les fichiers intermédiaires contenant des données du match ou des features frame-level doivent rester locaux et ne pas être publiés s'ils permettent de reconstruire les données protégées.

---

## 17. Next steps — plan d'exécution priorisé

### Sprint 1 — Reprise et contrôle des données

- [ ] Cloner le dépôt officiel et lire `CHALLENGE.md`, `DATA.md`, `EVALUATION.md`, `SUBMISSION.md` et `RULES.md`.
- [ ] Charger ATD, Smart Tagging et homographie avec les utilitaires officiels.
- [ ] Vérifier les tailles, index et décalages temporels.
- [ ] Reproduire les statistiques de présence du ballon et les intervalles évalués.
- [ ] Générer les premières visualisations ATD/vidéo/homographie.
- [ ] Produire le rapport QA.

**Livrable :** pipeline de chargement et rapport QA reproductibles.

### Sprint 2 — Baselines et HMM

- [ ] Reproduire les quatre baselines officielles.
- [ ] Reproduire les métriques et vérifier le contrat de soumission.
- [ ] Récupérer et auditer le code du HMM du collègue.
- [ ] Reproduire le résultat HMM sur les jeux externes dégradés.
- [ ] Documenter les états, émissions, transitions, alpha, EM et Viterbi.
- [ ] Mesurer la sensibilité aux graines et aux paramètres.

**Livrable :** baseline officielle + HMM reproductible.

### Sprint 3 — Features robustes et Smart Tagging

- [ ] Construire les features de ballon et indicateurs de missingness.
- [ ] Construire les features collectives par équipe.
- [ ] Ajouter les features cinématiques robustes.
- [ ] Transformer les intervalles de Smart Tagging en variables temporelles.
- [ ] Tester les ablations ATD seul / ATD + tags / prior d'équipe.
- [ ] Analyser spécifiquement les périodes sans ballon.
- [ ] Diagnostiquer la sur-prédiction de `D`.

**Livrable :** table de features documentée et HMM enrichi.

### Sprint 4 — Validation externe solide

- [ ] Identifier des matchs externes avec annotations de possession compatibles.
- [ ] Vérifier licences, disponibilité, sémantique et reproductibilité des labels.
- [ ] Construire un split par match.
- [ ] Développer un générateur de dégradation réaliste.
- [ ] Évaluer les modèles sur plusieurs matchs réellement indépendants.
- [ ] Calculer spellF1, macroF1 et les diagnostics par sous-cas.

**Livrable :** protocole de validation défendable et rapport d'incertitude.

### Sprint 5 — Modèles séquentiels et fusion

- [ ] Entraîner un TCN sur des matchs externes annotés.
- [ ] Comparer TCN direct, HMM enrichi et TCN + HMM/CRF.
- [ ] Tester une décomposition `D` vs `H/A`.
- [ ] Ajouter un module de fusion adaptative avec Smart Tagging.
- [ ] Réaliser une ablation vidéo tactique si le coût et la qualité des données le justifient.
- [ ] Conserver la meilleure architecture selon le spellF1 externe.

**Livrable :** modèle candidat et rapport d'ablation complet.

### Sprint 6 — Finalisation

- [ ] Choisir les hyperparamètres sur validation externe uniquement.
- [ ] Générer les 199 319 labels.
- [ ] Vérifier les frontières de mi-temps et les zones non évaluées.
- [ ] Exécuter le validateur officiel.
- [ ] Vérifier que le dépôt reproduit exactement le fichier généré.
- [ ] Rédiger le document technique : données utilisées, méthode, entraînement externe, règles de possession, limites et reproduction.
- [ ] Vérifier qu'aucune donnée protégée ni feature dérivée n'est publiée.

**Livrable :** soumission reproductible, dépôt public propre et documentation technique.

---

## 18. Décisions immédiates recommandées

Pour éviter de disperser le travail, les priorités immédiates sont :

1. **Auditer le HMM existant avant de développer une nouvelle architecture.** Il s'agit du meilleur point de départ expérimental de l'équipe.
2. **Construire les features collectives.** Elles répondent directement aux limites du tracking et à l'absence du ballon.
3. **Exploiter le Smart Tagging comme un prior incertain.** Il apporte un signal complémentaire, mais ne doit jamais être traité comme la vérité terrain frame-level.
4. **Résoudre le problème de validation externe.** Sans plusieurs matchs annotés, nous ne pouvons pas mesurer correctement la généralisation.
5. **N'ajouter la vidéo tactique qu'après avoir établi une référence tabulaire solide.** Cela permet de mesurer réellement sa valeur ajoutée.
6. **Optimiser le décodage pour les possessions complètes.** La segmentation et les transitions sont aussi importantes que les probabilités frame-level.

---

## 19. Références et documentation

### Documentation officielle du challenge

- [Repository — More Than a Hack 2027](https://github.com/metrica-sports/more-than-a-hack-2027)
- [CHALLENGE.md — tâche, labels et règles de possession](https://github.com/metrica-sports/more-than-a-hack-2027/blob/main/CHALLENGE.md)
- [DATA.md — inventaire, formats et alignement des données](https://github.com/metrica-sports/more-than-a-hack-2027/blob/main/DATA.md)
- [EVALUATION.md — définition des métriques et résultats des baselines](https://github.com/metrica-sports/more-than-a-hack-2027/blob/main/EVALUATION.md)
- [RULES.md — règles d'utilisation des données et de validation](https://github.com/metrica-sports/more-than-a-hack-2027/blob/main/RULES.md)
- [SUBMISSION.md — format et livrables](https://github.com/metrica-sports/more-than-a-hack-2027/blob/main/SUBMISSION.md)

### Travaux et expérimentations déjà partagés dans l'équipe

- `xp.md` — compte rendu des expériences baseline/HMM/Smart Tagging et résultats rapportés par le collègue.
- `lit-review-possession.md` — revue de littérature sur l'estimation de possession depuis les trajectoires, les TCN, HMM/CRF, le Pitch Control et les approches sans ballon.

### Références de recherche à approfondir

- Borghesi et al. (2023), *Using Temporal Convolutional Networks to estimate ball possession in soccer games*, Expert Systems with Applications. [DOI](https://doi.org/10.1016/j.eswa.2023.119780)
- Lang et al. (2022), *Predicting the in-game status in soccer with machine learning using spatiotemporal player tracking data*, Scientific Reports. [DOI](https://doi.org/10.1038/s41598-022-19948-1)
- Kim et al. (2026), *PathCRF: Ball-Free Soccer Event Detection via Possession Path Inference from Player Trajectories*. [DOI](https://doi.org/10.1145/3770855.3818152) · [arXiv](https://arxiv.org/abs/2602.12080)
- Spearman (2018), *Beyond Expected Goals*, MIT Sloan Sports Analytics Conference.
- Omidshafiei et al. (2022), *Multiagent off-screen behavior prediction in football*, Scientific Reports. [DOI](https://doi.org/10.1038/s41598-022-12547-0)

---

## Conclusion

Le plan proposé part des ressources les plus fiables et les plus directement alignées sur la tâche : ATD tactique, Smart Tagging et homographie. Il commence par reproduire les baselines, puis consolide le HMM déjà expérimenté, enrichit les features collectives et construit une validation externe rigoureuse. Les modèles TCN/CRF, la fusion multimodale et l'exploitation visuelle sont ensuite introduits progressivement, avec des ablations permettant de mesurer leur apport réel.

La réussite du projet dépendra moins de l'accumulation de modalités ou de la complexité du réseau que de trois capacités : **gérer les observations manquantes, estimer le contrôle réel plutôt que la simple proximité, et reconstruire des possessions temporellement cohérentes**.
