# 🤖 CESI AI Machine Learning - Prédiction de l'Attrition des Employés

Projet d'analyse prédictive visant à identifier les facteurs de départ des employés et à comparer différents algorithmes de Machine Learning pour prédire l'attrition.

---

## 📋 Table des Matières

- [Objectifs du Projet](#-objectifs-du-projet)
- [Structure du Projet](#-structure-du-projet)
- [Pipeline de Traitement](#-pipeline-de-traitement)
- [Analyse Exploratoire](#-analyse-exploratoire)
- [Modèles Implémentés](#-modèles-implémentés)
- [Métriques d'Évaluation](#-métriques-dévaluation)

---

## 🎯 Objectifs du Projet

1. **Prédire l'attrition** : Développer des modèles capables de prédire si un employé risque de quitter l'entreprise
2. **Identifier les facteurs clés** : Déterminer les variables les plus influentes dans la décision de départ via Feature Importance
3. **Comparer les algorithmes** : Évaluer et comparer 9 modèles de Machine Learning différents
4. **Fournir des insights actionnables** : Aider les RH à prendre des décisions éclairées pour améliorer la rétention
5. **Analyser visuellement** : Générer des graphiques pour diagnostiquer les causes de départ

---

## 📁 Structure du Projet

```
CESI_AI_Machine_Learning/
│
├── data/                                    # Données brutes et traitées
│   ├── employee_survey_data.csv            # Enquête satisfaction employés
│   ├── manager_survey_data.csv             # Évaluation des managers
│   ├── general_data.csv                    # Données démographiques et contractuelles
│   ├── in_time.csv                         # Horaires d'arrivée (badgeuse)
│   ├── out_time.csv                        # Horaires de départ (badgeuse)
│   ├── processed_hr_data.csv               # Données consolidées et nettoyées
│   ├── processed_hr_data_encoded_raw.csv   # Données encodées (non normalisées)
│   └── processed_hr_data_encoded_normalized.csv  # Données encodées et normalisées
│
├── picture/                                # Images pour la documentation
│   ├── SVM.webp                           # Diagramme SVM
│   ├── KNN.png                            # Diagramme KNN
│   ├── NAIVEBAYES.webp                    # Diagramme Naive Bayes
│   ├── DESICIONTREE.png                   # Diagramme Decision Tree
│   ├── RANDOMFOREST.jpg                   # Diagramme Random Forest
│   ├── XGBOOST.png                        # Diagramme XGBoost
│   └── RESEAUNEURONES.png                 # Diagramme Réseau de Neurones
│
├── Traitement.ipynb                        # Pipeline complet de traitement des données
├── Modele.ipynb                            # Implémentation et comparaison des 9 modèles
├── Plan_action.txt                         # Plan détaillé du projet
└── README.md                               # Documentation (ce fichier)
```

---

## 🔄 Pipeline de Traitement

### Étape 1 : Consolidation des Données (`process_hr_data()`)

**Objectif** : Centraliser les données dispersées dans 5 fichiers CSV et créer de nouvelles variables.

#### Sources de Données

1. **general_data.csv** : Informations démographiques, salaire, poste, ancienneté
2. **manager_survey_data.csv** : Évaluations de performance, implication
3. **employee_survey_data.csv** : Satisfaction environnement, équilibre vie pro/perso
4. **in_time.csv** : Horaires d'arrivée (badgeage entrant) - 365 jours
5. **out_time.csv** : Horaires de départ (badgeage sortant) - 365 jours

#### Fusion et Feature Engineering

- **Fusion** : Utilisation de `EmployeeID` comme clé de jointure (Left Merge)
- **Calcul de métriques temporelles** :
  - `AverageWorkingHours` : Moyenne des heures travaillées par jour (excluant les absences)
  - `TotalWorkingDays` : Nombre total de jours badgés dans l'année
- **Nettoyage** : Suppression des colonnes à valeur unique (ex: `Over18='Y'`, `StandardHours=8`)

**Sortie** : `processed_hr_data.csv` (données consolidées)

---

### Étape 2 : Encodage et Normalisation

**Objectif** : Transformer les données textuelles en numériques et générer deux versions du dataset.

#### Traitement de la Cible

- `Attrition` : Conversion binaire (Yes → 1 / No → 0)

#### Encodage des Variables Catégorielles

**Variables Ordinales** (ordre important) :
- `BusinessTravel` : Non-Travel (0) < Travel_Rarely (1) < Travel_Frequently (2)

**Variables Nominales** (pas d'ordre - One-Hot Encoding) :
- `Department`, `EducationField`, `Gender`, `JobRole`, `MaritalStatus`

#### Gestion des Valeurs Manquantes

- Imputation par la moyenne pour les colonnes numériques

#### Double Stratégie de Sortie

1. **`processed_hr_data_encoded_raw.csv`** (Non normalisé)
   - Pour : Random Forest, XGBoost, interprétation métier
   - Les valeurs restent réelles (salaire = 50000, âge = 30)

2. **`processed_hr_data_encoded_normalized.csv`** (Normalisé 0-1)
   - Pour : Réseaux de Neurones, KNN, SVM, Régression Logistique
   - Toutes les valeurs entre 0 et 1 (MinMaxScaler)

---

## 📊 Analyse Exploratoire

Le notebook `Traitement.ipynb` génère 5 graphiques clés pour diagnostiquer les causes de départ :

### 1. Tableau Statistique (Heatmap)
- Affiche les statistiques descriptives (moyenne, médiane, min, max, écart-type)
- Exclut les variables binaires (0/1) pour se concentrer sur les numériques

### 2. Répartition Globale (Countplot)
- Vérifie le déséquilibre des classes
- Affiche le pourcentage de départs vs restants

### 3. Taux de Départ par Métier (Barplot)
- Identifie les métiers les plus à risque
- Calcul du taux : (Départs / Total) × 100
- Tri décroissant pour mettre en avant les zones critiques

### 4. Heures de Travail (Boxplot)
- Compare la distribution des heures moyennes de travail
- Corrélation avec le burnout potentiel

### 5. Ancienneté (KDE Plot)
- Visualise à quel moment de la carrière les employés partent
- Superposition des courbes (Rouge = Départ, Bleu = Reste)

### 6. Matrice de Corrélation (Heatmap)
- Identifie les liens linéaires forts avec l'attrition
- Focus sur les 10 premières variables numériques
- **Affichage optimisé** : Étiquettes des colonnes en haut, rotées à 90°

---

## 🤖 Modèles Implémentés

Le notebook `Modele.ipynb` compare **9 modèles** de Machine Learning avec documentation complète pour chacun :

### 1. Régression Logistique
- **Formule** : P(y=1|x) = 1 / (1 + e^(-(w·x + b)))
- **Usage** : Classification binaire via fonction sigmoïde

### 2. Perceptron
- **Fonction** : f(x) = 1 si w·x + b > 0, sinon 0
- **Usage** : Neurone artificiel simple

### 3. Support Vector Machine (SVM)
- **Principe** : Trace un hyperplan avec marge maximale
- **Usage** : Séparation optimale des classes

### 4. K-Nearest Neighbors (KNN)
- **Principe** : Classification basée sur les K voisins les plus proches
- **Usage** : Prédiction locale sans règle globale

### 5. Naive Bayes
- **Formule** : P(A|B) = P(B|A)·P(A) / P(B)
- **Usage** : Probabilités bayésiennes avec hypothèse d'indépendance

### 6. Decision Tree
- **Principe** : Arbre de décisions binaires successives
- **Usage** : Règles de décision interprétables

### 7. Random Forest
- **Principe** : Ensemble de centaines d'arbres votant collectivement
- **Usage** : Robustesse par agrégation (Bagging)

### 8. XGBoost
- **Principe** : Arbres successifs corrigeant les erreurs précédentes
- **Usage** : Boosting pour performances maximales sur données tabulaires

### 9. Réseau de Neurones (MLP)
- **Principe** : Couches de neurones interconnectés
- **Usage** : Modélisation de relations complexes non linéaires

---

## 📈 Métriques d'Évaluation

### Métriques Principales

- **Accuracy** : Taux de prédictions correctes global
- **Precision** : Proportion de vrais positifs parmi les prédits positifs
- **Recall** : Proportion de vrais positifs détectés (critique pour l'attrition)
- **F1-Score** : Moyenne harmonique de Precision et Recall
- **AUC-ROC** : Qualité globale du modèle (aire sous la courbe ROC)

### Validation Croisée (K-Fold)

- **CV Recall Moyen** : Moyenne des scores sur 5 splits différents
- **CV Stabilité** : Écart-type pour évaluer la robustesse

### Prévention du Surapprentissage

- ✅ Validation croisée (5-Fold Cross-Validation)
- ✅ Séparation Train/Test (70/30 avec stratification)
- ✅ Techniques d'ensemble (Random Forest, XGBoost)
- ✅ `random_state=42` pour la reproductibilité

---

## 📊 Visualisations

### 1. Analyse des Facteurs d'Influence
- **Type** : Barplot de corrélations
- **Rouge** : Facteurs augmentant le départ (corrélation positive)
- **Vert** : Facteurs favorisant la rétention (corrélation négative)

### 2. Matrices de Confusion (3×3)
- Une matrice par modèle pour comparer les erreurs
- **Diagonale** : Prédictions correctes
- **Hors diagonale** : Faux positifs et faux négatifs

### 3. Feature Importance
- **Source** : XGBoost ou Random Forest
- **Affichage** : Top 15 des variables les plus influentes
- **Utilité** : Identifier pourquoi les employés partent

---