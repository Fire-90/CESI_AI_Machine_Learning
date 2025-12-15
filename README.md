# 🤖 CESI AI Machine Learning - Prédiction de l'Attrition des Employés

Projet d'analyse prédictive visant à identifier les facteurs de départ des employés et à comparer différents algorithmes de Machine Learning pour prédire l'attrition.

---

## 📋 Table des Matières

- [Objectifs du Projet](#-objectifs-du-projet)
- [Structure du Projet](#-structure-du-projet)
- [Données](#-données)
- [Méthodologie](#-méthodologie)
- [Modèles Implémentés](#-modèles-implémentés)
- [Installation](#-installation)
- [Utilisation](#-utilisation)
- [Éthique et Conformité](#-éthique-et-conformité)
- [Résultats Attendus](#-résultats-attendus)

---

## 🎯 Objectifs du Projet

1. **Prédire l'attrition** : Développer des modèles capables de prédire si un employé risque de quitter l'entreprise
2. **Identifier les facteurs clés** : Déterminer les variables les plus influentes dans la décision de départ
3. **Comparer les algorithmes** : Évaluer et comparer au minimum 8 modèles de Machine Learning différents
4. **Fournir des insights actionnables** : Aider les RH à prendre des décisions éclairées pour améliorer la rétention

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
│   ├── processed_hr_data.csv               # Données consolidées
│   ├── processed_hr_data_encoded_raw.csv   # Données encodées (non normalisées)
│   └── processed_hr_data_encoded_normalized.csv  # Données encodées et normalisées
│
├── Traitement.ipynb                        # Pipeline de traitement des données
├── Modele.ipynb                            # Implémentation et comparaison des modèles
├── Plan_action.txt                         # Plan détaillé du projet
└── README.md                               # Documentation (ce fichier)
```

---

## 📊 Données

### Sources de Données

Le projet utilise 5 fichiers sources distincts :

1. **general_data.csv** : Informations démographiques, salaire, poste, ancienneté
2. **manager_survey_data.csv** : Évaluations de performance, implication
3. **employee_survey_data.csv** : Satisfaction environnement, équilibre vie pro/perso
4. **in_time.csv / out_time.csv** : Données de badgeuse (année complète)

### Variables Créées

- **AverageWorkingHours** : Moyenne des heures travaillées par jour
- **TotalWorkingDays** : Nombre total de jours travaillés dans l'année

### Traitement Appliqué

- ✅ Fusion des 5 sources de données via `EmployeeID`
- ✅ Suppression des colonnes à valeur unique (sans variance)
- ✅ Gestion des valeurs manquantes (imputation par la moyenne)
- ✅ Encodage des variables catégorielles (Ordinal + One-Hot)
- ✅ Normalisation Min-Max (0-1) pour certains modèles

---

## ⚙️ Méthodologie

### 1. Traitement des Données
- **Normalisation** : MinMaxScaler pour mettre toutes les variables entre 0 et 1
- **Encodage** :
  - Ordinal pour `BusinessTravel` (Non < Rarely < Frequently)
  - One-Hot pour `Department`, `EducationField`, `Gender`, `JobRole`, `MaritalStatus`
- **Nettoyage** : Suppression des variables sans variance (ex: `Over18`, `StandardHours`)

### 2. Choix des Modèles (minimum 8)
- XGBoost
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)
- Perceptron
- Random Forest
- Régression Logistique
- Réseaux de Neurones
- *(autres à ajouter selon les besoins)*

### 3. Évaluation des Modèles
- **Métriques** : Accuracy, Precision, Recall, F1-Score
- **Visualisations** :
  - Matrice de confusion
  - Courbes ROC et AUC
  - Diagrammes de barres (comparaison des modèles)
  - Heatmap (corrélations)
  - Feature Importance (variables les plus influentes)

### 4. Prévention du Sur/Sous-Apprentissage
- Validation croisée (K-Fold Cross-Validation)
- Séparation Train/Test (80/20 ou 70/30)
- Techniques d'ensemble (Bagging, Boosting)
- Régularisation (L1, L2)

---

## 🤖 Modèles Implémentés

### Fichiers Normalisés vs Non-Normalisés

| Modèle | Fichier Recommandé | Raison |
|--------|-------------------|---------|
| XGBoost | `encoded_raw.csv` | Basé sur des arbres, insensible à l'échelle |
| Random Forest | `encoded_raw.csv` | Basé sur des arbres, insensible à l'échelle |
| KNN | `encoded_normalized.csv` | Sensible aux distances euclidiennes |
| SVM | `encoded_normalized.csv` | Nécessite des données normalisées |
| Régression Logistique | `encoded_normalized.csv` | Performance améliorée avec normalisation |
| Réseaux de Neurones | `encoded_normalized.csv` | Convergence plus rapide avec normalisation |
| Perceptron | `encoded_normalized.csv` | Nécessite des données normalisées |

---