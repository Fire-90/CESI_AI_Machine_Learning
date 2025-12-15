import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import warnings

# Imports Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, confusion_matrix

# On ignore les warnings inutiles
warnings.filterwarnings('ignore')

def charger_donnees(chemin):
    """Charge le fichier CSV."""
    print(f"📂 Chargement du fichier : {chemin}...")
    try:
        df = pd.read_csv(chemin)
        print(f" Données chargées : {df.shape[0]} lignes, {df.shape[1]} colonnes.")
        return df
    except FileNotFoundError:
        print(f" Erreur : Le fichier '{chemin}' est introuvable.")
        return None

def analyser_facteurs_influents(df):
    """
    Affiche les corrélations : 
    - Positives (Rouge) = Causes de départ
    - Négatives (Vert) = Raisons de rester
    """
    print("\n Analyse des facteurs d'influence (Corrélation)...")
    
    # Calcul des corrélations avec 'Attrition'
    # numeric_only=True évite les erreurs si des colonnes texte traînent
    corr = df.corr(numeric_only=True)['Attrition'].sort_values(ascending=False)
    
    # On retire la cible elle-même (qui vaut 1)
    corr = corr.drop('Attrition', errors='ignore')
    
    # On prend le Top 10 positif (Partent) et Top 10 négatif (Restent)
    top_positive = corr.head(10)
    top_negative = corr.tail(10)
    
    # On combine les deux pour le graphique
    top_corr = pd.concat([top_positive, top_negative])
    
    # Graphique
    plt.figure(figsize=(12, 8))
    # Couleur : Rouge si > 0 (Départ), Vert si < 0 (Reste)
    colors = ['red' if x > 0 else 'green' for x in top_corr.values]
    sns.barplot(x=top_corr.values, y=top_corr.index, palette=colors)
    
    plt.title("Facteurs d'influence : Rouge = Fait partir | Vert = Fait rester")
    plt.xlabel("Corrélation")
    plt.axvline(x=0, color='black', linestyle='--')
    plt.show()

def preparation_donnees(df):
    """Prépare les données pour l'IA (Split 70/30)."""
    print(" Préparation des données (Train/Test Split)...")
    
    y = df['Attrition']
    X = df.drop('Attrition', axis=1)
    
    # stratify=y est important pour garder la même proportion de départs
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    return X_train, X_test, y_train, y_test, X.columns

def entrainer_modeles(X_train, X_test, y_train, y_test):
    """Entraîne une liste de modèles et compare les résultats."""
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
        "Perceptron": Perceptron(random_state=42),
        "SVM": SVC(probability=True, random_state=42),
        "KNN": KNeighborsClassifier(),
        "NaiveBayes": GaussianNB(),
        "DecisionTree": DecisionTreeClassifier(random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
        "ReseauNeuronal": MLPClassifier(max_iter=500, random_state=42)
    }
    
    results = []
    trained_models = {}

    print("\n Début de l'entraînement des modèles...")
    print("-" * 60)

    for name, model in models.items():
        start_time = time.time()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        duration = time.time() - start_time
        
        # Calcul AUC si possible
        try:
            y_proba = model.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, y_proba)
        except:
            auc = 0

        results.append({
            'Modèle': name,
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred),
            'Recall': recall_score(y_test, y_pred),
            'F1-Score': f1_score(y_test, y_pred),
            'AUC': auc,
            'Temps (s)': duration
        })
        
        trained_models[name] = model
        print(f"   🔹 {name:<20} | F1-Score: {f1_score(y_test, y_pred):.4f} | Temps: {duration:.3f}s")

    return pd.DataFrame(results), trained_models

def afficher_matrice_confusion(y_test, trained_models, X_test):
    """Affiche les matrices de confusion."""
    print("\n📊 Génération des matrices de confusion...")
    plt.figure(figsize=(15, 10))
    
    for i, (name, model) in enumerate(trained_models.items()):
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        
        plt.subplot(3, 3, i+1)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title(name)
        plt.ylabel('Réel')
        plt.xlabel('Prédit')
    
    plt.tight_layout()
    plt.show()

def afficher_importance_variables(trained_models, feature_names):
    """
    Affiche l'importance des variables pour le meilleur modèle 'Arbre' disponible.
    Ne privilégie pas XGBoost dans le nom, mais prend le plus performant.
    """
    # On cherche un modèle capable de donner l'importance (RandomForest ou XGBoost)
    model_choisi = None
    nom_modele = ""

    # On vérifie si XGBoost est là, sinon Random Forest
    if "XGBoost" in trained_models:
        model_choisi = trained_models["XGBoost"]
        nom_modele = "XGBoost"
    elif "RandomForest" in trained_models:
        model_choisi = trained_models["RandomForest"]
        nom_modele = "Random Forest"
    
    if model_choisi:
        print(f"\n Analyse des causes réelles du départ (Basé sur le modèle : {nom_modele})...")
        
        importances = model_choisi.feature_importances_
        feature_imp_df = pd.DataFrame({
            'Variable': feature_names,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False).head(15)
        
        plt.figure(figsize=(10, 8))
        sns.barplot(x='Importance', y='Variable', data=feature_imp_df, palette='magma')
        plt.title(f"TOP 15 des variables les plus importantes ({nom_modele})")
        plt.xlabel("Poids dans la décision")
        plt.show()
    else:
        print("Aucun modèle de type 'Arbre' (Tree) n'a été entraîné pour l'analyse d'importance.")

# --- MAIN ---
if __name__ == "__main__":
    # 1. Chemin du fichier (Vérifie bien que c'est le bon !)
    fichier_csv = 'data/processed_hr_data_encoded_normalized.csv'
    
    # 2. Chargement
    df = charger_donnees(fichier_csv)
    
    if df is not None:
        # 3. Analyse Corrélation (Rouge vs Vert) - AVANT le split
        analyser_facteurs_influents(df)

        # 4. Préparation
        X_train, X_test, y_train, y_test, feature_names = preparation_donnees(df)
        
        # 5. Entraînement
        resultats_df, modeles_entraines = entrainer_modeles(X_train, X_test, y_train, y_test)
        
        # 6. Résultats
        print("\n CLASSEMENT FINAL (Trié par Recall & F1-Score) :")
        print(resultats_df.sort_values(by=['Recall', 'F1-Score'], ascending=False).to_string(index=False))
        
        # 7. Visualisations
        afficher_matrice_confusion(y_test, modeles_entraines, X_test)
        afficher_importance_variables(modeles_entraines, feature_names)