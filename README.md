# DataScientest_Project
Prédiction du Succès d'une Campagne Marketing

# Bank term deposit

## 📄 Description du Projet

Ce projet vise à prédire le succès d'une campagne marketing à l'aide de l'apprentissage supervisé. En analysant un ensemble de données comprenant des informations sur les clients et les campagnes précédentes, nous avons développé un modèle capable d'anticiper si un client répondra positivement à une campagne future.

##Dataset :
[Jeu de données disponible sur Kaggle](https://www.kaggle.com/datasets/janiobachmann/bank-marketing-dataset)

## 🔧 Technologies Utilisées

- **Python**
- **Scikit-Learn** : pour le prétraitement des données et l'entraînement du modèle
- **Matplotlib & Seaborn** : pour la visualisation des données
- **Plotly** : pour des visualisations interactives

## 🔍 Exploration des Données

L'analyse exploratoire a révélé plusieurs facteurs influençant le succès des campagnes, tels que :

- L'historique des interactions des clients
- Le canal de communication utilisé (email, appel, etc.)
- La répartition démographique des clients

## ⚖️ Modélisation

1. **Prétraitement des données** : gestion des valeurs manquantes, encodage des variables catégoriques, normalisation.
2. **Modèles testés** : Random Forest, Logistic Regression, Gradient Boosting.
3. **Meilleur modèle** : Random Forest avec un rappel de **84 %**, équilibrant les faux positifs et faux négatifs.

## 📊 Résultats

- **Métrique principale** : F1 score = 84 %
- **Courbe ROC/AUC** : Visualisation des performances du modèle
- **Feature Importance** : Analyse des variables les plus influentes

## 🎨 Visualisations!
