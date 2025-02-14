# DataScientest_Project

# Prédiction du Succès d'une Campagne Marketing

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

L'analyse exploratoire a révélé que les variables suivantes ont une influence significative sur le succès des campagnes :

Duration : durée du dernier contact avec le client
Housing : statut de possession d'un logement
Age : âge du client
Poutcome : résultat des campagnes marketing précédentes
Day : jour du mois où le contact a eu lieu

## ⚖️ Modélisation

1. **Prétraitement des données** : gestion des valeurs manquantes, encodage des variables catégoriques, normalisation.
2. **Modèles testés** : Régression Logistique, Arbre de Décision, Forêt Aléatoire, Bagging Classifier, AdaBoost Classifier, Gradient Boosting Classifier
3. **Meilleur modèle** : Random Forest avec un F1 de **85 %**, équilibrant les faux positifs et faux négatifs.

## 📊 Résultats

- **Métrique principale** : F1 score = 85 %
- **Courbe ROC/AUC** : Visualisation des performances du modèle
- **Feature Importance** : Analyse des variables les plus influentes

## 👥 Équipe du Projet
Ce projet a été réalisé par :

-Audrey Amiel
-Elyse Demeulemeester
-Manal Jewa
-David Legrand
-Manon Selle
