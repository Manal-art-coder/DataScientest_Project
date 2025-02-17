import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
url = "https://raw.githubusercontent.com/Manal-art-coder/DataScientest_Project/main/bank.csv"
df=pd.read_csv(url)
df.head()
st.title("Prédiction du succès d'une campagne Marketing")
st.sidebar.title("Sommaire")
pages=["Contexte et objectif", "Exploration du jeu de données", "Datavisulaization", "Préparation des données", "Modélisation", "Résultat et analyse"]
page=st.sidebar.radio("Aller vers", pages)
if page == pages[0] : 
  st.write("### Contexte et objectif")
  st.write("""
Ce projet a été réalisé dans le cadre de notre formation de **Data Analyst** avec l’organisme Datascientest.  

L’objectif est de prédire le succès d’une campagne marketing en analysant les facteurs influençant la souscription des clients à une offre spécifique.  
La variable cible étant la souscription ou non à l’offre, nous utilisons l’apprentissage supervisé pour résoudre ce problème de classification binaire.  

À partir de données labellisées, nous cherchons à identifier les profils les plus susceptibles de souscrire, afin d’optimiser les actions marketing et réduire les coûts de la campagne.  

**Le jeu de données** utilisé provient de la plateforme Kaggle et est accessible à l’adresse suivante : [Lien vers le dataset](#).  

Ce Streamlit retrace notre démarche, depuis l’exploration et le prétraitement des données jusqu’à la modélisation finale.  
Il permet de visualiser les différentes étapes du projet, d’analyser les variables explicatives sélectionnées et de tester plusieurs algorithmes de Machine Learning afin d’identifier le modèle le plus performant.
""")
if page == pages[1] : 
   st.write("### Exploration du jeu de données")
   st.dataframe(df.head(10))
   st.write("### Taille du Dataset")
   st.write(df.shape)
   st.write("### Statistiques descriptives")
   st.dataframe(df.describe())
   if st.checkbox("Afficher les NA") :
     st.dataframe(df.isna().sum())
if page == pages[2] : 
   st.write("### DataVisualization")
   st.title("Exploration des distributions des variables")
   selected_variable = st.selectbox("Sélectionnez une variable :", df.columns)

   if df[selected_variable].dtype in ["int64", "float64"]:  
      st.write(f"### Distribution de {selected_variable} (Numérique)")
      fig, ax = plt.subplots(figsize=(8, 5))
      df[selected_variable].hist(bins=20, ax=ax, color="royalblue", edgecolor="black")
      ax.set_xlabel(selected_variable)
      ax.set_ylabel("Fréquence")
      ax.set_title(f"Histogramme de {selected_variable}")
      st.pyplot(fig)
   else:  # Variable catégorielle
      st.write(f"### Distribution de {selected_variable} (Catégorielle)")
      fig, ax = plt.subplots(figsize=(8, 5))
      df[selected_variable].value_counts().plot(kind="bar", ax=ax, color="royalblue", edgecolor="black")
      ax.set_xlabel(selected_variable)
      ax.set_ylabel("Nombre d'occurrences")
      ax.set_title(f"Répartition des catégories de {selected_variable}")
      st.pyplot(fig)
