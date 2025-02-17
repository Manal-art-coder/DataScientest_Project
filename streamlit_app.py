import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df=pd.read_csv(r"C:\Users\manal\Downloads\bank.csv")
df.head()
st.title("Prédiction du succès d'une campagne Marketing")
st.sidebar.title("Sommaire")
pages=["Contexte et objectif", "Exploration du jeu de données", "Datavisulaization", "Préparation des données", "Modélisation", "Résultat et analyse"]
page=st.sidebar.radio("Aller vers", pages)
if page == pages[0] : 
  st.write("### Contexte et objectif")
  st.write("Ce projet a été réalisé dans le cadre de notre formation en Data Analyst avec l’organisme Datascientest. L’objectif est de prédire le succès d’une campagne marketing en analysant les facteurs influençant la souscription des clients à une offre spécifique. La variable cible étant la souscription ou non à l’offre, nous utilisons l’apprentissage supervisé pour résoudre ce problème de classification binaire. À partir de données labellisées, nous cherchons à identifier les profils les plus susceptibles de souscrire, afin d’optimiser les actions marketing et réduire les coûts de la campagne. Le jeu de données utilisé provient de la plateforme Kaggle et est accessible à l’adresse suivante : Lien vers le dataset. Ce Streamlit retrace notre démarche, depuis l’exploration et le prétraitement des données jusqu’à la modélisation finale. Il permet de visualiser les différentes étapes du projet, d’analyser les variables explicatives sélectionnées et de tester plusieurs algorithmes de Machine Learning afin d’identifier le modèle le plus performant.")
if page == pages[1] : 
  st.write("### Exploration du jeu de données")
  st.dataframe(df.head(10))
  st.write(df.shape)
  st.dataframe(df.describe())
if st.checkbox("Afficher les NA") :
  st.dataframe(df.isna().sum())
if page == pages[2] : 
  st.write("### DataVisualization")
  fig = plt.figure()
  sns.countplot(x = 'deposit', data = df)
  st.pyplot(fig)
  