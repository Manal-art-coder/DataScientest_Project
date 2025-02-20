import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import chi2_contingency
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import recall_score, precision_score, accuracy_score, f1_score, classification_report, precision_recall_curve, PrecisionRecallDisplay,  confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
import joblib
from sklearn.utils.validation import check_is_fitted
import os


df=pd.read_csv(r"C:\Users\manal\Desktop\Projets\Streamlit\bank.csv")
df.head()
st.title("Prédiction du succès d'une campagne Marketing")
st.sidebar.title("Sommaire")
pages=["Contexte & enjeux", "Présentation des données", "Visualisation des données", "Préprocessing", "Modèles ML", "Meilleur modèle", "Faites votre propre prédiction !","Conclusion & perspectives"]
page=st.sidebar.radio("Aller vers", pages)
if page == pages[0] : 
  st.write("### Contexte & enjeux")
  st.image(r"C:\Users\manal\Desktop\Projets\Streamlit\Screenshot 2025-02-19 111852.jpg", use_column_width=True)
  st.write("""
Ce projet a été réalisé dans le cadre de notre formation de **Data Analyst** avec l’organisme Datascientest.  

L’objectif est de prédire le succès d’une campagne marketing en analysant les facteurs influençant la souscription des clients à une offre spécifique.  
La variable cible étant la souscription ou non à l’offre, nous utilisons l’apprentissage supervisé pour résoudre ce problème de classification binaire.  

À partir de données labellisées, nous cherchons à identifier les profils les plus susceptibles de souscrire, afin d’optimiser les actions marketing et réduire les coûts de la campagne.  

**Le jeu de données** utilisé provient de la plateforme Kaggle et est accessible à l’adresse suivante : [Lien vers le dataset](#https://www.kaggle.com/datasets/janiobachmann/bank-marketing-dataset).  

Ce Streamlit retrace notre démarche, depuis l’exploration et le prétraitement des données jusqu’à la modélisation finale.  
Il permet de visualiser les différentes étapes du projet, d’analyser les variables explicatives sélectionnées et de tester plusieurs algorithmes de Machine Learning afin d’identifier le modèle le plus performant.
""")


if page == pages[1]:
    # Titre principal
    st.title("Présentation des données 📊")

    # Aperçu du dataset
    st.subheader("Aperçu des premières lignes")
    st.write("Voici un aperçu des 10 premières lignes du jeu de données :")
    st.dataframe(df.head(10))

    # Taille du dataset
    st.subheader("Taille du Dataset")
    st.write(f"**Nombre de lignes** : {df.shape[0]}")
    st.write(f"**Nombre de colonnes** : {df.shape[1]}")

    # Statistiques descriptives
    st.subheader("Statistiques Descriptives")
    st.write("Résumé statistique des variables numériques du dataset :")
    st.dataframe(df.describe())

    # Vérification des valeurs manquantes
    st.subheader("Valeurs Manquantes")
    if st.checkbox("Afficher les valeurs manquantes 🔍"):
        missing_values = df.isna().sum()
        if missing_values.sum() == 0:
            st.success("Aucune valeur manquante dans le dataset ! ✅")
        else:
            st.dataframe(missing_values[missing_values > 0])

# Vérification de la page actuelle

if page == pages[2]:
    # Titre principal
    st.title("Visualisation des Données 📊")

    # Sélection de la variable à explorer
    st.subheader("Exploration des distributions des variables")
    selected_variable = st.selectbox("Sélectionnez une variable :", df.columns)

    # Vérification du type de variable pour choisir le bon type de graphique
    if df[selected_variable].dtype in ["int64", "float64"]:  
        st.subheader(f"Distribution de '{selected_variable}' (Numérique)")

        # Création de l'histogramme
        fig, ax = plt.subplots(figsize=(8, 5))
        df[selected_variable].hist(bins=20, ax=ax, color="royalblue", edgecolor="black")
        ax.set_xlabel(selected_variable)
        ax.set_ylabel("Fréquence")
        ax.set_title(f"Histogramme de '{selected_variable}'")
        st.pyplot(fig)

    else: 
        st.subheader(f"Distribution de '{selected_variable}' (Catégorielle)")

        # Création du diagramme en barres
        fig, ax = plt.subplots(figsize=(8, 5))
        df[selected_variable].value_counts().plot(kind="bar", ax=ax, color="royalblue", edgecolor="black")
        ax.set_xlabel(selected_variable)
        ax.set_ylabel("Nombre d'occurrences")
        ax.set_title(f"Répartition des catégories de '{selected_variable}'")
        st.pyplot(fig)

    # Sélection des variables catégorielles et numériques
    categorical_vars = df.select_dtypes(include=["object"]).columns.tolist()
    numerical_vars = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

    # Suppression de 'deposit' si elle est présente
    if "deposit" in categorical_vars:
        categorical_vars.remove("deposit")
    if "deposit" in numerical_vars:
        numerical_vars.remove("deposit")

    # Analyse de la variable 'deposit'
    st.subheader("Analyse de la souscription ('deposit') en fonction des autres variables")
    selected_var = st.selectbox("Choisissez une variable à comparer avec 'deposit':", categorical_vars + numerical_vars)

    fig = go.Figure()

    if selected_var in numerical_vars:
        # Affichage d'un boxplot pour les variables numériques
        fig.add_trace(go.Box(
            x=df["deposit"], 
            y=df[selected_var], 
            name=selected_var, 
            boxmean=True
        ))
        fig.update_layout(title=f"Distribution de '{selected_var}' en fonction de 'deposit'")
        st.plotly_chart(fig)

    else:
        # Calcul des pourcentages de souscription et non-souscription
        grouped_data = df.groupby([selected_var, "deposit"]).size().reset_index(name="count")
        total_counts = grouped_data.groupby(selected_var)["count"].transform("sum")
        grouped_data["percentage"] = (grouped_data["count"] / total_counts) * 100

        # Affichage d'un barplot empilé pour visualiser les pourcentages
        colors = {"yes": "#003f5c", "no": "#d45087"}
        for deposit_value in ["yes", "no"]:
            filtered_data = grouped_data[grouped_data["deposit"] == deposit_value]
            fig.add_trace(go.Bar(
                x=filtered_data[selected_var], 
                y=filtered_data["percentage"], 
                name=f"{selected_var} - {deposit_value}",
                marker_color=colors[deposit_value],
                text=filtered_data["percentage"].round(1).astype(str) + '%',
                textposition="inside"
            ))

        fig.update_layout(
            title=f"Pourcentage de souscription ('deposit') en fonction de '{selected_var}'",
            barmode="stack",
            yaxis_title="Pourcentage (%)"
        )

        st.plotly_chart(fig)

if page == pages[3]:
    st.title("Préprocessing 🛠️")

      # Présentation des principales étapes de préparation
    st.write("### Principales étapes de préparation des données")
    st.write("- Les valeurs 'unknown' et 'other' ont été remplacées par NaN pour signaler la présence de valeurs manquantes.")
    st.write("- Les colonnes 'contact' et 'default' ont été supprimées car elles étaient dominées par une seule modalité.")
    st.write("- L'analyse des corrélations a révélé une forte corrélation entre 'pdays' et 'previous', nous avons donc conservé 'pdays'.")
    st.write("- Les variables binaires ont été encodées en valeurs numériques.")
    st.write("- Une nouvelle variable a été créée en combinant 'housing' et 'loan'.")
    st.write("- Une colonne binaire a été ajoutée pour indiquer si le client a été contacté ou non.")   
    
    if st.button("Afficher la matrice de corrélation 🔢"):
        df_numeric = df.select_dtypes(include=["number"])
        corr_matrix = df_numeric.corr()
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
        st.pyplot(fig)

    # Affichage du dataset avant nettoyage
    st.write("### Avant nettoyage")
    st.dataframe(df.head())
    
    # Affichage du dataset après nettoyage
    st.write("### Après nettoyage")
    df_clean=pd.read_csv("df_clean.csv")
    st.dataframe(df_clean.head())
    
import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

if page == pages[4]:
    st.title("Modèles de Machine Learning 📥")

    st.write("""Sélectionnez un modèle de Machine Learning entraîné et consultez ses performances.  
    Vous pouvez comparer plusieurs modèles et voir leurs scores de validation croisée.""")

    # 📂 Vérifier que le dossier "Models" existe
    model_dir = "Models"
    if not os.path.exists(model_dir):
        st.error(f"❌ Le dossier '{model_dir}' n'existe pas. Vérifiez votre structure de fichiers.")
    else:
        # 📄 Lister les modèles disponibles
        model_files = [f for f in os.listdir(model_dir) if f.endswith('.pkl')]

        if not model_files:
            st.error("❌ Aucun modèle trouvé. Assurez-vous d'avoir déplacé les fichiers dans ce dossier.")
        else:
            # 🛠 Retirer l'extension ".pkl" pour l'affichage
            model_names = [f.replace(".pkl", "") for f in model_files]

            # 📌 Sélection de modèle
            selected_model_name = st.selectbox("Choisissez un modèle :", model_names)
            selected_model_file = os.path.join(model_dir, selected_model_name + ".pkl")  # ✅ Ajouter le chemin complet

            # 🔍 Charger le modèle
            selected_model = joblib.load(selected_model_file)

            st.write(f"### Scores du modèle {selected_model_name}")

            # 📊 Vérifier l'existence du fichier de résultats
            scores_file = "cross_val_results.csv"
            if os.path.exists(scores_file):
                cross_val_df = pd.read_csv(scores_file)

                # 🔎 Filtrer les scores du modèle sélectionné
                scores = cross_val_df[cross_val_df['Model'] == selected_model_name]
                st.dataframe(scores)

                # 📈 Comparaison des modèles
                st.write("### 📊 Comparaison des modèles")
                fig, ax = plt.subplots(figsize=(8, 5))  # Ajuster la taille du graphique
                cross_val_df.plot(x="Model", y=["Mean Recall", "Mean Precision", "Mean F1"], 
                                  kind="bar", ax=ax, color=["#1f77b4", "#ff7f0e", "#2ca02c"])
                ax.set_xlabel("Modèles")
                ax.set_ylabel("Score")
                ax.set_title("Comparaison des scores des modèles")
                ax.legend(title="Métriques", loc="upper right", fontsize=8)
                ax.tick_params(axis="x", rotation=30)
                st.pyplot(fig)
            else:
                st.error("❌ Aucun score trouvé pour ce modèle.")

        
if page==pages[5]:
    st.title("Meilleur modèle 📥")
    model = joblib.load("final_model.pkl")
    performance_df = pd.read_csv("model_performance.csv")
    conf_matrix_before = np.load("conf_matrix_before.npy")
    conf_matrix_after = np.load("conf_matrix_after.npy")
    conf_matrix_optimal = np.load("conf_matrix_optimal.npy")
    feat_importance_before = pd.read_csv("feature_importance_before.csv")
    feat_importance_after = pd.read_csv("feature_importance_after.csv")
    st.title("Optimisation du Modèle de Machine Learning")
    st.sidebar.header("Navigation")
    step = st.sidebar.radio("Sélectionnez une étape :", ["1️⃣ Performances", "2️⃣ Feature Importance", "3️⃣ Matrice de Confusion", "4️⃣ Courbe ROC"])
    if step == "1️⃣ Performances":
        st.subheader("Comparaison des Performances du Modèle")
        st.dataframe(performance_df)
        fig, ax = plt.subplots(figsize=(10, 5))
        performance_df.set_index("Step").plot(kind="bar", ax=ax)
        plt.title("Évolution des Scores")
        plt.ylabel("Score")
        plt.xticks(rotation=0)
        st.pyplot(fig)
    elif step == "2️⃣ Feature Importance":
        st.subheader("Importance des Features")
        choix = st.radio("Choisissez :", ["Avant Optimisation", "Après Optimisation"])
        feat_data = feat_importance_before if choix == "Avant Optimisation" else feat_importance_after
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(data=feat_data.head(10), x="Importance", y="Variable", ax=ax, palette="Blues_r")
        plt.title(f"Top Features ({choix})")
        st.pyplot(fig)
    elif step == "3️⃣ Matrice de Confusion":
        st.subheader("Comparaison des Matrices de Confusion")
        choix = st.radio("Choisissez :", ["Avant Optimisation", "Après Hyperparamètres", "Après Seuil"])
        matrix = conf_matrix_before if choix == "Avant Optimisation" else conf_matrix_after if choix == "Après Hyperparamètres" else conf_matrix_optimal
        fig, ax = plt.subplots()
        ConfusionMatrixDisplay(matrix).plot(ax=ax, cmap='Blues')
        plt.title(f"Matrice de Confusion ({choix})")
        st.pyplot(fig)
    elif step == "4️⃣ Courbe ROC":
        st.subheader("Courbe ROC et Seuil Optimal")
        y_probs = np.load("y_probs.npy")
        y_test = np.load("y_test.npy")
        fpr, tpr, thresholds = roc_curve(y_test, y_probs)
        roc_auc = auc(fpr, tpr)
        best_f1 = 0
        best_threshold = 0
        for threshold in thresholds:
            y_pred_temp = (y_probs > threshold).astype(int)
            f1 = f1_score(y_test, y_pred_temp)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold

        st.write(f"**Seuil optimal basé sur le F1-score** : {best_threshold:.2f}")
        st.write(f"**Meilleur F1-score obtenu** : {best_f1:.4f}")
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(fpr, tpr, color='blue', label=f'ROC curve (AUC = {roc_auc:.2f})')
        ax.plot([0, 1], [0, 1], color='gray', linestyle='--')
        ax.set_xlabel('Taux de Faux Positifs')
        ax.set_ylabel('Taux de Vrais Positifs')
        ax.set_title('Courbe ROC')
        ax.legend()
        st.pyplot(fig)

if page == pages[6]:
   st.subheader("📝 Entrez les caractéristiques du client :")
   age = st.number_input("Âge", min_value=18, max_value=100, value=30)
   education = st.selectbox("Niveau d'éducation", options=["primary", "secondary", "tertiary"])
   balance = st.number_input("Solde du compte bancaire", value=0.0)
   housing = st.radio("Possède un prêt immobilier ?", options=["Oui", "Non"])
   day = st.number_input("Jour du contact", min_value=1, max_value=31, value=15)
   month = st.selectbox("Mois du contact", options=['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'])
   duration = st.number_input("Durée de l'appel (en secondes)", min_value=0, value=100)
   poutcome = st.radio("Résultat de la campagne précédente ?", options=["Success", "Failure", "missing"])
   pdays_contacted = st.radio("Déjà contacté ?", options=["Oui", "Non"])
   campaign = st.selectbox("Nombre de contacts durant cette campagne", options=["1 fois", "2 fois", "3-6 fois", "> 6 fois"])
   education_mapping = {"primary": 0, "secondary": 1, "tertiary": 2}
   education = education_mapping[education]
   housing = 1 if housing == "Oui" else 0
   pdays_contacted = 1 if pdays_contacted == "Oui" else 0
   month_mapping = {'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
                 'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12}
   month = month_mapping[month]
   user_data = pd.DataFrame([[age, education, balance, housing, day, month, duration, pdays_contacted, campaign, poutcome]],
                         columns=['age', 'education', 'balance', 'housing', 'day', 'month', 'duration', 'pdays_contacted', 'campaign', 'poutcome'])
   encoder = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
   one_hot_cols = ['campaign', 'poutcome']
   encoded_data = encoder.fit_transform(user_data[one_hot_cols])
   encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(one_hot_cols))
   user_data = pd.concat([user_data.drop(columns=one_hot_cols), encoded_df], axis=1)
   try:
       model = joblib.load("final_model.pkl")
       expected_columns = model.feature_names_in_
       for col in expected_columns:
        if col not in user_data.columns:
            user_data[col] = 0
       user_data = user_data[expected_columns]
       scaler = joblib.load("scaler.pkl")
       numerical_columns = ['age', 'balance', 'day', 'duration']
       user_data[numerical_columns] = scaler.transform(user_data[numerical_columns])
       st.write("🧐 Données après transformation :", user_data)
       if st.button("🔍 Prédire"):
           prediction = model.predict(user_data)[0]
           st.write("🔮 Prédiction brute :", prediction)
           result = "✅ Le client VA souscrire" if prediction == 1 else "❌ Le client NE souscrira PAS"
           st.success(result)
   except Exception as e:
      st.error(f"❌ Erreur : {e}")

   import io
   st.title("🔍 Prédiction de souscription d'un client à un compte à terme")

   uploaded_file = st.file_uploader("📥 Téléchargez votre fichier CSV", type=["csv"])

   if uploaded_file is not None:
        user_data = pd.read_csv(uploaded_file)
        st.write("✅ Données chargées avec succès !")

        user_data.replace(["unknown", "Unknown", "UNKNOWN", "other", "Other", "OTHER"], np.nan, inplace=True)
        if 'pdays' in user_data.columns:
            user_data['pdays_contacted'] = (user_data['pdays'] != -1).astype(int)

        columns_to_drop = ['default', 'contact', 'previous', 'pdays']
        user_data.drop(columns=[col for col in columns_to_drop if col in user_data.columns], inplace=True)

        def categorize_campaign(campaign):
            if campaign == 1:
                return '1 fois'
            elif campaign == 2:
                return '2 fois'
            elif 3 <= campaign <= 6:
                return '3-6 fois'
            else:
                return '> 6 fois'

        if 'campaign' in user_data.columns:
            user_data['campaign'] = user_data['campaign'].apply(categorize_campaign)

        month_mapping = {'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
                         'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12}
        if 'month' in user_data.columns:
            user_data['month'] = user_data['month'].str.lower().map(month_mapping)

        education_mapping = {'primary': 0, 'secondary': 1, 'tertiary': 2}
        if 'education' in user_data.columns:
            user_data['education'] = user_data['education'].str.lower().map(education_mapping)

        for col in ['housing', 'loan']:
            if col in user_data.columns:
                user_data[col].replace({'no': 0, 'yes': 1}, inplace=True)

        if 'duration' in user_data.columns:
            user_data['duration'] = user_data['duration'] / 60
            user_data['duration'] = user_data['duration'].round(2)

        if 'poutcome' in user_data.columns:
            user_data['poutcome'].fillna('missing', inplace=True)

        if 'housing' in user_data.columns and 'loan' in user_data.columns:
            user_data['housing_loan_interaction'] = user_data['housing'] & user_data['loan']

        categorical_imputer = SimpleImputer(strategy='most_frequent')
        cat_cols = ['job', 'education']
        if all(col in user_data.columns for col in cat_cols):
            user_data[cat_cols] = categorical_imputer.fit_transform(user_data[cat_cols])

        one_hot_cols = ['job', 'marital', 'poutcome', 'campaign']
        encoder = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')

        if all(col in user_data.columns for col in one_hot_cols):
            encoded_data = encoder.fit_transform(user_data[one_hot_cols])
            encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(one_hot_cols))
            user_data = pd.concat([user_data.drop(columns=one_hot_cols), encoded_df], axis=1)

        numerical_columns = ['age', 'balance', 'day', 'duration']
        scaler = RobustScaler()

        if all(col in user_data.columns for col in numerical_columns):
            user_data[numerical_columns] = scaler.fit_transform(user_data[numerical_columns])
      
        cols_to_remove = [
        'loan', 'housing_loan_interaction', 'job_blue-collar',
        'job_entrepreneur', 'job_housemaid', 'job_management', 'job_retired',
        'job_self-employed', 'job_services', 'job_student', 'job_technician',
        'job_unemployed', 'marital_married', 'marital_single']
        user_data.drop(columns=[col for col in cols_to_remove if col in user_data.columns], inplace=True)
   
        st.write("📊 Données après prétraitement :")
        st.dataframe(user_data.head())

        csv_buffer = io.BytesIO()
        user_data.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)  # Repositionner le curseur au début du fichier
        st.download_button(label="📥 Télécharger les données prétraitées",data=csv_buffer, file_name="transformed_data.csv", mime="text/csv")

        #entrainement 
        try:
            model = joblib.load("final_model.pkl")
            prediction = model.predict(user_data)
            user_data["Prédiction"] = ["Souscrit ✅" if p == 1 else "Ne souscrit pas ❌" for p in prediction]
            st.dataframe(user_data)
            csv_buffer = io.BytesIO()
            user_data.to_csv(csv_buffer, index=False)
            csv_buffer.seek(0)
            st.download_button(
            label="📥 Télécharger les prédictions",
            data=csv_buffer,
            file_name="predictions.csv",
            mime="text/csv")
        except Exception as e:
            st.error(f"❌ Erreur lors de la prédiction : {e}")

if page == pages[7] : 
  st.write("### Conclusion & perspectives")
  st.write("""Le modèle final utilisé ici est celui qui a montré les meilleures performances après optimisation des hyperparamètres. Cependant, pour garantir des prédictions optimales, il est impératif d'appliquer un seuil de décision de 0,45 avant d'interpréter les résultats.

📌 Attention : La variable "duration" peut entraîner une fuite de données, car la durée de l’appel n’est pas connue avant qu’il ait lieu. Pour une utilisation en conditions réelles, il est recommandé de ne pas l’inclure dans la prédiction.
           
💡 Votre avis compte ! Pour toute suggestion ou amélioration, n’hésitez pas à nous contacter. 🚀""")

      

      
        
      


    



        


