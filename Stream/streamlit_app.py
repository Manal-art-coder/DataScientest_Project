import streamlit as st
import pandas as pd
import requests
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
from io import BytesIO
import io
import os


df=pd.read_csv(r"https://raw.githubusercontent.com/Manal-art-coder/DataScientest_Project/dev/bank.csv")
df.head()
st.title("Prédiction du succès d'une campagne Marketing")
st.sidebar.title("Sommaire")
pages=["Contexte & enjeux", "Présentation des données", "Visualisation des données", "Préprocessing", "Modèles ML", "Meilleur modèle", "Faites votre propre prédiction !","Conclusion & Recommandations", "Difficultés & perspectives"]
page=st.sidebar.radio("Aller vers", pages)

if page == pages[0]: 
    # Section Contexte & Enjeux
    st.write("### 📌 Contexte & Enjeux")
    
    # Ajout d'une image d'illustration
    st.image(r"https://raw.githubusercontent.com/Manal-art-coder/DataScientest_Project/dev/Stream/Screenshot%202025-02-19%20111852.jpg", use_container_width=True)
    
    st.write("""
    Ce projet s’inscrit dans le cadre de notre formation de **Data Analyst** avec l’organisme **DataScientest**.  

    ### 🎯 Objectif du projet  
    L’objectif est d’analyser et prédire le succès d’une **campagne marketing bancaire** en identifiant les facteurs influençant la souscription des clients à une offre de dépôt à terme.  

    ### 📊 Données utilisées  
    - Le dataset provient de la plateforme **Kaggle**, accessible ici : [Bank Marketing Dataset](https://www.kaggle.com/datasets/janiobachmann/bank-marketing-dataset).  
    - Il contient des informations détaillées sur les clients, les interactions passées et les résultats des campagnes marketing.  

    ### 🔍 Démarche adoptée  
    Ce **Tableau de bord interactif Streamlit** retrace toutes les étapes du projet :  
    1. Exploration et prétraitement des données  
    2. Visualisation des variables explicatives  
    3. Mise en place et comparaison de plusieurs **modèles de Machine Learning**  
    4. Sélection du modèle le plus performant  
    5. **Prédiction en temps réel** sur de nouvelles données  

    📢 Ce projet permet ainsi de mieux comprendre les dynamiques des campagnes marketing et d'optimiser les stratégies commerciales ! 🚀
    """)

    #  Section Équipe du Projet
    st.write("### 👥 Équipe du Projet")

    #  Liste des membres de l'équipe
    team_members = [
        {"nom": "Jewa", "prenom": "Manal", "linkedin": "https://www.linkedin.com/in/manaljewa/"},
        {"nom": "Selle", "prenom": "Manon", "linkedin": "https://www.linkedin.com/in/manon-selle/"},
        {"nom": "Demeulemeester", "prenom": "Elyse", "linkedin": "https://www.linkedin.com/in/elyse-demeulemeester-aa41832b7/"},
        {"nom": "Amiel", "prenom": "Audrey", "linkedin": "https://www.linkedin.com/in/audrey-amiel/"},
        {"nom": "Legrand", "prenom": "David", "linkedin": "https://www.linkedin.com/in/david-legrand-89bb63247/"}
    ]

    #  Affichage des membres sur deux lignes
    col1, col2, col3 = st.columns(3)  # Première ligne (3 colonnes)
    col4, col5 = st.columns(2)  # Deuxième ligne (2 colonnes)

    columns = [col1, col2, col3, col4, col5]

    for col, member in zip(columns, team_members):
        with col:
            st.write(f"**{member['prenom']} {member['nom']}**")
            st.markdown(f"[LinkedIn]({member['linkedin']})")  # Lien vers le profil LinkedIn

if page == pages[1]:
    st.title("Présentation des données 📊")

    # Aperçu du dataset
    st.subheader("Aperçu des premières lignes")
    st.write("Voici un aperçu des 10 premières lignes du jeu de données :")
    st.dataframe(df.head(10))
    st.subheader("Taille du Dataset")
    st.write(f"**Nombre de lignes** : {df.shape[0]}")
    st.write(f"**Nombre de colonnes** : {df.shape[1]}")
    st.subheader("Types des Variables dans le Dataset")
    categorical_vars = df.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_vars = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    if st.button("Afficher les types des variables"):
        st.dataframe(df.dtypes.reset_index().rename(columns={"index": "Variable", 0: "Type"}))
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Afficher les variables catégorielles"):
            st.write("### Variables Catégorielles")
            st.write(categorical_vars)
    with col2:
        if st.button("Afficher les variables numériques"):
            st.write("### Variables Numériques")
            st.write(numerical_vars)
    st.subheader("Statistiques Descriptives")
    if st.checkbox("Afficher les statistiques détaillées"):
        st.write("Résumé statistique des variables numériques du dataset :")
        st.dataframe(df.describe())
    st.subheader("Valeurs Manquantes")
    if st.checkbox("Afficher les valeurs manquantes 🔍"):
        missing_values = df.isna().sum()
        if missing_values.sum() == 0:
            st.success("Aucune valeur manquante dans le dataset ! ✅")
        else:
            st.dataframe(missing_values[missing_values > 0].reset_index().rename(columns={"index": "Variable", 0: "Nombre de valeurs manquantes"}))
    st.subheader("Doublons dans le Dataset")
    if st.checkbox("Afficher les doublons 📋"):
        duplicate_rows = df[df.duplicated()]
        if duplicate_rows.empty:
            st.success("Aucun doublon trouvé dans le dataset ! ✅")
        else:
            st.write(f"Nombre de doublons : {duplicate_rows.shape[0]}")
            st.dataframe(duplicate_rows)

if page == pages[2]:
    st.title("Visualisation des Données 📊")
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

    def categorize_data(df):
        df = df.copy()
        df['age'] = pd.cut(df['age'], bins=[0, 25, 35, 45, 55, 65, float('inf')], 
                        labels=['<25', '25-35', '35-45', '45-55', '55-65', '>65'])
        df['campaign'] = pd.cut(df['campaign'], bins=[0, 1, 2, 6, float('inf')], 
                             labels=['1 fois', '2 fois', '3-6 fois', '>6 fois'])
        df['previous'] = pd.cut(df['previous'], bins=[-1, 0, 1, 5, float('inf')], 
                             labels=['jamais contacté (0)', '1 seul contact', '2 à 5 contacts', 'plus de 6 contacts'])
        df['pdays'] = pd.cut(df['pdays'], bins=[-2, -1, 180, 365, float('inf')], 
                          labels=['Jamais contacté (-1)', '0-6 mois', '6 mois-1 an', '> 1 an'])
        balance_bins = pd.qcut(df['balance'], q=4, duplicates='drop')
        balance_labels = [f"{int(interval.left)} - {int(interval.right)}" for interval in balance_bins.cat.categories]
        df['balance'] = pd.Categorical(balance_bins, categories=balance_bins.cat.categories, ordered=True)
        df['balance'] = df['balance'].cat.rename_categories(balance_labels)
        duration_bins = pd.qcut(df['duration'], q=4, duplicates='drop')
        duration_labels = [f"{int(interval.left)} - {int(interval.right)}" for interval in duration_bins.cat.categories]
        df['duration'] = pd.Categorical(duration_bins, categories=duration_bins.cat.categories, ordered=True)
        df['duration'] = df['duration'].cat.rename_categories(duration_labels)
        df['day'] = pd.to_datetime(df['day'], format='%d', errors='coerce').dt.day_name()
        return df
    new_1 = categorize_data(df)  # Appliquer la transformation
    categorical_vars = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 
                    'month', 'day', 'poutcome', 'age', 'campaign', 'previous', 'pdays', 'balance', 'duration']
    st.subheader("Analyse de la souscription ('deposit') en fonction des autres variables")
    selected_var = st.selectbox("Choisissez une variable à comparer avec 'deposit':", categorical_vars)
    fig = go.Figure()
    grouped_data = new_1.groupby([selected_var, "deposit"]).size().reset_index(name="count")
    total_counts = grouped_data.groupby(selected_var)["count"].transform("sum")
    grouped_data["percentage"] = (grouped_data["count"] / total_counts) * 100
    colors = {"yes": "#003f5c", "no": "#d45087"}
    for deposit_value in ["yes", "no"]:
        filtered_data = grouped_data[grouped_data["deposit"] == deposit_value]
        fig.add_trace(go.Bar(
        x=filtered_data[selected_var], 
        y=filtered_data["percentage"], 
        name=f"{deposit_value}",
        marker_color=colors[deposit_value],
        text=filtered_data["percentage"].round(1).astype(str) + '%',
        textposition="inside"))
    fig.update_layout(
    title=f"Pourcentage de souscription ('deposit') en fonction de '{selected_var}'",
    barmode="stack",
    yaxis_title="Pourcentage (%)")
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
    df_clean=pd.read_csv(r"https://raw.githubusercontent.com/Manal-art-coder/DataScientest_Project/dev/Stream/df_clean.csv")
    st.dataframe(df_clean.head())

if page == pages[4]:
    st.title("Modèles de Machine Learning 📥")

    st.write("""Sélectionnez un modèle de Machine Learning entraîné et consultez ses performances.  
    Vous pouvez comparer plusieurs modèles et voir leurs scores de validation croisée.""")

    MODEL_DIR_URL = "https://raw.githubusercontent.com/Manal-art-coder/DataScientest_Project/dev/Models/"
    model_files = ["AdaBoostClassifier.pkl", "BaggingClassifier.pkl", "DecisionTreeClassifier.pkl",
                   "GradientBoostingClassifier.pkl", "LogisticRegression.pkl", "RandomForestClassifier.pkl"]

    if not model_files:
        st.error("❌ Aucun modèle trouvé. Assurez-vous que les fichiers sont bien sur GitHub.")
    else:
        model_names = [f.replace(".pkl", "") for f in model_files]
        selected_model_name = st.selectbox("Choisissez un modèle :", model_names)
        selected_model_file = MODEL_DIR_URL + selected_model_name + ".pkl"  # ✅ Construire l'URL complète

        try:
            response = requests.get(selected_model_file)
            if response.status_code == 200:
                selected_model = joblib.load(BytesIO(response.content))
                st.success(f"✅ Modèle {selected_model_name} chargé avec succès !")
            else:
                st.error(f"❌ Erreur {response.status_code} lors du chargement de {selected_model_file}")
                selected_model = None
        except Exception as e:
            st.error(f"❌ Erreur lors du chargement du modèle {selected_model_name} : {e}")
            selected_model = None

        st.write(f"### Scores du modèle {selected_model_name}")

    SCORES_FILE_URL = "https://raw.githubusercontent.com/Manal-art-coder/DataScientest_Project/dev/Stream/cross_val_results.csv"
    try:
        response = requests.get(SCORES_FILE_URL)
        if response.status_code == 200:
            cross_val_df = pd.read_csv(BytesIO(response.content))  
            scores = cross_val_df[cross_val_df['Model'] == selected_model_name]
            st.dataframe(scores)

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
            st.error(f"❌ Erreur {response.status_code} : impossible de charger les résultats.")

    except Exception as e:
        st.error(f"❌ Erreur lors du chargement du fichier des scores : {e}")


if page == pages[5]:
    st.title("Meilleur modèle 📥")
    BASE_URL = "https://raw.githubusercontent.com/Manal-art-coder/DataScientest_Project/dev/Stream/"
    files = {
        "model": "final_model.pkl",
        "performance": "model_performance.csv",
        "conf_matrix_before": "conf_matrix_before.npy",
        "conf_matrix_after": "conf_matrix_after.npy",
        "conf_matrix_optimal": "conf_matrix_optimal.npy",
        "feat_importance_before": "feature_importance_before.csv",
        "feat_importance_after": "feature_importance_after.csv",
        "y_probs": "y_probs.npy",
        "y_test": "y_test.npy",
    }

    def load_file(url, is_numpy=False, is_pkl=False):
        response = requests.get(url)
        if response.status_code == 200:
            if is_numpy:
                return np.load(BytesIO(response.content), allow_pickle=True)
            elif is_pkl:
                with BytesIO(response.content) as file:
                    return joblib.load(file)  # Correction ici
            else:
                return pd.read_csv(BytesIO(response.content))
        else:
            st.error(f"❌ Erreur {response.status_code} : impossible de charger {url}")
            return None

    #  Chargement des fichiers
    model = load_file(BASE_URL + files["model"], is_pkl=True)
    performance_df = load_file(BASE_URL + files["performance"])
    conf_matrix_before = load_file(BASE_URL + files["conf_matrix_before"], is_numpy=True)
    conf_matrix_after = load_file(BASE_URL + files["conf_matrix_after"], is_numpy=True)
    conf_matrix_optimal = load_file(BASE_URL + files["conf_matrix_optimal"], is_numpy=True)
    feat_importance_before = load_file(BASE_URL + files["feat_importance_before"])
    feat_importance_after = load_file(BASE_URL + files["feat_importance_after"])
    y_probs = load_file(BASE_URL + files["y_probs"], is_numpy=True)
    y_test = load_file(BASE_URL + files["y_test"], is_numpy=True)

    st.title("Optimisation du Modèle de Machine Learning")
    st.sidebar.header("Navigation")
    step = st.sidebar.radio("Sélectionnez une étape :", ["1️⃣ Performances", "2️⃣ Feature Importance", "3️⃣ Matrice de Confusion", "4️⃣ Courbe ROC"])

    if step == "1️⃣ Performances":
        st.subheader("Comparaison des Performances du Modèle")
        if performance_df is not None:
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
        if feat_data is not None:
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.barplot(data=feat_data.head(10), x="Importance", y="Variable", ax=ax, palette="Blues_r")
            plt.title(f"Top Features ({choix})")
            st.pyplot(fig)

    elif step == "3️⃣ Matrice de Confusion":
        st.subheader("Comparaison des Matrices de Confusion")
        choix = st.radio("Choisissez :", ["Avant Optimisation", "Après Hyperparamètres", "Après Seuil"])
        matrix = conf_matrix_before if choix == "Avant Optimisation" else conf_matrix_after if choix == "Après Hyperparamètres" else conf_matrix_optimal
        if matrix is not None:
            fig, ax = plt.subplots()
            ConfusionMatrixDisplay(matrix).plot(ax=ax, cmap='Blues')
            plt.title(f"Matrice de Confusion ({choix})")
            st.pyplot(fig)

    elif step == "4️⃣ Courbe ROC":
        st.subheader("Courbe ROC et Seuil Optimal")
        if y_probs is not None and y_test is not None:
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
    st.title("🔍 Prédiction de souscription d'un client à un compte à terme")

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

    # Mapping des variables catégoriques
    education_mapping = {"primary": 0, "secondary": 1, "tertiary": 2}
    month_mapping = {'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
                     'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12}
    
    education = education_mapping[education]
    housing = 1 if housing == "Oui" else 0
    pdays_contacted = 1 if pdays_contacted == "Oui" else 0
    month = month_mapping[month]

    # Création du DataFrame utilisateur
    user_data = pd.DataFrame([[age, education, balance, housing, day, month, duration, pdays_contacted, campaign, poutcome]],
                             columns=['age', 'education', 'balance', 'housing', 'day', 'month', 'duration', 'pdays_contacted', 'campaign', 'poutcome'])

    # One-Hot Encoding
    one_hot_cols = ['campaign', 'poutcome']
    encoder = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')

    encoded_data = encoder.fit_transform(user_data[one_hot_cols])
    encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(one_hot_cols))
    user_data = pd.concat([user_data.drop(columns=one_hot_cols), encoded_df], axis=1)

    # Chargement du modèle et du scaler depuis GitHub
    try:
        main_dir = "https://raw.githubusercontent.com/Manal-art-coder/DataScientest_Project/dev/Stream/"
        model_url = main_dir + "final_model.pkl"
        scaler_url = main_dir + "scaler.pkl"

        # Télécharger le modèle
        response = requests.get(model_url)
        response.raise_for_status()
        model = joblib.load(BytesIO(response.content))

        # Ajouter les colonnes manquantes
        expected_columns = model.feature_names_in_
        for col in expected_columns:
            if col not in user_data.columns:
                user_data[col] = 0
        user_data = user_data[expected_columns]

        # Télécharger et appliquer le scaler
        response = requests.get(scaler_url)
        response.raise_for_status()
        scaler = joblib.load(BytesIO(response.content))

        numerical_columns = ['age', 'balance', 'day', 'duration']
        existing_numerical_columns = [col for col in numerical_columns if col in user_data.columns]
        user_data[existing_numerical_columns] = scaler.transform(user_data[existing_numerical_columns])

        # Prédiction
        if st.button("🔍 Prédire"):
            prediction = model.predict(user_data)[0]
            st.write("🔮 Prédiction brute :", prediction)
            result = "✅ Le client VA souscrire" if prediction == 1 else "❌ Le client NE souscrira PAS"
            st.success(result)

    except Exception as e:
        st.error(f"❌ Erreur : {e}")

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

if page == pages[7]:
    st.write("## 📊 Conclusion & Recommandations")

    st.write("### 🔎 Résumé des résultats")
    st.write("""
    Notre modèle final a confirmé plusieurs tendances observées lors de la phase d'exploration des données :
    - **Durée du dernier contact 📞** : Très corrélée avec la souscription, mais problématique pour une prédiction en amont.
    - **Période des campagnes 📅** : Mai est inefficace, alors que septembre et octobre sont plus favorables.
    - **Ciblage par âge 👥** : Les jeunes (-25 ans) et les seniors (+65 ans) souscrivent davantage.
    - **Jours de contact optimaux 📆** : mardi, mercredi et jeudi offrent les meilleurs taux de conversion.
    - **Solde du compte 💰** : Plus il est élevé, plus la souscription est probable.
    - **Impact des prêts 🏦** : Les clients avec un prêt immobilier sont moins enclins à souscrire à un DAT.
    """)

    st.write("### 📌 Recommandations stratégiques")
    st.markdown("""
    🔹 **Optimisation des campagnes**  
    - Cibler les campagnes en septembre/octobre plutôt qu'en mai.  
    - Planifier les appels principalement en milieu de semaine.  
      
    🔹 **Amélioration du ciblage client**  
    - Segmenter les clients en fonction de leur solde bancaire.  
    - Privilégier les profils ayant répondu positivement à des campagnes précédentes.  
      
    🔹 **Stratégie d’engagement client**  
    - Former les équipes pour prolonger la durée des appels et améliorer le taux de conversion.  
    - Personnaliser les offres en fonction des besoins spécifiques des tranches d’âge.  
    """)

if page ==pages[8]:
    st.write("## 🚀 Difficultés rencontrées & Perspectives")
    st.write("""
    🔹 **Utilisation de la variable duration**  
    - Variable informative mais inutilisable en amont.  
    - Son retrait a diminué les performances du modèle.  
      
    🔹 **Modélisation et choix des algorithmes**  
    - Test de plusieurs modèles avant d’identifier le meilleur.  
    - Difficulté à choisir entre prédiction avant ou après le premier contact.  
      
    🔹 **Problèmes de données**  
    - Valeurs inconnues dans certaines variables (`poutcome`, `education`).  
    - Déséquilibre des campagnes passées compliquant l'analyse des résultats.  
    """)

    st.write("### 🎯 Bilan et résultats obtenus")
    st.markdown("""
    - **Modèle final** : Précision **85%** et F1-score **0.83**.  
    - **Benchmark** : Résultats compétitifs par rapport aux standards du secteur bancaire.  
    - **Impact business** : Amélioration de l’efficacité des campagnes marketing.  
    """)

    st.write("### 🔍 Pistes d’amélioration")
    st.markdown("""
    ✅ **Améliorer les variables** en créant de nouvelles interactions (âge & statut pro, solde & prêt).   
    ✅ **Tester l’utilisation de duration** en la prédisant dans un modèle séparé.  
    ✅ **Test A/B** pour valider les recommandations sur un échantillon réel.  
    """)
    st.success("Ce projet nous a permis d'explorer des problématiques réelles de Machine Learning appliquées au marketing bancaire et de mieux comprendre l'importance des variables temporelles dans la prédiction.")



      

      
        
      


    



        


