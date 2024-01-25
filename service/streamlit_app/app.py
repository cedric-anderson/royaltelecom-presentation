import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import confusion_matrix, roc_auc_score, plot_confusion_matrix
import streamlit as st
import joblib


try:
    df = pd.read_csv("/app/streamlit_app/data/telecom_custumerDB.csv", encoding="utf-8")
except Exception as e:
    print(f"Erreur lors de la lecture du fichier : {e}")

st.sidebar.title("Sommaire:")

pages = ["Contexte du projet", "Exploration des données", "Data visualisation", "Modele performance"]

page = st.sidebar.radio("Aller vers la page :", pages)

if page == pages[0] : 
    st.title("Contexte du projet")

    # Titre de l'application
    st.title("Fidélisation Clientèle et Prédiction du Désabonnement")

    # Introduction
    st.write("Bienvenue dans un projet passionnant au cœur de la fidélisation clientèle de RoyalTelecom, une entreprise de télécommunication établie dans les pittoresques Alpes-Maritimes. Face à une concurrence intense offrant des services plus compétitifs, RoyalTelecom se lance dans une quête continue de nouveaux clients. Cependant, elle se rend compte que le coût d'acquisition de nouveaux clients est étonnamment élevé par rapport au coût de rétention.")

    # Enjeu et problématique
    st.markdown("### Problématique")
    st.write("L'enjeu ? Fidéliser la clientèle existante.")
    st.write("L'épineuse question à laquelle nous devons répondre : comment retenir nos clients actuels dans un monde où les alternatives abondent ? Pour résoudre ce défi, nous devons plonger dans l'analyse des comportements clients, comprendre leurs besoins spécifiques et développer des offres sur mesure.")

    # Mission du data scientist
    st.write("En tant que data scientist, je suis missionné par RoyalTelecom pour transformer cette vision en réalité. Notre objectif central est de prédire avec précision si un client choisira de résilier son contrat à la fin de l'échéance. Cette prédiction éclairée nous permettra d'anticiper les besoins du client et de lui offrir des services adaptés, renforçant ainsi notre relation avec lui.")

    # Exploration des données
    st.markdown("### Exploration des Données")
    st.write("Notre aventure commence par l'exploration approfondie des données historiques des clients. Chaque ligne de ce trésor d'informations représente un client unique, et chaque colonne expose une facette de sa relation avec RoyalTelecom. Une plongée visuelle dans ces données nous révélera des tendances, des motifs et des opportunités cachées.")

    # Analyse des insights
    st.markdown("### Analyse des Insights")
    st.write("Le prochain chapitre ? L'analyse approfondie des insights tirés de ces données. Nous allons décortiquer les comportements des clients, identifier les signaux faibles et les points forts, afin de mieux comprendre ce qui incite un client à rester fidèle ou à envisager la désertion.")

    # Mise en œuvre de l'intelligence artificielle
    st.markdown("### Mise en œuvre de l'Intelligence Artificielle")
    st.write("Enfin, le moment tant attendu de la mise en œuvre de l'intelligence artificielle. Des modèles de Machine Learning de pointe seront déployés pour prédire le risque de désabonnement de chaque client. Une avancée technologique qui permettra à RoyalTelecom d'anticiper les mouvements du marché et d'ajuster ses offres en conséquence.")

    # Conclusion
    st.write("Accrochez-vous, car ce projet promet des rebondissements, des découvertes fascinantes et, surtout, la clé pour maintenir la satisfaction client à des sommets inexplorés. Prêts à plonger dans l'avenir de la fidélisation clientèle ? Let the data journey begin! 🚀✨")


elif page == pages[1]:
    st.write("### Exploration des données")
    
    dfEst = df
    def statics():
        dfEst['Churn'].replace({'Yes': 1, 'No': 0}, inplace=True)

        num_retained = dfEst[dfEst.Churn == 0.0].shape[0]
        num_churned = dfEst[dfEst.Churn == 1.0].shape[0]
        retined = num_retained / (num_retained + num_churned) * 100
        churned = num_churned / (num_retained + num_churned) * 100

        col1, col2 = st.columns(2)
        col1.metric("Clients restés avec l'entreprise :", round(retined, 2), "%")
        col2.metric("Clients partis de l'entreprise :", round(churned, 2), "%", delta_color='inverse')


    statics()

# Utilisation de st.expander pour les statistiques
    with st.expander("Statistiques", expanded=True):
       if st.checkbox("Afficher les statistiques"):

         col1, col2, col3, col4 = st.columns(4)
         if col1.button("Head"):
             st.write("20 premières lignes du Datasets :")
             st.write(df.head(20))

         if col2.button("Tail"):
             st.write("20 dernières lignes du Datasets :")
             st.write(df.tail(20))

         if col3.button("Colonnes"):
             columns = df.columns
             mid_index = len(columns) // 4
             part1, part2, part3, part4 = columns[:mid_index], columns[mid_index:2*mid_index], columns[2*mid_index:3*mid_index], columns[3*mid_index:]

             col1, col2, col3, col4 = st.columns(4)

             # Affichage des noms de colonnes dans les différentes colonnes
             col1.write(part1)
             col2.write(part2)
             col3.write(part3)
             col4.write(part4)

         if col1.button("Description"):
             st.write(df.describe(include='all'))

         if col2.button("Shape"):
             st.write("Nombre de lignes", df.shape[0])
             st.write("Nombre de colonnes", df.shape[1])

elif page == pages[2]:
    st.write("### Analyse de données")

    # Plot churned
    with st.expander("Data visualization ", expanded=True):
        churned_charts = st.checkbox('Graphique, repartition de la variable cible')
        if churned_charts:
            chart_option = st.radio('Churned:', ['Diagramme en barres', 'Diagramme circulaire (Pie)'])

            if chart_option == 'Diagramme en barres':
                plot_churned = df['Churn'].value_counts()

                colors = ['#e74c3c', '#3498db']  # Couleurs pour Churn et No Churn respectivement

                fig = px.bar(x=plot_churned.index, y=plot_churned.values, color=plot_churned.index, text=plot_churned.values,
                            labels={'x': 'Churn', 'y': 'Nombre', 'color': 'Status'},
                            color_discrete_sequence=colors)

                fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')

                st.plotly_chart(fig)


            if chart_option == 'Diagramme circulaire (Pie)':
               # Diagramme circulaire churn
               pie_chart = px.pie(df, 
                                 names="Churn", 
                                 title="Diagramme circulaire sur le désabonnement (Churn)",
                                 color_discrete_sequence=['#3498db', '#e74c3c'],
                                 hole=0.4,
                                 labels={'1': 'Désabonné', '0': 'Abonné'},
                                 )
               st.plotly_chart(pie_chart)
  
        load = st.checkbox('Chargez les graphiques, variables independante Vs variable cible')
        if load:
            option = st.selectbox("Choisissez une option :", ["Gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService",
                             "InternetService", "DeviceProtection", "TechSupport"])
             
            mid_index = len(option) // 4
            part1, part2, part3, part4 = option[:mid_index], option[mid_index:2*mid_index], option[2*mid_index:3*mid_index], option[3*mid_index:]

            col1, col2, col3, col4 = st.columns(4)

             # Affichage des noms de colonnes dans les différentes colonnes
            col1.write(part1)
            col2.write(part2)
            col3.write(part3)
            col4.write(part4)

            if col1.button("Gender"):
                st.subheader("Exploration de la variable 'Genre':")

                # Informations sur la variable 'Genre'
                st.write("- Nous comptons 3 488 femmes dans notre échantillon, dont 2 549 restent fidèles, représentant un taux moyen de fidélité de 73,08%.")
                st.write("- De même, l'échantillon comprend 3 555 hommes, parmi lesquels 2 625 ne se désabonnent pas.")
                st.write("- Le taux moyen de désabonnement chez les hommes est légèrement plus élevé à 73,83%.")
                st.write("- Étonnamment, le taux de désabonnement semble comparable entre hommes et femmes, suggérant que le genre n'a pas d'impact significatif sur la décision de rompre un abonnement, comme le confirme notre graphique.")
                # plot churned clients by gender
                fig = px.histogram(df, x='gender', color='Churn', barmode='group')
                fig.update_layout(
                    title="Diagramme en barres des clients désabonnés par genre",
                    xaxis_title="Genre",
                    yaxis_title="Nombre de clients",
                    legend_title='Désabonnement'
                    )
    
                st.plotly_chart(fig)

            if col2.button('SeniorCitizen'):
                st.subheader("Analyse des Clients Seniors :")

                # Informations sur la variable 'Senior Citizen'
                st.write("- À travers nos données, un constat frappant se dégage : le taux de désabonnement est plus faible chez les clients plus âgés que chez les plus jeunes.")
                st.write("- Sur les 7 043 clients, 4 508 sont des clients seniors qui restent fidèles, représentant en moyenne 58,31% de fidélité, tandis que 1 393 clients plus jeunes ne se désabonnent pas.")
                st.write("- Si l'on regarde du côté des clients qui se désabonnent, 476 sont des clients seniors, contre 666 clients plus jeunes.")
                st.write("- Les clients plus jeunes semblent donc présenter un risque de désabonnement plus élevé. Notre graphique illustre clairement que l'âge est un facteur déterminant : plus le client est âgé, moins il est susceptible de se désabonner.")
                # Plot churned clients by SeniorCitizen
                fig = px.histogram(df, x='SeniorCitizen', color='Churn', barmode='group',
                                   labels={'SeniorCitizen': 'SeniorCitizen', 'Churn': 'Désabonnement', 'count': 'Nombre de clients'},
                                   title="Diagramme interactif des clients désabonnés par SeniorCitizen",
                                   category_orders={'SeniorCitizen': ['0', '1'], 'Churn': ['No', 'Yes']},
                                   color_discrete_map={'No': 'cyan', 'Yes': 'gold'})
                fig.update_layout(bargap=0.2)
                st.plotly_chart(fig)

            if col3.button("Partner"):
                st.subheader("Analyse de la Situation Matrimoniale des Clients :")

                # Informations sur la variable 'Partner'
                st.write("- Ce graphique révèle des informations particulièrement intrigantes sur la situation matrimoniale des clients en lien avec notre variable cible.")
                st.write("- Il est clair que les clients mariés ont considérablement moins de chances de se désabonner que leurs homologues célibataires.")
                st.write("- Ces données nous offrent une opportunité précieuse pour optimiser nos campagnes marketing. En ciblant davantage les jeunes célibataires, nous pourrions ajuster nos offres, anticipant ainsi leur propension plus élevée à se désabonner.")
                # Tracé des clients résiliés par partenaire
                fig = px.histogram(df, x='Partner', color='Churn', barmode='group',
                                   labels={'Partner': 'maried', 'Churn': 'Désabonnement', 'count': 'Nombre de clients'},
                                   title="Diagramme interactif de desabonnement des clients en fonction de leur situation marietale",
                                   category_orders={'Partner': ['No', 'Yes'], 'Churn': ['No', 'Yes']},
                                   color_discrete_map={'No': 'darkgray', 'Yes': 'firebrick'})
                fig.update_layout(bargap=0.2)
                st.plotly_chart(fig)

            if col4.button("Dependents"):
                st.subheader("Analyse de la Présence de Personnes à Charge :")

                # Informations sur la variable 'Dependents'
                st.write("- Parmi les 2 110 clients ayant des personnes à charge, seuls 326 ont résilié, affichant un remarquable taux moyen de non-résiliation de 84,54%.")
                st.write("- En revanche, sur les 4 933 clients sans personne à charge, le taux moyen de résiliation s'élève à 68,72%.")
                st.write("- La résiliation est donc plus rapide en moyenne chez les clients sans personne à charge, une donnée cruciale à considérer lors de la planification d'éventuelles campagnes de rétention client.")
                fig = px.histogram(df, x='Dependents', color='Churn', barmode='group',
                                   labels={'Dependents': 'Personne a charge', 'Churn': 'Désabonnement', 'count': 'Nombre de clients'},
                                   title="Diagramme de desabonnement des clients en fonction du nombre de personne a charge",
                                   category_orders={'Dependents': ['No', 'Yes'], 'Churn': ['No', 'Yes']},
                                   color_discrete_map={'No': 'goldenrod', 'Yes': 'fuchsia'})
                fig.update_layout(bargap=0.2)
                st.plotly_chart(fig)

            if col1.button("PhoneService"):
                st.subheader("Analyse du Service Téléphonique :")

                # Informations sur la variable 'PhoneService'
                st.write("- Parmi les 682 clients sans service téléphonique, 170 ont résilié, affichant un taux moyen de 75,07%.")
                st.write("- En revanche, parmi les 6 361 clients disposant du service téléphonique, 4 662 n'ont pas résilié, affichant un taux moyen de 73,29% de non-résiliation.")
                st.write("- Les personnes ayant souscrit à des services téléphoniques supplémentaires ont une propension plus élevée à se désabonner. On peut envisager que plus le coût de l'abonnement est élevé, plus les gens sont susceptibles de résilier.")
                # plot churned clients by phoneservice
                fig = px.histogram(df, x='PhoneService', color='Churn', barmode='group',
                                   labels={'PhoneService': 'PhoneService', 'Churn': 'Désabonnement', 'count': 'Nombre de clients'},
                                   title="Diagramme de desabonnement des clients en fonction des services telephoniques souscrit",
                                   category_orders={'PhoneService': ['No', 'Yes'], 'Churn': ['No', 'Yes']},
                                   color_discrete_map={'No': 'cyan', 'Yes': 'fuchsia'})
                fig.update_layout(bargap=0.2)
                st.plotly_chart(fig)

            if col2.button("InternetService"):
                st.subheader("Analyse du Service Internet :")

                # Informations sur la variable 'InternetService'
                st.write("- Selon nos données, parmi les 7 043 clients, 2 421 ont des services DSL, 3 096 ont la fibre optique et 1 526 n'ont pas de service Internet.")
                st.write("- Pour les clients avec le service DSL, le taux moyen de non-désabonnement atteint 81,04%, tandis que pour ceux avec la fibre optique, le taux moyen de désabonnement est de 58,10%. D'autre part, 92,59% de ceux sans service Internet sont moins enclins à se désabonner.")
                st.write("- Il est évident qu'une attention particulière doit être accordée aux abonnés de la fibre optique, car ils présentent un taux moyen de désabonnement plus élevé.")
                # plot churned clients by InternetService
                fig = px.histogram(df, x='InternetService', color='Churn', barmode='group',
                                   labels={'InternetService': 'PhoneService', 'Churn': 'Désabonnement', 'count': 'Nombre de clients'},
                                   title="Diagramme de desabonnement des clients en fonction du fournisseur de service",
                                   category_orders={'InternetService': ['No', 'Yes'], 'Churn': ['No', 'Yes']},
                                   color_discrete_map={'No': 'darkolivegreen', 'Yes': 'gold'})
                fig.update_layout(bargap=0.2)
                st.plotly_chart(fig)

            if col3.button("DeviceProtection"):
                st.subheader("Analyse de la Protection des Appareils :")

                # Informations sur la variable 'Device Protection'
                st.write("- En ce qui concerne la protection des appareils, on compte 3 095 clients sans protection d'appareil, affichant un taux moyen de non-résiliation de 60,87%, avec 939 résiliations.")
                st.write("- De plus, parmi les 1 526 clients sans service Internet, 113 ont résilié, affichant un taux moyen de non-résiliation de 92,59%.")
                st.write("- Sur les 2 422 clients avec une protection d'appareil, 1 877 n'ont pas résilié et 545 ont résilié, donnant un taux moyen de non-résiliation de 77,49%.")
                st.write("- Le taux moyen de résiliation le plus élevé concerne ceux sans protection d'appareil. La société devrait concentrer ses efforts sur cette catégorie pour éviter de perdre des clients.")
                # tracer les clients résiliés par protection d'appareil
                fig = px.histogram(df, x='DeviceProtection', color='Churn', barmode='group',
                                   labels={'DeviceProtection': 'DeviceProtection', 'Churn': 'Désabonnement', 'count': 'Nombre de clients'},
                                   title="Diagramme de desabonnement des clients en fonction des services suplementaire souscritent(protection contre la fraud)",
                                   category_orders={'DeviceProtection': ['No', 'Yes'], 'Churn': ['No', 'Yes']},
                                   color_discrete_map={'No': 'lightslategrey', 'Yes': 'lightcyan'})
                fig.update_layout(bargap=0.2)
                st.plotly_chart(fig)

            if col4.button("TechSupport"):
                st.subheader("Analyse du Support Technique :")

                # Informations sur la variable 'TechSupport'
                st.write("- Parmi les 7 043 clients, 2 044 ont souscrit au service de support technique. Parmi eux, 1 734 n'ont pas résilié, affichant un taux moyen de non-résiliation de 84,83%, tandis que 310 ont résilié.")
                st.write("- En revanche, parmi les 1 526 clients sans support technique, 1 413 n'ont pas résilié, avec seulement 113 résiliations, affichant un taux moyen de non-résiliation de 92%.")
                st.write("- Les clients sans support technique présentent un taux moyen de résiliation plus élevé, suggérant qu'ils sont plus susceptibles de résilier.")
                # plot churned clients by TechSupport
                plt.figure(figsize=(15, 10))
                sns.countplot(x='TechSupport', hue='Churn', data=df)
                st.pyplot(plt)
                fig = px.histogram(df, x='TechSupport', color='Churn', barmode='group',
                                   labels={'TechSupport': 'TechSupport', 'Churn': 'Désabonnement', 'count': 'Nombre de clients'},
                                   title="Diagramme de desabonnement des clients en fonction de la souscription a un support technique",
                                   category_orders={'TechSupport': ['No', 'Yes'], 'Churn': ['No', 'Yes']},
                                   color_discrete_map={'No': 'gold', 'Yes': 'darkgray'})
                fig.update_layout(bargap=0.2)
                st.plotly_chart(fig)

        
        matrix = st.checkbox('Chargez la matrix de correlation')
        if matrix:
            numerical_data = df.select_dtypes(include=['float64', 'int64'])
            correlation_matrix = numerical_data.corr()
            correlation_matrix = correlation_matrix.round(2)
            text_values = correlation_matrix.applymap(lambda x: f"{x:.2%}")

            # Création de la figure Plotly
            fig = make_subplots(rows=1, cols=1, subplot_titles=["Matrice de Corrélation"])

            heatmap = go.Heatmap(
                z=correlation_matrix.values,
                x=correlation_matrix.columns,
                y=correlation_matrix.index,
                colorscale='bluered',
                zmin=-1,
                zmax=1,
                text=text_values.values,
                hoverinfo="text+z",
            )

            fig.add_trace(heatmap)

            # Mise en forme de la figure
            fig.update_layout(
                title='Matrice de Corrélation',
                width=800,
                height=600,
                xaxis=dict(tickangle=-45),
            )

            # Affichage de la figure Plotly avec Streamlit
            st.plotly_chart(fig)
    

        barplot = st.checkbox('Chargez les graphiques de la distribution des variables')
        if barplot:
            numerical_data = ['tenure', 'MonthlyCharges', 'TotalCharges']
            df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
            barplot_option = st.radio('choose:', ['Histogrammes', 'Boxplot'])

            if barplot_option == 'Histogrammes':
                def hist_plot(Variable):
                    fig = px.histogram(df, x=Variable, title=f'Histogram of {Variable}')
                    st.plotly_chart(fig)
                selected_variable = st.sidebar.selectbox("Sélectionnez la variable", numerical_data)
                hist_plot(selected_variable)

            if barplot_option == 'Boxplot':
                def box_plot(Variable):
                    st.title(f'Boxplot de {Variable}')
                    fig = px.box(df, y=Variable, points="all", title=f'Boxplot de {Variable}')
                    st.plotly_chart(fig)
                selected_variable = st.sidebar.selectbox("Sélectionnez la variable", numerical_data)
                box_plot(selected_variable)

        
        violonplot = st.checkbox('Chargez les violons, variables independantes Vs Churn')
        if violonplot:
            numerical_data = ['tenure', 'MonthlyCharges', 'TotalCharges']
            df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

            violonlot_option = st.radio('choose violonplot:', ['TotalCharges Vs Churn', 'MonthlyCharges Vs Churn', 'tenure Vs Churn'])

            if violonlot_option == 'TotalCharges Vs Churn':
                filtered_df = df.dropna(subset=['TotalCharges'])
                fig = px.violin(filtered_df, x='Churn', y='TotalCharges', color='Churn', box=True, points="all", title='Graphique de violon - TotalCharges Vs Churn')
                st.plotly_chart(fig)

            elif violonlot_option == 'MonthlyCharges Vs Churn':
                fig = px.violin(df, x='Churn', y='MonthlyCharges', color='Churn', box=True, points="all", title='Graphique de violon - MonthlyCharges Vs Churn')
                st.plotly_chart(fig)

            elif violonlot_option == 'tenure Vs Churn':
                fig = px.violin(df, x='Churn', y='tenure', color='Churn',box=True, points="all", title='Graphique de violon - tenure Vs Churn')
                st.plotly_chart(fig) 



elif page == pages[3]:
    st.write("### Performance modele")

    train_feature_down = pd.read_csv("/app/streamlit_app/data/train_feature_down.csv")
    train_labels_down = pd.read_csv("/app/streamlit_app/data/train_labels_down.csv")
    val_labels = pd.read_csv("/app/streamlit_app/data/val_labels.csv")
    val_feature = pd.read_csv("/app/streamlit_app/data/val_feature.csv")

    # Charger les modèles
    logreg_model = joblib.load("/app/streamlit_app/modeles/logreg_model.joblib")
    rf_model = joblib.load("/app/streamlit_app/modeles/rf_model.joblib")
    xgboost_model = joblib.load("/app/streamlit_app/modeles/xgboost_model.joblib")

    # Fonction pour entraîner le modèle sélectionné
    def train_model(model_choisi, y_pred_reg, y_pred_rf, y_pred_xgboost, val_labels, val_feature):
        if model_choisi == 'Regression Logistique':
            model = logreg_model
            y_pred = y_pred_reg
        elif model_choisi == 'Random Forest':
            model = rf_model
            y_pred = y_pred_rf
        elif model_choisi == 'XGboost':
            model = xgboost_model
            y_pred = y_pred_xgboost

        # Calculer et renvoyer l'AUC
        auc_score = roc_auc_score(val_labels, y_pred)

        # Limiter à 2 chiffres après la virgule
        auc_score = round(auc_score, 2)

        # Afficher la matrice de confusion
        st.write("Matrice de Confusion:")
        confusion_mat = confusion_matrix(val_labels, (y_pred > 0.5))  # Vous pouvez ajuster le seuil si nécessaire
        st.table(confusion_mat)

        # Tracer la matrice de confusion
        st.write("Graphe de Matrice de Confusion:")
        fig, ax = plt.subplots()
        plot_confusion_matrix(model, val_feature, val_labels, ax=ax)
        st.pyplot(fig)

        # Tracer le graphe de prédiction
        st.write("Graphe de Prédiction:")
        fig_pred, ax_pred = plt.subplots()
        ax_pred.scatter(val_labels, y_pred)
        ax_pred.set_xlabel("Labels Réels")
        ax_pred.set_ylabel("Prédictions")
        ax_pred.set_title("Graphe de Prédiction")
        ax_pred.grid(True)
        st.pyplot(fig_pred)

        return auc_score

    # Faire des prédictions avec chaque modèle
    y_pred_reg = logreg_model.predict(val_feature)
    y_pred_rf = rf_model.predict(val_feature)
    y_pred_xgboost = xgboost_model.predict(val_feature)

    # Interface utilisateur pour choisir le modèle
    model_choisi = st.selectbox(label="Modèle", options=['Regression Logistique', 'Random Forest', 'XGboost'])

    # Afficher le résultat
    st.write("AUC Score:", train_model(model_choisi, y_pred_reg, y_pred_rf, y_pred_xgboost, val_labels, val_feature))


