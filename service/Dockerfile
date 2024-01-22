# image de base appropriée
FROM python:3.7-slim

# Copiez les fichiers nécessaires dans le conteneur
COPY requirements.txt /app/requirements.txt

COPY streamlit_app /app/streamlit_app

# Définissez le répertoire de travail
WORKDIR /app

# Installez les dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Exposez le port utilisé par Streamlit
EXPOSE 8501

# Commande pour démarrer l'application Streamlit
CMD ["streamlit", "run",  "streamlit_app/app.py"]
