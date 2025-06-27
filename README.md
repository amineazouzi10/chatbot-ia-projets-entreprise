# 🤖 Chatbot Intelligent IA - Gestion de Projets d'Entreprise

Un assistant conversationnel intelligent basé sur l'IA utilisant Rasa Pro CALM pour la **prédiction des compétences**, l'**évaluation des risques** et l'**estimation des coûts** de projets en entreprise.

## 🎯 Fonctionnalités Principales

- 🧠 **Prédiction des Compétences** - Analyse et recommande les compétences nécessaires pour les projets
- ⚠️ **Évaluation des Risques** - Identifie et évalue les risques potentiels des projets
- 💰 **Estimation des Coûts** - Calcule les coûts prévisionnels basés sur l'analyse IA
- 🔄 **Architecture CALM** - Conversations contextuelles avec logique métier structurée
- 🎤 **Reconnaissance Vocale** - Intégration Google Cloud Speech-to-Text
- 🔊 **Synthèse Vocale** - Réponses audio avec Google Cloud Text-to-Speech
- 🤖 **IA Générative** - Powered by Google Gemini AI
- 📊 **Analyse Vectorielle** - Recherche sémantique avec FAISS
- 🌐 **Interface Web** - Frontend Flask avec support multi-canal

## 🏗️ Architecture Technique

### Backend
- **Rasa Pro CALM** - Gestionnaire de conversations
- **Flask** - Serveur web et API REST
- **Google Gemini AI** - Modèle de langage pour l'analyse intelligente
- **Google Cloud Speech Services** - STT/TTS
- **FAISS** - Recherche vectorielle pour la similarité sémantique
- **NLTK** - Traitement du langage naturel

### Frontend
- **JavaScript** - Interface utilisateur interactive
- **CORS** - Support multi-domaines

## 📋 Prérequis

- Python 3.8+
- Licence Rasa Pro
- Google Cloud Account (Speech Services)
- Gemini API Key
- Node.js (pour le frontend)

## 🚀 Installation

### 1. Cloner le Repository
```bash
git clone https://github.com/votre-username/chatbot-ia-projets.git
cd chatbot-ia-projets
```

### 2. Environnement Virtuel
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 3. Installer les Dépendances
```bash
pip install -r requirements.txt
```

### 4. Configuration des Variables d'Environnement
```bash
cp .env.example .env
# Éditer .env avec vos clés API réelles
```

### 5. Configuration Google Cloud
```bash
# Configurez vos credentials Google Cloud
export GOOGLE_APPLICATION_CREDENTIALS="path/to/your/credentials.json"
```

### 6. Initialiser NLTK
```bash
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet')"
```

### 7. Entraîner le Modèle
```bash
rasa train
```

## ⚙️ Configuration

### Variables d'Environnement

Copiez `.env.example` vers `.env` et configurez :

```bash
# IA et APIs
GEMINI_API_KEY=votre_cle_gemini_api
GOOGLE_APPLICATION_CREDENTIALS=path/to/google-credentials.json

# Rasa Pro
RASA_PRO_LICENSE=votre_licence_rasa_pro

# Base de données
DATABASE_URL=postgresql://user:password@localhost:5432/rasa_pro

# Serveurs
RASA_SERVER_URL=http://localhost:5005
ACTION_SERVER_URL=http://localhost:5055
FLASK_SERVER_URL=http://localhost:8000

# Environnement
ENVIRONMENT=development
DEBUG=true
LOG_LEVEL=INFO
```

### Architecture CALM

#### Flows (`data/flows.yml`)
Contient les flux de conversation pour :
- Analyse de compétences projet
- Évaluation des risques
- Estimation des coûts
- Recommandations personnalisées

#### Patterns (`data/patterns.yml`)
Définit les patterns pour reconnaître :
- Demandes d'analyse de compétences
- Questions sur les risques
- Requêtes d'estimation de coûts

#### Prompts (`prompts/`)
Prompts LLM pour guider :
- L'analyse intelligente des projets
- La génération de recommandations
- L'évaluation contextuelle

## 🔧 Structure du Projet

```
├── __init__.py              # Initialisation du package
├── actions.py               # Actions personnalisées avec IA
├── backend.py               # Serveur Flask + APIs
├── app.js                   # Interface utilisateur
├── data/                    # Données d'entraînement CALM
│   ├── flows.yml           # Flux de conversation métier
│   └── patterns.yml        # Patterns de reconnaissance
├── prompts/                 # Prompts pour l'IA générative
├── static_audio/            # Fichiers audio pour les réponses
├── models/                  # Modèles entraînés (gitignored)
├── .env                     # Variables d'environnement (gitignored)
├── config.yml               # Configuration Rasa avec FlowPolicy
├── domain.yml               # Domaine de l'assistant
├── credentials.yml          # Identifiants des canaux
└── endpoints.yml            # Points de terminaison externes
```

## 🏃‍♂️ Utilisation

### Mode Développement

```bash
# Terminal 1: Action Server
rasa run actions --debug

# Terminal 2: Rasa Server
rasa run --enable-api --cors "*" --debug

# Terminal 3: Backend Flask
python backend.py

# Terminal 4: Test en ligne de commande
rasa shell
```

### Mode Production

```bash
# Démarrer tous les services
rasa run --enable-api --cors "*" --port 5005 &
rasa run actions --port 5055 &
python backend.py &
```

## 🧪 Cas d'Usage

### 1. Prédiction des Compétences
```
Utilisateur: "Quelles compétences sont nécessaires pour un projet de développement mobile ?"
Bot: Analyse le projet et recommande les compétences techniques et soft skills
```

### 2. Évaluation des Risques
```
Utilisateur: "Quels sont les risques pour ce projet de transformation digitale ?"
Bot: Identifie les risques techniques, financiers, temporels et humains
```

### 3. Estimation des Coûts
```
Utilisateur: "Combien coûterait le développement d'une plateforme e-commerce ?"
Bot: Calcule une estimation basée sur les paramètres du projet
```

## 🐳 Déploiement Docker

### Dockerfile
```dockerfile
FROM rasa/rasa:3.6.0-full

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 5005 5055 8000
CMD ["run", "--enable-api", "--cors", "*"]
```

### Docker Compose
```yaml
version: '3.8'
services:
  rasa:
    build: .
    ports:
      - "5005:5005"
    environment:
      - GEMINI_API_KEY=${GEMINI_API_KEY}
  
  actions:
    build: .
    ports:
      - "5055:5055"
    command: ["run", "actions"]
  
  backend:
    build: .
    ports:
      - "8000:8000"
    command: ["python", "backend.py"]
```

## 📊 APIs Disponibles

### Rasa API
- `POST /webhooks/rest/webhook` - Envoyer des messages
- `GET /model` - Informations sur le modèle

### Flask Backend
- `POST /api/chat` - Interface de chat
- `POST /api/voice` - Traitement vocal
- `GET /api/projects` - Analyse de projets
- `POST /api/skills/predict` - Prédiction de compétences
- `POST /api/risks/evaluate` - Évaluation de risques
- `POST /api/costs/estimate` - Estimation de coûts

## 🔍 Dépannage

### Problèmes Courants

**Erreur "Model not found":**
```bash
rasa train --force
```

**Erreur Google Cloud:**
```bash
export GOOGLE_APPLICATION_CREDENTIALS="path/to/credentials.json"
gcloud auth application-default login
```

**Erreur Gemini API:**
```bash
# Vérifiez votre clé API dans .env
# Assurez-vous que l'API Gemini est activée
```

**Erreur FAISS:**
```bash
pip install faiss-cpu  # ou faiss-gpu pour GPU
```

**Erreur NLTK:**
```bash
python -c "import nltk; nltk.download('all')"
```

## 🤝 Contribution

1. Fork le repository
2. Créer une branche feature (`git checkout -b feature/nouvelle-fonctionnalite`)
3. Commit les changements (`git commit -m 'Ajout nouvelle fonctionnalité'`)
4. Push la branche (`git push origin feature/nouvelle-fonctionnalite`)
5. Ouvrir une Pull Request

## 📝 Licence

Ce projet est sous licence - voir le fichier LICENSE pour plus de détails.

## 🏢 Cas d'Usage Entreprise

- **Consultation de Projets** - Analyse préliminaire des besoins projet
- **Gestion des Ressources** - Optimisation de l'allocation des compétences
- **Contrôle des Risques** - Identification proactive des problèmes potentiels
- **Budgétisation Intelligente** - Estimation précise des coûts projet
- **Support Décisionnel** - Aide à la prise de décision stratégique

## 📞 Support

Pour le support technique, créez une issue dans ce repository ou contactez [mohamedamineazouzi49@gmail.com]
