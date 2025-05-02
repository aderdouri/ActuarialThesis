# Tarification d'une Police d'Assurance Automobile

Ce projet s’inscrit dans le cadre du mémoire de Master M2 du C.N.A.M.  
**Master Droit, Économie et Gestion, mention Actuariat**

## 📄 Résumé

L’objectif de ce projet est de proposer une méthode de tarification pour une police d’assurance automobile en s’appuyant sur :

- L’historique des sinistres
- Les variables de tarification

Les travaux incluent des analyses statistiques et actuarielles ainsi que la mise en œuvre de modèles prédictifs.

## ⚙ Technologies et Méthodes

Le projet utilise plusieurs frameworks de **Machine Learning**, notamment :

- Le **Gradient Boosting**
- Le **Tweedie Compound Poisson Model** (modèle de Poisson Composé)
- Le **Zero-Inflated Tweedie Model** (version spécialisée pour gérer les zéros excédentaires)

Ces modèles sont appliqués pour modéliser à la fois la fréquence et le coût des sinistres, en vue d’estimer au mieux la prime pure.

## 📂 Contenu du dépôt

- `data/` → jeux de données et prétraitements
- `notebooks/` → notebooks Jupyter pour l’exploration et les tests
- `scripts/` → scripts Python pour l’entraînement et l’évaluation des modèles
- `README.md` → ce fichier
