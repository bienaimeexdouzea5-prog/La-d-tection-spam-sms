
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from preprocessing import load_dataset


# 1. Charger le dataset
data = load_dataset()

X = data["message"]
y = data["label"]


# 2. Diviser les données en entraînement et test
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# 3. Construire le modèle
model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("classifier", MultinomialNB())
])


# 4. Entraîner le modèle
print("Entraînement du modèle en cours...")

model.fit(X_train, y_train)


# 5. Tester le modèle
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\n===== RÉSULTATS =====")
print("Précision globale (Accuracy) :", accuracy)

print("\nRapport de classification :")
print(classification_report(
    y_test,
    y_pred,
    target_names=["NORMAL", "SPAM"]
))

print("\nMatrice de confusion :")
print(confusion_matrix(y_test, y_pred))


# 6. Sauvegarder le modèle
os.makedirs("models", exist_ok=True)

joblib.dump(model, "models/spam_model.pkl")

print("\nModèle sauvegardé avec succès !")
