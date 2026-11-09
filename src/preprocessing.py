
import pandas as pd
import re

def clean_text(text):
    """Nettoyer le contenu d'un SMS."""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def load_dataset(path="data/SMSSpamCollection"):
    """Charger et préparer le dataset."""

    df = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=["label", "message"],
        encoding="utf-8"
    )

    df["message"] = df["message"].apply(clean_text)

    df["label"] = df["label"].map({
        "ham": 0,
        "spam": 1
    })

    df = df.dropna()

    return df


if __name__ == "__main__":
    data = load_dataset()

    print("Dataset chargé avec succès !")
    print("\nNombre de messages :", len(data))
    print("\nRépartition des catégories :")
    print(data["label"].value_counts())
    print("\nExemples :")
    print(data.head())
