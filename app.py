
import gradio as gr
import joblib

# Charger le modèle entraîné
model = joblib.load("models/spam_model.pkl")


# Fonction de prédiction
def predict_sms(message):

    if not message or not message.strip():
        return {"Veuillez saisir un SMS": 1.0}

    prediction = model.predict([message])[0]
    probabilities = model.predict_proba([message])[0]

    if prediction == 1:
        return {
            "SPAM - Message indésirable": float(probabilities[1]),
            "NORMAL - Message légitime": float(probabilities[0])
        }

    return {
        "NORMAL - Message légitime": float(probabilities[0]),
        "SPAM - Message indésirable": float(probabilities[1])
    }


# Créer l'interface web
interface = gr.Interface(
    fn=predict_sms,
    inputs=gr.Textbox(
        lines=4,
        placeholder="Écrivez ou collez votre SMS ici...",
        label="Votre SMS"
    ),
    outputs=gr.Label(label="Résultat de la détection"),
    title="Détection des SMS indésirables",
    description="Cette application utilise le Machine Learning pour détecter les SMS SPAM et les messages normaux.",
    examples=[
        ["Congratulations! You have won a free prize. Call now!"],
        ["Hello, how are you? See you tomorrow."]
    ]
)


# Lancer l'application
if __name__ == "__main__":
    interface.launch()
