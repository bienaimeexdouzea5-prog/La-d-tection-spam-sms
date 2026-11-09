
import joblib

# Charger le modèle entraîné
model = joblib.load("models/spam_model.pkl")


def predict_sms(message):
    prediction = model.predict([message])[0]

    if prediction == 1:
        return "SPAM - Message indésirable"
    else:
        return "NORMAL - Message légitime"


# Tester avec quelques SMS
if __name__ == "__main__":

    sms1 = "Congratulations! You have won a free prize. Call now!"
    sms2 = "Hello, how are you? See you tomorrow."

    print("SMS 1 :", sms1)
    print("Résultat :", predict_sms(sms1))

    print("\nSMS 2 :", sms2)
    print("Résultat :", predict_sms(sms2))
