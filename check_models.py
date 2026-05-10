import google.generativeai as genai
import os
from dotenv import load_dotenv

# Charger la clé
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)

print("Vérification des modèles disponibles...")
try:
    available_models = genai.list_models()
    for m in available_models:
        if 'generateContent' in m.supported_generation_methods:
            print(f"✅ Modèle trouvé : {m.name}")
except Exception as e:
    print(f"❌ Erreur lors de la vérification : {e}")
    print("Vérifiez que votre clé API est correcte et que l'API 'Generative Language' est activée dans Google Cloud Console.")