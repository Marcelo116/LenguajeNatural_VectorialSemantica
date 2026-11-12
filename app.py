
import joblib
import spacy
import __main__

from fastapi import FastAPI
from pydantic import BaseModel


nlp = spacy.load(
    "en_core_web_sm",
    disable=["parser", "ner"]
)


def spacy_preprocessor(text):

    doc = nlp(text)

    tokens_limpios = [
        token.lemma_.lower()
        for token in doc
        if not token.is_stop and not token.is_punct
    ]

    return " ".join(tokens_limpios)


__main__.spacy_preprocessor = spacy_preprocessor


app = FastAPI(
    title="API de Análisis de Reseñas",
    description="API para clasificar reseñas de restaurantes",
    version="1.0.0"
)


modelo = joblib.load("modelo_sentimientos.pkl")


class Resena(BaseModel):

    texto: str



@app.get("/")
def inicio():

    return {
        "mensaje": "API de análisis de sentimientos funcionando"
    }


@app.post("/predict")
def predecir(resena: Resena):

    resultado = modelo.predict([resena.texto])

    sentimiento = (
        "Positiva"
        if resultado[0] == 1
        else "Negativa"
    )

    return {
        "resena": resena.texto,
        "prediccion": sentimiento
    }
