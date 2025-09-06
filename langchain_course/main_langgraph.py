from typing import TypedDict, Literal

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

modelo = ChatGoogleGenerativeAI(
    model='gemini-2.5-pro',
    temperature=0.5,
)

prompt_consultor_praia = ChatPromptTemplate(
    [
        ("system", "Apresente-se como Sra Praia. Voce eh uma especialista em viagens com destinos para praias"),
        ("human", "{query}")
    ]
)

prompt_consultor_montanha= ChatPromptTemplate(
    [
        ("system", "Apresente-se como Sra Montanha. Voce eh uma especialista em viagens com destinos para montanhas e atividades radicais"),
        ("human", "{query}")
    ]
)

cadeia_praia = prompt_consultor_praia | modelo | StrOutputParser()
cadeia_montanha = prompt_consultor_montanha | modelo | StrOutputParser()

prompt_roteador = ChatPromptTemplate(
    [
        ("system", "Responda apenas com 'praia' ou com 'montanha'"),
        ("human", "{query}")
    ]
)

class Rota(TypedDict):
    destino: Literal["praia", "montanha"]

roteador = prompt_roteador | modelo.with_structured_output(Rota)

def response(pergunta: str):
    rota = roteador.invoke({"query": pergunta})["destino"]
    print(rota)
    if rota == "praia":
        return cadeia_praia.invoke({"query": pergunta})
    else:
        return cadeia_montanha.invoke({"query": pergunta})

print(response("Quero surfar por lugares quentes"))
