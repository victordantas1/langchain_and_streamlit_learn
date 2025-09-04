from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

numero_dias = 7
numero_criancas = 2
atividade = "musica"

prompt = f"Crie um roteiro de viagens, para um periodo de {numero_dias}, para uma famili com {numero_criancas} que busca atividades relacionadas a {atividade}"

model = ChatGoogleGenerativeAI(
    model='gemini-2.5-pro',
    temperature=0.5,
)

resposta = model.invoke(prompt)

print(resposta.content)