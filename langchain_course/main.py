from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

dias = 14
numero_criancas = 4
atividade = "star wars"

prompt_template = PromptTemplate(
    template=f"""
    Crie um roteiro de viagem de {dias} dias,
    para uma familia com {numero_criancas} criancas,
    que gostam de {atividade}
    """
)

prompt = prompt_template.format(
    dias=dias,
    numero_criancas=numero_criancas,
    atividade=atividade,
)

model = ChatGoogleGenerativeAI(
    model='gemini-2.5-pro',
    temperature=0.5,
)

resposta = model.invoke(prompt)

print(resposta.content)