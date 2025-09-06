from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory

load_dotenv()

modelo = ChatGoogleGenerativeAI(
    model='gemini-2.5-pro',
    temperature=0.5,
)

prompt_sugestao = ChatPromptTemplate.from_messages(
    [
        ("system", "voce eh um guia de viagem especializado em destinos brasileiros. Apresente-se como Sr. Passeios"),
        ("placeholder", "{historico}"),
        ("human", "{query}")
    ]
)

lista_perguntas = [
    "Quero visitar um lugar no Brasil, famoso por praias e cultura. Pode sugerir?",
    "Qual a melhor epoca do ano para ir?"
]

cadeia = prompt_sugestao | modelo | StrOutputParser()

memoria = {}

sessao = "aula_langchain"

def historico_por_sessao(sessao: str):
    if sessao not in memoria:
        memoria[sessao] = InMemoryChatMessageHistory()
    return memoria[sessao]

cadeira_com_memoria = RunnableWithMessageHistory(
    runnable=cadeia,
    get_session_history=historico_por_sessao,
    input_messages_key="query",
    history_messages_key="historico",
)

for pergunta in lista_perguntas:
    resposta = cadeira_com_memoria.invoke(
        {
            "query": pergunta
        },
        config={"session_id": sessao}
    )
    print("Usuario:", pergunta)
    print("IA:", resposta)
