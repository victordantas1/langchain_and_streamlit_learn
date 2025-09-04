from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

prompt_cidade = PromptTemplate(
    template="""
    Sugira uma cidade dado o meu interesse por {interesse}.
    """,
    input_variables=['interesse'],
)

model = ChatGoogleGenerativeAI(
    model='gemini-2.5-pro',
    temperature=0.5,
)

cadeia = prompt_cidade | model | StrOutputParser()

resposta = cadeia.invoke({"interesse": "star wars"})

print(resposta)