from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain.globals import set_debug
from pydantic import Field, BaseModel
from dotenv import load_dotenv

set_debug(True)

class Destine(BaseModel):
    cidade: str = Field("A cidade recomendada para visitar")
    motivo: str = Field("A motivo pelo qual eh interessante visitar essa cidade")


load_dotenv()

model = ChatGoogleGenerativeAI(
    model='gemini-2.5-pro',
    temperature=0.5,
)

parser = JsonOutputParser(pydantic_object=Destine)

prompt_cidade = PromptTemplate(
    template="""
    Sugira uma cidade dado o meu interesse por {interesse}.
    {output_format}
    """,
    input_variables=['interesse'],
    partial_variables={'output_format': parser.get_format_instructions()}
)

cadeia = prompt_cidade | model | parser

resposta = cadeia.invoke({"interesse": "star wars"})

print(resposta)