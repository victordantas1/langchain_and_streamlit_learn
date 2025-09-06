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

class Restaurants(BaseModel):
    cidade: str = Field("A cidade recomendada para visitar")
    restaurantes: str = Field("Restaurantes recomendados na cidade")

load_dotenv()

model = ChatGoogleGenerativeAI(
    model='gemini-2.5-pro',
    temperature=0.5,
)

destine_parser = JsonOutputParser(pydantic_object=Destine)
restaurant_parser = JsonOutputParser(pydantic_object=Restaurants)

prompt_cidade = PromptTemplate(
    template="""
    Sugira uma cidade dado o meu interesse por {interesse}.
    {output_format}
    """,
    input_variables=['interesse'],
    partial_variables={'output_format': destine_parser.get_format_instructions()}
)

prompt_restaurantes = PromptTemplate(
    template="""
    Sugira restaurantes populares entre locais em {cidade}.
    {output_format}
    """,
    partial_variables={'output_format': restaurant_parser.get_format_instructions()}
)

prompt_cultural = PromptTemplate(
    template = "Sugira atividades e locais culturais em {cidade}."
)

cadeia_1 = prompt_cidade | model | destine_parser
cadeia_2 = prompt_restaurantes | model | restaurant_parser
cadeia_3 = prompt_cultural | model | StrOutputParser

cadeia = cadeia_1 | cadeia_2 | cadeia_3

resposta = cadeia.invoke({"interesse": "star wars"})

print(resposta)