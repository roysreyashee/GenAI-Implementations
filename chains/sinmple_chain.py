from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
load_dotenv()

prompt = ChatPromptTemplate.from_template(
    'Generate 5 interesting facts about {topic}'
)

model=ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite")

parser= StrOutputParser()

chain = prompt | model | parser

result = chain.invoke({'topic': 'Movies'})

print(result)

chain.get_graph().print_ascii()