from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
load_dotenv()

prompt1 = ChatPromptTemplate.from_template(
    'Generate 5 interesting facts about {topic}'
)

prompt2 = ChatPromptTemplate.from_template(
    'Generate 5 pointer summary from the following {text}'
)

model=ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite")

parser=StrOutputParser()

### Flow ---> PromptInput ---> ChatPromptTemplate ---> ChatModel ---> OutputParser ---> ParsedOutput ---> ChatPromptTemplate ---> ChatModel ---> OutputParser ---> FinalOutput
chain = prompt1 | model | parser | prompt2 | model | parser

result = chain.invoke({'topic': 'Unemployment in India'})

print(result)

chain.get_graph().print_ascii()