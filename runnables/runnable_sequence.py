from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence
from dotenv import load_dotenv
load_dotenv()

prompt = ChatPromptTemplate.from_template(
    'Write a joke about {topic}'
)

model=ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite")
parser= StrOutputParser()

chain = RunnableSequence(prompt , model , parser )

result = chain.invoke({'topic': 'Me'})

print(result)

chain.get_graph().print_ascii()