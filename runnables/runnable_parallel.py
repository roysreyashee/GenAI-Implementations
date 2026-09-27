from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence , RunnableParallel
from dotenv import load_dotenv
load_dotenv()

prompt1 = ChatPromptTemplate.from_template(
        'Generate a tweet about {topic}'
)

prompt2 = ChatPromptTemplate.from_template(
    'Generate a LinkedIn post about {topic}'
)

model= ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite")

parser= StrOutputParser()

parallel_chain= RunnableParallel({
    'tweet': RunnableSequence(prompt1, model, parser),
    'linkedin': RunnableSequence(prompt2, model, parser)
})

result = parallel_chain.invoke({'topic': 'Langchain'})

print(result)

