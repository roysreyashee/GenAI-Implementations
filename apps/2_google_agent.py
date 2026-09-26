from dotenv import load_dotenv
load_dotenv()

from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent

llm=ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite")
search= GoogleSerperAPIWrapper()

agent=create_agent(
    model=llm,
    tools=[search.run],
    system_prompt="You are a agent and can search any question on google"
)

while True:
    query = input("User: ")
    if query.lower() == 'quit' :
        print("GoodBye!")
        break

    response = agent.invoke({"messages":[{"role": "user", "content":query}]})
    print("AI:", response["messages"][-1])
        
    