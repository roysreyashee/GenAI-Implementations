from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool

from langchain_core.messages import HumanMessage

import requests

##tool create

@tool
def multiply(a:int , b: int) -> int :
    """Given two numbers this tool returns their product """
    return a * b 

print(multiply.invoke({"a": 4, "b": 5}))
print(multiply.args)

##tool binding 
llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite")

llm_with_tools = llm.bind_tools([multiply])

# print(llm_with_tools)

res = llm_with_tools.invoke('Can you multiply 9 with 6?').tool_calls[0]
print(res)
print(multiply.invoke(res))

res.get_graph().print_ascii()

