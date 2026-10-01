from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_core.tools import InjectedToolArg
from typing import Annotated
from langchain_core.messages import HumanMessage
import requests
#tool create 

@tool
def get_conversion_factor(base_currency: str, target_currency: str) -> float:
    """
    This function fetches the currency conversion factor between a given base currency and a target currency
    """
    url=f"https://v6.exchangerate-api.com/v6/161602639785fb808bdc3d6a/pair/{base_currency}/{target_currency}"

    response = requests.get(url)
    return response.json()

get_conversion_factor.invoke({'base_currency': 'USD' , 'target_currency': 'INR'})



###Second tool
@tool
def convert_currency(base_currency_val: int, conversion_rate: float) -> float:
    """
    Given a currency conversion rate this function calculates the target currency value from a given base currency value
    """

    return base_currency_val * conversion_rate

convert_currency.invoke({'base_currency_val': 10, 'conversion_rate': 96.01 })



#tool binding 
llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite")

llm_with_tools= llm.bind_tools([get_conversion_factor, convert_currency])

#tool execution
messages= [HumanMessage('What is the conversion factor between usd and inr . Based on that can you convert 10 usd to inr')]

print(messages)

ai_message = llm_with_tools.invoke(messages)
messages.append(ai_message)
print("Inital LLM decision: " ,ai_message.tool_calls)

tools_map = {
    "get_conversion_factor": get_conversion_factor,
    "convert_currency": convert_currency
}

for tool_call in ai_message.tool_calls:
    # A. Execute the first tool (get_conversion_factor)
    tool_name = tool_call["name"]
    selected_tool = tools_map[tool_name]
    
    # Run get_conversion_factor to fetch live API data
    tool_output = selected_tool.invoke(tool_call["args"])
    print(f"\n[Step 1] Executed {tool_name}. Output received.")
    
    # B. Extract the hidden field value (conversion_rate) from the API response
    # Exchangerate-API stores the rate in the "conversion_rate" key
    extracted_rate = tool_output.get("conversion_rate")
    
    # C. Manually prepare arguments for the second tool
    # We copy the model's intent or hardcode the expected payload structure 
    second_tool_args = {
        "base_currency_val": 30,                 # From your prompt
        "conversion_rate": extracted_rate       # Injected at runtime!
    }
    
    # D. Execute the second tool with the runtime injected argument
    final_val = convert_currency.invoke(second_tool_args)
    print(f"[Step 2] Executed convert_currency with injected rate ({extracted_rate}).")
    print(f"Final Calculation Result: {final_val}")