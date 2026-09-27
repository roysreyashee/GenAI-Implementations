from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel , RunnableBranch , RunnableLambda

from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal

load_dotenv()

model1= ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite")
parser = StrOutputParser()

class Feedback(BaseModel):
    sentiment: Literal['positive', 'negative'] = Field(description="Give me the sentiment of the feedback")

parser2 = PydanticOutputParser(pydantic_object=Feedback)

prompt1 = ChatPromptTemplate.from_template(
    'Classify the sentiment of following feedback text into positive or negative \n {feedback} \n {format_instruction}'
).partial(format_instruction=parser2.get_format_instructions())

classifier_chain = prompt1 | model1 | parser2

# result = classifier_chain.invoke({'feedback': 'This is a terrible smartphone'}).sentiment

# print(result)

prompt2 = ChatPromptTemplate.from_template(
      'Write an appropriate response to this positive feedback \n {feedback}'
)

prompt3 = ChatPromptTemplate.from_template(
      'Write an appropriate response to this negative feedback \n {feedback}'
)
branch_chain = RunnableBranch(
    (lambda x:x.sentiment == 'positive', prompt2 | model1 | parser),
    (lambda x:x.sentiment == 'negative', prompt3 | model1 | parser),
    RunnableLambda(lambda x: 'Could not find sentiment')
)

final_chain = classifier_chain | branch_chain

finalresult = final_chain.invoke({'feedback': 'This is a wonderful phone'})
print(finalresult)

final_chain.get_graph().print_ascii()


#### Flow ####

#  +-------------+      
#       | PromptInput |      
#       +-------------+      
#              *             
#              *             
#              *             
#   +--------------------+   
#   | ChatPromptTemplate |   
#   +--------------------+   
#              *             
#              *             
#              *             
# +------------------------+ 
# | ChatGoogleGenerativeAI | 
# +------------------------+ 
#              *             
#              *             
#              *             
#  +----------------------+  
#  | PydanticOutputParser |  
#  +----------------------+  
#              *             
#              *             
#              *             
#         +--------+         
#         | Branch |         
#         +--------+         
#              *             
#              *             
#              *             
#      +--------------+      
#      | BranchOutput |