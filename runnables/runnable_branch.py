from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence , RunnableParallel, RunnableBranch, RunnablePassthrough
from dotenv import load_dotenv
load_dotenv()

prompt1 = ChatPromptTemplate.from_template(
    'Write a detailed report on {topic}'
)

prompt2 = ChatPromptTemplate.from_template(
    'Summarize the following {text}'
)

model= ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite")

parser= StrOutputParser()

report_gen_chain= RunnableSequence(prompt1, model, parser)

branch_chain = RunnableBranch(
    (lambda x: len(x.split())>200 , RunnableSequence(prompt2, model, parser)),
    RunnablePassthrough()
)

final_chain = RunnableSequence(report_gen_chain, branch_chain)

result = final_chain.invoke({'topic': 'Russia Vs Ukraine'})

print(result)


final_chain.get_graph().print_ascii()


###Flow###
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
#     +-----------------+    
#     | StrOutputParser |    
#     +-----------------+    
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
#      +--------------+