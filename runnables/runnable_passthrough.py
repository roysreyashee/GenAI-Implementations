from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence , RunnableParallel, RunnablePassthrough
from dotenv import load_dotenv
load_dotenv()


prompt1 = ChatPromptTemplate.from_template(
        'Write a joke about {topic}'
)

model=ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite")

parser = StrOutputParser()

prompt2 = ChatPromptTemplate.from_template(
    'Explain the following joke {topic}'
)

joke_gen_chain= RunnableSequence(prompt1, model, parser)

parallel_chain = RunnableParallel({
    'joke': RunnablePassthrough(),
    'explanation': RunnableSequence(prompt2, model, parser)
})

final_chain = RunnableSequence(joke_gen_chain , parallel_chain)

result= final_chain.invoke({'topic': 'cricket'})

print(result)

final_chain.get_graph().print_ascii()

#### Flow ####

#  +-------------+                     
#                          | PromptInput |                     
#                          +-------------+                     
#                                 *                            
#                                 *                            
#                                 *                            
#                       +--------------------+                 
#                       | ChatPromptTemplate |                 
#                       +--------------------+                 
#                                 *                            
#                                 *                            
#                                 *                            
#                     +------------------------+               
#                     | ChatGoogleGenerativeAI |               
#                     +------------------------+               
#                                 *                            
#                                 *                            
#                                 *                            
#                        +-----------------+                   
#                        | StrOutputParser |                   
#                        +-----------------+                   
#                                 *                            
#                                 *                            
#                                 *                            
#                +---------------------------------+           
#                | Parallel<joke,explanation>Input |           
#                +---------------------------------+           
#                      ****                ****                
#                   ***                        ***             
#                 **                              ***          
#   +--------------------+                           **        
#   | ChatPromptTemplate |                            *        
#   +--------------------+                            *        
#              *                                      *        
#              *                                      *        
#              *                                      *        
# +------------------------+                          *        
# | ChatGoogleGenerativeAI |                          *        
# +------------------------+                          *        
#              *                                      *        
#              *                                      *        
#              *                                      *        
#     +-----------------+                     +-------------+  
#     | StrOutputParser |                     | Passthrough |  
#     +-----------------+                     +-------------+  
#                      ****                ****                
#                          ***          ***                    
#                             **      **                       
#               +----------------------------------+           
#               | Parallel<joke,explanation>Output |           
#               +----------------------------------+ 
