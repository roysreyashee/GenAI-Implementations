from langchain_community.document_loaders import WebBaseLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model= ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite")

prompt = ChatPromptTemplate.from_template(
    'Answer the following question - \n {question} from the following text - \n {text}'
)

parser = StrOutputParser()

url = "https://www.flipkart.com/apple-macbook-neo-a18-pro-2026-pro-8-gb-256-gb-ssd-tahoe-mhfh4hn-a/p/itmca8bd5b2e2477?pid=COMHZQX4DAXF5U9M&lid=LSTCOMHZQX4DAXF5U9MJGYTMC&marketplace=FLIPKART&q=Macbook&store=6bo%2Fb5g&srno=s_1_1&otracker=search&otracker1=search&fm=organic&iid=c0ac8a80-823e-447d-b1bf-57a6e1428fb3.COMHZQX4DAXF5U9M.SEARCH&ppt=None&ppn=None&ssid=eqrknetsqo0000001790597804112&qH=86c12cd177a3ce54&ov_redirect=true"

loader = WebBaseLoader(url)

docs = loader.load()
# print(len(docs))

# print(docs[0].page_content)

chain = prompt | model | parser

result = chain.invoke({'question': "What is the price of the product ?" , 'text': docs[0].page_content})

print(result)