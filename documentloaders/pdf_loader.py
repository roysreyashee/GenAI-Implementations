from langchain_community.document_loaders import PyPDFLoader

loader= PyPDFLoader('cricket_comprehensive_guide.pdf')

docs = loader.load()
print(len(docs))

print(docs[0].page_content)
print(docs[0].metadata)