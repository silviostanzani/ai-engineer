from langchain_community.llms import Ollama
from langchain_community.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from sys import argv
import pypdf

print('1')
# 1. Create the model
llm = Ollama(model='llama3.2')
embeddings = OllamaEmbeddings(model='llama3.2')
#llm = Ollama(model='mistral')
#embeddings = OllamaEmbeddings(model='mistral')
print('2')
# 2. Load the PDF file and create a retriever to be used for providing context
#loader = PyPDFLoader('500perguntasmilho.pdf')
loader = PyPDFLoader('menor500.pdf')
print('3')
pages = loader.load_and_split()
store = DocArrayInMemorySearch.from_documents(pages, embedding=embeddings)
retriever = store.as_retriever()
print('4')
# 3. Create the prompt template
template = """
Answer the question based only on the context provided.

Context: {context}

Question: {question}
"""

prompt = PromptTemplate.from_template(template)

print("prompt")
print(prompt)
print("prompt")

def format_docs(docs):
  print  ('!docs')
  print  (docs)
  print  ('!!!docs')
  return "\n\n".join(doc.page_content for doc in docs)

# 4. Build the chain of operations
chain = (
  {
    'context': retriever | format_docs,
    'question': RunnablePassthrough(),
  }
  | prompt
  | llm
  | StrOutputParser()
)

# 5. Start asking questions and getting answers in a loop
while True:
  question = input('What do you want to learn from the document?\n')
  print()
  print(chain.invoke({'question': question}))
  print()