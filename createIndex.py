from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
#from langchain.vectorstores import FAISS
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings





def split_paragraphs(rawText):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
    )

    return  text_splitter.split_text(rawText)

def load_pdfs(pdfs):
    text_chunks = []

    for pdf in pdfs:
        #print('p1')
        reader = PdfReader(pdf)
        for page in reader.pages:
            #print('p2')
            raw = page.extract_text()
            #print(raw)
            chunks = split_paragraphs(raw)
            #print(chunks)
            text_chunks += chunks
    return text_chunks

def obtainData():

    #list_of_pdfs = ["500perguntasmilho.pdf"]
    #list_of_pdfs = ["menor500.pdf"]
    
    list_of_pdfs = ["menor2.pdf"]
    
    text_chunks = load_pdfs(list_of_pdfs)

    #print(text_chunks)
    return(text_chunks)

def search(query, k, indext, encodert, datat):
    query_vector = encodert.encode([query])
    top_k = indext.search(query_vector, k)
    #print(top_k)
    return [
        datat[_id] for _id in top_k[1][0]
    ]


encoder = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')

#print('debug 1')

data= obtainData()
#print(data)
encoded_data = encoder.encode(data)
#print(encoded_data)

#print('debug 2')
# IndexFlatIP: Flat inner product (for small datasets)
# IndexIDMap: store document ids in the index as well
index = faiss.IndexIDMap(faiss.IndexFlatIP(768))

index.add_with_ids(encoded_data, np.arange(len(data)))

path = './faiss.index'

# Save index
faiss.write_index(index, path)

#ret = search("Como o clima influencia a cultura do milho?", 2, index, encoder, data)
'''
ret = search("O que é graus-dia?", 2, index, encoder, data)

print('!!!ret')
print(ret[0])
print('!ret')
print(ret[1])

pathn = './faiss.index'
indexn = faiss.read_index(pathn)

ret = search("o milho é uma planta C4?", 2, index, encoder, data)
print('!!!ret')
print(ret)
print('!ret')
'''


