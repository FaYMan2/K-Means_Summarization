from langchain.text_splitter import RecursiveCharacterTextSplitter
import numpy as np
from sklearn.cluster import KMeans
from langchain_together import ChatTogether
from langchain_together.embeddings import TogetherEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from time import time

api_key = '47d5d14ffbfdc4ab5cf873f7086e4fdd7215b8d217bb1f51ed14b1b68fa7ac2f'
llm = ChatTogether(
    api_key=api_key,
    model='mistralai/Mistral-7B-Instruct-v0.2'
)
embeddings = TogetherEmbeddings(model="togethercomputer/m2-bert-80M-8k-retrieval",api_key=api_key)


loader = PyPDFLoader("c:\\Users\\KIIT\\Downloads\\Sita_Soren_vs_Union_Of_India_on_4_March_2024.PDF")
pages = loader.load()
text = ""

for page in pages:
    text += page.page_content
    
text = text.replace('\t',' ')
print(llm.get_num_tokens(text))

text_splitter = RecursiveCharacterTextSplitter(
separators=["\n\n", "\n", "\t"],chunk_size = 10000,chunk_overlap=3000)
docs = text_splitter.create_documents([text])
print(len(docs))
# async embed documents
t1 = time()
vectors = embeddings.embed_documents([x.page_content for x in docs])
t2 = time()
print(f'len : {len(vectors)} shape : {len(vectors[0])} time taken = {abs(t2 - t1)}')

num_clusters = 15
t1 = time()
kmeans = KMeans(n_clusters=num_clusters,random_state=40).fit(vectors)
t2 = time()
print(f'labels = {kmeans.labels_} , time : {abs(t2 - t1)}' )


closest_indices = []


for i in range(num_clusters):
    distances = np.linalg.norm(vectors - kmeans.cluster_centers_[i], axis=1)    
    print(f'distance for centroid - {i} is {distances}') 
    closest_index = np.argmin(distances)
    closest_indices.append(closest_index)