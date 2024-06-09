
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains.summarize import load_summarize_chain
import numpy as np
from sklearn.cluster import KMeans
from langchain_community.document_loaders import PyPDFLoader
from langchain_together import ChatTogether
from langchain_together.embeddings import TogetherEmbeddings
from time import time
from langchain_core.prompts import PromptTemplate



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
    closest_index = np.argmin(distances)
    closest_indices.append(closest_index)
    
sorted_indices = sorted(closest_indices)
print(sorted_indices)

selected_docs = [docs[doc] for doc in sorted_indices] 

map_prompt = """
You will be given a single passage of a judgement. This section will be enclosed in triple backticks (```)
Your goal is to give a summary of this section so that a reader will have a full understanding of the premise
details.Your response should be at least three paragraphs and fully encompass what was said in the passage.

```{text}```
FULL SUMMARY:
"""

map_prompt_template = PromptTemplate(template=map_prompt, input_variables=["text"])
map_chain = load_summarize_chain(llm=llm,
                             chain_type="stuff",
                             prompt=map_prompt_template,
                             verbose=True)

summary_list = []

for i,doc in enumerate(selected_docs):
    chunk_summary = map_chain.run([doc])
    summary_list.append(chunk_summary)
    print (f"Summary #{i} (chunk #{sorted_indices[i]}) - Preview: {chunk_summary[:100]} length : {len(chunk_summary)} \n")

summaries = "\n".join(summary_list)

summaries = Document(page_content=summaries)
print (f"Your total summary has {llm.get_num_tokens(summaries.page_content)} tokens")

combine_prompt = """
You will be given a series of summaries from a judgement. The summaries will be enclosed in triple backticks (```)
Your goal is to give a verbose summary of what happened in the story.
The reader should be able to grasp the judgement in great detail.

```{text}```
VERBOSE SUMMARY:
"""
combine_prompt_template = PromptTemplate(template=combine_prompt, input_variables=["text"])

reduce_chain = load_summarize_chain(llm=llm,
                                    chain_type='stuff',
                                    prompt = combine_prompt_template,
                                    verbose=True)

output = reduce_chain.run([summaries])
print(output)