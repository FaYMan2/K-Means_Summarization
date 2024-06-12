import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { Document } from "langchain/document";
import { PromptTemplate } from "@langchain/core/prompts";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { loadSummarizationChain } from "langchain/chains";
import { distance, min, norm, sum, sumTransformDependencies} from "mathjs"
import skmeans from "skmeans";
import { ChatTogetherAI } from '@langchain/community/chat_models/togetherai';
import { TogetherAIEmbeddings } from "@langchain/community/embeddings/togetherai";
import { setEngine } from "crypto";

const model = new ChatTogetherAI({
    modelName: 'mistralai/Mistral-7B-Instruct-v0.3',
    apiKey: process.env.TO_API_KEY,
    temperature: 0,
});

const embeddings = new TogetherAIEmbeddings({
    apiKey: process.env.TO_API_KEY,
    modelName: 'togethercomputer/m2-bert-80M-8k-retrieval',
});


const loder = new PDFLoader(process.env.FILE_PATH,{
    splitPages:false,parsedItemSeparator: " ",
})
const pages = await loder.load()

const text_ = pages[0].pageContent
const text = text_.replace('\t',' ')

const text_splitter = new RecursiveCharacterTextSplitter({
    chunkSize : 10000,
    chunkOverlap : 3000,
    separators : ["\n\n", "\n", "\t"]
})

const docs = await text_splitter.createDocuments([text])
const vectors = []
let index = 1
for(const doc of docs){
    const embed = await embeddings.embedQuery(doc.pageContent)
    console.log(index,embed.length)
    vectors.push(embed)
    index += 1
}

const num_clusters = 15

const kmeans = skmeans(vectors,num_clusters)
const centroids = kmeans.centroids

const argFact = (compareFn) => (array) => array.map((el, idx) => [el, idx]).reduce(compareFn)[1]
const argMin = argFact((max, el) => (el[0] < max[0] ? el : max))

const selected_indices = []
for(let i =  0;i<num_clusters;i++){
    const distances =  vectors.map((value) => {
                return norm(distance(value, centroids[i])) 
            })
    selected_indices.push(argMin(distances))
}

console.log(selected_indices)
const map_prompt = 
`You will be given a single passage of a judgement. This section will be enclosed in triple hashtags (###)
Your goal is to give a summary of this section so that a reader will have a full understanding of the premise
details.Your response should be at least three paragraphs and fully encompass what was said in the passage.

###${text}###
FULL SUMMARY:
`
const map_prompt_template = new PromptTemplate({
    inputVariables : ['text'],
    template : map_prompt
})

const map_chain = loadSummarizationChain(model,{
    type : 'stuff',
    prompt : map_prompt_template,
})



const selected_docs = selected_indices.map((value) =>{
    const doc = docs[value]
    return doc
})


const summaries = await map_chain.invoke({input_documents : selected_docs})

console.log(summaries)










