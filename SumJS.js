import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { Document } from "langchain/document";
import { PromptTemplate } from "@langchain/core/prompts";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { loadSummarizationChain } from "langchain/chains";
import { norm } from "mathjs"
import skmeans from "skmeans";
import { ChatTogetherAI } from '@langchain/community/chat_models/togetherai';
import { TogetherAIEmbeddings } from "@langchain/community/embeddings/togetherai";

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

console.log(vectors.length,vectors[0].length)



