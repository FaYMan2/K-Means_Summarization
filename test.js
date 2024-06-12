import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { Document } from "langchain/document";
import { PromptTemplate } from "@langchain/core/prompts";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { loadSummarizationChain } from "langchain/chains";
import { distance, min, norm, sum} from "mathjs"
import skmeans from "skmeans";
import { ChatTogetherAI } from '@langchain/community/chat_models/togetherai';
import { TogetherAIEmbeddings } from "@langchain/community/embeddings/togetherai";

const model = new ChatTogetherAI({
    modelName: 'mistralai/Mistral-7B-Instruct-v0.3',
    apiKey: process.env.TO_API_KEY,
    temperature: 0,
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

const selected_indices = [
    7, 12,  4, 30, 10, 22,
   17, 28, 13, 33,  6, 20,
   18,  8,  2
 ]


const test = selected_indices.map( (value) =>{
    const doc = [docs[value].pageContent]
    console.log(value)
    console.log(doc)
    return doc
})

console.log(test)