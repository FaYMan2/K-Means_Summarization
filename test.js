import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { Document } from "langchain/document";
import { PromptTemplate } from "@langchain/core/prompts";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { MapReduceDocumentsChain, StuffDocumentsChain, loadSummarizationChain } from "langchain/chains";
import { distance, min, norm, sum} from "mathjs"
import skmeans from "skmeans";
import { ChatTogetherAI } from '@langchain/community/chat_models/togetherai';
import { TogetherAIEmbeddings } from "@langchain/community/embeddings/togetherai";

const model = new ChatTogetherAI({
    modelName: 'google/gemma-7b-it',
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
console.log(docs.length)

const selected_indices = [
    7, 12,  4, 30, 10, 22,
   17, 28, 13, 33,  6, 20,
   18,  8,  2
 ]
 const map_prompt = 
 `You will be given a single passage of a judgement. This section will be enclosed in triple hashtags (###)
 Your goal is to give a summary of this section so that a reader will have a full understanding of the premise
 details.Your response should be at least three paragraphs and fully encompass what was said in the passage.
 
 ###${text}###
 FULL SUMMARY:
 `

 const reduce_prompt = `You will be given a series of summaries from a judgement. The summaries will be enclosed in triple hashtags (###)
    Your goal is to give a verbose summary of what happened in the story.
    The reader should be able to grasp the judgement in great detail.
    
    ${text}
    VERBOSE SUMMARY:
    `
 const map_prompt_template = new PromptTemplate({
     inputVariables : ['text'],
     template : map_prompt
 })

 const reduce_prompt_template = new PromptTemplate({
    inputVariables : ['text'],
    template : reduce_prompt
 })

 const selected_docs =[]
 for(const doc in selected_indices){
    selected_docs.push(docs[doc])
 }


 const map_reduce_chain = loadSummarizationChain(model,{
    type : 'map_reduce',    
    combineMapPrompt : map_prompt_template,
    combinePrompt : reduce_prompt_template
 })


 const summary = map_reduce_chain.invoke({input_documents : selected_docs})
console.log(summary)