// 1. Import necessary modules and libraries
import { OpenAI } from "langchain/llms/openai";
import { RetrievalQAChain } from 'langchain/chains';
import { HNSWLib } from "langchain/vectorstores/hnswlib";
import { OpenAIEmbeddings } from 'langchain/embeddings/openai';
import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter';
import * as fs from 'fs';
import * as dotenv from 'dotenv';
import express from 'express';
import multer from "multer";
import cors from "cors";

const port = 4000

const app = express()
app.use(express.json());
dotenv.config();

app.use(cors());


const txtFilename = "doc";
const txtPath = `./documents/${txtFilename}.txt`;
const VECTOR_STORE_PATH = `${txtFilename}.index`;



export const runWithEmbeddings = async (question) => {

    // Initialize the OpenAI model with an empty configuration object
    const model = new OpenAI({});
    let vectorStore;
    if (fs.existsSync(VECTOR_STORE_PATH)) {
        console.log('Vector Exists..');
        vectorStore = await HNSWLib.load(VECTOR_STORE_PATH, new OpenAIEmbeddings());
    } else {
        const text = fs.readFileSync(txtPath, 'utf8');
        const textSplitter = new RecursiveCharacterTextSplitter({ chunkSize: 1000 });
        const docs = await textSplitter.createDocuments([text]);
        vectorStore = await HNSWLib.fromDocuments(docs, new OpenAIEmbeddings());
        await vectorStore.save(VECTOR_STORE_PATH);
    }

    const chain = RetrievalQAChain.fromLLM(model, vectorStore.asRetriever());

    const res = await chain.call({
        query: question,
    });


    return { res };
};


// Set up the storage configuration for Multer
const storage = multer.diskStorage({
    destination: (req, file, cb) => {
        cb(null, 'documents/'); // Directory where the uploaded files will be stored
    },
    filename: (req, file, cb) => {
        cb(null, 'doc.txt'); // Set the filename to 'doc.txt'
    },
});

const upload = multer({ storage: storage });



// API Routes 
app.get('/', (req, res) => {
    res.send('Hello World!!!')
})

app.post('/api/v1/question', async (req, res) => {
    let answer = await runWithEmbeddings(req.body.question);
    res.send(answer)
})

// API endpoint for Removing file from documents folder
app.post('/api/v1/removeDir', async (req, res) => {

    const folderPath = './doc.index';

    fs.rm(folderPath, { recursive: true }, (err) => {
        if (err) {
            res.send(`Error while removing folder: ${err}`);
        } else {
            res.send('Folder removed successfully.');

            // runWithEmbeddings()
        }
    });
})

// API endpoint for file upload
app.post('/api/v1/docUpdate', upload.single('file'), async (req, res) => {
    try {
        if (!req.file) {
            return res.status(400).send('No file uploaded.');
        }

        // File uploaded successfully
        return res.status(200).send('File uploaded!');
        runWithEmbeddings()
    } catch (error) {
        console.error(error);
        return res.status(500).send('Internal Server Error');
    }
})





app.listen(port, () => {
    console.log(`Example app listening on port ${port}`)
})