import path from "node:path";
import { fileURLToPath } from "node:url";
import { MongoClient } from "mongodb";
import {
  ChatGoogleGenerativeAI,
  GoogleGenerativeAIEmbeddings,
} from "@langchain/google-genai";
import { MongoDBAtlasVectorSearch } from "@langchain/mongodb";
import { TextLoader } from "@langchain/classic/document_loaders/fs/text";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";

// ---- __dirname for ESM ----
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// ---- MongoDB native client ----
let mongoClient: MongoClient | null = null;

const getMongoClient = async (): Promise<MongoClient> => {
  if (!mongoClient) {
    mongoClient = new MongoClient(process.env.MONGODB_URI || "");
    await mongoClient.connect();
  }
  return mongoClient;
};

// ---- Google GenAI Embeddings ----
const getEmbeddings = () => {
  if (!process.env.GOOGLE_API_KEY) {
    throw new Error("GOOGLE_API_KEY is not set in .env!");
  }

  return new GoogleGenerativeAIEmbeddings({
    apiKey: process.env.GOOGLE_API_KEY,
    model: "gemini-embedding-001",
  });
};

// ---- Vector Store ----
const getVectorStore = async () => {
  const client = await getMongoClient();
  const collection = client.db("edureach_db").collection("knowledge_docs");

  return new MongoDBAtlasVectorSearch(getEmbeddings(), {
    collection: collection as any,
    indexName: "edureach_vector_index",
    textKey: "text",
    embeddingKey: "embedding",
  });
};

// ============================================
// A) INDEXING — runs ONCE at server startup
// ============================================

export const initializeKnowledgeBase = async (): Promise<void> => {
  const client = await getMongoClient();
  const collection = client.db("edureach_db").collection("knowledge_docs");

  // Check if embeddings already exist
  const docWithEmbedding = await collection.findOne({
    embedding: { $exists: true, $not: { $size: 0 } },
  });

  if (docWithEmbedding) {
    const count = await collection.countDocuments();
    console.log(`Knowledge base ready (${count} chunks with embeddings)`);
    return;
  }

  const existingCount = await collection.countDocuments();

  if (existingCount > 0) {
    console.log(`Found ${existingCount} chunks with empty embeddings — reindexing`);
    await collection.deleteMany({});
  }

  console.log("Indexing knowledge base...");

  const embeddings = getEmbeddings();

  // Test embedding API
  try {
    const test = await embeddings.embedQuery("test");
    console.log(`Embedding API OK (${test.length} dimensions)`);
  } catch (error: any) {
    console.error("Embedding test failed:", error.message);
    throw error;
  }

  // Load document
  const filePath = path.join(
    __dirname,
    "../../knowledge-base/edureach-knowledge.txt"
  );

  const loader = new TextLoader(filePath);
  const docs = await loader.load();

  if (docs.length === 0) {
    throw new Error("Knowledge base file empty");
  }

  const totalCharacters = docs.reduce(
    (sum, doc) => sum + doc.pageContent.length,
    0
  );

  console.log(`Loaded ${totalCharacters} characters`);

  // Split documents
  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 1000,
    chunkOverlap: 200,
  });

  const splits = await splitter.splitDocuments(docs);

  console.log(`Split into ${splits.length} chunks`);

  // Store embeddings
  const vectorStore = new MongoDBAtlasVectorSearch(embeddings, {
    collection: collection as any,
    indexName: "edureach_vector_index",
    textKey: "text",
    embeddingKey: "embedding",
  });

  await vectorStore.addDocuments(splits);

  // Verify embeddings
  const verifyDoc = await collection.findOne({
    embedding: { $exists: true, $not: { $size: 0 } },
  });

  if (
    verifyDoc &&
    Array.isArray(verifyDoc.embedding) &&
    verifyDoc.embedding.length > 0
  ) {
    console.log(
      `${splits.length} chunks stored (${verifyDoc.embedding.length}D embeddings)`
    );
    console.log(
      `Create Atlas Vector Index with numDimensions: ${verifyDoc.embedding.length}`
    );
  } else {
    await collection.deleteMany({});
    throw new Error("Embeddings generation failed.");
  }
};

// ============================================
// B) RAG RESPONSE
// ============================================

export const getRAGResponse = async (question: string): Promise<string> => {
  try {
    const vectorStore = await getVectorStore();

    // Retrieve relevant documents
    const docs = await vectorStore.similaritySearch(question, 3);

    const context = docs.map((doc) => doc.pageContent).join("\n\n");

    const model = new ChatGoogleGenerativeAI({
      model: "gemini-2.5-flash-lite",
      temperature: 0.7,
      apiKey: process.env.GOOGLE_API_KEY,
    });

    const systemPrompt = `
You are EduReach Bot, an AI counselor for EduReach College Hyderabad.

Answer using the knowledge base.

If information is missing say:
"I don't have that information right now. Click Talk to Us to speak with a counselor."

Knowledge Base:
${context}
`;

    const messages = [
      { role: "system", content: systemPrompt },
      { role: "user", content: question },
    ];

    const result = await model.invoke(messages as any);

    const response =
      typeof result.content === "string"
        ? result.content
        : Array.isArray(result.content)
        ? result.content[0]?.text || ""
        : "";

    return response || "Please try again.";
  } catch (error) {
    console.error("RAG Error:", error);
    return "I'm having trouble right now. Please try again.";
  }
};