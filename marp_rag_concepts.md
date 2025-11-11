---
marp: true
theme: charcoal-lb
paginate: true
---

<!-- _class: lead -->

# Understanding RAG
## Retrieval-Augmented Generation

**A Complete Guide to Building AI Question-Answering Systems**

---

## What We'll Cover Today

1. What is RAG?
2. Why do we need it?
3. The RAG pipeline
4. Core components deep dive

---

## What We'll Cover Today (cont.)

5. Real-world applications
6. Common challenges & best practices
7. Your project overview

---

## What is RAG?

**RAG** = **R**etrieval-**A**ugmented **G**eneration

A technique that combines:
- **Retrieval**: Finding relevant information from documents
- **Augmented**: Enhancing with additional context
- **Generation**: Creating answers using an AI model

---

## The Open-Book Exam Analogy

### Without RAG 🚫
**Closed-book exam**
- AI only knows training data
- No access to new info
- May hallucinate answers
- No source citations

---

## The Open-Book Exam Analogy

### With RAG ✅
**Open-book exam**
- AI can look up documents
- Access to current data
- Grounded in facts
- Provides sources

---

## How RAG Works (Simple)

```
1. User asks a question
2. System searches documents
3. Retrieves relevant passages
4. Feeds passages to LLM as context
5. LLM generates informed answer
```

**Result**: Accurate, verifiable answers with citations!

---

## Why Do We Need RAG?

### Problem 1: LLMs Don't Know Everything

- Training data has cutoff dates
- No access to private documents
- Can't answer about proprietary data
- Information becomes outdated

**Example**: A 2023 LLM doesn't know 2024 events

---

## Why Do We Need RAG?

### Problem 2: Hallucinations

When LLMs don't know something, they make things up!

**Without RAG:**
> Q: "What's our return policy?"
> A: *Invents a generic policy*

---

## Why Do We Need RAG?

### Problem 2: Hallucinations (cont.)

**With RAG:**
> Q: "What's our return policy?"
> A: *Reads actual policy document* ✅

**Result**: Accurate answers grounded in your actual documents!

---

## Why Do We Need RAG?

### Problem 3: No Source Attribution

**Without RAG**:
- Can't verify where info came from
- No transparency
- Trust issues

---

## Why Do We Need RAG?

### Problem 3: No Source Attribution (cont.)

**With RAG**:
- ✅ Direct citations to documents
- ✅ Verifiable answers
- ✅ Transparent process
- ✅ Auditable sources

---

## The RAG Solution

RAG solves these problems by:

- ✅ Giving LLMs access to up-to-date information
- ✅ Grounding answers in actual documents
- ✅ Providing source citations

---

## The RAG Solution (cont.)

- ✅ Reducing hallucinations
- ✅ Enabling domain-specific knowledge

**Let's see how it works!**

---

## The RAG Pipeline (High-Level)

```
User Question
    ↓
Search Documents
    ↓
Retrieve Context
    ↓
LLM Generation
    ↓
Answer + Sources
```

---

## The RAG Pipeline (Detailed)

**Two Phases:**

1. **Indexing Phase** (done once)
   - Load documents → Chunk text → Create embeddings → Store in DB

2. **Query Phase** (every question)
   - Embed question → Search DB → Retrieve chunks → Build prompt → Generate answer

---

## Indexing Phase

```
┌─────────────────┐
│  Your Documents │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Text Chunking  │ ← Split into pieces
└────────┬────────┘
```

---

## Indexing Phase (cont.)

```
┌─────────────────┐
│  Text Chunking  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Embeddings     │ ← Convert to vectors
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Vector DB      │ ← Store for searching
└─────────────────┘
```

---

## Query Phase

```
┌─────────────────┐
│  User Question  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Embed Question │ ← Same format as docs
└────────┬────────┘
```

---

## Query Phase (cont.)

```
┌─────────────────┐
│  Embed Question │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Search Vector  │ ← Find similar chunks
│      DB         │
└────────┬────────┘
```

---

## Query Phase (cont.)

```
┌─────────────────┐
│  Search Vector  │
│      DB         │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  LLM + Context  │ ← Generate answer
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Answer         │
└─────────────────┘
```

---

<!-- _class: lead -->

# Core Components
## Let's dive into the 7 key parts

---

## Component 1: Document Loading

**What**: Reading documents from files

**Why**: Get your data into the system

**How**:
- Read files (PDF, TXT, DOCX, etc.)
- Extract text content
- Preserve metadata (filename, date, author)

---

## Component 1: Document Loading

```python
documents = load_documents("./docs/")
```

**In your project**: This is TODO #1!

---

## Component 2: Text Chunking

**What**: Splitting long documents into smaller pieces

**Why**:
- Embedding models have token limits
- Smaller chunks = more precise retrieval
- LLMs have context window limits

---

## Chunking Strategies

| Strategy | Pros | Cons |
|----------|------|------|
| **Fixed-Size** | Simple, predictable | May break sentences |
| **Semantic** | Preserves meaning | More complex |
| **Overlapping** | Keeps context | Some redundancy |

---

## Chunking Strategies (cont.)

**Best Practice**: 300-500 characters with 10-20% overlap

**In your project**: This is TODO #2!

---

## Chunking Example

**Document**:
"The cat sat on the mat. The dog ran in the yard. The bird flew away."

**Sentence-based chunks**:
- Chunk 1: "The cat sat on the mat."
- Chunk 2: "The dog ran in the yard."
- Chunk 3: "The bird flew away."

---

## Chunking Example (cont.)

**Result**: Each chunk is searchable independently!

**Why this matters**: Better retrieval precision

---

## Chunk Size Trade-offs

| Size | Characters | Pros | Cons |
|------|-----------|------|------|
| **Small** | 100-300 | Precise | Less context |
| **Medium** | 300-700 | Balanced ⭐ | Standard |
| **Large** | 700-1500 | More context | Less precise |

---

## Chunk Size Trade-offs (cont.)

**Recommendation**: Start with medium, experiment!

**Key insight**: There's no perfect size - it depends on your documents

---

## Component 3: Embeddings

**What**: Converting text into numbers (vectors)

**Why**:
- Computers can't compare meaning directly
- Numbers can be compared mathematically
- Similar meanings → similar vectors

---

## The Magic of Embeddings

Imagine words as points in space:

```
      Animal ────────────────→

  car ·                    · dog

         · house  · home

  plane·                   · cat

      ← ────────────────────
```

**Similar concepts cluster together!**

---

## Real Embedding Example

```
"king"   → [0.2, 0.8, 0.3, ..., 0.1]  (384 numbers)
"queen"  → [0.3, 0.7, 0.3, ..., 0.2]  (similar!)
"car"    → [0.7, 0.1, 0.8, ..., 0.4]  (different)
```

**Key Insight**: The model learned that "king" and "queen" are related!

---

## How Embeddings Work

**Training** (already done for you):
- Neural network trained on millions of texts
- Learns that "king" and "queen" appear in similar contexts
- Encodes that similarity in vectors

---

## How Embeddings Work (cont.)

**Using** (what you do):
- Pass text through the model
- Get back a vector of numbers
- Each dimension captures meaning

**In your project**: Pre-built with SentenceTransformer!

---

## Popular Embedding Models

| Model | Dimensions | Best For |
|-------|-----------|----------|
| `all-MiniLM-L6-v2` | 384 | Fast, efficient ⭐ |
| `all-mpnet-base-v2` | 768 | More accurate |
| `instructor-large` | 768 | Specialized retrieval |

**Your Project**: Uses `all-MiniLM-L6-v2`

---

## Component 4: Vector Database

**What**: Database optimized for storing & searching vectors

**Why**:
- Traditional DBs search by exact matches
- Vector DBs search by similarity
- Much faster than manual comparison

---

## Component 4: Vector Database (cont.)

**Popular Options**:
- ChromaDB ⭐ (your project)
- Pinecone
- Weaviate
- Qdrant

---

## How Vector Search Works

**Cosine Similarity** (most common):
- Measures angle between vectors
- Range: -1 (opposite) to 1 (identical)
- Focuses on direction, not magnitude

---

## How Vector Search Works (cont.)

```python
query_vector = [0.5, 0.8, 0.3]
doc1_vector  = [0.6, 0.7, 0.4]  # Similar!
doc2_vector  = [0.1, 0.2, 0.9]  # Different

similarity(query, doc1) = 0.95  # High ✅
similarity(query, doc2) = 0.43  # Low ❌
```

---

## The Search Process

1. User asks: **"What are parking rules?"**
2. Convert question to vector: `[0.3, 0.7, 0.2, ...]`
3. Compare to all document vectors
4. Return top-K most similar (e.g., top 3)

---

## The Search Process (cont.)

**Why It's Fast**: Uses approximate nearest neighbor algorithms

**Speed**: Can search millions of vectors in milliseconds!

---

## Component 5: Retrieval

**Key Concept**: Top-K Retrieval

- **K** = number of results to return
- Typical values: 3-10
- Trade-off: More context vs. noise

---

## Component 5: Retrieval (cont.)

```python
results = vector_db.search(question_embedding, k=3)

# Results ranked by similarity:
# 1. "Students must park..." (0.92)
# 2. "Parking permits..." (0.87)
# 3. "Visitor parking..." (0.81)
```

---

## Component 6: Prompt Construction

**What**: Building the input for the LLM

**Structure**:
```
System Prompt
    ↓
Retrieved Context (chunks 1-3)
    ↓
User Question
    ↓
Instructions
```

---

## Example RAG Prompt (Part 1)

```
You are a helpful assistant.

Context:
---
[From School_Parking_Rules.txt]:
Students must park in designated lots.
Permits cost $50/semester.
---
```

---

## Example RAG Prompt (Part 2)

```
Question: What are parking rules for students?

Answer based only on the context above.
```

**Notice**: We explicitly tell it to use only the context!

---

## Prompt Engineering Tips

✅ Be explicit: "Answer ONLY based on context"

✅ Request citations: "Include source references"

✅ Set the tone: formal, casual, technical

---

## Prompt Engineering Tips (cont.)

✅ Specify format: bullets, paragraphs, etc.

**Good prompts = Better answers!**

**In your project**: This is part of TODO #4!

---

## Component 7: LLM Generation

**What**: The language model creates the answer

**How**:
1. Receives prompt (context + question)
2. Processes using patterns from training
3. Generates coherent answer

---

## Component 7: LLM Generation (cont.)

**Key Parameters**:
- Temperature
- Max tokens
- Top-p

---

## LLM Parameters

**Temperature** (0.0 - 2.0):
- **Low (0.0-0.3)**: Deterministic, factual ⭐
- **Medium (0.5-0.8)**: Balanced
- **High (0.9-2.0)**: Creative, unpredictable

---

## LLM Parameters (cont.)

**Max Tokens**: Limits response length

**For RAG**: Use low temperature for factual answers!

**In your project**: Using Mistral 7B via Ollama

---

## Example RAG Response

```
Based on the parking rules document:

Students must:
1. Park in designated student lots only
2. Display parking permit on dashboard
3. Purchase permits for $50 per semester
```

---

## Example RAG Response (cont.)

```
Visitors may use free parking near the
main entrance for up to 2 hours.

Source: School_Parking_Rules.txt
```

✅ Accurate ✅ Cited ✅ Verifiable

---

<!-- _class: lead -->

# Real-World Applications
## Where is RAG used?

---

## Application 1: Customer Support

**Customer Support Chatbots**
- Answer questions from knowledge base
- Cite company policies
- Provide accurate product information

**Example**: "What's your warranty policy?"

---

## Application 2: Legal Analysis

**Legal Document Analysis**
- Search case law and statutes
- Find relevant precedents
- Summarize complex legal documents

**Example**: Finding similar court cases

---

## Application 3: Medical Systems

**Medical Information Systems**
- Query medical literature
- Retrieve treatment guidelines
- Access research papers

**Example**: Evidence-based treatment recommendations

---

## Application 4: Company Knowledge

**Internal Company Knowledge**
- Employee handbook Q&A
- Technical documentation search
- Onboarding assistance

**Example**: "How do I submit expenses?"

---

## Application 5: Education

**Educational Tools**
- Textbook Q&A systems
- Research paper search
- Study guides

**Your turn**: You'll build one in this project!

---

<!-- _class: lead -->

# Common Challenges
## And how to solve them

---

## Challenge 1: Chunking Issues

**Problem**: Splitting important info across chunks

**Solution**:
- Use overlapping chunks
- Experiment with different chunk sizes
- Consider semantic chunking

---

## Challenge 2: Irrelevant Retrievals

**Problem**: Retrieved chunks don't answer question

**Solution**:
- Tune similarity threshold
- Use better embedding models
- Implement query rewriting

---

## Challenge 3: Context Window Limits

**Problem**: Can't fit all chunks in LLM

**Solution**:
- Retrieve fewer chunks (reduce K)
- Use smaller chunks
- Implement re-ranking

---

## Challenge 4: Hallucinations

**Problem**: LLM still makes things up

**Solution**:
- Explicit prompt instructions
- Post-generation verification
- Add confidence scores

---

## Challenge 5: Slow Response Times

**Problem**: System is too slow

**Solution**:
- Cache embeddings
- Optimize vector search
- Use faster embedding models

---

<!-- _class: lead -->

# Best Practices
## How to build better RAG systems

---

## Best Practices: Document Prep

✅ **Clean your data**
- Remove headers/footers
- Fix encoding issues
- Standardize formatting

---

## Best Practices: Document Prep (cont.)

✅ **Add metadata**
- Source filename, date, author
- Section/chapter information

✅ **Structure matters**
- Clear section breaks
- Consistent formatting
- Meaningful headings

---

## Best Practices: Chunking

✅ **Start with 300-500 characters**
- Good balance for most use cases
- Adjust based on testing

✅ **Use 10-20% overlap**
- Preserves context
- Not too much redundancy

---

## Best Practices: Chunking (cont.)

✅ **Test different strategies**
- Experiment with sizes
- Measure retrieval quality
- Optimize for your domain

**Key**: There's no one-size-fits-all!

---

## Best Practices: Retrieval

✅ **Tune K (number of results)**
- Start with 3-5
- More isn't always better
- Balance context and noise

---

## Best Practices: Retrieval (cont.)

✅ **Set similarity thresholds**
- Filter low-relevance results
- Reduce hallucinations

✅ **Add metadata filters**
- Filter by date, category, etc.

---

## Best Practices: Evaluation

✅ **Create test questions**
- Cover different topics
- Include edge cases
- Mix easy and hard

---

## Best Practices: Evaluation (cont.)

✅ **Measure performance**
- Answer accuracy
- Retrieval precision
- Response time

✅ **Iterate and improve**
- Analyze failures
- Adjust parameters

---

<!-- _class: lead -->

# RAG vs. Alternatives
## How does it compare?

---

## RAG vs. Fine-Tuning

| Aspect | RAG | Fine-Tuning |
|--------|-----|-------------|
| **Cost** | Low ✅ | High 💰 |
| **Updates** | Easy ✅ | Hard 🔧 |

---

## RAG vs. Fine-Tuning (cont.)

| Aspect | RAG | Fine-Tuning |
|--------|-----|-------------|
| **Sources** | Citations ✅ | None ❌ |
| **Flexibility** | Very flexible ✅ | Locked ❌ |

**RAG is better for dynamic knowledge!**

---

## RAG vs. Long Context LLMs

| Aspect | RAG | Long Context |
|--------|-----|--------------|
| **Scale** | Millions of docs ✅ | Limited ❌ |
| **Speed** | Fast ✅ | Slow ❌ |

---

## RAG vs. Long Context LLMs (cont.)

| Aspect | RAG | Long Context |
|--------|-----|--------------|
| **Cost/query** | Lower ✅ | Higher 💰 |
| **Relevance** | Selective ✅ | All context |

**Best approach**: Hybrid! Combine both techniques.

---

<!-- _class: lead -->

# Your Group Project
## Time to build!

---

## Your Project Overview

You're building a complete RAG system!

**7 TODOs**:
1. Document Loading
2. Text Chunking
3. Document Processing
4. RAG Query Function

---

## Your Project Overview (cont.)

**7 TODOs** (continued):
5. Create Test Dataset
6. Evaluation Metrics
7. Run Evaluation

**Result**: A working RAG system you can demo!

---

## Your Project's RAG Pipeline

```
TODO #1: Load Documents
    ↓
TODO #2: Chunk Text
    ↓
TODO #3: Process All Docs
    ↓
Pre-built: Create Embeddings
```

---

## Your Project's RAG Pipeline (cont.)

```
Pre-built: Store in ChromaDB
    ↓
TODO #4: RAG Query Function
    ↓
TODOs #5-7: Test & Evaluate
```

---

## What You'll Learn

### Python Skills:
- File I/O and text processing
- String manipulation
- Functions and data structures
- Loops and conditionals

---

## What You'll Learn (cont.)

### AI/ML Concepts:
- Embeddings and vector search
- Retrieval-augmented generation
- System evaluation

### Teamwork:
- Collaboration, code review, documentation

---

## Team Responsibilities

**Recommended for 3 people**:
- **Person 1**: TODOs #1-2 (Loading & chunking)
- **Person 2**: TODOs #3-4 (Processing & RAG query)
- **Person 3**: TODOs #5-7 (Testing & evaluation)

---

## Team Responsibilities (cont.)

**Everyone**:
- Code review
- Testing
- Report writing
- Understanding all the code!

---

## Project Timeline

| Week | Task | Time/Person |
|------|------|-------------|
| **Week 1** | Form team, setup | 2-3 hours |
| **Week 2** | Complete TODOs | 4-8 hours |

---

## Project Timeline (cont.)

| Week | Task | Time/Person |
|------|------|-------------|
| **Week 3** | Testing & analysis | 2-3 hours |
| **Week 4** | Write report | 2-3 hours |

**Total**: ~12-17 hours per person

---

## Getting Started: Step 1-3

1. **Clone the repository**
2. **Install Docker** (see `docker_starter.md`)
3. **Create conda environment** (see `README.md`)

---

## Getting Started: Step 4-7

4. **Build Docker image** with Ollama
5. **Run `test_setup.ipynb`** to verify setup
6. **Read `STUDENT_PROJECT_GUIDE.md`**
7. **Start coding!**

---

<!-- _class: lead -->

# Key Takeaways
## What to remember

---

## Key Takeaways (1-3)

1. **RAG** = Retrieval + LLM for better answers

2. **Two phases**: Indexing (once) + Query (every question)

3. **Core components**: Chunking, embeddings, vector DB, prompt, LLM

---

## Key Takeaways (4-6)

4. **Benefits**: Accurate, verifiable, up-to-date answers

5. **Applications**: Everywhere! ChatGPT uses this!

6. **Challenges**: Chunking, retrieval quality, speed

---

## Key Takeaways (7-10)

7. **Best practices**: Clean data, tune parameters, evaluate

8. **Your project**: Hands-on experience with all components

9. **Skills gained**: Python + AI/ML + teamwork

10. **Career value**: Portfolio project for interviews!

---

## Further Reading

### Papers:
- [Original RAG Paper](https://arxiv.org/abs/2005.11401)
- [Dense Passage Retrieval](https://arxiv.org/abs/2004.04906)

---

## Further Reading (cont.)

### Resources:
- [LangChain RAG Tutorial](https://python.langchain.com/docs/tutorials/rag/)
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [Sentence Transformers](https://www.sbert.net/)

---

## Further Reading (cont.)

### Advanced Topics:
- Hybrid search strategies
- Re-ranking retrieved results
- Query expansion
- Multi-hop reasoning

---

## Questions?

**Remember**:
- Review `rag_concepts.md` for detailed explanations
- Use `STUDENT_PROJECT_GUIDE.md` for assignment details

---

## Questions? (cont.)

- Check `docker_starter.md` for setup help
- Run `test_setup.ipynb` to verify your environment

**You're learning the tech behind ChatGPT, Copilot, and modern AI!**

---

<!-- _class: lead -->

# Let's Build Something Amazing!

## Good luck with your RAG project! 🚀

---
