# RAG Group Project Guide

## Welcome! 👋

Your team is about to build a **RAG (Retrieval-Augmented Generation)** system - an AI that can answer questions about YOUR documents. This is the same technology used by ChatGPT, GitHub Copilot, and other AI assistants!

**This is a GROUP PROJECT:** Work in teams of 2-4 students. Collaboration is expected and encouraged!

---

## What Your Team Will Build

A question-answering system that:
1. Reads your documents
2. Breaks them into smaller chunks
3. Finds relevant information when you ask a question
4. Uses an AI to generate accurate answers

**Example:**
- You provide: School policy documents
- You ask: "What are the parking rules?"
- System: Searches documents → Finds relevant sections → Generates answer

---

## What You'll Learn (Python Fundamentals)

### Core Skills You'll Practice:
✅ **File I/O** - Reading documents from your computer
✅ **String Manipulation** - Splitting and processing text
✅ **Functions** - Writing reusable code
✅ **Lists & Dictionaries** - Organizing data
✅ **Loops** - Processing multiple items
✅ **Conditionals** - Making decisions in code
✅ **Basic Math** - Calculating metrics

### What's Pre-Built for You:
✅ Embedding model (converts text to numbers)
✅ Vector database (stores and searches)
✅ LLM connection (generates answers)

**Why?** These use complex libraries that would take weeks to learn. We want you to focus on Python fundamentals first!

---

## Project Files

### Files You'll Use:
1. **`rag_helpers.py`** - Pre-built library code (DON'T MODIFY)
2. **`student_rag_project.ipynb`** - Your main notebook (COMPLETE THE TODOs)
3. **Your documents** - Add your own text files to test with

### Files You'll Create:
1. Test questions dataset
2. Evaluation results
3. Final report

---

## Getting Started

### Step 1: Set Up Your Environment

Make sure your team has:
- Anaconda or Miniconda installed
- Python 3.12+ (comes with Anaconda/Miniconda)
- Jupyter notebook or VSCode
- Docker Desktop installed and running

Create the conda environment:
```bash
# Create environment from file
conda env create -f rag_env.yml

# Activate environment
conda activate rag3_313
```

If the environment file doesn't work, create manually:
```bash
conda create -n rag3_313 python=3.13
conda activate rag3_313
pip install sentence-transformers chromadb requests
```

### Step 2: Prepare Your Documents

1. Create a folder for your documents (e.g., `my_docs/`)
2. Add 5-10 text files (.txt format)
3. Choose a topic you know well (makes testing easier!)

**Document Ideas:**
- Course notes or textbooks
- Technical documentation
- Company policies
- Sports rules
- Recipes
- Historical documents
- Anything with factual information!

**Requirements:**
- At least 5 documents
- Minimum 1000 words total
- Plain text format (.txt)

### Step 3: Open the Student Notebook

1. Open `student_rag_project.ipynb`
2. Read through the entire notebook first
3. Look for sections marked `# TODO:`

### Step 4: Complete the 7 TODOs

Work through each TODO section in order. Each one builds on the previous:

---

## The 7 TODO Tasks

### TODO #1: Document Loading (15 points)
**What:** Write a function to load text files from a folder

**Python Skills:**
- File I/O (`open()`, `.read()`)
- Loops (`for` loop)
- Dictionaries (creating and storing data)
- Lists (collecting items)

**Hints Provided:** Yes! The notebook walks you through it.

**Estimated Time:** 20-30 minutes

---

### TODO #2: Text Chunking (15 points)
**What:** Write a function to split long text into smaller chunks with overlap

**Python Skills:**
- String slicing (`text[start:end]`)
- While loops
- Basic math
- Lists

**Why It Matters:** Documents are too long for AI models. We need smaller pieces.

**Estimated Time:** 30-40 minutes

---

### TODO #3: Process Documents (15 points)
**What:** Use your chunking function on all documents and create metadata

**Python Skills:**
- Nested loops (loop within a loop)
- Dictionaries (metadata)
- `enumerate()` function
- Combining functions

**Estimated Time:** 25-35 minutes

---

### TODO #4: RAG Query Function (20 points)
**What:** Write the main function that answers questions!

**Python Skills:**
- Function composition (calling multiple functions)
- String formatting (building prompts)
- Dictionaries (returning structured data)
- Using pre-built classes

**Why It Matters:** This is the heart of your RAG system!

**Estimated Time:** 30-40 minutes

---

### TODO #5: Create Test Dataset (10 points)
**What:** Make a list of test questions and expected answers

**Python Skills:**
- Lists and dictionaries
- Data structures
- Critical thinking

**Tips:**
- Create questions YOU know the answer to
- Mix easy and hard questions
- Include questions that require combining information

**Estimated Time:** 20-30 minutes

---

### TODO #6: Evaluation Metrics (15 points)
**What:** Write functions to measure system performance

**Python Skills:**
- Functions
- Mathematical calculations
- Lists and sets
- Aggregation

**Metrics You'll Calculate:**
- Average response time
- Success rate
- Sources used

**Estimated Time:** 20-30 minutes

---

### TODO #7: Run Evaluation (10 points)
**What:** Test your system with all questions and collect results

**Python Skills:**
- Loops
- Function calls
- Data collection

**Estimated Time:** 15-20 minutes

---

## Group Work & Responsibilities

### Team Size
- **Recommended:** 3 students

### Suggested Work Division

#### Option 1: Task-Based Split (Recommended for 3-4 people)
- **Person 1:** TODOs #1-2 (Document loading & chunking)
- **Person 2:** TODOs #3-4 (Document processing & RAG query)
- **Person 3:** TODOs #5-7 (Testing & evaluation)
- **Everyone:** Code review, testing, report writing

#### Option 2: Role-Based Split (Works for any size)
- **Developer(s):** Write the main code for TODOs
- **Tester:** Create test dataset, verify results
- **Analyst:** Run evaluations, analyze results, write report
- **All:** Pair programming, code review, documentation

#### Option 3: Rotate (Good for learning)
- Rotate through TODOs as a team
- Each person leads 2-3 TODOs while others review
- Everyone contributes to all parts

### Collaboration Expectations

**Everyone should:**
- ✅ Understand all the code (not just the parts they wrote)
- ✅ Contribute to debugging and problem-solving
- ✅ Participate in testing and analysis
- ✅ Review code written by teammates
- ✅ Contribute to the final report

**Best Practices:**
- Meet regularly (in-person or virtual)
- Use Git/GitHub for version control (recommended)
- Document who did what in your report
- Test each other's code
- Ask questions if you don't understand something

### Individual Accountability

Your submission should include:
- A section in the report listing each member's contributions
- All members should be able to explain any part of the code
- Instructor may ask individual questions during demos/presentations

---

## Grading Rubric (100 points total)

### Implementation (60 points)

| TODO | Points | What's Graded |
|------|--------|---------------|
| #1: Document Loading | 15 | Correctly loads all .txt files, handles errors, returns proper format |
| #2: Text Chunking | 15 | Chunks text correctly with overlap, handles edge cases |
| #3: Process Documents | 15 | Processes all docs, creates proper metadata, correct structure |
| #4: RAG Query | 20 | Complete pipeline, proper prompt, returns all required data |
| #5: Test Dataset | 10 | At least 10 questions, diverse types, clear expected answers |
| #6: Metrics | 15 | All functions work correctly, accurate calculations |
| #7: Evaluation | 10 | Runs all tests, collects results properly |

### Code Quality (20 points)

| Criterion | Points | Description |
|-----------|--------|-------------|
| **Readability** | 5 | Clear variable names, proper formatting |
| **Comments** | 5 | Explains your logic where needed |
| **Correctness** | 5 | Code runs without errors |
| **Completeness** | 5 | All TODOs completed |

### Results & Analysis (20 points)

| Criterion | Points | Description |
|-----------|--------|-------------|
| **System Performance** | 8 | System successfully answers questions |
| **Test Coverage** | 6 | Good variety of test questions |
| **Analysis** | 6 | Thoughtful discussion of results |

---

## Submission Checklist

### Required Files:
- [ ] `student_rag_project.ipynb` - Your completed notebook
- [ ] `evaluation_results.json` - Saved results
- [ ] `my_docs/` folder - Your documents
- [ ] `REPORT.md` or `REPORT.pdf` - Your team writeup
- [ ] `GROUP_INFO.txt` - List of all team members and contributions

### In Your Report (3-4 pages):

#### 0. Group Information (NEW - Required!)
- **Team Name:** [Your team name]
- **Team Members:** [List all names and student IDs]
- **Individual Contributions:**
  - Member 1: [What they did]
  - Member 2: [What they did]
  - etc.

#### 1. Introduction
- What documents did you choose and why?
- What questions can your system answer?
- How did your team collaborate?

#### 2. Implementation
- Describe your chunking strategy (chunk size, overlap)
- Why did you choose these parameters?
- Any challenges you faced?

#### 3. Results
- Show your evaluation metrics
- Include 3-5 example Q&A pairs
- Which questions worked well? Which didn't?

#### 4. Analysis
- What did you learn about RAG systems?
- What would you improve with more time?
- What surprised you?

#### 5. Python Reflection
- Which Python concepts did you use most?
- What was most challenging?
- What did you learn?

---

## Tips for Success

### Before You Start:
1. ✅ Read the entire notebook as a team
2. ✅ Make sure Docker/Ollama is running on at least one machine
3. ✅ Choose documents your team understands well
4. ✅ Plan your time (expect 10-15 hours total team effort)
5. ✅ Decide on work division early

### While Working:
1. ✅ Test each TODO individually before moving on
2. ✅ Read the hints carefully
3. ✅ Use `print()` statements to debug
4. ✅ Review each other's code regularly
5. ✅ Ask for help if stuck for more than 30 minutes
6. ✅ Save your work frequently (use Git!)
7. ✅ Meet regularly to sync progress

### For Better Results:
1. ✅ Start with small chunk sizes (200-300) and test
2. ✅ Try different overlap values (10%, 15%, 20%)
3. ✅ Create questions that test different aspects
4. ✅ Review answers critically - are they accurate?
5. ✅ Have team members test the system independently

### Common Mistakes to Avoid:
❌ Not testing code before moving to next TODO
❌ Copy-pasting without understanding
❌ Choosing documents that are too similar
❌ Creating only easy test questions
❌ Not reading error messages carefully
❌ Waiting until the last minute
❌ Only one person doing all the work
❌ Not communicating with teammates
❌ Forgetting to document contributions

---

## Debugging Help

### "No documents loaded"
- Check your folder path is correct
- Make sure files are .txt format
- Check file encoding (should be UTF-8)

### "List index out of range"
- Probably an empty list somewhere
- Add checks: `if my_list:` before accessing items

### "LLM connection failed"
- Is Docker running? Check with `docker ps`
- Is Ollama accessible? Try `curl http://127.0.0.1:11434`
- Did you start the container? See COMMANDS.txt

### "Chunks are empty"
- Check your chunking logic
- Make sure `chunk_size` is reasonable (100-1000)
- Verify you're slicing strings correctly

### Jupyter Kernel Crashes
- Embedding model might be too large
- Restart kernel and run cells in order
- Make sure you have enough RAM

---

## Frequently Asked Questions

### Q: Can I use documents in other languages?
**A:** Yes! But make sure your embedding model supports it. The default model works best with English.

### Q: How many documents do I need?
**A:** Minimum 5, but 8-12 is better for testing.

### Q: How should we divide the work in our group?
**A:** See the "Group Work & Responsibilities" section below. Each member should contribute to coding, testing, and documentation. Recommended split: 1 person per TODO or rotate responsibilities.

### Q: What if my answers are wrong?
**A:** That's okay! Analyze WHY in your report. Did it retrieve wrong chunks? Is the prompt unclear?

### Q: How do I know if my chunk size is good?
**A:** Experiment! Try 200, 500, and 1000. See which gives better answers.

### Q: Can I modify `rag_helpers.py`?
**A:** No, that's pre-built code. Focus on completing the TODOs.

### Q: My evaluation is slow. Is that normal?
**A:** Yes! The LLM takes time. Expect 3-5 seconds per question.

### Q: Can I add extra features?
**A:** Yes! After completing all TODOs, feel free to experiment. Document extras in your report.

---

## Example Document Sets

### Good Choices:  

✅ School policies (attendance, parking, academic integrity)  
✅ Game rules (sports, board games, video games)  
✅ Technical guides (software docs, manuals)  
✅ Historical documents (speeches, letters, articles)  
✅ Recipe collections (structured, factual)  

### Poor Choices:  

❌ Single very long document (hard to test)  
❌ Highly creative/narrative text (facts work better)  
❌ Documents with lots of tables/formatting  
❌ Mixed topics with no connection  

---

## Timeline Suggestion (Per Team)

| Week | Task | Time (per person) |
|------|------|-------------------|
| **Week 1** | Form team, choose documents, set up environments | 2-3 hours |
| **Week 2** | Complete TODOs #1-7, team review | 4-8 hours |
| **Week 3** | Testing, optimization, analysis | 2-3 hours |
| **Week 4** | Write report, document contributions, submit | 2-3 hours |

**Total Time Per Person:** ~12-17 hours
**Total Team Effort:** ~36-68 hours (depending on team size)

---

## Getting Help

### Within Your Team (Try This First!):
- Debug together using pair programming
- Explain the problem to teammates
- Review each other's code
- Search for solutions as a team

### During Office Hours:
- Attend as a team (or send a representative)
- Bring your notebook with specific errors
- Explain what you've tried
- Show your thought process

### In Discussion Forum:
- Post code snippets (not entire solutions)
- Describe the error message
- Explain expected vs. actual behavior
- Mention what your team has already tried

### Team Resources:
- Read error messages carefully
- Use `print()` to check values
- Test small pieces of code separately
- Search Python documentation together
- Review each other's understanding

---

## Learning Resources

### Python Basics:
- [Python Official Tutorial](https://docs.python.org/3/tutorial/)
- [W3Schools Python](https://www.w3schools.com/python/)

### Understanding RAG:
- "What is RAG?" - Search for beginner guides
- LangChain RAG documentation
- Your class notes!

### Debugging:
- [How to Debug Python Code](https://realpython.com/python-debugging-pdb/)
- Stack Overflow (search before posting)

---

## After Completion

### Showcase Your Work:
- Add to your GitHub portfolio
- Demonstrate in job interviews
- Explain how RAG works to friends

### Next Steps:
- Try different embedding models
- Experiment with prompt engineering
- Build a web interface
- Try with larger document collections
- Compare different LLMs

### Skills You Gained:
✅ Python programming fundamentals
✅ File handling and text processing
✅ Working with AI/ML libraries
✅ System evaluation and metrics
✅ Technical writing
✅ Problem-solving and debugging

---

## Final Thoughts

This project combines **Python fundamentals** with **cutting-edge AI technology** and **teamwork**.

Your team isn't just learning to code - you're building something real and useful together!

Take your time, communicate well, ask questions, and enjoy the process. By the end, your team will have:
- A working AI system
- Strong Python skills
- Understanding of modern AI techniques
- Collaboration and teamwork experience
- A portfolio project to show employers

**Good luck! Your team has got this! 🚀**