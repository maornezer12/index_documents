# Document Indexer

A Python tool that extracts text from PDF and DOCX documents, chunks the text using configurable strategies, generates embeddings using Google Gemini API, and stores the results in PostgreSQL for semantic search and retrieval.

## What is this project?

Document Indexer is a complete pipeline for converting documents into searchable vector representations. It:

1. **Extracts text** from PDF and DOCX files
2. **Chunks the text** into manageable pieces using one of three strategies
3. **Generates embeddings** (vector representations) for each chunk using Google's Gemini API
4. **Stores everything** in PostgreSQL for efficient semantic search

The embeddings enable semantic search - finding documents based on meaning rather than just keywords. This makes it ideal for building RAG (Retrieval-Augmented Generation) systems, document search, and knowledge bases.

## Why Chunking?

Documents are split into smaller chunks because embedding models have input size limits, and smaller, focused chunks enable more precise semantic search and better retrieval of relevant content. Each chunk gets its own embedding, allowing you to find specific sections of documents rather than just entire files.

## Chunking Strategies

The tool offers three different approaches to split documents into chunks. Each strategy has different strengths:

### 1. Fixed-size (`fixed`)
Splits text into fixed-size character windows with overlap.

- **Best for**: Uniform chunk sizes, when you need consistent embedding costs
- **Overlap**: 200 characters (default) to maintain context between chunks
- **Trade-off**: May split sentences/paragraphs in the middle

**Example use case**: Large documents where you want predictable chunk sizes.

### 2. Sentence-based (`sentence`) - Default
Splits on sentence boundaries using NLTK Punkt tokenizer, then merges sentences to approach target size.

- **Best for**: Natural language documents, preserving semantic units
- **Overlap**: None - each sentence appears only once
- **Trade-off**: More natural splits, but chunk sizes may vary

**Example use case**: Articles, papers, and text where sentence boundaries matter.

### 3. Paragraph-based (`paragraph`)
Splits on double newlines (paragraph breaks), then merges short paragraphs to target size.

- **Best for**: Structured documents with clear paragraph boundaries
- **Overlap**: None - each paragraph appears only once
- **Trade-off**: Preserves document structure, but may create very small or large chunks

**Example use case**: Formatted documents, reports, and structured content.

## Embeddings

### What are embeddings?

Embeddings are numerical vector representations of text. They convert words and sentences into arrays of numbers (vectors) that capture semantic meaning. Similar texts produce similar vectors, enabling:

- **Semantic search**: Find documents by meaning, not just keywords
- **Similarity matching**: Compare texts mathematically
- **RAG systems**: Retrieve relevant context for AI models

### Embedding dimensions

This tool uses **768-dimensional vectors** (768 numbers per chunk). This dimension size offers:

- **Good balance**: Sufficient representation power without excessive storage
- **Performance**: Fast similarity calculations while maintaining accuracy
- **Storage efficiency**: Smaller vectors mean less database space

According to Google's benchmarks, 768 dimensions achieve comparable performance to higher dimensions (1536, 3072) for most tasks, making it an efficient choice.

### Model details

- **Primary model**: `text-embedding-004` (Gemini API)
- **Fallback model**: `embedding-001` (automatic fallback if primary fails)
- **Input limit**: 2,048 tokens per chunk (approximately 3,000 characters)
- **Output dimension**: 768 (fixed)

### Learn more

For detailed information about Gemini embeddings, including advanced features like task types and batch processing, see the [official Gemini Embeddings documentation](https://ai.google.dev/gemini-api/docs/embeddings).

## Technical Details

### Architecture

This modular Python application is organized into focused modules:

- **`utils/config.py`** - Configuration and constants
- **`utils/io_utils.py`** - File I/O (PDF/DOCX reading, text normalization)
- **`utils/cli_utils.py`** - Command-line interface parsing
- **`utils/embeddings_utils.py`** - Embedding generation (Gemini API)
- **`utils/db_utils.py`** - operations
- **`utils/errors_utils.py`** - Error handling and user-friendly messages

### Default Configuration

- **Default chunk size**: 1,500 characters
- **Default overlap**: 200 characters (for fixed strategy only)
- **Default strategy**: `sentence`
- **Maximum chunk size**: 3,000 characters (enforced safety limit)

### Database Schema

The tool automatically creates the `document_chunks` table and indexes on first run:

```sql
CREATE TABLE document_chunks (
  id BIGSERIAL PRIMARY KEY,
  filename TEXT NOT NULL,
  strategy_split TEXT NOT NULL,
  chunk_text TEXT NOT NULL,
  embedding DOUBLE PRECISION[] NOT NULL,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```

**Indexes created automatically:**
- `ix_filename` on `filename` column
- `ix_strategy_split` on `strategy_split` column  
- `ix_created_at` on `created_at` column

**Duplicate Prevention:**
When re-indexing the same file with the same strategy, the tool automatically deletes existing chunks for that filename-strategy combination before inserting new ones, preventing duplicates.

This means you can index the same file with different strategies (e.g., `fixed`, `sentence`, `paragraph`) simultaneously - each strategy will have its own chunks stored separately.

### Features

- **Automatic retry logic**: Exponential backoff (up to 5 attempts) for transient API errors
- **Error handling**: Friendly error messages with actionable suggestions
- **Text normalization**: Automatic whitespace cleanup and normalization
- **Safety limits**: Automatic truncation of chunks exceeding model limits with warnings

### Command-Line Arguments

**Required:**
- `--file`: Path to PDF or DOCX file

**Chunking Options:**
- `--strategy`: Chunking strategy - `fixed`, `sentence`, or `paragraph` (default: `sentence`)

## Installation

### Requirements

- **Python**: >= 3.10
- **PostgreSQL**: Any recent version (9.5+)
- **Internet**: Required for embedding API calls and NLTK tokenizer download (first run)

### PostgreSQL Setup

**Check if PostgreSQL is installed:**

```bash
psql --version
```

If you see a version number (e.g., `psql (PostgreSQL) 15.3`), PostgreSQL is already installed.

**If PostgreSQL is NOT installed:**

**Mac (using Homebrew):**
```bash
brew install postgresql@17
brew services start postgresql@17
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt update
sudo apt install postgresql postgresql-contrib
sudo systemctl start postgresql
sudo systemctl enable postgresql
```

**Windows:**
1. Download PostgreSQL installer from https://www.postgresql.org/download/windows/
2. Run the installer and follow the setup wizard
3. Remember the password you set for the `postgres` user
4. PostgreSQL service will start automatically

**Create a database (if needed):**

After installation, create a database for the project:

```bash
psql -U postgres
CREATE DATABASE document_indexer;
\q
```

**If PostgreSQL IS already installed:**

1. Verify it's running:
   ```bash
   # Mac/Linux
   brew services list | grep postgresql
   # or
   sudo systemctl status postgresql
   
   # Windows
   # Check Services (services.msc) for "postgresql" service
   ```

2. Create database if needed (see commands above)

### Setup Steps

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd index_documents
   ```

2. **Create virtual environment**:
   ```bash
   # Mac/Linux
   python -m venv .venv
   source .venv/bin/activate
   
   # Windows
   python -m venv .venv
   .venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment**:
   ```bash
   # Mac/Linux
   cp .env.example .env
   
   # Windows
   copy .env.example .env
   ```
   
   Then edit `.env` and fill in your values:
   
   **Example `.env` file:**
   ```ini
   GEMINI_API_KEY=AIzaSyXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
   POSTGRES_URL=postgresql://user:password@localhost:5432/document_indexer
   ```

   - `GEMINI_API_KEY`: Get from https://makersuite.google.com/app/apikey
   - `POSTGRES_URL`: Your PostgreSQL connection string (format: `postgresql://[user[:password]@][host][:port][/database]`)

5. **NLTK Data** (first time only):
   If using `sentence` strategy, NLTK will download Punkt tokenizer automatically on first run. If it fails:
   ```bash
   python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
   ```

## Running Tests

**Recommended:** Run tests before using the tool to ensure everything is working correctly.

**Run all tests:**
```bash
pytest
```

**Run with verbose output:**
```bash
pytest -v
```

**Run specific test file:**
```bash
pytest tests/test_embeddings_utils.py -v
pytest tests/test_chunking.py -v
pytest tests/test_db_utils.py -v
```

**Run specific test:**
```bash
pytest tests/test_embeddings_utils.py::TestEmbedTexts::test_embeds_single_text -v
```



The test suite includes tests covering:
- Chunking strategies (fixed, sentence, paragraph)
- Database operations
- Embedding generation (Gemini API integration)

## Usage

### Quick Start Examples

Ready-to-use commands with the included sample files:

**Indexing `docs/space.pdf`:**

```bash
# Fixed-size chunking
python index_documents.py --file ./docs/space.pdf --strategy fixed

# Sentence-based chunking (default)
python index_documents.py --file ./docs/space.pdf --strategy sentence

# Paragraph-based chunking
python index_documents.py --file ./docs/space.pdf --strategy paragraph
```

**Indexing `docs/football.docx`:**

```bash
# Fixed-size chunking
python index_documents.py --file ./docs/football.docx --strategy fixed

# Sentence-based chunking (default)
python index_documents.py --file ./docs/football.docx --strategy sentence

# Paragraph-based chunking
python index_documents.py --file ./docs/football.docx --strategy paragraph
```

### Viewing Data in PostgreSQL

After indexing, you can inspect the data in `psql`:

```sql
-- Switch to expanded view to avoid column misalignment:
\x on

-- View sample rows neatly:
SELECT id, filename, strategy_split,
       LEFT(REPLACE(chunk_text, E'\n',' '), 20) AS preview_chunk_text,
       embedding[1:5] AS embedding_preview,
       created_at::date AS created_date
FROM document_chunks;

-- Exit expanded view
\x off
```

