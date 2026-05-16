# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **Bug Solutions MCP** project with two main components:

1. **Root-level MCP Server** (`mcp_server.py`): A FastMCP server that provides semantic search over bug solutions using vector embeddings. It searches a PostgreSQL database (Neon) for similar issues and returns matched bug solutions with full metadata.

2. **CLI Chat Application** (`cli_project/`): An interactive command-line interface that integrates with Claude through the Anthropic API and MCP clients. It supports document retrieval (@doc syntax) and MCP commands (/command syntax).

The project ingests GitHub issues, generates OpenAI embeddings, stores them in PostgreSQL, and provides semantic search capabilities for bug resolution assistance.

## Architecture

### Root Level
- **mcp_server.py**: FastMCP server exposing `search_bug_solution` tool. Uses OpenAI embeddings for semantic similarity and vector search.
- **ingest_issues.py**: GitHub issue scraper fetching issues from configured repos.
- **embed_and_store.py**: Creates embeddings via OpenAI API and stores in PostgreSQL.
- **migrate_to_neon_direct.py**: Data migration utilities to Neon PostgreSQL.
- **export_chroma.py**, **import_to_neon.py**, **verify_neon.py**: Utility scripts for data management.
- **requirements.txt**: Python dependencies (requests, python-dotenv, openai, psycopg2, mcp, fastapi, uvicorn, numpy).

### CLI Project (`cli_project/`)
- **main.py**: Entrypoint that initializes Claude service, MCP clients, and launches the CLI.
- **mcp_server.py**: Document MCP server (contains stubs with TODOs for tool/resource implementation).
- **mcp_client.py**: Client for communicating with MCP servers.
- **core/claude.py**: Claude service wrapper around Anthropic SDK.
- **core/cli.py**: CLI UI with command autocompletion using prompt-toolkit.
- **core/cli_chat.py**: Chat logic handling message formatting, document extraction (@mentions), and MCP command processing.
- **core/chat.py**: Base chat class managing message history and tool execution.
- **core/tools.py**: Tool integration layer.
- **pyproject.toml**: Project metadata and dependencies.

## Setup & Development

### Environment Setup
1. **Create/update `.env`** with:
   - `ANTHROPIC_API_KEY`: Your Anthropic API key (required for CLI)
   - `OPENAI_API_KEY`: OpenAI API key (required for embeddings)
   - `DATABASE_URL`: PostgreSQL connection string (defaults to `postgresql://localhost/bug_solutions` locally, requires Neon for production)
   - `GITHUB_TOKEN`: GitHub PAT for issue scraping
   - `CLAUDE_MODEL`: Model name for Claude (e.g., `claude-opus-4-7`)
   - `PORT`: Port for MCP server (defaults to 8000)

2. **Install dependencies**:
   ```bash
   # Using pip in the root directory
   pip install -r requirements.txt
   
   # For CLI project (uses uv.lock, install uv first)
   cd cli_project && pip install -e . && cd ..
   ```

### Running the MCP Server
- **Production HTTP transport**: `python mcp_server.py` (runs on port from env, defaults to 8000)
- The server expects a PostgreSQL database with tables: `issues` (with columns: id, repo, issue_number, url, title, body, labels, created_at, closed_at, resolution_text, comments) and `embeddings` (id, embedding as float array).

### Running the CLI Application
- **Root directory**: `python cli_project/main.py [additional_mcp_servers]`
- Accepts optional additional MCP server scripts as arguments
- Interactive prompt supports: natural text queries, document references (`@doc_id`), and MCP commands (`/command args`)
- Tab autocompletion for commands and resources

### Common Development Tasks

**Ingest new issues from GitHub**:
```bash
python ingest_issues.py
```
(Fetches from repos defined in REPOS list, outputs to issues_corpus.json)

**Generate embeddings and populate database**:
```bash
python embed_and_store.py
```

**Migrate data to Neon**:
```bash
python migrate_to_neon_direct.py
```

**Export database embeddings**:
```bash
python export_chroma.py
```

**Verify Neon database**:
```bash
python verify_neon.py
```

**Run embedding tests**:
```bash
bash test_embeddings.sh  # or python test_embeddings.py
```

## Key Implementation Details

### Vector Search (mcp_server.py)
- Uses OpenAI's `text-embedding-3-small` for embeddings (1536 dimensions)
- Cosine similarity for ranking results
- Returns top 3 most similar issues with full metadata (title, body, resolution, comments, labels)

### Database Schema
- **issues table**: GitHub issue metadata + resolution_text (string) and comments (JSONB)
- **embeddings table**: issue_id references issues, embedding stored as float array
- PostgreSQL with Neon pooling for connection management
- JSONB columns may be pre-parsed by psycopg2 (handle both dict and JSON string formats)

### MCP Integration
- Uses mcp[cli] library for protocol implementation
- FastMCP for simplified server creation
- stdio transport for CLI communication, HTTP for root server
- Document MCP server has TODO stubs for reading/editing/summarizing documents

### CLI Chat Flow
1. Parse user input for commands (/) and mentions (@)
2. Extract document content for mentioned resources
3. Build context-aware prompts with documents
4. Send to Claude via Anthropic SDK
5. Stream responses back to user

## Known Issues & TODOs

**CLI Project Document Server** (`cli_project/mcp_server.py`):
- TODO: Implement tool to read documents
- TODO: Implement tool to edit documents
- TODO: Add resource to list all doc IDs
- TODO: Add resource to return document contents
- TODO: Add prompt to rewrite documents in markdown
- TODO: Add prompt to summarize documents

**Recent Fixes** (check commits):
- Fixed JSON deserialization for JSONB columns (psycopg2 pre-parses some types)
- Fixed FastMCP to pass host/port to constructor, not run()
- Migrated from Chroma to direct PostgreSQL + Neon

## Important Notes

- **.env is in .gitignore**: Never commit API keys or database credentials. Use environment variables.
- **Database connection**: Requires PostgreSQL with vector/JSONB support (Neon works). Local dev uses `sslmode=disable`, production uses Neon with `sslmode=require`.
- **Rate limiting**: GitHub API scraper includes rate limit checks and backoff sleep.
- **Vector similarity**: Embeddings are 1536-dim vectors; similarity is computed on-demand (not indexed).
- **MCP Transport**: Root server uses HTTP for deployment, CLI project uses stdio for stdio transport in subprocess communication.

## Testing

Run embedding tests to verify OpenAI integration and database connectivity:
```bash
bash test_embeddings.sh
```

No lint or type checks are currently configured. Consider adding: `pylint`, `mypy`, or `ruff` for code quality.
