#!/usr/bin/env python3
"""Build nimllm's combined training set from all available sources.

Combines:
  1. Chat conversations (chat_input.txt)
  2. Book chunks (all .txt in ~/data/books/ and ~/data/books/gutenberg/)
  3. Claude Code conversation logs

Usage: python3 build_data.py
Output: ../training_data.txt (relative to script location)
"""

import os, sys, glob

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
BOOKS_DIR = os.path.expanduser("~/data/books")
GUTENBERG_DIR = os.path.join(BOOKS_DIR, "gutenberg")
CLAUDE_DIR = os.path.expanduser("~/.claude/projects")
CHAT_INPUT = os.path.join(ROOT, "chat_input.txt")
OUTPUT = os.path.join(ROOT, "training_data.txt")

def chunk_book(path, max_chunk=800, min_chunk=50):
    """Break a book into paragraph-sized chunks."""
    text = open(path).read()
    paras = [p.strip().replace('\n', ' ') for p in text.split('\n\n') if p.strip()]
    chunks = []
    buf = ''
    for p in paras:
        if len(buf) + len(p) < max_chunk:
            buf += ' ' + p if buf else p
        else:
            if buf and len(buf) > min_chunk:
                chunks.append(buf)
            buf = p
    if buf and len(buf) > min_chunk:
        chunks.append(buf)
    return chunks

def extract_claude_conversations():
    """Extract user/assistant text from Claude Code JSONL logs."""
    import json
    convos = []
    for root, dirs, files in os.walk(CLAUDE_DIR):
        if "subagents" in root:
            continue
        for f in files:
            if not f.endswith(".jsonl"):
                continue
            turns = []
            try:
                for line in open(os.path.join(root, f)):
                    line = line.strip()
                    if not line: continue
                    try: obj = json.loads(line)
                    except: continue
                    msg = obj.get("message", {})
                    role = msg.get("role")
                    if role not in ("user", "assistant"): continue
                    content = msg.get("content", "")
                    if isinstance(content, list):
                        texts = [p["text"] for p in content if isinstance(p, dict) and p.get("type") == "text"]
                        content = " ".join(texts)
                    if not isinstance(content, str) or not content.strip(): continue
                    if content.startswith("<system-reminder>"): continue
                    if len(content) > 5000: content = content[:5000]
                    tag = "<|user|>" if role == "user" else "<|assistant|>"
                    turns.append(f"{tag} {content.strip()}")
            except: continue
            if len(turns) >= 2:
                convos.append(" ".join(turns))
    return convos

def main():
    total_docs = 0
    total_bytes = 0

    with open(OUTPUT, 'w') as out:
        # 1. Chat conversations
        if os.path.exists(CHAT_INPUT):
            lines = open(CHAT_INPUT).readlines()
            for line in lines:
                out.write(line)
            print(f"  chat: {len(lines)} docs")
            total_docs += len(lines)
            total_bytes += sum(len(l) for l in lines)

        # 2. Books (~/data/books/*.txt + gutenberg/*.txt)
        book_files = glob.glob(os.path.join(BOOKS_DIR, "*.txt"))
        book_files += glob.glob(os.path.join(GUTENBERG_DIR, "*.txt"))
        # Exclude claude_conversations.txt (handled separately)
        book_files = [f for f in book_files if "claude_conversations" not in f]
        book_chunks = 0
        for bf in sorted(set(book_files)):
            chunks = chunk_book(bf)
            for c in chunks:
                out.write(c + '\n')
                total_bytes += len(c) + 1
            book_chunks += len(chunks)
        print(f"  books: {book_chunks} chunks from {len(set(book_files))} files")
        total_docs += book_chunks

        # 3. Claude Code logs
        convos = extract_claude_conversations()
        for c in convos:
            out.write(c + '\n')
            total_bytes += len(c) + 1
        print(f"  claude: {len(convos)} conversations")
        total_docs += len(convos)

    print(f"  total: {total_docs} docs, {total_bytes // (1024*1024)}MB")
    print(f"  written to {OUTPUT}")

if __name__ == "__main__":
    main()
