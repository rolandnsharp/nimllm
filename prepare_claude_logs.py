#!/usr/bin/env python3
"""Convert Claude Code JSONL logs into nimllm training format.

Extracts user messages and assistant text responses from JSONL conversation
logs. Skips tool calls, system messages, and binary content. Formats as
one conversation per line (matching chat_input.txt format).

Usage: python3 prepare_claude_logs.py > claude_conversations.txt
"""

import json, os, sys, glob

CLAUDE_DIR = os.path.expanduser("~/.claude/projects")
MIN_CHARS = 50  # skip very short conversations

def extract_conversations(jsonl_path):
    """Extract user/assistant text turns from a JSONL file."""
    turns = []
    try:
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue

                msg = obj.get("message", {})
                role = msg.get("role")
                if role not in ("user", "assistant"):
                    continue

                # Extract text content
                content = msg.get("content", "")
                if isinstance(content, list):
                    # Multi-part content — extract text blocks only
                    texts = []
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "text":
                            texts.append(part["text"])
                        elif isinstance(part, str):
                            texts.append(part)
                    content = " ".join(texts)
                elif not isinstance(content, str):
                    continue

                content = content.strip()
                if not content:
                    continue
                # Skip system-reminder-only messages
                if content.startswith("<system-reminder>"):
                    continue
                # Skip very long tool outputs
                if len(content) > 5000:
                    content = content[:5000]

                tag = "<|user|>" if role == "user" else "<|assistant|>"
                turns.append(f"{tag} {content}")
    except Exception:
        return []

    return turns

def main():
    # Find all JSONL files (skip subagents — those are tool internals)
    jsonl_files = []
    for root, dirs, files in os.walk(CLAUDE_DIR):
        if "subagents" in root:
            continue
        for f in files:
            if f.endswith(".jsonl"):
                jsonl_files.append(os.path.join(root, f))

    print(f"Found {len(jsonl_files)} conversation logs", file=sys.stderr)

    total_convos = 0
    total_chars = 0
    for path in sorted(jsonl_files):
        turns = extract_conversations(path)
        if len(turns) < 2:
            continue
        # Join into one line per conversation
        convo = " ".join(turns)
        if len(convo) < MIN_CHARS:
            continue
        print(convo)
        total_convos += 1
        total_chars += len(convo)

    print(f"Extracted {total_convos} conversations, {total_chars // 1024}KB",
          file=sys.stderr)

if __name__ == "__main__":
    main()
