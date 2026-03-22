#!/bin/bash
## feed_books.sh — Teach nimllm from a library of books
##
## Each book is fed using --read mode: lower LR (0.0001), 5000 steps,
## chunks at sentence boundaries. The model absorbs the text into its
## weights without catastrophic forgetting (elastic pull planned for later).
##
## Usage: ./feed_books.sh
## Run AFTER base training completes.

set -e
cd "$(dirname "$0")"

MICROGPT="./src/microgpt"
BOOKS_DIR="/home/roland/data/books"

# Rebuild with latest code
echo "=== rebuilding microgpt ==="
~/.nimble/bin/nim c -d:release src/microgpt.nim

# Feed each book
for book in \
  "$BOOKS_DIR/beyond_good_and_evil.txt" \
  "$BOOKS_DIR/meditations.txt" \
  "$BOOKS_DIR/zarathustra.txt" \
  "$BOOKS_DIR/republic.txt" \
  "$BOOKS_DIR/prince.txt" \
  "$BOOKS_DIR/art_of_war.txt" \
  "$BOOKS_DIR/walden.txt" \
  "$BOOKS_DIR/frankenstein.txt" \
  "$BOOKS_DIR/alice.txt" \
  "$BOOKS_DIR/origin_of_species.txt" \
  "$BOOKS_DIR/leviathan.txt" \
  "$BOOKS_DIR/tao_te_ching.txt" \
  "$BOOKS_DIR/siddhartha.txt" \
  "$BOOKS_DIR/prophet.txt" \
  "$BOOKS_DIR/flatland.txt" \
  "$BOOKS_DIR/time_machine.txt" \
  "$BOOKS_DIR/war_of_worlds.txt" \
  "$BOOKS_DIR/common_sense.txt" \
  "$BOOKS_DIR/kjv_new_testament.txt" \
  "$BOOKS_DIR/claude_conversations.txt"
do
  if [ -f "$book" ]; then
    echo ""
    echo "=== feeding: $(basename "$book") ==="
    $MICROGPT --read "$book"
    echo "=== done: $(basename "$book") ==="
  else
    echo "skip: $book (not found)"
  fi
done

echo ""
echo "=== all books fed ==="
echo "run ./src/chat to talk to your model"
