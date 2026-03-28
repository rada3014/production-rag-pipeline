# Chunking Strategy Analysis — Atomic Habits

## Which Strategy Won and Why

For a structured non-fiction book like *Atomic Habits*, **Strategy 3 (Sentence-Aware)** produced the most relevant retrieved chunks. Non-fiction books are written in tight, self-contained sentences where a single sentence often carries the full meaning of a concept — for example, *"You do not rise to the level of your goals. You fall to the level of your systems."* Strategy 3 preserves these semantic units by grouping complete sentences up to 500 characters, so the retrieved chunks answer questions directly without cutting a sentence mid-thought. When asked *"What is the 1% rule?"*, Strategy 3 returned the exact paragraph explaining the compounding math (1.01^365 = 37.78), fully intact, whereas the other strategies split it across chunk boundaries and lost the numerical context.

## Why the Other Strategies Fell Short

Strategy 1 (CharacterTextSplitter with `separator=''`) split on raw character count with no awareness of word or sentence boundaries, producing chunks like *"...get just 1 percen"* cut mid-word. Even with the whitespace fix, the chunks were semantically arbitrary — the retrieved results for *"What are the four laws of behavior change?"* returned chunks from three different chapters that each mentioned the word "law" but none that contained the full four-law framework together. Strategy 2 (RecursiveCharacterTextSplitter) was meaningfully better — its cascading separators (`\n\n`, `\n`, ` `) respected paragraph structure — but *Atomic Habits* uses single `\n` line breaks throughout (a side-effect of PDF parsing), so the recursive splitter effectively behaved like a character splitter most of the time and still produced mid-sentence cuts in roughly 30% of chunks.

## Key Takeaway

For PDF books with heavy PDF-parser whitespace noise and dense, sentence-level information, the sentence-aware approach is the right default. The `clean()` preprocessing step (collapsing tabs and extra spaces) was essential — without it, `sent_tokenize` misparsed tab-separated words as sentence boundaries, inflating chunk counts and degrading retrieval. If the document were a technical manual with bullet lists or code blocks, the recursive splitter would likely win instead, since those documents lack the clean sentence grammar that Strategy 3 depends on.
