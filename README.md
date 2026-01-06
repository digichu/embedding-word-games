# Word Games with Vector Embeddings

A collection of interactive word games that use vector embeddings to explore semantic relationships between words. Transform words by adding and subtracting word vectors to navigate from a start word to an end word.

## Features

- **Word Transformation Game**: Start with one word and try to reach another by adding/subtracting word vectors
- **Real-time Similarity Tracking**: See how close you are to the target word using cosine similarity
- **Cheat Mode**: Automatically find the best next move
- **Analogy Maker**: Create word analogies (e.g., "king is to queen as man is to woman")
- **Multiple Backends**: Choose from different embedding models

## Implementations

### 1. Word Game with GloVe Vectors (`word_game.py`)

Uses pre-trained GloVe word vectors loaded from a text file.

**Requirements:**
- GloVe vector file: `wiki_giga_2024_300_MFT20_vectors_seed_2024_alpha_0.75_eta_0.05_combined.txt`
- The file should be in GloVe format (space-separated, no header) or word2vec format

**Run:**
```bash
python word_game.py
```

### 2. Word Game with BERT (`wordgame_bert.py`)

Uses BERT-based sentence transformers for embeddings. Automatically loads vocabulary from the model's tokenizer.

**Features:**
- Uses `all-mpnet-base-v2` model (optimized for embeddings)
- Pre-computes embeddings for vocabulary tokens
- Caches embeddings to disk for faster subsequent runs

**Run:**
```bash
python wordgame_bert.py
```

**Note:** First run will download the model and pre-compute embeddings, which may take several minutes.

### 3. Word Game with OpenAI Embeddings (`wordgame_gpt.py`)

Uses OpenAI's embedding API (text-embedding-3-small) for word embeddings.

**Requirements:**
- OpenAI API key (set as `OPENAI_API_KEY` environment variable or in `openai_api_key.txt` file)
- Requires internet connection for API calls
- Caches embeddings locally to reduce API calls

**Run:**
```bash
python wordgame_gpt.py
```

**Note:** This implementation requires an OpenAI API key and will make API calls. Embeddings are cached to minimize API usage.

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. For `word_game.py`, download a GloVe vector file and place it in the project directory with the expected filename, or modify the filename in the code.

## How to Play

1. Enter a **Start Word** and an **End Word**
2. Click **Start Game**
3. Add or subtract words to transform the current vector toward the end word
4. Watch the **Cosine Similarity** increase as you get closer
5. Win when similarity exceeds 0.71!

### Controls

- **Add**: Add a word's vector to the current vector
- **Subtract**: Subtract a word's vector from the current vector
- **Reset**: Start over from the beginning word
- **Cheat**: Automatically make the best next move
- **Make Analogy**: Create word analogies using the relationship between start and end words

## Requirements

- Python 3.7+
- tkinter (usually included with Python)
- numpy
- gensim (for word_game.py)
- openai (for wordgame_gpt.py)
- sentence-transformers (for wordgame_bert.py)
- torch and transformers (dependencies of sentence-transformers)
- tqdm (for progress bars)

See `requirements.txt` for specific versions.

## Technical Details

The game works by:
1. Representing words as high-dimensional vectors (embeddings)
2. Adding/subtracting word vectors to transform meaning
3. Using cosine similarity to measure how close the current vector is to the target
4. Finding the best next word by comparing all available word vectors

## License

This project is provided as-is for educational and entertainment purposes.

