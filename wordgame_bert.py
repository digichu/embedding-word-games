import tkinter as tk
from tkinter import ttk, scrolledtext
import numpy as np
import os
import json
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


class WordGameBERT:
    def __init__(self, root):
        self.root = root
        self.root.title("Word Game with BERT")
        self.root.geometry("800x700")
        
        # Load BERT model
        print("Loading BERT model... This may take a moment on first run.")
        try:
            # Use a BERT-based model optimized for embeddings
            # all-mpnet-base-v2 is a good choice, or paraphrase-MiniLM-L6-v2 for faster inference
            self.model = SentenceTransformer('all-mpnet-base-v2')
            print("BERT model loaded successfully.")
        except Exception as e:
            raise Exception(f"Failed to load BERT model: {e}")
        
        # Cache for embeddings to avoid repeated computations
        self.embedding_cache = {}
        self.embedding_cache_file = "bert_embedding_cache.json"
        self.load_embedding_cache()
        
        # Load vocabulary from BERT tokenizer and pre-compute embeddings
        print("Loading BERT tokenizer vocabulary...")
        self.words_list, self.word_to_index, self.vectors_matrix = self._load_bert_vocabulary()
        print(f"Loaded {len(self.words_list)} tokens from BERT vocabulary")
        
        # Game state
        self.start_word = None
        self.end_word = None
        self.current_vector = None
        self.end_vector = None
        self.game_started = False
        self.word_history = []
        self.victory_shown = False
        
        # Build GUI
        self.build_gui()
        
        # Initially disable game controls
        self.disable_game_controls()
    
    def _load_bert_vocabulary(self):
        """Load vocabulary from BERT tokenizer and pre-compute embeddings"""
        # Access the tokenizer from the model
        tokenizer = self.model.tokenizer
        
        # Get all tokens from vocabulary
        vocab_dict = tokenizer.get_vocab()
        all_tokens = list(vocab_dict.keys())
        
        # Filter out special tokens, subwords, and non-English characters
        valid_tokens = []
        special_tokens = {'[CLS]', '[SEP]', '[MASK]', '[UNK]', '[PAD]'}
        
        def is_english_token(token):
            """Check if token contains only English characters"""
            # Allow ASCII letters, numbers, and common punctuation
            # Exclude tokens with non-ASCII characters (like Japanese, Chinese, etc.)
            try:
                # Check if token can be encoded as ASCII (English characters only)
                token.encode('ascii')
                # Also check if it's mostly alphabetic (allow some punctuation)
                # Remove common punctuation and check if rest is alphabetic
                cleaned = ''.join(c for c in token if c.isalnum())
                if len(cleaned) == 0:
                    return False  # Only punctuation
                # Check if at least 50% of non-punctuation chars are letters
                return True
            except UnicodeEncodeError:
                return False  # Contains non-ASCII characters
        
        for token in all_tokens:
            # Skip special tokens
            if token in special_tokens or (token.startswith('[') and token.endswith(']')):
                continue
            # Skip subword tokens (those starting with ##)
            if token.startswith('##'):
                continue
            # Skip non-English tokens
            if not is_english_token(token):
                continue
            valid_tokens.append(token)
        
        print(f"Found {len(valid_tokens)} valid tokens (excluding special tokens)")
        print("Pre-computing embeddings for vocabulary... This may take a few minutes.")
        
        # Pre-compute embeddings for all tokens
        words_list = []
        word_to_index = {}
        embeddings_list = []
        
        # Separate tokens into cached and uncached
        tokens_to_encode = []
        
        for token in valid_tokens:
            token_lower = token.lower()
            if token_lower in self.embedding_cache:
                # Use cached embedding
                words_list.append(token_lower)
                word_to_index[token_lower] = len(words_list) - 1
                embeddings_list.append(self.embedding_cache[token_lower])
            else:
                # Need to encode
                tokens_to_encode.append(token)
        
        # Batch encode uncached tokens for efficiency
        if tokens_to_encode:
            batch_size = 100
            for i in tqdm(range(0, len(tokens_to_encode), batch_size), desc="Computing embeddings"):
                batch = tokens_to_encode[i:i+batch_size]
                # Encode batch
                batch_embeddings = self.model.encode(
                    batch, 
                    convert_to_numpy=True, 
                    show_progress_bar=False,
                    batch_size=batch_size
                )
                
                for token, embedding in zip(batch, batch_embeddings):
                    token_lower = token.lower()
                    words_list.append(token_lower)
                    word_to_index[token_lower] = len(words_list) - 1
                    embedding_array = embedding.astype(np.float32)
                    embeddings_list.append(embedding_array)
                    
                    # Also cache it
                    self.embedding_cache[token_lower] = embedding_array
        
        # Save cache after pre-computation
        self.save_embedding_cache()
        
        # Convert to matrix
        vectors_matrix = np.array(embeddings_list, dtype=np.float32)
        
        print(f"Pre-computed {len(words_list)} embeddings ({len(tokens_to_encode)} new, {len(words_list) - len(tokens_to_encode)} from cache)")
        
        return words_list, word_to_index, vectors_matrix
    
    def load_embedding_cache(self):
        """Load embedding cache from file"""
        if os.path.exists(self.embedding_cache_file):
            try:
                with open(self.embedding_cache_file, "r") as f:
                    cache_data = json.load(f)
                    # Convert lists back to numpy arrays
                    self.embedding_cache = {k: np.array(v, dtype=np.float32) for k, v in cache_data.items()}
                print(f"Loaded {len(self.embedding_cache)} cached embeddings")
            except Exception as e:
                print(f"Error loading cache: {e}")
                self.embedding_cache = {}
    
    def save_embedding_cache(self):
        """Save embedding cache to file"""
        try:
            # Convert numpy arrays to lists for JSON
            cache_data = {k: v.tolist() for k, v in self.embedding_cache.items()}
            with open(self.embedding_cache_file, "w") as f:
                json.dump(cache_data, f)
        except Exception as e:
            print(f"Error saving cache: {e}")
    
    def get_embedding(self, word):
        """Get embedding for a word, using cache if available"""
        word_lower = word.lower()
        
        # Check cache first
        if word_lower in self.embedding_cache:
            return self.embedding_cache[word_lower].copy()
        
        # Get embedding from BERT model
        embedding = self.model.encode(word_lower, convert_to_numpy=True, show_progress_bar=False)
        embedding = embedding.astype(np.float32)
        
        # Cache it
        self.embedding_cache[word_lower] = embedding.copy()
        self.save_embedding_cache()
        
        return embedding
    
    def build_gui(self):
        # Top section: Start Word, End Word, Start Game button
        top_frame = ttk.Frame(self.root, padding="10")
        top_frame.pack(fill=tk.X)
        
        # Start Word
        ttk.Label(top_frame, text="Start Word:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.start_word_entry = ttk.Entry(top_frame, width=20)
        self.start_word_entry.grid(row=0, column=1, padx=5, pady=5)
        self.start_error_label = ttk.Label(top_frame, text="", foreground="red")
        self.start_error_label.grid(row=1, column=1, sticky=tk.W, padx=5)
        
        # End Word
        ttk.Label(top_frame, text="End Word:").grid(row=0, column=2, sticky=tk.W, padx=5, pady=5)
        self.end_word_entry = ttk.Entry(top_frame, width=20)
        self.end_word_entry.grid(row=0, column=3, padx=5, pady=5)
        self.end_error_label = ttk.Label(top_frame, text="", foreground="red")
        self.end_error_label.grid(row=1, column=3, sticky=tk.W, padx=5)
        
        # Start Game button
        self.start_game_btn = ttk.Button(top_frame, text="Start Game", command=self.start_game)
        self.start_game_btn.grid(row=0, column=4, padx=10, pady=5)
        
        # Make Analogy button
        self.make_analogy_btn = ttk.Button(top_frame, text="Make Analogy", command=self.open_analogy_window)
        self.make_analogy_btn.grid(row=0, column=5, padx=10, pady=5)
        
        # Separator
        ttk.Separator(self.root, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=10, pady=10)
        
        # Input section: New Word, Add, Subtract
        input_frame = ttk.Frame(self.root, padding="10")
        input_frame.pack(fill=tk.X)
        
        ttk.Label(input_frame, text="New Word:").pack(side=tk.LEFT, padx=5)
        self.new_word_entry = ttk.Entry(input_frame, width=20)
        self.new_word_entry.pack(side=tk.LEFT, padx=5)
        
        self.add_btn = ttk.Button(input_frame, text="Add", command=self.add_word)
        self.add_btn.pack(side=tk.LEFT, padx=5)
        
        self.subtract_btn = ttk.Button(input_frame, text="Subtract", command=self.subtract_word)
        self.subtract_btn.pack(side=tk.LEFT, padx=5)
        
        self.new_word_error_label = ttk.Label(input_frame, text="", foreground="red")
        self.new_word_error_label.pack(side=tk.LEFT, padx=5)
        
        # Separator
        ttk.Separator(self.root, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=10, pady=10)
        
        # Cosine Similarity and Difference Vector display
        similarity_frame = ttk.Frame(self.root, padding="10")
        similarity_frame.pack(fill=tk.X)
        
        ttk.Label(similarity_frame, text="Cosine Similarity:").pack(side=tk.LEFT, padx=5)
        self.similarity_label = ttk.Label(similarity_frame, text="0.0000", font=("Arial", 12, "bold"))
        self.similarity_label.pack(side=tk.LEFT, padx=5)
        
        ttk.Label(similarity_frame, text="Difference Vector:").pack(side=tk.LEFT, padx=5)
        self.difference_label = ttk.Label(similarity_frame, text="0.0000", font=("Arial", 12, "bold"))
        self.difference_label.pack(side=tk.LEFT, padx=5)
        
        # Current Vector display
        vector_frame = ttk.Frame(self.root, padding="10")
        vector_frame.pack(fill=tk.X)
        
        ttk.Label(vector_frame, text="Current Vector:").pack(anchor=tk.W, padx=5)
        self.current_vector_text = scrolledtext.ScrolledText(vector_frame, height=4, width=80, wrap=tk.WORD, state=tk.DISABLED)
        self.current_vector_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Output area
        output_frame = ttk.Frame(self.root, padding="10")
        output_frame.pack(fill=tk.BOTH, expand=True)
        
        ttk.Label(output_frame, text="Output:").pack(anchor=tk.W, padx=5)
        self.output_text = scrolledtext.ScrolledText(output_frame, height=10, width=80, wrap=tk.WORD, state=tk.DISABLED)
        self.output_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Reset and Cheat buttons
        reset_frame = ttk.Frame(self.root, padding="10")
        reset_frame.pack(fill=tk.X)
        
        self.reset_btn = ttk.Button(reset_frame, text="Reset", command=self.reset_game)
        self.reset_btn.pack(side=tk.LEFT, padx=5, pady=5)
        
        self.cheat_btn = ttk.Button(reset_frame, text="Cheat", command=self.cheat)
        self.cheat_btn.pack(side=tk.LEFT, padx=5, pady=5)
    
    def disable_game_controls(self):
        """Disable game controls until game starts"""
        self.new_word_entry.config(state=tk.DISABLED)
        self.add_btn.config(state=tk.DISABLED)
        self.subtract_btn.config(state=tk.DISABLED)
        self.reset_btn.config(state=tk.DISABLED)
        self.cheat_btn.config(state=tk.DISABLED)
        self.make_analogy_btn.config(state=tk.DISABLED)
    
    def enable_game_controls(self):
        """Enable game controls after game starts"""
        self.new_word_entry.config(state=tk.NORMAL)
        self.add_btn.config(state=tk.NORMAL)
        self.subtract_btn.config(state=tk.NORMAL)
        self.reset_btn.config(state=tk.NORMAL)
        self.cheat_btn.config(state=tk.NORMAL)
        self.make_analogy_btn.config(state=tk.NORMAL)
    
    def start_game(self):
        """Validate start and end words, then initialize the game"""
        # Clear previous error messages
        self.start_error_label.config(text="")
        self.end_error_label.config(text="")
        
        start_word = self.start_word_entry.get().strip().lower()
        end_word = self.end_word_entry.get().strip().lower()
        
        if not start_word or not end_word:
            if not start_word:
                self.start_error_label.config(text="Please enter a word")
            if not end_word:
                self.end_error_label.config(text="Please enter a word")
            return
        
        # Get embeddings (this validates the words)
        try:
            self.root.config(cursor="wait")
            self.root.update()
            
            start_vector = self.get_embedding(start_word)
            end_vector = self.get_embedding(end_word)
            
            self.root.config(cursor="")
            
            # Initialize game
            self.start_word = start_word
            self.end_word = end_word
            self.current_vector = start_vector.copy()
            self.end_vector = end_vector.copy()
            self.game_started = True
            self.word_history = [start_word]
            self.victory_shown = False
            
            # Enable game controls
            self.enable_game_controls()
            
            # Update displays
            self.update_similarity()
            self.update_current_vector()
            self.update_output()
            
        except Exception as e:
            self.root.config(cursor="")
            self.start_error_label.config(text="not in database")
            self.end_error_label.config(text="not in database")
    
    def cosine_similarity(self, vec1, vec2):
        """Calculate cosine similarity between two vectors"""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot_product / (norm1 * norm2)
    
    def update_similarity(self):
        """Update the cosine similarity and difference vector displays"""
        if self.current_vector is not None and self.end_vector is not None:
            similarity = self.cosine_similarity(self.current_vector, self.end_vector)
            self.similarity_label.config(text=f"{similarity:.4f}")
            
            # Calculate difference vector magnitude
            difference_vector = self.end_vector - self.current_vector
            difference_magnitude = np.linalg.norm(difference_vector)
            self.difference_label.config(text=f"{difference_magnitude:.4f}")
            
            # Check for victory
            if similarity > 0.71 and self.game_started and not self.victory_shown:
                self.output_text.config(state=tk.NORMAL)
                self.output_text.insert(tk.END, "\nvictory!")
                self.output_text.config(state=tk.DISABLED)
                self.victory_shown = True
        else:
            self.similarity_label.config(text="0.0000")
            self.difference_label.config(text="0.0000")
    
    def update_current_vector(self):
        """Update the current vector display"""
        self.current_vector_text.config(state=tk.NORMAL)
        self.current_vector_text.delete(1.0, tk.END)
        if self.current_vector is not None:
            # Show first 10 dimensions and summary
            vector_str = f"Shape: {self.current_vector.shape}\n"
            vector_str += f"First 10 values: {self.current_vector[:10]}\n"
            vector_str += f"Norm: {np.linalg.norm(self.current_vector):.4f}"
            self.current_vector_text.insert(1.0, vector_str)
        self.current_vector_text.config(state=tk.DISABLED)
    
    def update_output(self):
        """Update the output area with word history"""
        self.output_text.config(state=tk.NORMAL)
        self.output_text.delete(1.0, tk.END)
        if self.word_history:
            output_str = self.word_history[0]  # Start word
            for word, op in self.word_history[1:]:
                output_str += f" {op} {word}"
            self.output_text.insert(1.0, output_str)
        self.output_text.config(state=tk.DISABLED)
    
    def add_word(self):
        """Add a word vector to the current vector"""
        self.new_word_error_label.config(text="")
        word = self.new_word_entry.get().strip().lower()
        
        if not word:
            return
        
        try:
            self.root.config(cursor="wait")
            self.root.update()
            
            word_vector = self.get_embedding(word)
            
            self.root.config(cursor="")
            
            # Add the word vector
            self.current_vector = self.current_vector + word_vector
            
            # Update history
            self.word_history.append((word, "+"))
            
            # Update displays
            self.update_similarity()
            self.update_current_vector()
            self.update_output()
            
            # Clear the entry
            self.new_word_entry.delete(0, tk.END)
            
        except Exception as e:
            self.root.config(cursor="")
            self.new_word_error_label.config(text="not in database")
    
    def subtract_word(self):
        """Subtract a word vector from the current vector"""
        self.new_word_error_label.config(text="")
        word = self.new_word_entry.get().strip().lower()
        
        if not word:
            return
        
        try:
            self.root.config(cursor="wait")
            self.root.update()
            
            word_vector = self.get_embedding(word)
            
            self.root.config(cursor="")
            
            # Subtract the word vector
            self.current_vector = self.current_vector - word_vector
            
            # Update history
            self.word_history.append((word, "-"))
            
            # Update displays
            self.update_similarity()
            self.update_current_vector()
            self.update_output()
            
            # Clear the entry
            self.new_word_entry.delete(0, tk.END)
            
        except Exception as e:
            self.root.config(cursor="")
            self.new_word_error_label.config(text="not in database")
    
    def reset_game(self):
        """Reset the game to initial state"""
        if not self.game_started:
            return
        
        # Reset to start word
        try:
            self.current_vector = self.get_embedding(self.start_word).copy()
            self.word_history = [self.start_word]
            self.victory_shown = False
            
            # Update displays
            self.update_similarity()
            self.update_current_vector()
            self.update_output()
            
            # Clear new word entry
            self.new_word_entry.delete(0, tk.END)
            self.new_word_error_label.config(text="")
        except Exception as e:
            print(f"Error resetting game: {e}")
    
    def cheat(self):
        """Find the best word to add or subtract to get closer to the end word"""
        if not self.game_started:
            return
        
        print("Calculating best move...")
        
        # Calculate target vector (direction from current to end)
        # We want: current ± word ≈ end
        # So: ±word ≈ end - current = target
        # So: word ≈ ±target
        target_vector = self.end_vector - self.current_vector
        
        # Normalize target vector for cosine similarity calculation
        target_norm = np.linalg.norm(target_vector)
        if target_norm == 0:
            return
        target_normalized = target_vector / target_norm
        
        # Get all word vectors as a matrix (like Glove version)
        words = self.words_list
        word_vectors = self.vectors_matrix
        
        # Collect all words to exclude: start word, end word, and all words already used
        exclude_words = set()
        exclude_words.add(self.start_word.lower())
        exclude_words.add(self.end_word.lower())
        
        # Add all words that have already been used (from word_history)
        for item in self.word_history[1:]:  # Skip the first item (start word)
            if isinstance(item, tuple):
                word, _ = item
                exclude_words.add(word.lower())
        
        # Get indices of all words to exclude
        exclude_indices = set()
        for word in exclude_words:
            idx = self.word_to_index.get(word)
            if idx is not None:
                exclude_indices.add(idx)
        
        # Create mask to exclude all forbidden words
        valid_mask = np.ones(len(words), dtype=bool)
        for idx in exclude_indices:
            valid_mask[idx] = False
        
        # Calculate cosine similarities with target vector (vectorized)
        # Normalize word vectors
        word_norms = np.linalg.norm(word_vectors, axis=1, keepdims=True)
        # Avoid division by zero
        word_norms = np.where(word_norms == 0, 1, word_norms)
        word_vectors_normalized = word_vectors / word_norms
        
        # Normalize start and end vectors for similarity comparison
        start_vector = self.get_embedding(self.start_word)
        start_norm = np.linalg.norm(start_vector)
        if start_norm > 0:
            start_normalized = start_vector / start_norm
        else:
            start_normalized = start_vector
        
        end_norm = np.linalg.norm(self.end_vector)
        if end_norm > 0:
            end_normalized = self.end_vector / end_norm
        else:
            end_normalized = self.end_vector
        
        # Calculate cosine similarities with start and end words
        # Exclude words that are too similar to start or end (e.g., pluralizations)
        similarity_threshold = 0.8  # Threshold for "substantially similar"
        similarities_to_start = np.dot(word_vectors_normalized, start_normalized)
        similarities_to_end = np.dot(word_vectors_normalized, end_normalized)
        
        # Mark words that are too similar to start or end
        too_similar_to_start = similarities_to_start > similarity_threshold
        too_similar_to_end = similarities_to_end > similarity_threshold
        too_similar_mask = ~(too_similar_to_start | too_similar_to_end)
        
        # Combine with existing valid_mask
        valid_mask = valid_mask & too_similar_mask
        
        # Calculate similarities with +target and -target
        similarities_positive = np.dot(word_vectors_normalized, target_normalized)
        similarities_negative = np.dot(word_vectors_normalized, -target_normalized)
        
        # Mask out excluded words by setting their similarities to -inf
        similarities_positive = np.where(valid_mask, similarities_positive, -np.inf)
        similarities_negative = np.where(valid_mask, similarities_negative, -np.inf)
        
        # Find best match for positive (add) and negative (subtract)
        best_idx_positive = np.argmax(similarities_positive)
        best_sim_positive = similarities_positive[best_idx_positive]
        
        best_idx_negative = np.argmax(similarities_negative)
        best_sim_negative = similarities_negative[best_idx_negative]
        
        # Determine which is better
        if best_sim_positive > best_sim_negative:
            best_word = words[best_idx_positive]
            best_operation = "+"
            word_vector = word_vectors[best_idx_positive]
        else:
            best_word = words[best_idx_negative]
            best_operation = "-"
            word_vector = word_vectors[best_idx_negative]
        
        # Safety check: ensure the selected word is not in the exclude list
        if best_word.lower() in exclude_words:
            print(f"Error: Selected word '{best_word}' is in exclude list. This should not happen.")
            return
        
        # Apply the best move
        if best_operation == "+":
            self.current_vector = self.current_vector + word_vector
        else:
            self.current_vector = self.current_vector - word_vector
        
        # Update history
        self.word_history.append((best_word, best_operation))
        
        # Update displays
        self.update_similarity()
        self.update_current_vector()
        self.update_output()
        
        print(f"Cheat: {best_operation} {best_word}")
    
    def open_analogy_window(self):
        """Open a window to make an analogy"""
        if not self.game_started:
            return
        
        # Create new window
        analogy_window = tk.Toplevel(self.root)
        analogy_window.title("Make Analogy")
        analogy_window.geometry("400x200")
        
        # Frame for input
        input_frame = ttk.Frame(analogy_window, padding="20")
        input_frame.pack(fill=tk.BOTH, expand=True)
        
        ttk.Label(input_frame, text="Enter word X:").pack(pady=10)
        
        word_entry = ttk.Entry(input_frame, width=30)
        word_entry.pack(pady=10)
        word_entry.focus()
        
        error_label = ttk.Label(input_frame, text="", foreground="red")
        error_label.pack(pady=5)
        
        def calculate_analogy(event=None):
            """Calculate and display the analogy"""
            x_word = word_entry.get().strip().lower()
            error_label.config(text="")
            
            if not x_word:
                error_label.config(text="Please enter a word")
                return
            
            try:
                analogy_window.config(cursor="wait")
                analogy_window.update()
                
                # Get embeddings
                start_vector = self.get_embedding(self.start_word)
                end_vector = self.get_embedding(self.end_word)
                x_vector = self.get_embedding(x_word)
                
                # Calculate relationship vector and target
                relationship_vector = end_vector - start_vector
                target_vector = x_vector + relationship_vector
                
                # Normalize target
                target_norm = np.linalg.norm(target_vector)
                if target_norm == 0:
                    error_label.config(text="Could not calculate analogy")
                    analogy_window.config(cursor="")
                    return
                target_normalized = target_vector / target_norm
                
                # Get all word vectors as a matrix (like Glove version)
                words = self.words_list
                word_vectors = self.vectors_matrix
                
                # Exclude words: X word, start word, end word, and words substantially similar to start/end/X
                exclude_words = {self.start_word.lower(), self.end_word.lower(), x_word.lower()}
                
                # Get indices of all words to exclude
                exclude_indices = set()
                for word in exclude_words:
                    idx = self.word_to_index.get(word)
                    if idx is not None:
                        exclude_indices.add(idx)
                
                # Create mask to exclude all forbidden words
                valid_mask = np.ones(len(words), dtype=bool)
                for idx in exclude_indices:
                    valid_mask[idx] = False
                
                # Normalize word vectors
                word_norms = np.linalg.norm(word_vectors, axis=1, keepdims=True)
                word_norms = np.where(word_norms == 0, 1, word_norms)
                word_vectors_normalized = word_vectors / word_norms
                
                # Normalize start, end, and X vectors for similarity comparison
                start_emb = self.get_embedding(self.start_word)
                end_emb = self.get_embedding(self.end_word)
                x_emb = self.get_embedding(x_word)
                
                start_norm = np.linalg.norm(start_emb)
                end_norm = np.linalg.norm(end_emb)
                x_norm = np.linalg.norm(x_emb)
                
                start_normalized = start_emb / start_norm if start_norm > 0 else start_emb
                end_normalized = end_emb / end_norm if end_norm > 0 else end_emb
                x_normalized = x_emb / x_norm if x_norm > 0 else x_emb
                
                # Calculate similarities with start, end, and X words
                similarity_threshold = 0.8  # Threshold for "substantially similar"
                similarities_to_start = np.dot(word_vectors_normalized, start_normalized)
                similarities_to_end = np.dot(word_vectors_normalized, end_normalized)
                similarities_to_x = np.dot(word_vectors_normalized, x_normalized)
                
                # Mark words that are too similar to start, end, or X
                too_similar_to_start = similarities_to_start > similarity_threshold
                too_similar_to_end = similarities_to_end > similarity_threshold
                too_similar_to_x = similarities_to_x > similarity_threshold
                
                # Also exclude words too similar to start, end, or X
                valid_mask = valid_mask & ~(too_similar_to_start | too_similar_to_end | too_similar_to_x)
                
                # Calculate similarities with target
                similarities = np.dot(word_vectors_normalized, target_normalized)
                
                # Mask out excluded words by setting their similarities to -inf
                similarities = np.where(valid_mask, similarities, -np.inf)
                
                # Find best match (next closest word after exclusions)
                best_idx = np.argmax(similarities)
                best_word = words[best_idx]
                
                analogy_window.config(cursor="")
                
                # Check if we found a valid word (similarity should be > -inf)
                if similarities[best_idx] == -np.inf:
                    error_label.config(text="Could not find analogy completion")
                    return
                
                # Display result
                analogy_text = f"{self.start_word} is to {self.end_word} as {x_word} is to {best_word}"
                self.output_text.config(state=tk.NORMAL)
                self.output_text.insert(tk.END, f"\n{analogy_text}")
                self.output_text.config(state=tk.DISABLED)
                
                # Close window
                analogy_window.destroy()
                
            except Exception as e:
                analogy_window.config(cursor="")
                error_label.config(text=f"Error: {str(e)}")
        
        # Bind Enter key
        word_entry.bind('<Return>', calculate_analogy)
        
        # Submit button
        submit_btn = ttk.Button(input_frame, text="Enter", command=calculate_analogy)
        submit_btn.pack(pady=10)


def main():
    root = tk.Tk()
    app = WordGameBERT(root)
    root.mainloop()


if __name__ == "__main__":
    main()

