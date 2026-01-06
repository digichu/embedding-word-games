import tkinter as tk
from tkinter import ttk, scrolledtext
import numpy as np
from gensim.models import KeyedVectors
import os
from tqdm import tqdm


class WordGame:
    def __init__(self, root):
        self.root = root
        self.root.title("Word Game with Glove Vectors")
        self.root.geometry("800x700")
        
        # Load vectors
        vector_file = "wiki_giga_2024_300_MFT20_vectors_seed_2024_alpha_0.75_eta_0.05_combined.txt"
        if not os.path.exists(vector_file):
            raise FileNotFoundError(f"Vector file not found: {vector_file}")
        
        print("Loading vectors... This may take a moment.")
        # Load Glove format (space-separated, no header)
        # Glove format: word value1 value2 ... valueN
        try:
            # Try loading as Glove format (no header, space-separated) with progress
            self.vectors = self._load_vectors_with_progress(
                vector_file,
                binary=False,
                no_header=True
            )
        except Exception as e:
            # If that fails, try with header (word2vec format)
            try:
                self.vectors = self._load_vectors_with_progress(
                    vector_file,
                    binary=False,
                    no_header=False
                )
            except Exception as e2:
                raise Exception(f"Failed to load vectors. Error 1: {e}, Error 2: {e2}")
        
        print(f"\nLoaded {len(self.vectors)} vectors")
        
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
    
    def _load_vectors_with_progress(self, vector_file, binary=False, no_header=True, encoding='utf-8'):
        """Load vectors with progress tracking using gensim's optimized loader"""
        file_size = os.path.getsize(vector_file)
        
        print("Loading vectors...")
        # Use gensim's fast loader - it's much faster than manual parsing
        # We'll show progress by monitoring file access
        with tqdm(total=file_size, unit='B', unit_scale=True, desc="Loading vectors", initial=0) as pbar:
            # Gensim loads the file internally, so we can't track progress directly
            # But we can update the bar to show it's working
            vectors = KeyedVectors.load_word2vec_format(
                vector_file,
                binary=False,
                no_header=no_header,
                encoding=encoding
            )
            # Update to 100% when done
            pbar.update(file_size)
        
        return vectors
    
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
        
        # Validate words
        start_valid = start_word in self.vectors.key_to_index
        end_valid = end_word in self.vectors.key_to_index
        
        if not start_valid:
            self.start_error_label.config(text="not in database")
        
        if not end_valid:
            self.end_error_label.config(text="not in database")
        
        # Only proceed if both words are valid
        if not (start_valid and end_valid):
            return
        
        # Initialize game
        self.start_word = start_word
        self.end_word = end_word
        self.current_vector = self.vectors[start_word].copy()
        self.end_vector = self.vectors[end_word].copy()
        self.game_started = True
        self.word_history = [start_word]
        self.victory_shown = False
        
        # Enable game controls
        self.enable_game_controls()
        
        # Update displays
        self.update_similarity()
        self.update_current_vector()
        self.update_output()
    
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
        
        if word not in self.vectors.key_to_index:
            self.new_word_error_label.config(text="not in database")
            return
        
        # Add the word vector
        word_vector = self.vectors[word]
        self.current_vector = self.current_vector + word_vector
        
        # Update history
        self.word_history.append((word, "+"))
        
        # Update displays
        self.update_similarity()
        self.update_current_vector()
        self.update_output()
        
        # Clear the entry
        self.new_word_entry.delete(0, tk.END)
    
    def subtract_word(self):
        """Subtract a word vector from the current vector"""
        self.new_word_error_label.config(text="")
        word = self.new_word_entry.get().strip().lower()
        
        if not word:
            return
        
        if word not in self.vectors.key_to_index:
            self.new_word_error_label.config(text="not in database")
            return
        
        # Subtract the word vector
        word_vector = self.vectors[word]
        self.current_vector = self.current_vector - word_vector
        
        # Update history
        self.word_history.append((word, "-"))
        
        # Update displays
        self.update_similarity()
        self.update_current_vector()
        self.update_output()
        
        # Clear the entry
        self.new_word_entry.delete(0, tk.END)
    
    def reset_game(self):
        """Reset the game to initial state"""
        if not self.game_started:
            return
        
        # Reset to start word
        self.current_vector = self.vectors[self.start_word].copy()
        self.word_history = [self.start_word]
        self.victory_shown = False
        
        # Update displays
        self.update_similarity()
        self.update_current_vector()
        self.update_output()
        
        # Clear new word entry
        self.new_word_entry.delete(0, tk.END)
        self.new_word_error_label.config(text="")
    
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
        
        # Get all word vectors as a matrix (much faster than looping)
        # Use index_to_key to ensure proper word-vector mapping
        if hasattr(self.vectors, 'index_to_key'):
            words = self.vectors.index_to_key
        else:
            # Fallback: create ordered list from indices
            words = [None] * len(self.vectors.key_to_index)
            for word, idx in self.vectors.key_to_index.items():
                words[idx] = word
        
        word_vectors = self.vectors.vectors  # This is the matrix of all vectors
        
        # Collect all words to exclude: start word, end word, and all words already used
        exclude_words = set()
        exclude_words.add(self.start_word.lower())
        exclude_words.add(self.end_word.lower())
        
        # Add all words that have already been used (from word_history)
        # word_history format: [start_word, (word1, op1), (word2, op2), ...]
        for item in self.word_history[1:]:  # Skip the first item (start word)
            if isinstance(item, tuple):
                word, _ = item
                exclude_words.add(word.lower())
        
        # Get indices of all words to exclude
        exclude_indices = set()
        for word in exclude_words:
            idx = self.vectors.key_to_index.get(word)
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
        start_vector = self.vectors[self.start_word]
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
            
            # Validate word
            if not x_word:
                error_label.config(text="Please enter a word")
                return
            
            if x_word not in self.vectors.key_to_index:
                error_label.config(text="not in database")
                return
            
            # Calculate analogy
            # A is to B as X is to Y
            # Vector relationship: B - A, then add to X
            relationship_vector = self.end_vector - self.vectors[self.start_word]
            x_vector = self.vectors[x_word]
            target_vector = x_vector + relationship_vector
            
            # Find closest word to target_vector
            # Normalize target vector
            target_norm = np.linalg.norm(target_vector)
            if target_norm == 0:
                error_label.config(text="Could not calculate analogy")
                return
            target_normalized = target_vector / target_norm
            
            # Get all word vectors
            if hasattr(self.vectors, 'index_to_key'):
                words = self.vectors.index_to_key
            else:
                words = [None] * len(self.vectors.key_to_index)
                for word, idx in self.vectors.key_to_index.items():
                    words[idx] = word
            
            word_vectors = self.vectors.vectors
            
            # Normalize word vectors
            word_norms = np.linalg.norm(word_vectors, axis=1, keepdims=True)
            word_norms = np.where(word_norms == 0, 1, word_norms)
            word_vectors_normalized = word_vectors / word_norms
            
            # Calculate cosine similarities
            similarities = np.dot(word_vectors_normalized, target_normalized)
            
            # Exclude words: X word, start word, end word, and words substantially similar to start/end
            exclude_indices = set()
            
            # Exclude X word
            x_idx = self.vectors.key_to_index.get(x_word)
            if x_idx is not None:
                exclude_indices.add(x_idx)
            
            # Exclude start word
            start_idx = self.vectors.key_to_index.get(self.start_word)
            if start_idx is not None:
                exclude_indices.add(start_idx)
            
            # Exclude end word
            end_idx = self.vectors.key_to_index.get(self.end_word)
            if end_idx is not None:
                exclude_indices.add(end_idx)
            
            # Exclude words substantially similar to start or end word
            similarity_threshold = 0.8  # Threshold for "substantially similar"
            
            # Normalize start and end vectors
            start_vector = self.vectors[self.start_word]
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
            
            # Calculate similarities with start and end words
            similarities_to_start = np.dot(word_vectors_normalized, start_normalized)
            similarities_to_end = np.dot(word_vectors_normalized, end_normalized)
            
            # Calculate similarities with X word (to exclude variations)
            x_vector = self.vectors[x_word]
            x_norm = np.linalg.norm(x_vector)
            if x_norm > 0:
                x_normalized = x_vector / x_norm
            else:
                x_normalized = x_vector
            similarities_to_x = np.dot(word_vectors_normalized, x_normalized)
            
            # Mark words that are too similar to start, end, or X
            too_similar_to_start = similarities_to_start > similarity_threshold
            too_similar_to_end = similarities_to_end > similarity_threshold
            too_similar_to_x = similarities_to_x > similarity_threshold
            
            # Create mask to exclude all forbidden words
            valid_mask = np.ones(len(words), dtype=bool)
            for idx in exclude_indices:
                valid_mask[idx] = False
            
            # Also exclude words too similar to start, end, or X
            valid_mask = valid_mask & ~(too_similar_to_start | too_similar_to_end | too_similar_to_x)
            
            # Mask out excluded words by setting their similarities to -inf
            similarities = np.where(valid_mask, similarities, -np.inf)
            
            # Find best match (next closest word after exclusions)
            best_idx = np.argmax(similarities)
            y_word = words[best_idx]
            
            # Display result in output area
            # Format: "A is to B as X is to Y"
            analogy_text = f"{self.start_word} is to {self.end_word} as {x_word} is to {y_word}"
            self.output_text.config(state=tk.NORMAL)
            self.output_text.insert(tk.END, f"\n{analogy_text}")
            self.output_text.config(state=tk.DISABLED)
            
            # Close window
            analogy_window.destroy()
        
        # Bind Enter key
        word_entry.bind('<Return>', calculate_analogy)
        
        # Submit button
        submit_btn = ttk.Button(input_frame, text="Enter", command=calculate_analogy)
        submit_btn.pack(pady=10)


def main():
    root = tk.Tk()
    app = WordGame(root)
    root.mainloop()


if __name__ == "__main__":
    main()

