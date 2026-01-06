import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import numpy as np
import os
from openai import OpenAI
import json


class WordGameGPT:
    def __init__(self, root):
        self.root = root
        self.root.title("Word Game with ChatGPT API")
        self.root.geometry("800x700")
        
        # Initialize OpenAI client
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            # Try to load from file
            if os.path.exists("openai_api_key.txt"):
                with open("openai_api_key.txt", "r") as f:
                    api_key = f.read().strip()
        
        if not api_key:
            messagebox.showerror("API Key Required", 
                                "Please set OPENAI_API_KEY environment variable or create openai_api_key.txt file with your API key.")
            root.destroy()
            return
        
        self.client = OpenAI(api_key=api_key)
        self.embedding_model = "text-embedding-3-small"  # or "text-embedding-ada-002"
        
        # Cache for embeddings to avoid repeated API calls
        self.embedding_cache = {}
        self.embedding_cache_file = "embedding_cache.json"
        self.load_embedding_cache()
        
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
    
    def load_embedding_cache(self):
        """Load embedding cache from file"""
        if os.path.exists(self.embedding_cache_file):
            try:
                with open(self.embedding_cache_file, "r") as f:
                    self.embedding_cache = json.load(f)
                print(f"Loaded {len(self.embedding_cache)} cached embeddings")
            except Exception as e:
                print(f"Error loading cache: {e}")
                self.embedding_cache = {}
    
    def save_embedding_cache(self):
        """Save embedding cache to file"""
        try:
            with open(self.embedding_cache_file, "w") as f:
                json.dump(self.embedding_cache, f)
        except Exception as e:
            print(f"Error saving cache: {e}")
    
    def get_embedding(self, word):
        """Get embedding for a word, using cache if available"""
        word_lower = word.lower()
        
        # Check cache first
        if word_lower in self.embedding_cache:
            return np.array(self.embedding_cache[word_lower], dtype=np.float32)
        
        # Get embedding from API
        try:
            response = self.client.embeddings.create(
                model=self.embedding_model,
                input=word_lower
            )
            embedding = np.array(response.data[0].embedding, dtype=np.float32)
            
            # Cache it
            self.embedding_cache[word_lower] = embedding.tolist()
            self.save_embedding_cache()
            
            return embedding
        except Exception as e:
            print(f"Error getting embedding for '{word}': {e}")
            raise
    
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
        
        # Reset button
        reset_frame = ttk.Frame(self.root, padding="10")
        reset_frame.pack(fill=tk.X)
        
        self.reset_btn = ttk.Button(reset_frame, text="Reset", command=self.reset_game)
        self.reset_btn.pack(side=tk.LEFT, padx=5, pady=5)
    
    def disable_game_controls(self):
        """Disable game controls until game starts"""
        self.new_word_entry.config(state=tk.DISABLED)
        self.add_btn.config(state=tk.DISABLED)
        self.subtract_btn.config(state=tk.DISABLED)
        self.reset_btn.config(state=tk.DISABLED)
    
    def enable_game_controls(self):
        """Enable game controls after game starts"""
        self.new_word_entry.config(state=tk.NORMAL)
        self.add_btn.config(state=tk.NORMAL)
        self.subtract_btn.config(state=tk.NORMAL)
        self.reset_btn.config(state=tk.NORMAL)
    
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
            error_msg = str(e)
            if "not in database" in error_msg.lower() or "not found" in error_msg.lower():
                if "start" in error_msg.lower() or start_word not in str(e):
                    self.end_error_label.config(text="not in database")
                else:
                    self.start_error_label.config(text="not in database")
            else:
                messagebox.showerror("Error", f"Failed to get embeddings: {e}")
    
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
            messagebox.showerror("Error", f"Failed to reset: {e}")
    
def main():
    root = tk.Tk()
    app = WordGameGPT(root)
    root.mainloop()


if __name__ == "__main__":
    main()

