"""
Wordle Environment for training ML Agents.
"""

import pandas as pd
import random
from typing import List, Tuple, Dict


class WordleEnv:
    """
    Wordle game environment for ML agents to learn.

    State representation:
    - Feedback history from previous guesses
    - Remaining valid words

    Action space: Choose a word from the valid words list
    """

    def __init__(
        self, word_list_path: str = "data/filtered_words.csv", max_attempts: int = 6
    ):
        """
        Initialize Wordle environment.

        Args:
            word_list_path: Path to CSV file with 5-letter words
            max_attempts: Maximum number of guesses allowed (default: 6)
        """
        # Load word list
        df = pd.read_csv(word_list_path)
        self.all_words = df["word"].str.lower().tolist()
        self.word_frequencies = dict(zip(df["word"].str.lower(), df["count"]))

        # Game state
        self.max_attempts = max_attempts
        self.reset()

    def reset(self, target_word: str = None) -> Dict:
        """
        Reset the game with a new target word.

        Args:
            target_word: Specific word to use as target (default: random)

        Returns:
            Initial state dictionary
        """
        if target_word is None:
            self.target_word = random.choice(self.all_words)
        else:
            self.target_word = target_word.lower()

        self.attempts_used = 0
        self.guess_history = []
        self.feedback_history = []
        self.valid_words = self.all_words.copy()
        self.solved = False

        return self.get_state()

    def get_feedback(self, guess: str) -> List[str]:
        """
        Get string-encoded color feedback for a guess.

        Args:
            guess: The guessed word

        Returns:
            List of feedback: ["green", "yellow", "gray"] for each position
        """
        guess = guess.lower()
        target = self.target_word
        feedback = ["gray"] * 5

        # Count letter freq in target
        target_letter_counts = {}
        for letter in target:
            target_letter_counts[letter] = target_letter_counts.get(letter, 0) + 1

        # First pass through 5-letter guess: mark greens (exact matches)
        for i in range(5):
            if guess[i] == target[i]:
                feedback[i] = "green"
                target_letter_counts[guess[i]] -= 1

        # Second pass: mark yellows (wrong position)
        for i in range(5):
            if feedback[i] == "gray" and guess[i] in target:
                if target_letter_counts.get(guess[i], 0) > 0:
                    feedback[i] = "yellow"
                    target_letter_counts[guess[i]] -= 1

        return feedback

    def step(self, guess: str) -> Tuple[Dict, bool, Dict]:
        """
        Take a step in the environment by making a guess.

        Args:
            guess: The word to guess

        Returns:
            Tuple of (state, done, info)
            - state: Current game state
            - done: Whether the game is over
            - info: Additional information (solved, target_word)
        """
        guess = guess.lower()

        # Validate guess
        if len(guess) != 5 or guess not in self.all_words:
            return self.get_state(), False, {"valid_guess": False}

        # Get feedback
        feedback = self.get_feedback(guess)

        # Update history
        self.guess_history.append(guess)
        self.feedback_history.append(feedback)
        self.attempts_used += 1

        # Filter valid words based on feedback
        self.valid_words = self._filter_valid_words(guess, feedback)

        # Check if solved
        done = False

        if guess == self.target_word:
            self.solved = True
            done = True
        elif self.attempts_used >= self.max_attempts:
            done = True

        state = self.get_state()
        info = {
            "solved": self.solved,
            "target_word": self.target_word if done else None,
            "attempts_used": self.attempts_used,
            "valid_words_remaining": len(self.valid_words),
            "valid_guess": True,
        }

        return state, done, info

    def _filter_valid_words(self, guess: str, feedback: List[str]) -> List[str]:
        """
        Filter valid words based on feedback from a guess.

        Args:
            guess: The guessed word
            feedback: Feedback for each letter

        Returns:
            List of words still possible given the feedback
        """
        valid = []

        for word in self.valid_words:
            if self._is_consistent(word, guess, feedback):
                valid.append(word)

        return valid

    def _is_consistent(self, word: str, guess: str, feedback: List[str]) -> bool:
        """
        Check if a word is consistent with the guess and feedback.

        Args:
            word: Word to check
            guess: The guess that was made
            feedback: Feedback received

        Returns:
            True if word could still be the answer
        """
        # Count letter freq in word and guess
        word_letter_counts = {}
        for letter in word:
            word_letter_counts[letter] = word_letter_counts.get(letter, 0) + 1

        # Check green positions
        for i in range(5):
            if feedback[i] == "green":
                if word[i] != guess[i]:
                    return False

        # Check yellow and gray
        guess_letter_info = {}
        for i in range(5):
            letter = guess[i]
            if letter not in guess_letter_info:
                # accumulate min and max number of occurrences of letter based on feedback
                guess_letter_info[letter] = {
                    "min": 0,
                    "max": float("inf"),
                    "not_positions": set(),
                }

            if feedback[i] == "green":
                guess_letter_info[letter]["min"] += 1
            elif feedback[i] == "yellow":
                guess_letter_info[letter]["min"] += 1
                guess_letter_info[letter]["not_positions"].add(i)
            elif feedback[i] == "gray":
                # Gray means this letter doesn't appear MORE times than already found
                greens_yellows = sum(
                    1
                    for j, f in enumerate(feedback)
                    if f in ["green", "yellow"] and guess[j] == letter
                )
                guess_letter_info[letter]["max"] = greens_yellows

        # Validate word against constraints learned from feedback
        for letter, info in guess_letter_info.items():
            count_in_word = word_letter_counts.get(letter, 0)

            # Check minimum occurrences
            if count_in_word < info["min"]:
                return False

            # Check maximum occurrences
            if count_in_word > info["max"]:
                return False

            # Check positions for yellow letters
            for pos in info["not_positions"]:
                if word[pos] == letter:
                    return False

        return True

    def get_state(self) -> Dict:
        """
        Get current state representation.

        Returns:
            Dictionary with game state information
        """
        return {
            "guess_history": self.guess_history.copy(),
            "feedback_history": [f.copy() for f in self.feedback_history],
            "valid_words": self.valid_words.copy(),
            "attempts_used": self.attempts_used,
            "attempts_remaining": self.max_attempts - self.attempts_used,
        }

    def get_valid_actions(self) -> List[str]:
        """
        Get list of valid words that can be guessed.

        Returns:
            List of valid words
        """
        return self.valid_words.copy()

    def render(self):
        """
        Print current game state to console.
        """
        print(f"\nAttempt {self.attempts_used}/{self.max_attempts}")
        print(f"Valid words remaining: {len(self.valid_words)}")

        for guess, feedback in zip(self.guess_history, self.feedback_history):
            emoji_feedback = []
            for f in feedback:
                if f == "green":
                    emoji_feedback.append("🟩")
                elif f == "yellow":
                    emoji_feedback.append("🟨")
                else:
                    emoji_feedback.append("⬜")
            print(f"{guess.upper()}: {''.join(emoji_feedback)}")


# Helper function to test the environment
def test_environment():
    """Test the Wordle environment with a simple game."""
    env = WordleEnv("data/filtered_words.csv")

    # Test with a known word
    state_ = env.reset(target_word="creed")
    print(f"Target word: {env.target_word}")

    test_guesses = ["about", "creed"]

    for guess in test_guesses:
        print(f"\nGuessing: {guess}")
        state, done, info = env.step(guess)
        env.render()
        print(f"Done: {done}")
        print(f"Info: {info}")

        if done:
            break


if __name__ == "__main__":
    test_environment()
