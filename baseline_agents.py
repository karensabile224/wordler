"""
Baseline Wordle Agents for Comparison
These agents provide benchmarks to evaluate RL performance against.
"""

import random
from typing import Dict


class BaseAgent:
    """Base class for Wordle agents."""

    def __init__(self, word_frequencies: Dict[str, int] = None):
        """
        Initialize agent.

        Args:
            word_frequencies: Dictionary mapping words to frequency counts
        """
        self.word_frequencies = word_frequencies or {}

    def choose_word(self, state: Dict) -> str:
        """
        Choose next word to guess.

        Args:
            state: Current game state

        Returns:
            Word to guess
        """
        raise NotImplementedError


class RandomAgent(BaseAgent):
    """Agent that guesses randomly from valid words."""

    def choose_word(self, state: Dict) -> str:
        """Choose random word from valid words."""
        valid_words = state["valid_words"]
        if not valid_words:
            return random.choice(list(self.word_frequencies.keys()))
        return random.choice(valid_words)


class FrequencyAgent(BaseAgent):
    """Agent that always guesses the most frequent valid word."""

    def choose_word(self, state: Dict) -> str:
        """Choose most frequent word from valid words."""
        valid_words = state["valid_words"]

        if not valid_words:
            return max(
                self.word_frequencies,
                key=lambda w: self.word_frequencies.get(w, 0),
            )

        return max(valid_words, key=lambda w: self.word_frequencies.get(w, 0))


def evaluate_agent(
    agent: BaseAgent, env, num_games: int = 100, verbose: bool = False
) -> Dict:
    """
    Evaluate an agent's performance.

    Args:
        agent: Agent to evaluate
        env: Wordle environment
        num_games: Number of games to play
        verbose: Whether to print each game

    Returns:
        Dictionary with performance metrics
    """
    results = {
        "games_played": num_games,
        "games_won": 0,
        "total_guesses": 0,
        "guess_distribution": {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, "failed": 0},
    }

    for game_num in range(num_games):
        state = env.reset()
        done = False

        if verbose:
            print(f"\n--- Game {game_num + 1} ---")
            print(f"Target: {env.target_word}")

        while not done:
            guess = agent.choose_word(state)
            state, done, info = env.step(guess)

            if verbose:
                print(f"Guess {info['attempts_used']}: {guess}")

        # Record results
        if info["solved"]:
            results["games_won"] += 1
            results["total_guesses"] += info["attempts_used"]
            results["guess_distribution"][info["attempts_used"]] += 1
        else:
            results["guess_distribution"]["failed"] += 1
            results["total_guesses"] += 6  # Count as max attempts

        if verbose:
            print(
                f"{'Won' if info['solved'] else 'Lost'} in {info['attempts_used']} guesses"
            )

    # Calculate statistics
    results["win_rate"] = results["games_won"] / num_games
    results["avg_guesses"] = results["total_guesses"] / num_games
    if results["games_won"] > 0:
        results["avg_guesses_when_won"] = (
            results["total_guesses"] / results["games_won"]
        )

    return results


def print_results(agent_name: str, results: Dict):
    """Print formatted results."""
    print(f"\n{'='*50}")
    print(f"{agent_name} Results")
    print(f"{'='*50}")
    print(f"Games played: {results['games_played']}")
    print(f"Win rate: {results['win_rate']:.1%}")
    print(f"Average guesses (all games): {results['avg_guesses']:.2f}")
    if "avg_guesses_when_won" in results:
        print(f"Average guesses (won games): {results['avg_guesses_when_won']:.2f}")
    print(f"\nGuess distribution:")
    for i in range(1, 7):
        count = results["guess_distribution"][i]
        # bar graph with at most 50 bars wide
        bar = "🟫" * (count * 50 // results["games_played"])
        print(f"  {i}: {count:3d} {bar}")
    print(f"  X: {results['guess_distribution']['failed']:3d}")


if __name__ == "__main__":
    from wordle_env import WordleEnv

    # Initialize environment
    env = WordleEnv("data/filtered_words.csv")

    # Test each agent
    agents = [
        ("Random Agent", RandomAgent(env.word_frequencies)),
        ("Frequency Agent", FrequencyAgent(env.word_frequencies)),
    ]

    for agent_name, agent in agents:
        results = evaluate_agent(agent, env, num_games=10, verbose=False)
        print_results(agent_name, results)
