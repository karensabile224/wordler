"""
Comprehensive evaluation script for A2C Wordle solver.

This script evaluates a trained model on different word sets and generates
detailed metrics and visualizations.

Usage in Colab:
    python evaluate_model.py --checkpoint checkpoints/a2c-epoch=10.ckpt

Features:
- Evaluation on training vocabulary (check for learning)
- Evaluation on held-out words (check for generalization)
- Detailed metrics: win rate, avg turns, turn distribution
- Visualizations: histograms, per-word analysis
- Export results to CSV for further analysis
"""

import argparse
import json
from collections import defaultdict, Counter
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

import a2c.play
import wordle.state


def evaluate_on_words(
    agent,
    env,
    word_list: List[str],
    max_episodes: Optional[int] = None,
    verbose: bool = True,
) -> Dict:
    """
    Evaluate agent on a list of goal words.

    Args:
        agent: GreedyActorCriticAgent (deterministic)
        env: Wordle environment
        word_list: List of words to use as goals
        max_episodes: Optional limit on number of words to test
        verbose: Whether to show progress bar

    Returns:
        Dictionary with detailed metrics
    """
    n_words = len(word_list) if max_episodes is None else min(len(word_list), max_episodes)

    results = {
        'wins': 0,
        'losses': 0,
        'total_turns': 0,
        'winning_turns': 0,
        'turn_distribution': Counter(),  # Distribution of turns to win (1-6)
        'failed_words': [],  # List of (word, guesses) for failures
        'per_word_results': [],  # List of dicts for each word
    }

    iterator = tqdm(word_list[:n_words], desc="Evaluating") if verbose else word_list[:n_words]

    for goal_word in iterator:
        try:
            win, outcomes = a2c.play.goal(agent, env, goal_word)
            n_turns = len(outcomes)

            word_result = {
                'goal_word': goal_word,
                'win': win,
                'turns': n_turns,
                'guesses': [guess for guess, _ in outcomes],
            }

            results['per_word_results'].append(word_result)
            results['total_turns'] += n_turns

            if win:
                results['wins'] += 1
                results['winning_turns'] += n_turns
                results['turn_distribution'][n_turns] += 1
            else:
                results['losses'] += 1
                results['failed_words'].append((goal_word, [g for g, _ in outcomes]))

        except Exception as e:
            print(f"Error evaluating {goal_word}: {e}")
            results['losses'] += 1
            results['failed_words'].append((goal_word, []))

    # Compute aggregate metrics
    total_games = results['wins'] + results['losses']
    results['win_rate'] = results['wins'] / total_games if total_games > 0 else 0.0
    results['avg_turns_all'] = results['total_turns'] / total_games if total_games > 0 else 0.0
    results['avg_turns_wins'] = results['winning_turns'] / results['wins'] if results['wins'] > 0 else 0.0

    return results


def print_results(results: Dict, title: str = "Evaluation Results"):
    """Pretty print evaluation results."""
    print("\n" + "="*60)
    print(f"{title:^60}")
    print("="*60)

    total = results['wins'] + results['losses']
    print(f"\nGames Played: {total}")
    print(f"Wins: {results['wins']} ({results['win_rate']*100:.2f}%)")
    print(f"Losses: {results['losses']} ({(1-results['win_rate'])*100:.2f}%)")
    print(f"\nAverage Turns (all games): {results['avg_turns_all']:.2f}")
    print(f"Average Turns (wins only): {results['avg_turns_wins']:.2f}")

    # Turn distribution
    if results['turn_distribution']:
        print("\nTurn Distribution (wins only):")
        for turn in sorted(results['turn_distribution'].keys()):
            count = results['turn_distribution'][turn]
            pct = count / results['wins'] * 100 if results['wins'] > 0 else 0
            bar = '█' * int(pct / 2)  # Scale bar to 50 chars max
            print(f"  {turn} turns: {count:4d} ({pct:5.1f}%) {bar}")

    # Failed words sample
    if results['failed_words']:
        print(f"\nFailed Words (showing first 10 of {len(results['failed_words'])}):")
        for word, guesses in results['failed_words'][:10]:
            guess_str = ' → '.join(guesses) if guesses else '(error)'
            print(f"  {word}: {guess_str}")

    print("="*60 + "\n")


def plot_turn_distribution(results: Dict, title: str = "Turn Distribution", save_path: Optional[str] = None):
    """Plot histogram of turns to win."""
    if not results['turn_distribution']:
        print("No wins to plot!")
        return

    turns = sorted(results['turn_distribution'].keys())
    counts = [results['turn_distribution'][t] for t in turns]

    plt.figure(figsize=(10, 6))
    plt.bar(turns, counts, color='steelblue', edgecolor='black', alpha=0.7)
    plt.xlabel('Number of Turns', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xticks(turns)
    plt.grid(axis='y', alpha=0.3)

    # Add percentage labels on bars
    for turn, count in zip(turns, counts):
        pct = count / results['wins'] * 100 if results['wins'] > 0 else 0
        plt.text(turn, count, f'{pct:.1f}%', ha='center', va='bottom', fontsize=10)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {save_path}")

    plt.show()


def compare_results(results_dict: Dict[str, Dict], save_path: Optional[str] = None):
    """Compare multiple evaluation results side-by-side."""
    print("\n" + "="*80)
    print(f"{'Comparison Across Evaluation Sets':^80}")
    print("="*80)

    # Create comparison table
    metrics = ['win_rate', 'avg_turns_all', 'avg_turns_wins']
    headers = ['Evaluation Set', 'Win Rate', 'Avg Turns (All)', 'Avg Turns (Wins)']

    print(f"\n{headers[0]:30s} {headers[1]:>15s} {headers[2]:>18s} {headers[3]:>18s}")
    print("-" * 80)

    for name, results in results_dict.items():
        total = results['wins'] + results['losses']
        print(f"{name:30s} {results['win_rate']*100:14.2f}% "
              f"{results['avg_turns_all']:18.2f} {results['avg_turns_wins']:18.2f}")

    print("="*80 + "\n")

    # Plot comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    names = list(results_dict.keys())
    win_rates = [results_dict[n]['win_rate'] * 100 for n in names]
    avg_turns_all = [results_dict[n]['avg_turns_all'] for n in names]
    avg_turns_wins = [results_dict[n]['avg_turns_wins'] for n in names]

    # Win rate
    axes[0].bar(names, win_rates, color='green', alpha=0.7, edgecolor='black')
    axes[0].set_ylabel('Win Rate (%)', fontsize=12)
    axes[0].set_title('Win Rate Comparison', fontweight='bold')
    axes[0].set_ylim(0, 100)
    axes[0].grid(axis='y', alpha=0.3)
    for i, v in enumerate(win_rates):
        axes[0].text(i, v + 1, f'{v:.1f}%', ha='center', fontsize=10)

    # Avg turns (all)
    axes[1].bar(names, avg_turns_all, color='steelblue', alpha=0.7, edgecolor='black')
    axes[1].set_ylabel('Average Turns', fontsize=12)
    axes[1].set_title('Avg Turns (All Games)', fontweight='bold')
    axes[1].set_ylim(0, 6)
    axes[1].grid(axis='y', alpha=0.3)
    for i, v in enumerate(avg_turns_all):
        axes[1].text(i, v + 0.1, f'{v:.2f}', ha='center', fontsize=10)

    # Avg turns (wins)
    axes[2].bar(names, avg_turns_wins, color='orange', alpha=0.7, edgecolor='black')
    axes[2].set_ylabel('Average Turns', fontsize=12)
    axes[2].set_title('Avg Turns (Wins Only)', fontweight='bold')
    axes[2].set_ylim(0, 6)
    axes[2].grid(axis='y', alpha=0.3)
    for i, v in enumerate(avg_turns_wins):
        axes[2].text(i, v + 0.1, f'{v:.2f}', ha='center', fontsize=10)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved comparison plot to {save_path}")

    plt.show()


def save_results_to_csv(results: Dict, output_path: str):
    """Save per-word results to CSV for further analysis."""
    df = pd.DataFrame(results['per_word_results'])
    df['guesses_str'] = df['guesses'].apply(lambda x: ' → '.join(x))
    df = df[['goal_word', 'win', 'turns', 'guesses_str']]
    df.to_csv(output_path, index=False)
    print(f"Saved detailed results to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate A2C Wordle solver")
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint (.ckpt file)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='evaluation_results',
        help='Directory to save results and plots'
    )
    parser.add_argument(
        '--max_words',
        type=int,
        default=None,
        help='Maximum number of words to evaluate (for quick testing)'
    )
    parser.add_argument(
        '--skip_plots',
        action='store_true',
        help='Skip generating plots (useful for headless environments)'
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    print(f"\n{'='*60}")
    print(f"{'A2C Wordle Evaluation':^60}")
    print(f"{'='*60}\n")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Output dir: {output_dir}")
    print(f"\nLoading model...")

    # Load model and create greedy agent
    model, agent, env = a2c.play.load_from_checkpoint(args.checkpoint)

    print(f"✓ Model loaded successfully")
    print(f"  Environment: {env.unwrapped.spec.id if hasattr(env, 'unwrapped') else 'Unknown'}")
    print(f"  Total vocabulary: {len(env.words)} words")
    print(f"  Allowable goal words: {env.allowable_words} words")

    # Prepare word sets
    training_words = env.words[:env.allowable_words]
    all_words = env.words

    # If vocabulary is large, also test on held-out subset
    held_out_words = []
    if len(all_words) > env.allowable_words:
        held_out_words = all_words[env.allowable_words:env.allowable_words + min(100, len(all_words) - env.allowable_words)]

    results_dict = {}

    # ========================================
    # Evaluation 1: Training vocabulary
    # ========================================
    print(f"\n{'─'*60}")
    print("📊 Evaluating on TRAINING vocabulary")
    print(f"{'─'*60}")

    results_train = evaluate_on_words(
        agent, env, training_words,
        max_episodes=args.max_words,
        verbose=True
    )
    print_results(results_train, title="Training Vocabulary Results")
    results_dict['Training Words'] = results_train

    # Save detailed results
    save_results_to_csv(results_train, output_dir / 'training_words_detailed.csv')

    if not args.skip_plots:
        plot_turn_distribution(
            results_train,
            title=f"Turn Distribution - Training Words (n={results_train['wins']})",
            save_path=output_dir / 'training_turn_distribution.png'
        )

    # ========================================
    # Evaluation 2: Held-out words (if available)
    # ========================================
    if held_out_words:
        print(f"\n{'─'*60}")
        print("📊 Evaluating on HELD-OUT vocabulary (generalization test)")
        print(f"{'─'*60}")

        results_heldout = evaluate_on_words(
            agent, env, held_out_words,
            max_episodes=args.max_words,
            verbose=True
        )
        print_results(results_heldout, title="Held-Out Vocabulary Results")
        results_dict['Held-Out Words'] = results_heldout

        save_results_to_csv(results_heldout, output_dir / 'heldout_words_detailed.csv')

        if not args.skip_plots:
            plot_turn_distribution(
                results_heldout,
                title=f"Turn Distribution - Held-Out Words (n={results_heldout['wins']})",
                save_path=output_dir / 'heldout_turn_distribution.png'
            )

    # ========================================
    # Evaluation 3: Full vocabulary (optional, if not too large)
    # ========================================
    if len(all_words) <= 1000 and len(all_words) > env.allowable_words:
        print(f"\n{'─'*60}")
        print("📊 Evaluating on FULL vocabulary")
        print(f"{'─'*60}")

        results_full = evaluate_on_words(
            agent, env, all_words,
            max_episodes=args.max_words,
            verbose=True
        )
        print_results(results_full, title="Full Vocabulary Results")
        results_dict['Full Vocabulary'] = results_full

        save_results_to_csv(results_full, output_dir / 'full_vocab_detailed.csv')

    # ========================================
    # Comparison & Summary
    # ========================================
    if len(results_dict) > 1 and not args.skip_plots:
        print(f"\n{'─'*60}")
        print("📈 Generating comparison plots")
        print(f"{'─'*60}")
        compare_results(results_dict, save_path=output_dir / 'comparison.png')

    # Save summary JSON
    summary = {
        'checkpoint': args.checkpoint,
        'model_params': {
            'vocab_size': len(env.words),
            'allowable_words': env.allowable_words,
        },
        'results': {
            name: {
                'win_rate': res['win_rate'],
                'avg_turns_all': res['avg_turns_all'],
                'avg_turns_wins': res['avg_turns_wins'],
                'wins': res['wins'],
                'losses': res['losses'],
            }
            for name, res in results_dict.items()
        }
    }

    summary_path = output_dir / 'summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n✓ Saved summary to {summary_path}")

    # Final summary
    print(f"\n{'='*60}")
    print(f"{'Evaluation Complete!':^60}")
    print(f"{'='*60}")
    print(f"\nResults saved to: {output_dir.absolute()}")
    print(f"  - summary.json (aggregate metrics)")
    print(f"  - *_detailed.csv (per-word results)")
    if not args.skip_plots:
        print(f"  - *.png (visualizations)")
    print()


if __name__ == '__main__':
    main()
