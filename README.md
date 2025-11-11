# wordler

### Histogram of Top 20 Most Frequent 5-Letter English Words

To view the historgram, run

1. Start a local HTTP server from the project root directory:

```
python3 -m http.server 8080
```

2. Open your browser and visit:

```
http://localhost:8080/data/
```

You should see something like this:
<img width="868" height="431" alt="Screenshot 2025-10-25 at 3 44 36 PM" src="https://github.com/user-attachments/assets/fab46fcd-fab7-4b12-918c-3ffbf63450e6" />

## wordle_game.py vs wordle_env.py

- `wordle_game.py` allows human input and manual testing
- `wordle_env.py` allows ML agents to programmatically control the wordle game
- `wordle_env.py` keeps track of the final answer in the environment state so we can provide feedback to the ML agent for every guess (check how closely the guess matches the target and determine if the agent won)

## baseline agents

As a baseline for comparison, so far we have a random agent and frequency agent to compare our future RL agent against.

The random agent just guesses randomly and the frequency agent guesses the most frequent word based on the linguistic heuristic that common words are more likely to be answers according to Zipf's law. Zipf's law says "the most frequent word will occur approximately twice as often as the second most frequent word, which will occur approximately twice as often as the fourth most frequent word" [(source)](https://demonstrations.wolfram.com/ZipfsLawAppliedToWordAndLetterFrequencies/)

You can control how many games the baseline agents play by changing around these parameters in `baseline_agents.py`

```
for agent_name, agent in agents:
        results = evaluate_agent(agent, env, num_games=10, verbose=False)

```

To run the script:

```
python3 baseline_agents.py
```
