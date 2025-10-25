'''
Script to launch wordle game in terminal. User is given the option to play the
game themselves or run it through a pre-built deep-learning model.
'''

import time

def welcome():
  '''
  Helper function to welcome user to Wordler and offer an explanation for how
  Wordle puzzles work. 
  '''
  print('- Welcome to Wordler! :D -')
  time.sleep(1.5)
  input('Press the enter key to continue')
  print('Enter \'i\' to show game instructions; enter \'s\' to skip')
  key = input()
  while key != 'i' and key != 's':
      print('Invalid input; please enter \'i\' or \'s\'')
      key = input()
  if key == 'i':
      print_instructions()
  print()
  print('Press the \'u\' key on your keyboard to solve a Wordle yourself,'
  ' or press the \'a\' key to have AI play for you.')
  key = input()
  while key != 'u' and key != 'a':
      print('Invalid input; please enter \'u\' or \'a\'')
      key = input()
  print()
  print('Have fun!')
  print()

def print_instructions():
  '''
  Helper function for printing Wordle rules.
  '''
  print()
  print('WORDLE RULES')
  print('- You have six attempts to correctly guess a five-letter English word.')
  input('Press the enter key to continue')
  input('- For each turn, guess any valid five-letter word; guesses are valid' \
  ' if and only if they appear in a standard, up-to-date English dictionary.')
  input('- After a guess is submitted, every letter in the guess will be' \
  ' labeled with an appropriate color block to indicate if the letter is in' \
  ' the solution word.')
  input('- Letters that are in the solution word and in the correct spot will' \
  ' be marked as green, letters that are in the solution word and not in the' \
  ' correct spot will be marked as yellow, and letters that are not in the' \
  ' solution word at all will be marked as gray.')
  input('- If a guess contains m occurrences of a given letter and the' \
  ' solution word contains n occurrences of the letter, where m > n, the' \
  ' letter will be labeled as yellow n times, not m times.')
  
def run_game():
  '''
  Helper function to run Wordle game.
  '''
  # TODO: update this to pull from dataset
  solution_word = 'creed'
  # define map to store counts of letters in solution word
  solution_dict = {}
  for letter in solution_word:
    if solution_dict.get(letter) is None:
      solution_dict[letter] = 1
    else:
      solution_dict[letter] += 1
  print(solution_dict)
  attempt = 0
  guess = ''
  print('Enter your first guess:')
  while guess != solution_word and attempt < 6:
    guess = input()
    while(len(guess) != 5 or not guess.isalpha()):
       print('Invalid guess; please enter a five-letter word.')
       guess = input()
    feedback_str = ''
    guess_dict = {}
    for i in range(5):
      if guess[i] == solution_word[i]:
        letter = guess[i]
        if guess_dict.get(letter) is None:
          guess_dict[letter] = 1
        else:
          guess_dict[letter] += 1
        feedback_str += '🟩'
      elif guess[i] in solution_word:    
        letter = guess[i]
        if guess_dict.get(letter) is None:
          guess_dict[letter] = 1
          feedback_str += '🟨'
        elif guess_dict[letter] < solution_dict[letter]:
          guess_dict[letter] += 1
          feedback_str += '🟨'
        else:
          feedback_str += '⬜️'
      else:
        feedback_str += '⬜️' 
    print(feedback_str)
    attempt += 1
  if guess == solution_word:
    print('Congrats! You solved the Wordle!')
  if guess != solution_word and i == 6:
    print(f'Failed to solve; the answer was {solution_word}')

def main():
  welcome()
  run_game()

if __name__ == '__main__':
  main()