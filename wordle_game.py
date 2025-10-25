'''
Script to launch wordle game in terminal. User is given the option to play the
game themselves or run it through a pre-built deep-learning model.
'''

import time
import keyboard

def print_instructions():
  print('- Welcome to Wordler! :D -')
  time.sleep(1)
  print('Press the \'i\' key to show instructions; press the \'s\'' \
  ' key to skip')
  while(True):
        a = keyboard.read_key()

        if a == 'i':
           print('You can choose to play the game yourself or run it through' \
            ' AI. In each turn, guess a five-letter word; guessses are valid if' \
            ' and only if they appear in a standard, up-to-date English ' \
            ' dictionary. After a guess is submitted, every letter in the guess' \
            ' will be labeled with an appropriate color block to indicate if the' \
            ' letter is in the solution word. Letters that are in the solution' \
            ' word and in the correct spot will be marked as green, letters that' \
            ' are in the solution word and not in the correct spot will be' \
            ' marked as yellow, and letters that are not in the solution word' \
            ' at all will be marked as green.')
        elif a == 's':
            break

  time.sleep(0.3)
  print('Press the \'p\' key on your keyboard to solve a Wordle yourself, and' \
  'press the \'a\' key to have AI play for you.')
  print('Have fun!')

def run_game():
  solution_word = 'crane'
  i = 0
  guess = ''
  print('Enter your first guess:')
  while guess != solution_word and i < 6:
    guess = input()
    while(len(guess) != 5 or not guess.isalpha()):
       print('Invalid guess; please enter a five-letter word.')
       guess = input()
    feedback_str = ''
    for l in range(5):
      if guess[l] == solution_word[l]:
        feedback_str += '🟩'
      elif guess[l] in solution_word:             
        feedback_str += '🟨'
      else:
        feedback_str += '⬜️' 
    print(feedback_str)
    i += 1
  if guess == solution_word:
    print('Congrats! You solved the Wordle!')
  if guess != solution_word and i == 6:
    print(f'Failed to solve; the answer was {solution_word}')

def main():
  # print_instructions()
  run_game()

if __name__ == '__main__':
  main()