#!/usr/bin/env python3

#1st Attempt at tic tac toe:

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np
import random

'''
MODEL
'''
#initial training data
trainingData = [[[0,0,0,0,0,1,1,1,1], 8], [[1,1,1,1,1,2,2,2,2], 2]]

def loadData():
  random.shuffle(trainingData)
  global X
  global y
  X = []
  y = []
  for Xd, Yd in trainingData:
    X.append(Xd)
    y.append(Yd)
  X = tf.constant(X)
  y = tf.constant(y)
loadData()
    


model = tf.keras.models.Sequential([
  tf.keras.layers.Dense(3, activation='relu'),
  tf.keras.layers.Dense(3, activation='relu'),
  tf.keras.layers.Dense(3, activation='relu'),
  tf.keras.layers.Dense(9, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

def train():
  model.fit(X, y, epochs=100, verbose=0)
  print("\n")

'''
GAME
'''

AI_is_playing = "X"

def resetGame():
  global board
  global game_still_going
  global winner
  global current_player
  # Will hold our game board data
  board = ["-", "-", "-",
           "-", "-", "-",
           "-", "-", "-"]
  # Lets us know if the game is over yet
  game_still_going = True
  # Tells us who the winner is
  winner = None
  # Tells us who the current player is (X goes first)
  current_player = "X"

def gen_machine_board(board):
  machineBoard = []
  for field in board:
    if field == current_player:
      machineBoard.append(1)
    elif field == "-":
      machineBoard.append(0)
    else:
      machineBoard.append(-1)
  return machineBoard
  
def machine_move(board):
  machineBoard = tf.constant([gen_machine_board(board)])
  return np.argmax(model.predict(machineBoard))+1

# ------------- Functions ---------------

# Play a game of tic tac toe
def play_game():

  resetGame()

  # Loop until the game stops (winner or tie)
  while game_still_going:

    # Handle a turn
    handle_turn(current_player)

    # Check if the game is over
    check_if_game_over()

    # Flip to the other player
    flip_player()
    
    #train AI
    train()
  
  # Since the game is over, print the winner or tie
  display_board(board)
  if winner == "X" or winner == "O":
    print(winner + " won.")
    #boostAIdata
    for i in range(10):
      trainingData.append([potWinningMoveX, potWinningMoveY])
    loadData()
  elif winner == None:
    print("Tie.")
  print("AI played " + AI_is_playing)

# Display the game board to the screen
def display_board(board):
  print("\n")
  print(str(board[6]) + " | " + str(board[7]) + " | " + str(board[8]) + "     7 | 8 | 9")
  print(str(board[3]) + " | " + str(board[4]) + " | " + str(board[5]) + "     4 | 5 | 6")
  print(str(board[0]) + " | " + str(board[1]) + " | " + str(board[2]) + "     1 | 2 | 3")
  print("\n")

# Handle a turn for an arbitrary player
def handle_turn(player):
  display_board(board)
  
  global game_still_going
  global winner
  global Xlist
  global ylist
  global X
  global y
  global potWinningMoveX
  global potWinningMoveY
  valid = False
  
  position = None
  while not valid:
    valid = True
    if AI_is_playing == current_player:
      position = str(machine_move(board))
    else:
      # Get position from player
      print(player + "'s turn.")
      position = None
      while position not in ["1", "2", "3", "4", "5", "6", "7", "8", "9"]:
        position = input("Choose a position from 1-9: ")

    # Get correct index in our board list
    position = int(position) - 1
      
    potWinningMoveX = gen_machine_board(board)
    potWinningMoveY = position

    # Then also make sure the spot is available on the board
    if board[position] != "-":
      print(str(position+1) + " is already taken")
      valid = False
      #generate data for AI
      if AI_is_playing == current_player:
        i = 0
        for field in board:
          if field == "-":
            trainingData.append([gen_machine_board(board), i])
          i += 1
        loadData()
        train()
        
  if valid:
    # Generate Data for AI
    if AI_is_playing != current_player:
      trainingData.append([gen_machine_board(board), position])
      loadData()
    # Put the game piece on the board
    board[position] = player
    # Show the game board
    display_board(board) 
  
# Check if the game is over
def check_if_game_over():
  check_for_winner()
  check_for_tie()

# Check to see if somebody has won
def check_for_winner():
  # Set global variables
  global winner
  # Check if there was a winner anywhere
  row_winner = check_rows()
  column_winner = check_columns()
  diagonal_winner = check_diagonals()
  # Get the winner
  if row_winner:
    winner = row_winner
  elif column_winner:
    winner = column_winner
  elif diagonal_winner:
    winner = diagonal_winner

# Check the rows for a win
def check_rows():
  # Set global variables
  global game_still_going
  # Check if any of the rows have all the same value (and is not empty)
  row_1 = board[0] == board[1] == board[2] != "-"
  row_2 = board[3] == board[4] == board[5] != "-"
  row_3 = board[6] == board[7] == board[8] != "-"
  # If any row does have a match, flag that there is a win
  if row_1 or row_2 or row_3:
    game_still_going = False
  # Return the winner
  if row_1:
    return board[0] 
  elif row_2:
    return board[3] 
  elif row_3:
    return board[6] 
  # Or return None if there was no winner
  else:
    return None

# Check the columns for a win
def check_columns():
  # Set global variables
  global game_still_going
  # Check if any of the columns have all the same value (and is not empty)
  column_1 = board[0] == board[3] == board[6] != "-"
  column_2 = board[1] == board[4] == board[7] != "-"
  column_3 = board[2] == board[5] == board[8] != "-"
  # If any row does have a match, flag that there is a win
  if column_1 or column_2 or column_3:
    game_still_going = False
  # Return the winner
  if column_1:
    return board[0] 
  elif column_2:
    return board[1] 
  elif column_3:
    return board[2] 
  # Or return None if there was no winner
  else:
    return None

# Check the diagonals for a win
def check_diagonals():
  # Set global variables
  global game_still_going
  # Check if any of the columns have all the same value (and is not empty)
  diagonal_1 = board[0] == board[4] == board[8] != "-"
  diagonal_2 = board[2] == board[4] == board[6] != "-"
  # If any row does have a match, flag that there is a win
  if diagonal_1 or diagonal_2:
    game_still_going = False
  # Return the winner
  if diagonal_1:
    return board[0] 
  elif diagonal_2:
    return board[2]
  # Or return None if there was no winner
  else:
    return None

# Check if there is a tie
def check_for_tie():
  # Set global variables
  global game_still_going
  # If board is full
  if "-" not in board:
    game_still_going = False
    return True
  # Else there is no tie
  else:
    return False

# Flip the current player from X to O, or O to X
def flip_player():
  # Global variables we need
  global current_player
  # If the current player was X, make it O
  if current_player == "X":
    current_player = "O"
  # Or if the current player was O, make it X
  elif current_player == "O":
    current_player = "X"
    
def flip_AI():
  global AI_is_playing
  if AI_is_playing == "X":
    AI_is_playing = "O"
  elif AI_is_playing == "O":
    AI_is_playing = "X"

while True:
  play_game()
  #print(Xlist,ylist)
  #print(display_board(trainingData[-1][0]), trainingData[-1][1]+1)
  input()
  flip_AI()
