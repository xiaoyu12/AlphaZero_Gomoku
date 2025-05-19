# -*- coding: utf-8 -*-
"""
human VS AI models
Input your move in the format: 2,3

"""

from __future__ import print_function
import pickle
import sys
import argparse
from game import Board, Game
from mcts_pure import MCTSPlayer as MCTS_Pure
from mcts_alphaZero import MCTSPlayer
from policy_value_net_numpy import PolicyValueNetNumpy
from policy_value_net_pytorch import PolicyValueNet  # PyTorch
from policy_value_resnet_pytorch import PolicyValueResNet  # PyTorch ResNet

class Human:
    """
    Human player class to interact with the game.
    """

    def __init__(self):
        self.player = None

    def set_player_ind(self, p):
        """
        Set the player index (1 or 2).
        """
        self.player = p

    def get_action(self, board):
        """
        Get the human player's move from input.
        Validates the move and ensures it is legal.
        """
        try:
            location = input("Your move: ")
            if isinstance(location, str):  # For Python 3
                location = [int(n, 10) for n in location.split(",")]
            move = board.location_to_move(location)
        except Exception as e:
            move = -1

        # Check if the move is valid
        if move == -1 or move not in board.availables:
            print("Invalid move. Try again.")
            move = self.get_action(board)
        return move

    def __str__(self):
        """
        String representation of the human player.
        """
        return f"Human {self.player}"


def run(model_type='pytorch', model_file='best_policy_8_8_5.model2', width=8, start_player=0):
    """
    Main function to set up and run the game.
    Allows human vs AI gameplay with different model types.
    """
    n = 5  # Number of pieces in a row to win
    height = width  # Board height (square board assumed)

    try:
        # Initialize the board and game
        board = Board(width=width, height=height, n_in_row=n)
        game = Game(board, show_gui=True)

        # Load the AI model based on the specified type
        if model_type == 'resnet':
            # Use PyTorch ResNet model
            best_policy = PolicyValueResNet(width, height, model_file=model_file, use_gpu=False, model_type='pytorch')
            mcts_player = MCTSPlayer(best_policy.policy_value_fn, c_puct=5, n_playout=800)
        elif model_type == 'pytorch':
            # Use standard PyTorch model
            best_policy = PolicyValueNet(width, height, model_file=model_file, use_gpu=False, model_type='pytorch')
            mcts_player = MCTSPlayer(best_policy.policy_value_fn, c_puct=5, n_playout=800)
        else:
            # Use a pure NumPy model (trained in Theano/Lasagne)
            try:
                policy_param = pickle.load(open(model_file, 'rb'))
            except:
                policy_param = pickle.load(open(model_file, 'rb'), encoding='bytes')  # Python 3 compatibility
            best_policy = PolicyValueNetNumpy(width, height, policy_param)
            mcts_player = MCTSPlayer(best_policy.policy_value_fn, c_puct=5, n_playout=800)

        # Uncomment the following line to play with pure MCTS (weaker performance)
        # mcts_player = MCTS_Pure(c_puct=5, n_playout=1000)

        # Initialize the human player
        human = Human()

        # Start the game (human vs AI)
        game.start_play(human, mcts_player, start_player=start_player, is_shown=1, show_gui=True)
    except KeyboardInterrupt:
        print('\nGame interrupted. Exiting...')


if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='AlphaZero Gomoku')
    parser.add_argument('--model_type', type=str, default='pytorch',
                        help='Model type: pytorch, theano, or resnet')
    parser.add_argument('--model_file', type=str, default='best_policy_8_8_5.model2',
                        help='Path to the model file')
    parser.add_argument('--width', type=int, default=8,
                        help='Board width (default: 8)')
    parser.add_argument('--ai_first', action='store_true',
                        help='Set AI to play first')

    args = parser.parse_args()

    # Determine the starting player (0 for human, 1 for AI)
    start_player = 1 if args.ai_first else 0

    # Run the game with the specified parameters
    run(model_type=args.model_type, model_file=args.model_file, width=args.width, start_player=start_player)

