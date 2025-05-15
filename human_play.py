# -*- coding: utf-8 -*-
"""
human VS AI models
Input your move in the format: 2,3

@author: Junxiao Song
"""

from __future__ import print_function
import pickle
from game import Board, Game
from mcts_pure import MCTSPlayer as MCTS_Pure
from mcts_alphaZero import MCTSPlayer
from policy_value_net_numpy import PolicyValueNetNumpy
# from policy_value_net import PolicyValueNet  # Theano and Lasagne
from policy_value_net_pytorch import PolicyValueNet  # Pytorch
# from policy_value_net_tensorflow import PolicyValueNet # Tensorflow
# from policy_value_net_keras import PolicyValueNet  # Keras
import sys
import argparse

class Human(object):
    """
    human player
    """

    def __init__(self):
        self.player = None

    def set_player_ind(self, p):
        self.player = p

    def get_action(self, board):
        try:
            location = input("Your move: ")
            if isinstance(location, str):  # for python3
                location = [int(n, 10) for n in location.split(",")]
            move = board.location_to_move(location)
        except Exception as e:
            move = -1
        if move == -1 or move not in board.availables:
            print("invalid move")
            move = self.get_action(board)
        return move

    def __str__(self):
        return "Human {}".format(self.player)


def run(use_pytorch=False, model_file='best_policy_8_8_5.model2', width=8, start_player=0):
    n = 5
    height = width
    try:
        board = Board(width=width, height=height, n_in_row=n)
        game = Game(board, show_gui=True)

        # ############### human VS AI ###################
        # load the trained policy_value_net in either Theano/Lasagne, PyTorch or TensorFlow
        if use_pytorch:
            best_policy = PolicyValueNet(width, height, model_file = model_file, use_gpu=False, model_type='pytorch')
            mcts_player = MCTSPlayer(best_policy.policy_value_fn, c_puct=5, n_playout=400)
        else:
        # load the provided model (trained in Theano/Lasagne) into a MCTS player written in pure numpy
            try:
                policy_param = pickle.load(open(model_file, 'rb'))
            except:
                policy_param = pickle.load(open(model_file, 'rb'),
                                       encoding='bytes')  # To support python3
            best_policy = PolicyValueNetNumpy(width, height, policy_param)
            mcts_player = MCTSPlayer(best_policy.policy_value_fn,
                                 c_puct=5,
                                 n_playout=400)  # set larger n_playout for better performance"""

        # uncomment the following line to play with pure MCTS (it's much weaker even with a larger n_playout)
        # mcts_player = MCTS_Pure(c_puct=5, n_playout=1000)

        # human player, input your move in the format: 2,3
        human = Human()

        # set start_player=0 for human first
        game.start_play(human, mcts_player, start_player=start_player, is_shown=1, show_gui=True)
    except KeyboardInterrupt:
        print('\n\rquit')

use_pytorch = False
if __name__ == '__main__':
    # if run with --pytorch, use PyTorch
    if '--pytorch' in sys.argv:
        use_pytorch = True

    parser = argparse.ArgumentParser(description='AlphaZero Gomoku')
    parser.add_argument('--pytorch', action='store_true',
                        help='use pytorch')
    parser.add_argument('--model_file', type=str,  default='best_policy_8_8_5.model2',
                        help='model file')
    parser.add_argument('--width', type=int, default=8, 
                        help='board width')
    parser.add_argument('--ai_first', action='store_true',
                        help='AI first')

    args = parser.parse_args()
    start_player = 0
    if args.ai_first:
        start_player = 1
    run(use_pytorch=use_pytorch, model_file=args.model_file, width=args.width, 
        start_player=start_player)
    
