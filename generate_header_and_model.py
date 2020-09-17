import tensorflow as tf
import numpy as np
import argparse
import os
import enum
from models.model_espcn import ESPCN
from models.model_ldsp import LDSP
from models.model_srcnn import SRCNN
from models.model_vespcn import VESPCN
from models.model_vsrnet import VSRnet
from collections import OrderedDict

@enum.unique
class Padding(enum.Enum):
    Valid = 0
    Same = 1
    Same_clamp_to_edge = 2

def get_arguments():
    parser = argparse.ArgumentParser(description='generate c header with model weights and binary model file')
    parser.add_argument('--model', type=str, default='srcnn', choices=['srcnn', 'ldsp', 'espcn', 'vespcn', 'vsrnet'],
                        help='What model to use for generation')
    parser.add_argument('--output_folder', type=str, default='./',
                        help='where to put generated files')
    parser.add_argument('--ckpt_path', default=None,
                        help='Path to the model checkpoint, from which weights are loaded')
    parser.add_argument('--random_weights', action='store_true',
                        help='Loads random weights into the model, useful for prototyping and testing code')
    parser.add_argument('--use_mc', action='store_true',
                        help='Whether motion compensation is used in video super resolution model')
    parser.add_argument('--scale_factor', type=int, default=2, choices=[2, 3, 4, 6],
                        help='What scale factor was used for chosen model')

    return parser.parse_args()

def main():
    args = get_arguments()

    if not os.path.exists(args.output_folder):
        os.mkdir(args.output_folder)

    if (not args.random_weights) and (args.ckpt_path is None):
        print("Path to the checkpoint file was not provided")
        exit(1)

    if args.model == 'srcnn':
        model = SRCNN(args)
    elif args.model == 'espcn':
        model = ESPCN(args)
    elif args.model == 'ldsp':
        model = LDSP(args)
    elif args.model == 'vespcn':
        model = VESPCN(args)
    elif args.model == 'vsrnet':
        model = VSRnet(args)
    else:
        exit(1)

    with tf.Session() as sess:
        input_ph = model.get_placeholder()
        predicted = model.load_model(input_ph)

        if args.model == 'vespcn':
            predicted = predicted[2]
        predicted = tf.identity(predicted, name='y')

        if args.random_weights:
            print("Random Weights Loaded.")
            init = tf.global_variables_initializer()
            sess.run(init)
        else:
            print("Checkpoint Weights Loaded.")
            if os.path.isdir(args.ckpt_path):
                args.ckpt_path = tf.train.latest_checkpoint(args.ckpt_path)
            saver = tf.train.Saver()
            saver.restore(sess, args.ckpt_path)

        output_graph_def = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def, ['y'])
        tf.train.write_graph(output_graph_def, args.output_folder, args.model + '.pb', as_text=False)


if __name__ == '__main__':
    main()

