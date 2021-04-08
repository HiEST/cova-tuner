import argparse
import tensorflow as tf
from tensorflow.keras.models import save_model, Sequential

def main():
    args = argparse.ArgumentParser()
    args.add_argument("-i", "--input", default="frozen_graph.pb", help="path to the frozen pb graph.")
    args.add_argument("-o", "--output", default="frozen_graph.h5", help="path to the output h5 file.")
    config = args.parse_args()

    model = tf.keras.models.load_model(config.input)
    save_model(model,config.output, save_format='h5')

if __name__ == "__main__":
    main()