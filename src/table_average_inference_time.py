import argparse
import pandas

from preprocess import preprocess

# Parse the arguments of the command line
parser = argparse.ArgumentParser(description="Produces a table of the average inference time according to the batch size.")

parser.add_argument("results", help="The results of the processing")

args = parser.parse_args()

results = pandas.read_csv(args.results)
results = preprocess(results)

print(results["average_time"])
