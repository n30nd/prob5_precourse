# TODO: [part d]
# Calculate the accuracy of a baseline that simply predicts "London" for every
#   example in the dev set.
# Hint: Make use of existing code.
# Your solution here should only be a few lines.

import argparse
import utils

def main():
    accuracy = 0.0

    # Compute accuracy in the range [0.0, 100.0]
    ### YOUR CODE HERE ###
    # pass
    dev_file_path = "birth_dev.tsv"
    
    # Create predictions list with "London" for each example
    with open(dev_file_path, encoding='utf-8') as f:
        lines = f.readlines()
    
    num_examples = len(lines)
    london_predictions = ["London"] * num_examples
    
    # Use existing evaluate_places function to calculate accuracy
    total, correct = utils.evaluate_places(dev_file_path, london_predictions)
    
    if total > 0:
        accuracy = (correct / total) * 100.0
    ### END YOUR CODE ###

    return accuracy

if __name__ == '__main__':
    accuracy = main()
    with open("london_baseline_accuracy.txt", "w", encoding="utf-8") as f:
        f.write(f"{accuracy}\n")
