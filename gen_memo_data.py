import itertools
import random
import sys
def gen_memo_data(vocabLength, repeatTimes, fileName):
    if vocabLength > 52:
        raise ValueError("vocabLength cannot exceed 52 (26 lowercase + 26 uppercase letters).")
    if vocabLength > 26:
        alphabet = [chr(i) for i in range(97, 97 + 26)] + [chr(i) for i in range(65, 65 + (vocabLength - 26))]
    else:
        alphabet = [chr(i) for i in range(97, 97 + vocabLength)]
    # Inputs and Outputs are direct copys, where they are all 2 letter combinations of the given alphabet.
    inputs = ["".join(p) for p in itertools.product(alphabet, repeat=2)]
    outputs = inputs.copy()
    # Randomized outputs, but with a fixed seed for reproducibility.
    random.seed(42)
    random.shuffle(outputs)
    # Create the unique pairs of inputs and outputs, and then repeat them accordingly, and later shuffle all data afterwards.
    unique_pairs = []
    for input, output in zip(inputs, outputs):
        unique_pairs.append(f"{input}={output}\n")
    dataset = unique_pairs * repeatTimes
    random.shuffle(dataset)
    # Write to file.
    with open(fileName, "w", encoding="utf-8") as f:
        f.writelines(dataset)
    print(f"Generated {len(unique_pairs)} unique mappings.")
if __name__ == "__main__":
    gen_memo_data(int(sys.argv[1]), 100, "input.txt")