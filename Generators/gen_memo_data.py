import itertools
import random
def gen_memo_data(vocabLength, repeatTimes, fileName):
    # Only works within range of alphabet.
    if vocabLength > 26:
        raise ValueError("vocabLength must be less than or equal to 26")
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
    gen_memo_data(6, 50, "input.txt")