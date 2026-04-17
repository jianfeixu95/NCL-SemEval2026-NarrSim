from datasets import load_dataset

dataset = load_dataset("ficsim/ficsim")

print(dataset["train"][0])
print(dataset["train"][1])
print(dataset["train"][2])