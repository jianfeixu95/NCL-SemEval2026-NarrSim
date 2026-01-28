from datasets import load_dataset
import csv

ds = load_dataset("euclaise/writingprompts")
print(ds)

print(ds["train"][0])
print(ds["validation"][0])
print(ds["test"][0])

writingprompts_train_output_file = "writingprompts_story_train.csv"
writingprompts_validation_output_file = "writingprompts_story_validation.csv"
writingprompts_test_output_file = "writingprompts_story_test.csv"

with open(writingprompts_train_output_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["prompt","story"])  # header

    i = 0
    for ex in ds["train"]:
        prompt = ex["prompt"].replace("\n", " ").strip()
        story = ex["story"].replace("\n", " ").strip()
        writer.writerow([prompt, story])
        i+=1

    print(f"train dataset length: {len(ds['train'])}, actual length: {i}")

with open(writingprompts_validation_output_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["prompt","story"])  # header

    i = 0
    for ex in ds["validation"]:
        prompt = ex["prompt"].replace("\n", " ").strip()
        story = ex["story"].replace("\n", " ").strip()
        writer.writerow([prompt, story])
        i+=1

    print(f"validation dataset length: {len(ds['validation'])}, actual length: {i}")


with open(writingprompts_test_output_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["prompt","story"])  # header

    i = 0
    for ex in ds["test"]:
        prompt = ex["prompt"].replace("\n", " ").strip()
        story = ex["story"].replace("\n", " ").strip()
        writer.writerow([prompt, story])
        i+=1

    print(f"test dataset length: {len(ds['test'])}, actual length: {i}")