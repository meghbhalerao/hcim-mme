import os

dataset = "office"
domain = "dslr"
imbalance_type = "exp"
if dataset == "office":
    data_path = "../data/%s/%s/images/"%(dataset,domain)

num_examples = []
for class_ in os.listdir(data_path):
    num_examples.append(len(os.listdir(os.path.join(data_path,class_))))

num_examples.sort(reverse = True)
print(num_examples)