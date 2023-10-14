import utility

items = utility.read_json("datasets/train.json")
print("number of items: ", len(items))

items = utility.read_json("datasets/validation.json")
print("number of items: ", len(items))
