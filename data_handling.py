import pandas as pd
from torch.utils.data import Dataset
import json

def load_tinystories_data(path):
    texts = []
    with open(path, 'r', encoding='utf8') as file:
        lines = ''
        for line in file.readlines():
            if line == '<|endoftext|>\n':
                texts.append(lines)
                lines = ''
                continue
            lines += line.replace('\n', ' ')

    return texts



class Labels():
    def __init__(self):
        with open("data/ekman_mapping.json", "r") as f:
            json_data = json.load(f)
        with open("data/labels.txt", "r") as f:
            x = f.read().split("\n")
        self.label_categories = json_data
        self.all_labels = x
        self.flipped_dict = {emotion: category for category, emotions in json_data.items() for emotion in emotions}

        self.index_to_label = {index: label for index, label in enumerate(x)}
        self.label_to_index = {label: index for index, label in enumerate(x)}
        
    def get_label_name(self, index:int):
        return self.index_to_label[index]
    
    def get_label_groupings(self, label:str):
        labels = self.label_categories[label]
        indexes = [self.label_to_index[x] for x in self.all_labels if x in labels]
        return labels, indexes
    
    def get_group_name(self, label:str):
        return self.flipped_dict[label]


class CustomTextDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        """
        Args:
            dataframe (pd.DataFrame): DataFrame containing your data.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.df = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Get the row at the specified index
        row = self.df.iloc[idx]
        
        # Extract the text and the label (label_as_string)
        text = row['processed_text_column']
        label = row['processed_label_column']
        
        # Build the sample dictionary
        sample = {"text": text, "label": label}
        
        if self.transform:
            sample = self.transform(sample)
            
        return sample
    


def load_sentiment_data(path):
    lab = Labels()

    column_names = ["text","label","idk"]
    data = pd.read_csv(path, sep = "\t", names = column_names)
    d = dict()
    d["processed_label_column"] = []
    d["processed_text_column"]  = []
    d["label_as_string"] = []
    d["label_as_group"] = []
    d["has_multiple_labels"] = []
    for index, row in data.iterrows():
        try:
            label = int(row["label"])
            d["processed_label_column"].append(label)
            d["processed_text_column"].append(row["text"])
            d["label_as_string"].append(lab.get_label_name(label))
            d["label_as_group"].append(lab.get_group_name(lab.get_label_name(label)))
            d["has_multiple_labels"].append(False)
        except:
            for label_element in row["label"].split(","):
                label = int(label_element)
                d["processed_text_column"].append(row["text"])
                d["processed_label_column"].append(label)
                d["label_as_string"].append(lab.get_label_name(label))
                d["label_as_group"].append(lab.get_group_name(lab.get_label_name(label)))
                d["has_multiple_labels"].append(True)

    cleaned_df = pd.DataFrame(d)
    dataset = CustomTextDataset(cleaned_df)

    return dataset