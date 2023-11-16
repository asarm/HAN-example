import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import numpy as np
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

from HAN import HierarchicalAttentionNetwork

data = {
    "sport": [
        "The soccer team had a rigorous training session in preparation for the championship match. The coach emphasized the importance of teamwork and strategy.",
        "In the thrilling basketball game, the star player scored a three-pointer in the final seconds, securing a narrow victory for the team. The fans erupted in cheers as they celebrated the hard-fought win.",
        "The Olympic swimmer broke a world record, showcasing incredible speed and skill in the pool. The athlete expressed gratitude for the support received from fans and teammates.",
        "The tennis player showcased exceptional skill and agility, dominating the opponent in a straight-set victory. The athlete thanked the coaching staff for their guidance and support.",
        "A local marathon brought together runners from diverse backgrounds, fostering a sense of community and healthy competition. Participants shared stories of personal achievements and perseverance.",
        "The upcoming sports event promises to attract a global audience, with top athletes from around the world competing for prestigious titles. Organizers are focused on delivering a memorable and inclusive experience.",
        "In a surprising turn of events, the underdog team emerged victorious in a high-stakes rugby match, defying expectations. The win was celebrated as a triumph of determination and teamwork.",
        "The professional cyclist overcame physical challenges to finish a grueling race, highlighting the importance of resilience and mental toughness in the world of competitive sports."
    ],
    "business": [
        "The tech company announced a groundbreaking innovation in artificial intelligence, revolutionizing the industry. Investors showed strong interest in the company's forward-thinking approach.",
        "Amidst economic challenges, the business successfully implemented cost-cutting measures to maintain profitability. The CEO emphasized the importance of adaptability and resilience.",
        "A global corporation expanded its market presence by establishing strategic partnerships with key industry players. The move was part of a long-term growth strategy to capture new market opportunities.",
        "A start-up company secured significant funding from venture capitalists, enabling them to scale their operations and launch new products. The founders expressed gratitude for the investors' belief in their vision.",
        "The real estate market experienced a surge in activity as businesses adapted to remote work, leading to increased demand for flexible office spaces. Real estate developers capitalized on this trend to meet evolving needs.",
        "A multinational corporation implemented sustainable business practices, earning accolades for their commitment to environmental responsibility. Consumers praised the company's efforts to reduce carbon footprint and promote eco-friendly initiatives.",
        "Entrepreneurs shared success stories at a business conference, offering insights into overcoming challenges and achieving growth. Attendees gained valuable knowledge and inspiration for their own ventures.",
        "An industry leader unveiled a cutting-edge technology that promises to revolutionize supply chain management. Analysts predict that this innovation will streamline operations and enhance efficiency in various sectors."
    ]
}

val_data = {
    "sport": [
        "The gymnast performed a flawless routine, earning high scores from the judges. Spectators marveled at the athlete's precision and grace.",
        "A local cycling competition brought together enthusiasts from the community. Participants showcased their passion for the sport and enjoyed friendly competition.",
        "The professional golfer demonstrated exceptional skill, sinking a challenging putt to secure a tournament victory. The crowd erupted in applause for the triumphant moment."
    ],
    "business": [
        "A successful entrepreneur shared insights on navigating the business world during a panel discussion. Attendees gained valuable advice for achieving success in their ventures.",
        "A tech startup received recognition for developing an innovative app that addresses a pressing market need. Investors expressed interest in supporting the company's growth.",
        "The CEO of a leading finance company outlined strategies for financial resilience in uncertain times during a business conference. The audience appreciated the practical insights shared."
    ]
}

labels = ['sport', 'sport', 'sport', 'sport', 'sport', 'sport', 'sport', 'sport', 
          'business', 'business', 'business', 'business', 'business', 'business', 'business', 'business']
val_labels = ['sport', 'sport', 'sport', 'business', 'business', 'business']

label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

flattened_data = [document for category in data.values() for document in category]
flatten_validation = [document for category in val_data.values() for document in category]

X_train, y_train = flattened_data, encoded_labels
X_val, y_val = flatten_validation, label_encoder.transform(val_labels)

def tokenize(text):
    return text.split() 

# Build vocabulary
vocab = build_vocab_from_iterator(map(tokenize, flattened_data), specials=["<unk>", "<pad>", "<bos>", "<eos>"])
vocab.set_default_index(vocab["<unk>"])

# Define a custom dataset
class CustomDataset(Dataset):
    def __init__(self, data, labels, vocab, max_length):
        self.data = data
        self.labels = labels
        self.vocab = vocab
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]
        label = self.labels[idx]

        tokens = [self.vocab[token] for token in tokenize(text)]
        tokens = tokens[:self.max_length] + [self.vocab["<pad>"]] * (self.max_length - len(tokens))

        return {'tokens': torch.tensor(tokens), 'label': label}

train_dataset = CustomDataset(X_train, y_train, vocab, max_length=128)
val_dataset = CustomDataset(X_val, y_val, vocab, max_length=128)

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

word_embedding_dim = 100
sentence_embedding_dim = 100
hidden_size = 50
num_classes = len(label_encoder.classes_)

han_model = HierarchicalAttentionNetwork(len(vocab), word_embedding_dim, sentence_embedding_dim, hidden_size, num_classes)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(han_model.parameters(), lr=0.001)

num_epochs = 10

for epoch in range(num_epochs):
    han_model.train()
    total_loss = 0

    for batch in train_dataloader:
        tokens = batch['tokens']
        labels = batch['label']

        optimizer.zero_grad()

        logits, _, _ = han_model(tokens)
        loss = criterion(logits, labels)
        total_loss += loss.item()

        loss.backward()
        optimizer.step()

    average_loss = total_loss / len(train_dataloader)

    # Validation
    han_model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in val_dataloader:
            tokens = batch['tokens']
            labels = batch['label']

            logits, _, _ = han_model(tokens)
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    val_accuracy = accuracy_score(all_labels, all_preds)

    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {average_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}')
