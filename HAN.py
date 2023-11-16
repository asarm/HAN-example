import torch
import torch.nn as nn
import torch.nn.functional as F

class WordAttention(nn.Module):
    def __init__(self, hidden_size, word_embedding_dim):
        super(WordAttention, self).__init__()

        self.W_word = nn.Linear(hidden_size, hidden_size)
        self.u_word = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, word_encoder_output):
        u_it = torch.tanh(self.W_word(word_encoder_output))
        attention_scores = F.softmax(self.u_word(u_it), dim=1)
        word_attention_output = torch.sum(attention_scores * word_encoder_output, dim=1)

        return word_attention_output, attention_scores

class SentenceAttention(nn.Module):
    def __init__(self, hidden_size, sent_embedding_dim):
        super(SentenceAttention, self).__init__()

        self.W_sentence = nn.Linear(hidden_size, hidden_size)
        self.u_sentence = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, sentence_encoder_output):
        u_is = torch.tanh(self.W_sentence(sentence_encoder_output))
        attention_scores = F.softmax(self.u_sentence(u_is), dim=1)
        sentence_attention_output = torch.sum(attention_scores * sentence_encoder_output, dim=1)

        return sentence_attention_output, attention_scores

class HierarchicalAttentionNetwork(nn.Module):
    def __init__(self, vocab_size, word_embedding_dim, sentence_embedding_dim, hidden_size, num_classes):
        super(HierarchicalAttentionNetwork, self).__init__()

        self.word_embedding = nn.Embedding(vocab_size, word_embedding_dim)
        self.word_lstm = nn.GRU(word_embedding_dim, hidden_size, bidirectional=True, batch_first=True)
        self.word_attention = WordAttention(hidden_size * 2, word_embedding_dim)

        self.sentence_lstm = nn.GRU(hidden_size * 2, hidden_size, bidirectional=True, batch_first=True)
        self.sentence_attention = SentenceAttention(hidden_size * 2, sentence_embedding_dim)

        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, input_data):
        word_embedded = self.word_embedding(input_data)
        word_output, _ = self.word_lstm(word_embedded)
        word_attention_output, word_attention_scores = self.word_attention(word_output)

        sentence_output, _ = self.sentence_lstm(word_attention_output.unsqueeze(1))
        sentence_attention_output, sentence_attention_scores = self.sentence_attention(sentence_output)

        logits = self.fc(sentence_attention_output)

        return logits, word_attention_scores, sentence_attention_scores
