import torch
from torch.autograd import Variable
from torch import nn

class Embedder(nn.Module):
    def __init__(self, window, char_voca, word_voca=None, jaso_dim=40, char_dim=40, word_dim=0,
                 gazet=None, gazet_embed=True, pos_enc=True, phoneme=True):
        """
        :param  window:  left/right window size from current character
        :param  voca:  vocabulary
        """
        super().__init__()
        self.window = window
        self.char_voca = char_voca
        self.word_voca = word_voca
        self.gazet = gazet
        self.phoneme = phoneme
        self.gazet_embed = gazet_embed
        self.pos_enc = pos_enc
        
        # Context length
        self.context_len = 2 * window + 1
        
        # Vocab size
        self.char_voca_size = len(self.char_voca)
        # self.word_voca_size = len(self.word_voca)
        self.gazet_voca_size = 15

        # Dimensions
        self.jaso_dim = jaso_dim
        self.char_dim = char_dim
        self.word_dim = word_dim
        self.gazet_dim = self.gazet_voca_size
        self.embed_dim = self.char_dim + self.gazet_dim
        # self.embed_dim = self.char_dim + self.word_dim + self.gazet_dim
            
        # Use word vocab
        if word_voca:
            self.word_voca = word_voca
        
        # Use positional encoding
        if pos_enc:
            self.pe_tensor = self.positional_encoding(self.context_len, self.embed_dim)
            
        # Modules
        #if self.phoneme:
        #    self.jaso2char = nn.Sequential(
        #        nn.Conv1d(jaso_dim, char_dim, 3, 3))
            
        # char embedding
        self.char_embedding = nn.Embedding(self.char_voca_size, self.char_dim)
        
        # word embedding
        # self.word_embedding = nn.Embedding(self.word_vocab_size, self.word_dim)

        
    @classmethod
    def positional_encoding(cls, context_len, embed_dim):
        """
        Positional encoding Variable 출력
        embeds [batch_size, context_len, embed_dim]에 Broadcasting 으로 더해짐
        :return: pe [context_len, embed_dim]
        """
        pe_tensor = torch.zeros([context_len, embed_dim])
        for j in range(context_len):
            j += 1 # 1-based indexing
            for k in range(embed_dim):
                k += 1 # 1-based indexing
                pe_tensor[j-1, k-1] = (1-j/context_len) - (k/embed_dim)*(1-2*j/context_len)

        pe_tensor = Variable(pe_tensor)
        if torch.cuda.is_available():
            pe_tensor = pe_tensor.cuda()
        return pe_tensor
        
    def forward(self, sentence, gazet):
        """
        Args:
            sentence [seq_len, context_len=21]
            gazet [seq_len, context_len=21, gazet_vocab_size=15]
        Return:
            sentence_embed [seq_len, context_len=21, embed_dim]
        """
        # char vec
        char_vec = self.char_embedding(sentence) # [seq_len, context_len, char_dim]
        
        # word vec
        # word_vec = self.word_embedding(sentences) # [seq_len, context_len, word_dim]
        
        # gazet vec
        gazet_vec = gazet # [seq_len, context_len, gazet_dim]
        
        # sentence_embed = torch.cat([char_vec, word_vec, gazet], dim=2) # [seq_len, context_len, embed_dim]
        sentence_embed = torch.cat([char_vec, gazet], dim=2) # [seq_len, context_len, embed_dim]
        
        if self.pos_enc:
            sentence_embed += self.pe_tensor
        
        return sentence_embed
        

