from collections import defaultdict

class Vocab(object):
    """
    Defines a Vocab class, similar to Vocab in torchtext, but not that integrated with a field, cause field is associated with padding and batching, if batching and padding is not implemented, than it is a simple Vocab. I am not prepared to make a Field, cause it is complicated to some extent.
    """
    
    def __init__(self, counter, max_size=None, min_freq=1, specials=['<pad>'], vectors=None, unk_init=None, vectors_cache=None):
        self.freqs = counter
        counter = counter.copy()
        min_freq = max(min_freq, 1)
        
        self.itos = list(specials)
        for tok in specials:
            del counter[tok]
        
        max_size = None if max_size is None else max_size + len(self.itos)
        
        words_and_frequencies = sorted(counter.items(), key=lambda tup: tup[0])
        words_and_frequencies.sort(key=lambda tup: tup[1], reverse=True)
        
        for word, freq in words_and_frequencies:
            if freq < min_freq or len(self.itos) == max_size:
                break
            self.itos.append(word)
        
        self.stoi = defaultdict(_default_unk_index)
        self.stoi.update({tok:i for i, tok in enumerate(self.itos)})
        
    def __len__(self):
        return len(self.itos)

def _default_unk_index():
    return 0
