import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F

class BigramModel:
    def __init__(self, filename: str):
        self._words = open("names.txt", "r").read().splitlines()
        chars = sorted(list(set("".join(self._words))))
        self._stoi = {s: i+1 for i,s in enumerate(chars)}
        self._stoi["."] = 0
        self._itos = {i: s for s,i in self._stoi.items()}
        self._xs = None
        self._ys = None
        self._num_classes = 27
        self._num_samples = 0
        self._W = None
        self._generator = torch.Generator().manual_seed(2147483647)
        self._generate_training_set()
     
    def _generate_training_set(self):
        xs, ys = [], []
        for word in self._words:
            chs = ['.'] + list(word) + ['.']
            for ch1, ch2 in zip(chs, chs[1:]):        
                idx1 = self._stoi[ch1]
                idx2 = self._stoi[ch2]
                xs.append(idx1)
                ys.append(idx2)   

        self._xs = torch.tensor(xs)
        self._ys = torch.tensor(ys)
        self._num_samples = self._xs.nelement()
        print(f"num_samples: {self._num_samples}")

    def train(self, epochs=200):
        self._W = torch.randn((self._num_classes, self._num_classes), generator=self._generator, requires_grad=True)
        loss = None
        for k in range(epochs):
            # forward pass

            # 1. input to the network, one-hot encoding
            xenc = F.one_hot(self._xs, num_classes=self._num_classes).float()

            # 2. predict log-counts
            logits = xenc @ self._W

            # 3. counts, equivlanet to N in simple bigram model
            counts = logits.exp()
            probs = counts / counts.sum(1, keepdims=True)

            # 4. (negative log likelihood) using probabilites assigned by the nn for the next character
            loss = -probs[torch.arange(self._num_samples), self._ys].log().mean()

            # backware pass
            self._W.grad = None
            loss.backward()

            self._W.data += -50 * self._W.grad

        print(f"loss:{loss.item()}")

    def generate_name(self):
        idx = 0 # "."
        name = []
        while True:
            xenc = F.one_hot(torch.tensor([idx]), num_classes=self._num_classes).float()
            logits = xenc @ self._W
            counts = logits.exp()
            p = counts / counts.sum(1, keepdims=True)
            idx = torch.multinomial(p, num_samples=1, replacement=True, generator=self._generator).item()
            name.append(self._itos[idx])
            # check for end token
            if idx == 0:
                break
        print("".join(name))

    def generate_names(self, num=10):
        self._generator = torch.Generator().manual_seed(2147483647)
        for i in range(num):
            self.generate_name()

def main():
    model = BigramModel("names.txt")
    model.train()

    print("Generating names:")
    model.generate_names()

main()