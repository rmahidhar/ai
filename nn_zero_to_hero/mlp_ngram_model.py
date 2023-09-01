import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
import random

# Implementation based on the following paper
# https://papers.nips.cc/paper_files/paper/2000/file/728f206c2a01bf572b5940d7d9a8fa4c-Paper.pdf
class NgramModel:
    def __init__(self, filename: str):
        self._words = open("names.txt", "r").read().splitlines()
        chars = sorted(list(set("".join(self._words))))
        self._stoi = {s: i+1 for i,s in enumerate(chars)}
        self._stoi["."] = 0
        self._itos = {i: s for s,i in self._stoi.items()}

        self._X = None
        self._Y = None
        self._Xtrain = None
        self._Ytrain = None
        self._Xval = None
        self._Yval = None
        self._Xtest = None
        self._Ytest = None

        self._num_classes = 27
        self._num_samples = 0

        # context length: how many characters do we take to predict the next one.
        self._context_length = 3
        self._generator = torch.Generator().manual_seed(2147483647)

        # Embedding Layer: 27x2
        self._C = torch.randn((self._num_classes, 2), generator=self._generator)
        # input size = context length 3 * embedding dimension 2 => 3 * 2 = 6
        # Input Layer: 6
        # Hidden Layer: 300
        self._W1 = torch.randn((6, 300), generator=self._generator)
        self._b1 = torch.randn(300, generator=self._generator)
        # Output Layer: 27
        self._W2 = torch.randn((300, self._num_classes), generator=self._generator)
        self._b2 = torch.randn(self._num_classes, generator=self._generator)
        self._parameters = [self._C, self._W1, self.b1, self._W2, self._b2]
        self._generate_training_set()
     
    def _generate_dataset(self, words):
        X, Y = [], []
        for word in self._words:
            context = [0] * self._context_length
            for ch in word + ".": # "." is a separator
                ix = self._stoi[ch]
                X.append(context)
                Y.append(ix)
                # print("".join(self._itos[i] for i in context), '--->', self._itos[ix])
                # crop and append
                context = context[1:] + [ix]

        X = torch.tensor(X)
        Y = torch.tensor(Y)
        return X, Y

    def _generate_training_dataset(self):
        random.seed(43)
        random.shuffle(self._words)
        n1 = int(0.8 * len(self._words))
        n2 = int(0.9 * len(self._words))
        self._Xtrain, self._Ytrain = self._generate_dataset(self._words[:n1])
        self._Xval, self._Yval = self._generate_dataset(self._words[n1:n2])
        self._Xtest, self._Ytest = self._generate_dataset(self._words[n2:])

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