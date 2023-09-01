import torch
import matplotlib.pyplot as plt


def get_bigram_frequency_counts(words, stoi):
    # torch 2-d tensor for storing the bigram count 
    # rows - 1st char in bigrams
    # cols - 2nd char in bigram
    # value - bigram count
    # total chars = 27 (a-z, .)
    N = torch.zeros(27, 27, dtype=torch.int32)
    for word in words:
        chs = ['.'] + list(word) + ['.']
        for ch1, ch2 in zip(chs, chs[1:]):
            idx1 = stoi[ch1]
            idx2 = stoi[ch2]
            N[idx1, idx2] += 1
    return N


def get_bigram_frequency_counts_probability_distribution(N):
    P = N.float()
    # inplace operations are efficient
    P /= P.sum(dim=1, keepdim=True)
    return P


def plot_bigram_frequency_counts(N, itos):
    # create a square image with sides of 16 each. 
    # The first value (16) is width 
    # The second value (16) is height
    plt.figure(figsize=(16, 16)) 
    plt.imshow(N, cmap='Blues')
    for r in range(27):
        for c in range(27):
            chstr = itos[r] + itos[c]
            plt.text(c, r, chstr, ha="center", va="bottom", color="gray")
            plt.text(c, r, N[r, c].item(), ha="center", va="top", color="gray")
    plt.axis('off')


def generate_names_from_model(P, itos, g):
    idx = 0
    name = []
    while True:
        # Generate using bigram model probability distrubution for next char generation 
        p = P[idx]
        idx = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
        name.append(itos[idx])
        # check for end token
        if idx == 0:
            break
    print("".join(name))

def generate_names_with_uniform_distribution(itos, g):
    idx = 0
    name = []
    while True:
        # Generate using uniform distrubution where all characters
        # are equally likely be the next character.
        p = torch.ones(27) / 27 # generate using uniform distrubution
        idx = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
        name.append(itos[idx])
        # check for end token
        if idx == 0:
            break
    print("".join(name))


def compute_model_log_likelihood(P, words, stoi):
    n = 0
    log_likelihood = 0.0
    for word in words:
        chs = ['.'] + list(word) + ['.']
        for ch1, ch2 in zip(chs, chs[1:]):
            idx1 = stoi[ch1]
            idx2 = stoi[ch2]
            prob = P[idx1, idx2]
            logprob = torch.log(prob)
            log_likelihood += logprob
            n += 1

    nll = -log_likelihood
    avg_nll = nll/n

    print(f'{log_likelihood=}')
    print(f'{nll=}')
    print(f'{avg_nll=}')


def main():
    words = open("names.txt", "r").read().splitlines()
    chars = sorted(list(set("".join(words))))
    stoi = {s: i+1 for i,s in enumerate(chars)}
    stoi["."] = 0
    itos = {i: s for s,i in stoi.items()}

    N = get_bigram_frequency_counts(words, stoi)
    plot_bigram_frequency_counts(N, itos)
    P = get_bigram_frequency_counts_probability_distribution(N)
    compute_model_log_likelihood(P, words, stoi)

    print("\nGenerating names from model:")
    g = torch.Generator().manual_seed(2147483647)
    for i in range(10):
        generate_names_from_model(P, itos, g)

    print("\nGenerating names without model:")
    g = torch.Generator().manual_seed(2147483647)
    for i in range(10):
        generate_names_with_uniform_distribution(itos, g)

main()