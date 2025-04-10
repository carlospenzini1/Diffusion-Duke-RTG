import torch
import argparse

from load_model import load_model
from transformers import GPT2TokenizerFast
import torch.nn.functional as F
import sampling
import torch
import random
import numpy as np
def main():
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
    
    parser = argparse.ArgumentParser(description="Generate some samples")
    parser.add_argument("--model_path", default="louaaron/sedd-medium", type=str)
    parser.add_argument("--dataset", default="wikitext103", type=str)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--steps", type=int, default=1024)
    args = parser.parse_args()

    
    device = torch.device('cuda')
    #device=torch.device('cpu')
    model, graph, noise = load_model(args.model_path, device)
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')

    sampling_fn = sampling.get_pc_sampler(
        graph, noise, (args.batch_size, 1024), 'analytic', args.steps, device=device
    )

    samples = sampling_fn(model)

    text_samples = tokenizer.batch_decode(samples)
    for i in text_samples:
        print(i)
        print("=================================================")

if __name__=="__main__":
    main()