###############################################################################
# Language Modeling on Wikitext-2
#
# This file generates new sentences sampled from the language model.
#
###############################################################################
import argparse
import data
import torch

parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 Language Model')
# Model parameters.
parser.add_argument('--data', type=str, default='./data/wikitext-2',
                    help='location of the data corpus')
parser.add_argument('--checkpoint', type=str, default='./model.pt',
                    help='model checkpoint to use')
parser.add_argument('--outf', type=str, default='generated.txt',
                    help='output file for generated text')
parser.add_argument('--words', type=int, default='1000',
                    help='number of words to generate')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--temperature', type=float, default=1.0,
                    help='temperature - higher will increase diversity')
parser.add_argument('--log-interval', type=int, default=100,
                    help='reporting interval')
parser.add_argument('--input', default=None,
                    help='Specify words for generation to start from.')
args = parser.parse_args()

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda.")

device = torch.device("cuda" if args.cuda else "cpu")

if args.temperature < 1e-3:
    parser.error("--temperature has to be greater or equal 1e-3.")

with open(args.checkpoint, 'rb') as f:
    model = torch.load(f, map_location=device)
model.eval()

corpus = data.Corpus(args.data)
ntokens = len(corpus.dictionary)

is_transformer_model = hasattr(model, 'model_type') and model.model_type == 'Transformer'
if not is_transformer_model:
    hidden = model.init_hidden(1)


with open(args.outf, 'w') as outf:
    # no tracking history
    with torch.no_grad():  

        # Start of the modified code. 
        if not args.input:
            input = torch.randint(ntokens, (1, 1), dtype=torch.long).to(device)

        else:
            input_words = args.input
            input_words = input_words.split() if " " in input_words else [input_words]
            input_length = len(input_words) 

            # Assertion in case " " is provided as input, aka. covering myself for line 66. :)
            assert input_length > 0, f"\033[92mEmpty input. Please enter word(s)! Example words: I, you, know.\033[0m"

            for word in input_words:
                # A little longer line, it breakes very ugly in the output otherwise.
                assert word in corpus.dictionary.word2idx.keys(), \
                f"\033[92mWord '{word}' not in a vocabulary. Please try again with different word(s)! Example words: I, you, know.\033[0m"
    
        for i in range(input_length if args.input else 0, args.words):

            if args.input and i == input_length:
                idx = 0

                while True:
                    word = input_words[idx]
                    word_idx = corpus.dictionary.word2idx[word]
                    input = torch.Tensor([[word_idx]]).long().to(device)
                    output, hidden = model(input, hidden)

                    outf.write(word + ('\n' if i % 20 == 19 else ' '))
                    idx += 1

                    if idx == input_length:
                        break

            output, hidden = model(input, hidden) 
            word_weights = output.squeeze().div(args.temperature).exp().cpu()
            word_idx = torch.multinomial(word_weights, 1)[0] 
            input.fill_(word_idx)

            word = corpus.dictionary.idx2word[word_idx]
            outf.write(word + ('\n' if i % 20 == 19 else ' '))

            if i % args.log_interval == 0:
                print('| Generated {}/{} words'.format(i, args.words))
                

