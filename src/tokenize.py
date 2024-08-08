from copy import deepcopy

class AggregatedTokenizer:
    def __init__(
        self, 
        tokenizers: list, 
        tokenization_kwargs: list, 
        decoder_tokenization_postprocessing: list,            
    ):
        self.tokenizers = tokenizers
        self.tokenization_kwargs = tokenization_kwargs
        self.decoder_tokenization_postprocessing = decoder_tokenization_postprocessing
        
        self.vocabs = [tokenizer.get_vocab() for tokenizer in self.tokenizers]  # str --> id
        self.reverse_vocabs = [{v: k for k, v in vocab.items()} for vocab in self.vocabs]  #  id --> str
        
        self._agg_tokenizer = self._init_agg_tokenizer()
        self.vocab = self._agg_tokenizer.get_vocab()  # str --> id
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}

        self.unk_tokens = [tokenizer.unk_token for tokenizer in self.tokenizers]
        self.unk_token_ids = [tokenizer.unk_token_id for tokenizer in self.tokenizers]
        self.unk_token_id_set = set([self.vocab[unk_token] for unk_token in self.unk_tokens])

        self.eos_tokens = [tokenizer.eos_token for tokenizer in self.tokenizers]
        self.eos_tokens_ids = [tokenizer.eos_token_id for tokenizer in self.tokenizers] # 
        self.eos_token_id_set = set([self.vocab[eos_token] for eos_token in self.eos_tokens])
    
    @staticmethod
    def _get_unique_tokens(tokenizer) -> set:
        return set(tokenizer.get_vocab().keys())

    def _init_agg_tokenizer(self):
        print("Adding tokens from tokenizer 0 to aggregated tokenizer", end="")
        agg_tokenizer = deepcopy(self.tokenizers[0])
        print(" --> Done")
        for i, tokenizer_i in enumerate(self.tokenizers[1:], start=1):
            toks_to_add: set[str] = self._get_unique_tokens(tokenizer_i) - self._get_unique_tokens(agg_tokenizer) 
            print(f"Adding tokens from tokenizer {i} to aggregated tokenizer", end="")
            agg_tokenizer.add_tokens(list(toks_to_add))
            print(" --> Done")
        return agg_tokenizer

    def tokenize_single(
        self, 
        tokenizer_id: int, 
        input_text: str,
        is_decoder_input: bool = False,
        add_special_tokens=False,
        **kwargs
    ):
        tokenized_inputs = self.tokenizers[tokenizer_id](
            input_text, 
            add_special_tokens=add_special_tokens,
            return_tensors="pt",
            **self.tokenization_kwargs[tokenizer_id],
            **kwargs,
        )
        if is_decoder_input and self.decoder_tokenization_postprocessing[tokenizer_id] is not None:
            self.decoder_tokenization_postprocessing[tokenizer_id](
                tokenized_inputs
            )
        return tokenized_inputs

    def __call__(self, *args, **kwargs):
        return self._agg_tokenizer(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self._agg_tokenizer.decode(*args, **kwargs)
    