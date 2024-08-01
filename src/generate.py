import time
import torch
import numpy as np
import torch.nn.functional as F

from .tokenize import AggregatedTokenizer

class AggregatedGenerator:
    def __init__(
        self, 
        models: list, 
        agg_tokenizer: AggregatedTokenizer, 
        generation_kwargs: list[dict], 
        decoder_prompts: list[str | None],
        encoder_prompts: list[str | None]
    ):
        self.agg_tokenizer = agg_tokenizer
        self.models = models
        self.gen_kwargs: list[dict] = generation_kwargs
        self.decoder_prompts: list[str | None] = decoder_prompts
        self.encoder_prompts: list[str | None] = encoder_prompts
        
        self.normalization_coef = 1 / len(self.models)  

    def _text_to_inds(self, model_id, text: str, return_tensors="pt"):
        return self.agg_tokenizer.tokenizers[model_id](text, return_tensors=return_tensors).input_ids

    def _init_tokenizer_inds_mappings(
        self,
        encoders_inputs,
    ):
        n_models = len(self.models)
        valid_tokenizes_inds = [[] for _ in range(n_models)]
        agg_tokenizer_inds = [[] for _ in range(n_models)]
        valid_to_agg_inds = [[] for _ in range(n_models)]
        
        
        
        for model_id in range(len(self.models)):
            next_token_logits_tuple = self.models[model_id].generate(
                **encoders_inputs[model_id], 
                max_new_tokens=1,
                num_beams=1,
                output_logits=True,
                return_dict_in_generate=True,
                **self.gen_kwargs[model_id]
            )['logits']  
            next_token_logits = torch.concat(next_token_logits_tuple, dim=0)[0].cpu().numpy()
            
            for token_ind in range(next_token_logits.shape[0]):
                token_str = self.agg_tokenizer.reverse_vocabs[model_id].get(token_ind)  # token in original tokenizer
                if token_str is not None:
                    agg_token_id = self.agg_tokenizer.vocab[token_str]   
                    agg_tokenizer_inds[model_id].append(agg_token_id)
                    valid_tokenizes_inds[model_id].append(token_ind)
            
            valid_logits = next_token_logits[valid_tokenizes_inds[model_id]]
            
            for valid_token_ind, agg_token_id in enumerate(agg_tokenizer_inds[model_id]):
                valid_to_agg_inds[model_id].append(agg_token_id) # valid_token_ind --> agg_token_id
            
        self._valid_tokenizes_inds = valid_tokenizes_inds
        self._agg_tokenizer_inds = agg_tokenizer_inds
        self._valid_to_agg_inds = valid_to_agg_inds

        
    
    @torch.no_grad()
    def _generate_agg_next_top_k_p(
        self, 
        encoders_inputs: list[list[int]] | list[torch.Tensor], 
        decoder_input_text: list[int] | torch.Tensor | None,
        agg_decoder_input_ids: list[int] | torch.Tensor | None,
        device: str = "cuda",
        top_k: int | None = None,
        top_p: float | None = None,
    ):
        assert top_k is None or top_p is None , "top_k, top_p cannot be chosen at the same time"

        # for each model
        agg_probs = np.zeros(len(self.agg_tokenizer.vocab))
        
        for model_id, model in enumerate(self.models):
            decoder_input_text_local = f"{self.decoder_prompts[model_id]} {decoder_input_text}" if decoder_input_text else self.decoder_prompts[model_id]
            start = time.time()
            decoder_input_ids = self.agg_tokenizer.tokenize_single(
                tokenizer_id=model_id, 
                input_text=decoder_input_text_local,
                is_decoder_input=True,
                add_special_tokens=False,
            ).to(device)['input_ids'] if decoder_input_text_local is not None else None
            total = time.time() - start 
            # print(f"{model_id=} decoding time (s): {total}")
            
            start = time.time()
            next_token_logits_tuple = model.generate(
                **encoders_inputs[model_id], 
                decoder_input_ids=decoder_input_ids, 
                max_new_tokens=1,
                num_beams=1,
                output_logits=True,
                return_dict_in_generate=True,
                **self.gen_kwargs[model_id]
            )['logits']  
            total = time.time() - start 
            # print(f"{model_id=} generate time (s): {total}")
         
            
            start = time.time()
            next_token_logits = torch.concat(next_token_logits_tuple, dim=0)[0]
            logits_filtered = next_token_logits[
                self._valid_tokenizes_inds[model_id]
            ]
            scores_filtered = F.softmax(logits_filtered, dim=-1).cpu().numpy()
            agg_probs[self._valid_to_agg_inds[model_id]] += self.normalization_coef * scores_filtered
            
            total = time.time() - start 
            # print(f"{model_id=} agg_probs compute time (s): {total}")

        # print("sum probs =", agg_probs.sum())
        # make agg_probs sum up to 1. due to not existent tokens before 
        agg_probs = agg_probs / agg_probs.sum()
                
        # top-p, top-k or beam search on agg_token_probas
        if top_k is not None:
            inds = np.argsort(-agg_probs)[:top_k]
            next_agg_token_id = np.random.choice(inds, p = agg_probs[inds] / agg_probs[inds].sum())

        elif top_p is not None:
            inds = np.argsort(-agg_probs)
            while agg_probs[inds].sum() > top_p and len(inds) > 1:
                inds = inds[:-1]
            next_agg_token_id = np.random.choice(inds, p = agg_probs[inds] / agg_probs[inds].sum())
        
        else:
            next_agg_token_id = np.random.choice(np.arange(agg_probs.shape[0]), p=agg_probs) 

        decoder_output_ids = agg_decoder_input_ids + [next_agg_token_id] if agg_decoder_input_ids is not None else [next_agg_token_id] 
        decoder_output_text = self.agg_tokenizer.decode(decoder_output_ids, skip_special_tokens=True)
        return next_agg_token_id, decoder_output_text, decoder_output_ids
    
    
    @torch.no_grad()
    def _generate_next(
        self, 
        model_id: int,
        encoder_input_text: str, 
        decoder_input_text: str | None,
        device: str = "cuda",
        max_new_tokens: int = 1,
    ):
        encoder_input_text = f"{self.encoder_prompts[model_id]} {encoder_input_text}" if self.encoder_prompts[model_id] is not None else encoder_input_text
        encoder_inputs = self.agg_tokenizer.tokenize_single(
            tokenizer_id=model_id, 
            input_text=encoder_input_text,
            add_special_tokens=True
        ).to(device)

        if decoder_input_text:
            decoder_input_ids = self.agg_tokenizer.tokenize_single(
                tokenizer_id=model_id, 
                input_text=decoder_input_text,
                is_decoder_input=True,
                add_special_tokens=False,
            ).to(device)['input_ids'] 
        else: 
            decoder_input_ids = None
        print(decoder_input_text, decoder_input_ids)

        print(encoder_inputs)
        print(self.gen_kwargs[model_id])
               
        next_token_logits_tuple = self.models[model_id].generate(
            **encoder_inputs, 
            decoder_input_ids=decoder_input_ids,
            max_new_tokens=max_new_tokens,
            num_beams=1,
            output_logits=True,
            return_dict_in_generate=True,
            **self.gen_kwargs[model_id]
        )['logits']  
        print(next_token_logits_tuple)
        print(next_token_logits_tuple[0].shape)
        
        next_token_logits = torch.concat(next_token_logits_tuple, dim=0)
        next_tokens_ids = torch.max(next_token_logits, dim=-1).indices.cpu().tolist()
        decoder_input_ids = decoder_input_ids.cpu().tolist()[0] if decoder_input_ids is not None else []
        decoder_output_ids = decoder_input_ids + next_tokens_ids
        decoder_output_text = self.agg_tokenizer.tokenizers[model_id].decode(
            decoder_output_ids, skip_special_tokens=True
        )
        return next_tokens_ids, decoder_output_text, decoder_output_ids
     
            
    def generate_agg(
        self, 
        encoder_input_text: str, 
        max_new_tokens=256, 
        device="cuda",
        top_k: int | None = None,
        top_p: float | None = None,
        num_beams: int = 1,
        beam_top_k: int | None = None
    ):
        if beam_top_k is None:
            beam_top_k = num_beams
        
        encoder_input_text = encoder_input_text.replace("\n", "")
        encoders_inputs: list[torch.Tensor] = [
            self.agg_tokenizer.tokenize_single(
                tokenizer_id=model_id, 
                input_text=f"{self.encoder_prompts[model_id]} {encoder_input_text}" if self.encoder_prompts[model_id] is not None else encoder_input_text,
                add_special_tokens=True
            ).to(device)
            for model_id in range(len(self.models))
        ]
        self._init_tokenizer_inds_mappings(encoders_inputs=encoders_inputs)
        
        stop_token_ids: set[int] = self.agg_tokenizer.eos_token_id_set
        next_agg_token_id = None
        beam_sequences = [[] for _ in range(num_beams)]
        beam_is_eos = [False for _ in range(num_beams)]
        beam_input_texts = ["" for _ in range(num_beams)]
        beam_scores = [0] * num_beams
        decoder_output_ids = []
        decoder_output_text = ""
        generated_count = 0
        eos_criteria = False
        
        while not eos_criteria and generated_count + 1 < max_new_tokens:
            decoder_input_text = decoder_output_text
            decoder_input_ids = decoder_output_ids
            if num_beams > 1:
                next_agg_token_id, decoder_output_text, decoder_output_ids, beam_sequences, beam_input_texts, beam_scores, beam_is_eos = self._generate_agg_next_beams(
                    encoders_inputs,
                    beam_input_texts,
                    beam_sequences,
                    beam_scores,
                    beam_is_eos,
                    is_first=(generated_count == 0),
                    device=device,
                    num_beams=num_beams,
                    beam_top_k=beam_top_k,
                )
                eos_criteria = all(beam_is_eos)
                
            else:
                next_agg_token_id, decoder_output_text, decoder_output_ids = self._generate_agg_next_top_k_p(
                    encoders_inputs,
                    decoder_input_text,
                    device=device,
                    top_k=top_k,
                    top_p=top_p,
                    agg_decoder_input_ids=decoder_input_ids
                )
                eos_criteria = (next_agg_token_id in stop_token_ids)
                print(self.agg_tokenizer.reverse_vocab[next_agg_token_id].replace("▁", " "), end="")

            generated_count += 1 
        return decoder_output_text, decoder_output_ids

    @torch.no_grad()
    def generate_single(self, model_id: int, encoder_input_text: str, device: str = "cuda"):
        encoder_input_text = encoder_input_text.replace("\n", "")
        encoder_input_text = f"{self.encoder_prompts[model_id]} {encoder_input_text}" if self.encoder_prompts[model_id] is not None else encoder_input_text
        encoder_inputs = self.agg_tokenizer.tokenize_single(
            tokenizer_id=model_id, 
            input_text=encoder_input_text, 
            add_special_tokens=True
        ).to(device)
        decoder_input_ids = self.agg_tokenizer.tokenize_single(
            tokenizer_id=model_id, 
            input_text=self.decoder_prompts[model_id],
            is_decoder_input=True
        ).to(device)['input_ids'] if self.decoder_prompts[model_id] is not None else None

        tok_preds = self.models[model_id].generate(
            **encoder_inputs, 
            decoder_input_ids=decoder_input_ids, 
            num_beams=1,
            max_new_tokens=256,
            output_logits=True,
            return_dict_in_generate=True,
            **self.gen_kwargs[model_id],
        )['sequences']
        decoder_output_text = self.agg_tokenizer.tokenizers[model_id].decode(
            tok_preds.detach().cpu()[0], skip_special_tokens=True
        )
        return decoder_output_text
    
    
    @torch.no_grad()
    def _generate_agg_next_beams(
        self, 
        encoders_inputs: list[list[int]] | list[torch.Tensor], 
        beam_input_texts: list[str], 
        beam_sequences: list[list[int]],
        beam_scores: list[float],
        beam_is_eos: list[bool],
        is_first: bool,
        device: str = "cuda",
        num_beams: int = 1,
        beam_top_k: int | None = 1,
    ):
        all_candidates = []
        for beam_idx in range(num_beams):
            if beam_is_eos[beam_idx]:
                continue
            
            agg_probs = np.zeros(len(self.agg_tokenizer.vocab))
            
            for model_id, model in enumerate(self.models):
                decoder_input_text_local = f"{self.decoder_prompts[model_id]} {beam_input_texts[beam_idx]}" if beam_input_texts[beam_idx] else self.decoder_prompts[model_id]
                decoder_input_ids = self.agg_tokenizer.tokenize_single(
                    tokenizer_id=model_id, 
                    input_text=decoder_input_text_local,
                    is_decoder_input=True,
                    add_special_tokens=False,
                ).to(device)['input_ids'] if decoder_input_text_local is not None else None
    
                next_token_logits_tuple = model.generate(
                    **encoders_inputs[model_id], 
                    decoder_input_ids=decoder_input_ids, 
                    num_beams=1,
                    max_new_tokens=1,
                    output_logits=True,
                    return_dict_in_generate=True,
                    **self.gen_kwargs[model_id]
                )['logits']  
    
                next_token_logits = torch.concat(next_token_logits_tuple, dim=0)[0]
                logits_filtered = next_token_logits[
                    self._valid_tokenizes_inds[model_id]
                ]
                scores_filtered = F.softmax(logits_filtered, dim=-1).cpu().numpy()
                agg_probs[self._valid_to_agg_inds[model_id]] += self.normalization_coef * scores_filtered

            # Normalize agg_probs
            agg_probs = agg_probs / agg_probs.sum()

            # Collect all candidate tokens and their scores
            candidate_ids = np.arange(len(agg_probs))
            for candidate_id in candidate_ids:
                new_sequence = beam_sequences[beam_idx] + [candidate_id]
                new_score = beam_scores[beam_idx] + np.log(agg_probs[candidate_id])
                all_candidates.append((new_sequence, new_score))
            
            if is_first:
                # break, because first token should be [:top_k] from one-beam scores
                break
        

        # Select the top num_beams sequences
        all_candidates.sort(key=lambda x: x[1], reverse=True)
        top_k_candidates = all_candidates[:beam_top_k]

        top_k_scores = np.array([score for seq, score in top_k_candidates])
        top_k_scores_normalized = top_k_scores / top_k_scores.sum()

        ranked_candidate_inds_unsorted = np.random.choice(
            range(beam_top_k), 
            size=num_beams, 
            p=top_k_scores_normalized, 
            replace=False
        )
        ranked_candidate_inds_sorted = sorted(ranked_candidate_inds_unsorted)

        beam_sequences = []
        beam_scores = []
        for ranked_candidate_ind in ranked_candidate_inds_sorted:
            beam_sequences.append(top_k_candidates[ranked_candidate_ind][0])
            beam_scores.append(top_k_candidates[ranked_candidate_ind][1])
            
                
        # Update beam_input_texts for the next step
        beam_input_texts = [self.agg_tokenizer.decode(seq, skip_special_tokens=True) for seq in beam_sequences]
        beam_is_eos = [seq[-1] in self.agg_tokenizer.eos_token_id_set for seq in beam_sequences]
        
        # Return the best beam
        best_beam_idx = np.argmax(beam_scores)
        decoder_output_ids = beam_sequences[best_beam_idx]
        decoder_output_text = self.agg_tokenizer.decode(decoder_output_ids, skip_special_tokens=True)

        print()
        for beam_input_text, beam_score in zip(beam_input_texts, beam_scores):
            print(f"score = {beam_score: 0.5f} - {beam_input_text}")
        print()

        return decoder_output_ids[-1], decoder_output_text, decoder_output_ids, beam_sequences, beam_input_texts, beam_scores, beam_is_eos


    def generate_all_single(self, encoder_input_text: str, device: str = "cuda"):
        decoders_output_texts = [
            self.generate_single(model_id=model_id, encoder_input_text=encoder_input_text, device=device) for model_id in range(len(self.models))
        ]
        return decoders_output_texts