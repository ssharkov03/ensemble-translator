import time
import torch
import numpy as np
import torch.nn.functional as F

from .tokenize import AggregatedTokenizer
from .evaluate import SimilarityChecker

class EnsembleGenerator:
    def __init__(
        self,
        models: list,
        similarity_checker: SimilarityChecker, 
        agg_tokenizer: AggregatedTokenizer,
        generation_kwargs: list[dict],
        decoder_prompts: list[str | None],
        encoder_prompts: list[str | None],
    ):
        self.agg_tokenizer = agg_tokenizer
        self.similarity_checker = similarity_checker
        self.models = models
        self.gen_kwargs: list[dict] = generation_kwargs
        self.decoder_prompts: list[str | None] = decoder_prompts
        self.encoder_prompts: list[str | None] = encoder_prompts
        self.num_models = len(self.models)

        self.EMPTY_CANDIDATE_IND = -1
        
    def _text_to_inds(self, model_id, text: str, return_tensors="pt"):
        return self.agg_tokenizer.tokenizers[model_id](
            text, return_tensors=return_tensors
        ).input_ids

    def _init_tokenizer_inds_mappings(
        self,
        dummy_encoders_inputs,
    ):
        n_models = len(self.models)
        valid_tokenizes_inds = [[] for _ in range(n_models)]
        agg_tokenizer_inds = [[] for _ in range(n_models)]
        valid_to_agg_inds = [[] for _ in range(n_models)]

        for model_id in range(len(self.models)):
            next_token_logits_tuple = self.models[model_id].generate(
                **dummy_encoders_inputs[model_id],
                max_new_tokens=1,
                num_beams=1,
                output_logits=True,
                return_dict_in_generate=True,
                **self.gen_kwargs[model_id],
            )["logits"]
            next_token_logits = (
                torch.concat(next_token_logits_tuple, dim=0)[0].cpu().numpy()
            )

            for token_ind in range(next_token_logits.shape[0]):
                token_str = self.agg_tokenizer.reverse_vocabs[model_id].get(
                    token_ind
                )  # token in original tokenizer
                if token_str is not None:
                    agg_token_id = self.agg_tokenizer.vocab[token_str]
                    agg_tokenizer_inds[model_id].append(agg_token_id)
                    valid_tokenizes_inds[model_id].append(token_ind)

            for valid_token_ind, agg_token_id in enumerate(
                agg_tokenizer_inds[model_id]
            ):
                valid_to_agg_inds[model_id].append(
                    agg_token_id
                )  # valid_token_ind --> agg_token_id

        self._valid_tokenizes_inds = valid_tokenizes_inds
        self._agg_tokenizer_inds = agg_tokenizer_inds
        self._valid_to_agg_inds = valid_to_agg_inds


    @torch.no_grad()
    def _get_encoder_hidden_states(self, model_id: int, encoder_inputs: list[dict]):
        try:
            encoder_output_vectors = (
                self.models[model_id]
                .base_model.encoder(**encoder_inputs[model_id], return_dict=True)
                .last_hidden_state
            )
        except AttributeError:
            encoder_output_vectors = (
                self.models[model_id]
                .base_model.text_encoder(**encoder_inputs[model_id], return_dict=True)
                .last_hidden_state
            )
        return encoder_output_vectors

    def _init_eos_not_eos_ids(self):
        self.eos_ids = list(self.agg_tokenizer.eos_token_id_set)
        self.not_eos_ids = list(set(range(len(self.agg_tokenizer.vocab))) - set(self.eos_ids))    

    @staticmethod
    def _repeat_first_dim(t, n_repeat):
        other_dims = [1] * (t.ndim - 1)
        return t.repeat(n_repeat, *other_dims)
    
    def ensemble_generate(
        self,
        encoder_input_text: str,
        max_new_tokens=256,
        device="cuda",
        num_beams: int = 1,
        beam_top_k: int | None = None,
        verbose: bool = False,
    ):
        assert num_beams >= 1
        
        if beam_top_k is None:
            beam_top_k = num_beams


        # Encoder outputs precomputation
        encoder_input_text = encoder_input_text.replace("\n", "")
        encoders_inputs: list[dict] = [
            self.agg_tokenizer.tokenize_single(
                tokenizer_id=model_id,
                input_text=(
                    f"{self.encoder_prompts[model_id]} {encoder_input_text}"
                    if self.encoder_prompts[model_id] is not None
                    else encoder_input_text
                ),
                add_special_tokens=True,
            ).to(device)
            for model_id in range(len(self.models))
        ]

        encoders_last_hidden_states = [
            self._get_encoder_hidden_states(model_id, encoders_inputs)
            for model_id in range(len(self.models))
        ]

        self._init_eos_not_eos_ids()
        self._init_tokenizer_inds_mappings(dummy_encoders_inputs=encoders_inputs)

        stop_gen_mask = np.zeros((num_beams, self.num_models))  # num_beams x num_models
        stop_token_ids: set[int] = self.agg_tokenizer.eos_token_id_set
        beam_sequences = [[] for _ in range(num_beams)]
        beam_input_texts = ["" for _ in range(num_beams)]
        beam_is_eos = [False for _ in range(num_beams)]
        beam_scores = [0] * num_beams
        decoder_output_ids = []
        decoder_output_text = ""
        generated_count = 0
        eos_criteria = False

        while not eos_criteria and generated_count + 1 < max_new_tokens:
            decoder_input_text = decoder_output_text
            decoder_input_ids = decoder_output_ids
            (
                beam_sequences,
                beam_input_texts,
                beam_scores,
                beam_is_eos,
                stop_gen_mask,
            ) = self._generate_agg_next_beams(
                encoders_last_hidden_states,
                beam_input_texts,
                beam_sequences,
                beam_scores,
                stop_gen_mask,
                beam_is_eos,
                device=device,
                num_beams=num_beams,
                beam_top_k=beam_top_k,
                verbose=verbose,
            )
            eos_criteria = all(beam_is_eos) or (
                stop_gen_mask.sum() == num_beams * self.num_models
            )
            generated_count += 1
        
        # Return the best beam
        best_beam_idx = np.argmax(beam_scores)
        decoder_output_ids = beam_sequences[best_beam_idx]
        decoder_output_text = self.agg_tokenizer.decode(
            decoder_output_ids, 
            skip_special_tokens=True
        )
        del encoders_last_hidden_states
        return decoder_output_text

    
    @torch.no_grad()
    def _generate_agg_next_beams(
        self,
        encoders_output_vectors: list[torch.Tensor],
        beam_input_texts: list[str],
        beam_sequences: list[list[int]],
        beam_scores: list[float],
        stop_gen_mask: np.ndarray,
        beam_is_eos: list[bool],
        device: str = "cuda",
        num_beams: int = 1,
        beam_top_k: int | None = 1,
        verbose: bool = False,
    ):
        beams_input_texts_set = set()
        other_candidates_list = []
        running_beams = []

        ### Check for running/stopped beams
        for beam_idx in range(num_beams):            
            if beam_input_texts[beam_idx] in beams_input_texts_set:
                # beam has duplicate => skip it
                continue
            elif beam_is_eos[beam_idx]:
                # beam is full already => just use it as candidate
                sequence = beam_sequences[beam_idx]
                score = beam_scores[beam_idx]
                local_candidates = np.full((1, 3), fill_value=np.nan)  # just one not changed candidate
                # candidate_id, score, beam_idx
                local_candidates[-1, :] = [self.EMPTY_CANDIDATE_IND, beam_scores[beam_idx], beam_idx]   
                local_candidates = local_candidates[~np.isnan(local_candidates).any(axis=1)]
                other_candidates_list.append(local_candidates)
                continue
            
            beams_input_texts_set.add(beam_input_texts[beam_idx])
            running_beams.append(beam_idx)

        
        beam_bs = len(running_beams)
        next_probs = np.zeros((num_beams, len(self.agg_tokenizer.vocab)))

        for model_id, model in enumerate(self.models):

            ### Tokenization of decoder input
            decoder_input_texts_local = [
                f"{self.decoder_prompts[model_id]} {beam_input_texts[beam_idx]}"
                if beam_input_texts[beam_idx]
                else self.decoder_prompts[model_id] 
                for beam_idx in running_beams
            ]
                        
            decoder_inputs = self.agg_tokenizer.tokenize_single(
                model_id, 
                decoder_input_texts_local, 
                is_decoder_input=True, 
                add_special_tokens=False,
                padding=True,
                pad_to_multiple_of=1,
            ).to(device)

            ### Generating next token logits
            outputs = self.models[model_id](
                input_ids=None,
                decoder_input_ids=decoder_inputs.input_ids,
                decoder_attention_mask=decoder_inputs.attention_mask,
                encoder_outputs=(self._repeat_first_dim(encoders_output_vectors[model_id], beam_bs),),
                return_dict=True,
                **self.gen_kwargs[model_id],
            )

            ### Processing predicted probabilities
            beam_next_token_logits = outputs.logits[:, -1, :]
            beam_next_token_ids = torch.argmax(beam_next_token_logits, dim=-1).cpu() 
            
            stop_gen_mask[running_beams, model_id] = (
                beam_next_token_ids == self.agg_tokenizer.eos_tokens_ids[model_id]
            )

            logits_filtered = beam_next_token_logits[:, self._valid_tokenizes_inds[model_id]]
            
            scores_filtered = F.softmax(logits_filtered, dim=-1).cpu().numpy()
            next_probs[np.ix_(running_beams, self._valid_to_agg_inds[model_id])] += (
                scores_filtered
            )

        ### Adding candidates for beams
        next_probs = next_probs / next_probs.sum(axis=1, keepdims=True)
        next_probs_clipped = np.clip(next_probs, 0.0, 1.0)
        next_log_probs = np.log(next_probs_clipped)
        
        # Here condition of eos can be defined: [... == self.num_models] [... > self.num_models / 2]
        beam_is_eos_new = stop_gen_mask.sum(axis=1) == self.num_models  

        beam_candidates = np.full((len(running_beams), len(self.agg_tokenizer.vocab), 3), fill_value=np.nan)
        beam_candidates[:, :, 0] = range(len(self.agg_tokenizer.vocab))
        for i, beam_idx in enumerate(running_beams):
            beam_candidates[i, :, 2] = beam_idx
            beam_candidates[i, :, 1] = beam_scores[beam_idx] + next_log_probs[beam_idx, :]
            if not beam_is_eos_new[beam_idx]:
                beam_candidates[i, self.eos_ids, 1] = np.nan
        
        ### Updating beams
        all_candidates = np.vstack([np.vstack(beam_candidates), *other_candidates_list])
        all_candidates = all_candidates[~np.isnan(all_candidates).any(axis=1)]

        if all_candidates.shape[0] > beam_top_k:
            top_k_candidates = all_candidates[np.argpartition(-all_candidates[:, 1], beam_top_k, axis=0)[:beam_top_k], :]
        else:
            top_k_candidates = all_candidates
        
        num_top_k_candidates = top_k_candidates.shape[0]

        top_k_scores = top_k_candidates[:, 1]
        top_k_scores_normalized = top_k_scores / top_k_scores.sum()
        
        if beam_top_k > num_beams:
            ranked_candidate_inds = np.random.choice(
                range(min(num_top_k_candidates, beam_top_k)), 
                size=min(num_top_k_candidates, num_beams), 
                p=top_k_scores_normalized, 
                replace=False
            )
        else:
            ranked_candidate_inds = range(min(num_top_k_candidates, num_beams))

        beam_sequences_old = beam_sequences
        beam_sequences = []
        beam_scores = top_k_candidates[ranked_candidate_inds, 1].tolist()
        beam_old_inds = top_k_candidates[ranked_candidate_inds, 2].astype(int).tolist()
        for ranked_candidate_ind in ranked_candidate_inds:
            candidate_id = int(top_k_candidates[ranked_candidate_ind][0])
            candidate_beam_idx = beam_old_inds[ranked_candidate_ind]
            if candidate_id != self.EMPTY_CANDIDATE_IND:
                new_sequence = beam_sequences_old[candidate_beam_idx] + [candidate_id]
            else:
                new_sequence = beam_sequences_old[candidate_beam_idx]
            beam_sequences.append(new_sequence)

        stop_gen_mask = stop_gen_mask[beam_old_inds, :]
        beam_input_texts = [
            self.agg_tokenizer.decode(seq, skip_special_tokens=True)
            for seq in beam_sequences
        ]
        beam_is_eos = [
            seq[-1] in self.agg_tokenizer.eos_token_id_set for seq in beam_sequences
        ]
        if verbose:
            print()
            for beam_input_text, beam_score in zip(beam_input_texts, beam_scores):
                print(f"score = {beam_score: 0.5f} - {beam_input_text}")
            print()
        
        del logits_filtered, outputs, beam_next_token_logits
        return (
            beam_sequences,
            beam_input_texts,
            beam_scores,
            beam_is_eos,
            stop_gen_mask,
        )

    @torch.no_grad()
    def generate_single(
        self, 
        model_id: int, 
        encoder_input_text: str, 
        device: str = "cuda", 
        num_beams: int = 5, 
        max_new_tokens: int = 256,  
        **kwargs
    ):
        encoder_input_text = encoder_input_text.replace("\n", "")
        encoder_input_text = (
            f"{self.encoder_prompts[model_id]} {encoder_input_text}"
            if self.encoder_prompts[model_id] is not None
            else encoder_input_text
        )
        encoder_inputs = self.agg_tokenizer.tokenize_single(
            tokenizer_id=model_id,
            input_text=encoder_input_text,
            add_special_tokens=True,
        ).to(device)
        decoder_input_ids = (
            self.agg_tokenizer.tokenize_single(
                tokenizer_id=model_id,
                input_text=self.decoder_prompts[model_id],
                is_decoder_input=True,
            ).to(device)["input_ids"]
            if self.decoder_prompts[model_id] is not None
            else None
        )
        tok_preds = self.models[model_id].generate(
            **encoder_inputs,
            decoder_input_ids=decoder_input_ids,
            num_beams=5,
            max_new_tokens=max_new_tokens,
            **self.gen_kwargs[model_id],
            **kwargs,
        )
        if isinstance(tok_preds, torch.Tensor):
            decoder_output_text = self.agg_tokenizer.tokenizers[model_id].decode(
                tok_preds.cpu()[0], skip_special_tokens=True
            )
        else:
            decoder_output_text = self.agg_tokenizer.tokenizers[model_id].decode(
                tok_preds.sequences.cpu()[0], skip_special_tokens=True
            )

        
        return decoder_output_text

    def generate_all_single(
        self, 
        encoder_input_text: str, 
        device: str = "cuda",
        num_beams: int = 5, 
        max_new_tokens: int = 256,  
        **kwargs
    ):
        decoders_output_texts = [
            self.generate_single(
                model_id=model_id, 
                encoder_input_text=encoder_input_text, 
                device=device,
                num_beams=num_beams,
                max_new_tokens=max_new_tokens,
                **kwargs
            )
            for model_id in range(self.num_models)
        ]
        return decoders_output_texts

    def translate(
        self, 
        src_text: str, 
        device: str = "cuda",
        ensemble_num_beams: int = 3, 
        instance_num_beams: int = 5,
        max_new_tokens: int = 256,
        verbose: bool = False,
        **kwargs
    ):
        response = dict()
        if verbose:
            print("Generating ensemble translation...")
        
        response['ensemble_translation'] = self.ensemble_generate(
            encoder_input_text=src_text,
            device=device,
            max_new_tokens=max_new_tokens,
            num_beams=ensemble_num_beams,
            verbose=verbose,
        )
        if verbose:
            print("Generating instance translations...")

        response['instance_translations'] = self.generate_all_single(
            encoder_input_text=src_text, 
            device=device,
            num_beams=instance_num_beams, 
            max_new_tokens=max_new_tokens,  
            **kwargs
        )

        response["similarity"] = self.similarity_checker.check_similarity(texts=response['instance_translations'])
        return response
        

