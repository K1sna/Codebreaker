import os
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm


class StarCoder2Querier:

    def __init__(self):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cache_dir = "./model_tmp/starcoder2"
        if not os.path.exists(cache_dir):
            os.system(r"touch {}".format(cache_dir))
        self.tokenizer = AutoTokenizer.from_pretrained("bigcode/starcoder2-3b", padding_side='left', cache_dir = cache_dir)
        self.model = AutoModelForCausalLM.from_pretrained("bigcode/starcoder2-3b", cache_dir = cache_dir).to(self.device)

    def cal_token_ppl(log_prob):
        perplexity = 2 ** (-log_prob)
        return perplexity

    def simple_batch_query(self, prompts, batch_size=4, m=2):
        """Batch query the prompts and keep records in the log files (simple version)."""
        n = len(prompts)
        # Ensure the tokenizer has a pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        num_batches = len(prompts) // batch_size + (len(prompts) % batch_size > 0)

        # Initialize dynamic lists for responses
        generated_texts_list = [[] for _ in range(m)]

        for i in tqdm(range(num_batches), desc="Batch Querying", unit="batch"):
            batch_start = i * batch_size
            batch_end = min((i + 1) * batch_size, len(prompts))
            batch_prompts = prompts[batch_start:batch_end]

            input_ids_batch = self.tokenizer(batch_prompts, return_tensors='pt', padding=True, truncation=True).to(self.device)
            input_lengths = (input_ids_batch['attention_mask'].sum(dim=1)).tolist()
            max_input_length = max(input_lengths)

            outputs = self.model.generate(
                input_ids_batch['input_ids'],
                attention_mask=input_ids_batch['attention_mask'],
                max_length=max_input_length + 50,
                num_return_sequences=m,  # Generate top-m sequences
                output_scores=True,
                temperature=0.8,
                no_repeat_ngram_size=2,
                return_dict_in_generate=True,
                pad_token_id=self.tokenizer.pad_token_id,
                do_sample=True
            )

            generated_ids_batch = outputs.sequences.view(batch_size, m, -1)  # Reshape to (batch_size, num_return_sequences, sequence_length)

            for j in range(batch_size):
                generated_texts = []
                for k in range(m):
                    generated_ids = generated_ids_batch[j, k]
                    generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
                    generated_texts.append(generated_text)      
                # Append the generated texts to the corresponding lists
                for k in range(m):
                    generated_texts_list[k].append(generated_texts[k])

        results = []
        for i in range(n):
            result = {}
            for k in range(m):
                result[f'response{k+1}'] = generated_texts_list[k][i]
            results.append(result)

        return results

    def batch_query(self, prompts, batch_size=4, m=2):
        """Batch query the prompts and keep records in the log files."""
        n = len(prompts)
        try:
            # Ensure the tokenizer has a pad token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            num_batches = len(prompts) // batch_size + (len(prompts) % batch_size > 0)

            # Initialize dynamic lists for responses, probabilities, and perplexities
            generated_texts_list = [[] for _ in range(m)]
            log_probs_list = [[] for _ in range(m)]
            perplexities_list = [[] for _ in range(m)]

            for i in tqdm(range(num_batches), desc="Batch Querying", unit="batch"):
                batch_start = i * batch_size
                batch_end = min((i + 1) * batch_size, len(prompts))
                batch_prompts = prompts[batch_start:batch_end]

                input_ids_batch = self.tokenizer(batch_prompts, return_tensors='pt', padding=True, truncation=True).to(self.device)
                input_lengths = (input_ids_batch['attention_mask'].sum(dim=1)).tolist()
                max_input_length = max(input_lengths)

                outputs = self.model.generate(
                    input_ids_batch['input_ids'],
                    attention_mask=input_ids_batch['attention_mask'],
                    max_length=max_input_length + 100,
                    num_return_sequences=m,  # Generate top-m sequences
                    output_scores=True,
                    temperature=0.8,
                    no_repeat_ngram_size=2,
                    return_dict_in_generate=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    do_sample=True
                )

                generated_ids_batch = outputs.sequences.view(batch_size, m, -1)  # Reshape to (batch_size, num_return_sequences, sequence_length)
                logits = self.model.compute_transition_scores(
                    outputs.sequences, outputs.scores, normalize_logits=True)

                for j in range(batch_size):
                    generated_texts, log_probs_lists, perplexities = [], [], []
                    for k in range(m):
                        generated_ids = generated_ids_batch[j, k]
                        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
                        generated_texts.append(generated_text)      

                        log_probs = logits[j * m + k].cpu().numpy().tolist()  # 将 logit 转换为列表
                        log_probs_lists.append(log_probs)

                        # Calculate perplexity for each generated sequence
                        encodings = self.tokenizer(generated_text, return_tensors='pt').to(self.device)
                        max_length = self.model.config.max_position_embeddings
                        stride = 512
                        seq_len = encodings.input_ids.size(1)
                    
                        nlls = []
                        prev_end_loc = 0
                        for begin_loc in range(0, seq_len, stride):
                            end_loc = min(begin_loc + max_length, seq_len)
                            trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
                            input_ids = encodings.input_ids[:, begin_loc:end_loc].to(self.device)
                            target_ids = input_ids.clone()
                            target_ids[:, :-trg_len] = -100

                            with torch.no_grad():
                                outputs = self.model(input_ids, labels=target_ids)

                                # loss is calculated using CrossEntropyLoss which averages over valid labels
                                neg_log_likelihood = outputs.loss

                            nlls.append(neg_log_likelihood)

                            prev_end_loc = end_loc
                            if end_loc == seq_len:
                                break

                        ppl = torch.exp(torch.stack(nlls).mean()).item()
                        perplexities.append(ppl)

                    # Append the generated texts, probabilities, and perplexities to the corresponding lists
                    for k in range(m):
                        generated_texts_list[k].append(generated_texts[k])
                        log_probs_list[k].append(log_probs_lists[k])
                        perplexities_list[k].append(perplexities[k])

            results = []
            for i in range(n):
                result = {}
                for k in range(m):
                    result[f'response{k+1}'] = generated_texts_list[k][i]
                    result[f'log_probs{k+1}'] = log_probs_list[k][i]
                    result[f'wppl{k}'] = perplexities_list[k][i]
                results.append(result)
 
            return results

        except Exception as e:
            print(f"Exception during batch querying: {str(e)}")
            exit(1)