import math
import jax
import jax.numpy as jnp
import numpy as np
import datasets
from jax.sharding import NamedSharding, PartitionSpec as P
from math_verify import parse, verify
from sampler import sample


def load_datasets(vocab, seq_len=1024):
    pad_id = vocab.pad_id()
    bos_id = vocab.bos_id()
    eos_id = vocab.eos_id()

    # load MATH dataset
    print('loading datasets…')
    ds_name = 'EleutherAI/hendrycks_math'
    configs = datasets.get_dataset_config_names(ds_name) # ['algebra', 'counting_and_probability', 'geometry', 'intermediate_algebra', 'number_theory', 'prealgebra', 'precalculus']
    ds_train = datasets.concatenate_datasets([datasets.load_dataset(ds_name, config, split='train') for config in configs]) # ['problem', 'solution']
    ds_valid = datasets.concatenate_datasets([datasets.load_dataset(ds_name, config, split='test') for config in configs]) # ['problem', 'solution']

    # tokenize trainind dataset
    print('tokenizing training dataset…')
    train_tokens = np.full([len(ds_train), seq_len], pad_id, dtype=np.int32)
    train_pos = np.zeros([len(ds_train), seq_len], dtype=np.int32)
    train_loss_mask = np.zeros([len(ds_train), seq_len], dtype=np.bool_)
    train_attn_mask = np.zeros([len(ds_train), seq_len, seq_len], dtype=np.bool_)
    seq_idx = 0
    tok_idx = 0
    skipped = 0
    for example in ds_train:

        # tokenize example
        prompt = f'Problem: {example["problem"]}\nSolution: '
        solution = f'{example["solution"]}'
        prompt_tokenized, solution_tokenized = vocab.EncodeAsIds([prompt, solution])
        example_tokenized = [bos_id] + prompt_tokenized + solution_tokenized + [eos_id]
        
        # if example is too long, skip it
        if len(example_tokenized) > seq_len:
            skipped += 1
            continue
            
        # if example doesn't fit in current sequence, start next sequence
        if tok_idx + len(example_tokenized) > seq_len: 
            seq_idx += 1
            tok_idx = 0

        # store tokens
        train_tokens[seq_idx, tok_idx:tok_idx+len(example_tokenized)] = example_tokenized
        train_pos[seq_idx, tok_idx:tok_idx+len(example_tokenized)] = np.arange(len(example_tokenized))
        train_loss_mask[seq_idx, tok_idx+len(prompt_tokenized):tok_idx+len(example_tokenized)-1] = True
        train_attn_mask[seq_idx, tok_idx:tok_idx+len(example_tokenized), tok_idx:tok_idx+len(example_tokenized)] = True
        tok_idx += len(example_tokenized)
    train_attn_mask = np.tril(train_attn_mask)
    train_tokens = train_tokens[:seq_idx+1]
    train_pos = train_pos[:seq_idx+1]
    train_attn_mask = train_attn_mask[:seq_idx+1]
    train_loss_mask = train_loss_mask[:seq_idx+1]
    print(f'skipped train. seq.: {skipped / len(ds_train):.1%}')

    # tokenize eval dataset
    print('tokenizing eval dataset…')
    skipped = 0
    prompts_eval = []
    problems_eval = []
    solutions_eval = []
    tokens_eval = np.full([len(ds_valid), seq_len], pad_id, dtype=np.int32)
    for i, example in enumerate(ds_valid):
        problems_eval += [example['problem']]
        solutions_eval += [example['solution']]
        prompt = f'Problem: {example["problem"]}\nSolution: '
        prompt_tokenized = [bos_id] + vocab.EncodeAsIds(prompt)
        if len(prompt_tokenized) < seq_len:
            tokens_eval[i, :len(prompt_tokenized)] = prompt_tokenized
        else:
            skipped += 1
    problems_eval = np.array(problems_eval)
    solutions_eval = np.array(solutions_eval)
    print(f'skipped valid. seq.: {skipped / len(ds_valid):.1%}')
    
    return train_tokens, train_pos, train_attn_mask, train_loss_mask, tokens_eval, problems_eval, solutions_eval


def benchmark_model(key, model, tokens, problems_eval, solutions_eval, vocab, batch_size, n_eval_samples=None, temperature=1):
    pad_id = vocab.pad_id()
    eos_id = vocab.eos_id()
    key_decoding, key_questions = jax.random.split(key)
    mesh = model.in_embed.embedding.value.sharding.mesh
    if n_eval_samples is None: n_eval_samples = len(tokens)
    n_batches = int(math.ceil(n_eval_samples / batch_size))
    n_eval_samples = n_batches * batch_size
    sample_idxs = jax.random.choice(key_questions, max(n_eval_samples, len(tokens)), shape=[n_batches, batch_size], replace=False)
    lengths_list = []
    correct_list = []
    finished_list = []
    for batch_idx in sample_idxs:
        tokens_batch = jax.device_put(tokens[batch_idx], NamedSharding(mesh, P('data', None)))
        completions_tokens = sample(key_decoding, model, tokens_batch, temperature)
        completions_text = vocab.DecodeIds(completions_tokens)
        for sample_idx, completion_tokens, completion_text in zip(batch_idx, completions_tokens, completions_text):
            if sample_idx < len(problems_eval):
                problem = problems_eval[sample_idx]
                gold = solutions_eval[sample_idx]
                parsed = parse(completion_text)
                finished = eos_id in completion_tokens
                correct = verify(parse(gold), parsed)
                lengths_list += [len(completion_tokens)]
                finished_list += [finished]
                correct_list += [correct]
                print('------------')
                print(f'PROMPT:\n{problem}\nCOMPLETION:\n{completion_text}\nPARSED: {parsed}\nGOLD: {gold}\nCORRECT: {correct}')

    return dict(length=np.mean(lengths_list), finished=np.mean(finished_list), accuracy=np.mean(correct_list))
