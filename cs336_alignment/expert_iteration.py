import argparse

from vllm import SamplingParams

def training_loop(n_ei_steps, G, sampling_min_tokens, sampling_max_tokens, batch_size_per_ei_step, train_prompts, train_answers, train_ground_truths):
    sampling_params = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        max_tokens=sampling_max_tokens,
        min_tokens=sampling_min_tokens,
        n=G,
        seed=42,
    )

    for i in range(n_ei_steps):
        print(f"Expert iteration {i}")
        batch_indices = random.sample(range(len(train_prompts)), batch_size_per_ei_step)
        batch_prompts = [train_prompts[i] for i in batch_indices]
        batch_answers = [train_answers[i] for i in batch_indices]
        batch_ground_truths = [train_ground_truths[i] for i in batch_indices]

        # train the model
        # evaluate the model
        # save the model
        # load the model
        # train the model

if __name__ == "__main__":
    # add arguments for n_ei_steps, G, sampling_min_tokens, sampling_max_tokens, seed
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_ei_steps", type=int, default=5)
    parser.add_argument("--G", type=int, default=4)
    parser.add_argument("--sampling_min_tokens", type=int, default=4)
    parser.add_argument("--sampling_max_tokens", type=int, default=1024)
    parser.add_argument("--batch_size_per_ei_step", type=int, default=100)
    args = parser.parse_args()

    training_loop(args.n_ei_steps, args.G, args.sampling_min_tokens, args.sampling_max_tokens)