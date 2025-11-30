import os
import signal
import re
import json
import dataclasses
import time
import contextlib
from typing import Callable, Dict, List, Union, Tuple
from pathlib import Path


import submitit
from submitit.helpers import Checkpointable
import numpy as np
import torch
import torch.distributions
import torch.nn.functional as F

import transformers
from datasets import load_dataset


def cat_if_not_none(a, b):
    if a is None or b is None:
        return None
    else:
        return torch.cat([a, b], dim=0)


def calc_xentropy(logits, input_ids):
    logits_offset = logits[:, :-1]
    return (
        torch.nn.CrossEntropyLoss(reduction="none")(
            logits_offset.reshape(-1, logits_offset.shape[-1]),
            input_ids[:, 1:].reshape(-1),
        )
        .view(*logits_offset.shape[:2])
        .mean(dim=-1)
    )


@dataclasses.dataclass
class State:
    ids: torch.Tensor
    target: torch.Tensor
    xentropy: torch.Tensor
    final_token: torch.Tensor
    token_grads: torch.Tensor
    extra: Dict[str, torch.Tensor]

    def cat(self, other):
        return State(
            ids=torch.cat([self.ids, other.ids], dim=0),
            target=torch.cat([self.target, other.target], dim=0),
            xentropy=torch.cat([self.xentropy, other.xentropy], dim=0),
            final_token=torch.cat([self.final_token, other.final_token], dim=0),
            token_grads=cat_if_not_none(self.token_grads, other.token_grads),
            extra={
                k: cat_if_not_none(self.extra[k], other.extra[k]) for k in self.extra
            },
        )

    def subset(self, keep):
        return State(
            ids=self.ids[keep],
            target=self.target[keep],
            xentropy=self.xentropy[keep],
            final_token=self.final_token[keep],
            token_grads=(
                self.token_grads[keep.to("cpu")]
                if self.token_grads is not None
                else None
            ),
            extra={k: self.extra[k][keep] for k in self.extra},
        )


# based on https://github.com/llm-attacks/llm-attacks/blob/main/llm_attacks/gcg/gcg_attack.py
def token_grads(
    model: torch.nn.Module,
    cache_run: Callable,
    input_ids: torch.Tensor,
    x_penalty: torch.Tensor,
    batch_size: int,
):
    """
    Compute gradients with respect to one-hot encoded input tokens. This is a
    infinitesimal approximation to the token influence on the loss so it's a
    very noisy indicator of which tokens might reduce loss.
    """
    embed = model.get_input_embeddings()

    token_grads = torch.empty(
        (input_ids.shape[0], input_ids.shape[1], embed.num_embeddings),
        dtype=torch.float,
    )
    loss = torch.empty(input_ids.shape[0], device=model.device)
    xentropy = torch.empty(input_ids.shape[0], device=model.device)
    target = torch.empty(input_ids.shape[0], device=model.device)
    final_token = torch.empty(input_ids.shape[0], device=model.device, dtype=torch.long)
    extra = dict()

    with torch.enable_grad():
        model.zero_grad()

        for i in range(0, input_ids.shape[0], batch_size):
            imax = min(i + batch_size, input_ids.shape[0])

            # using a one hot matrix as input to the model gives us gradients with
            # respect to potential input tokens.
            one_hot = F.one_hot(
                input_ids[i:imax].clone(), num_classes=embed.num_embeddings
            ).to(embed.weight.dtype)
            one_hot.requires_grad = True

            cache = cache_run(inputs_embeds=torch.matmul(one_hot, embed.weight))

            logits_offset = cache["logits"][:, :-1]
            this_xentropy = (
                -(torch.log_softmax(logits_offset, dim=-1) * one_hot[:, 1:])
                .sum(dim=-1)
                .mean(dim=-1)
            )

            this_loss = -cache["target"] + this_xentropy * x_penalty[i:imax]
            this_loss.sum().backward()

            loss[i:imax] = this_loss
            target[i:imax] = cache["target"]
            xentropy[i:imax] = this_xentropy
            final_token[i:imax] = cache["logits"][:, -1, :].argmax(dim=-1)
            token_grads[i:imax] = one_hot.grad

            for k in cache:
                if k not in ["target", "logits"]:
                    e = cache[k]
                    if k not in extra:
                        extra[k] = torch.empty(
                            (input_ids.shape[0], *e.shape[1:]),
                            dtype=e.dtype,
                            device=e.device,
                        )
                    extra[k][i:imax] = e

            # important to zero out gradients here to release memory
            model.zero_grad()

    return State(input_ids, target, xentropy, final_token, token_grads, extra)


@dataclasses.dataclass
class History:
    """
    The `epo` function returns a History objet that contains the full history of the
    population members at each iteration.
    """

    # The token ids for each population member at each iteration.
    ids: List = dataclasses.field(default_factory=lambda: [])
    # The cross-entropy loss for each population member at each iteration.
    xentropy: List = dataclasses.field(default_factory=lambda: [])
    # The target objective for each popultion member at each iteration.
    target: List = dataclasses.field(default_factory=lambda: [])
    # The indices of the population members that were retained at each iteration.
    keep: List = dataclasses.field(default_factory=lambda: [])
    # The runtime for each iteration.
    runtime: List = dataclasses.field(default_factory=lambda: [])

    def subset(self, slc):
        """
        Return a History object sliced along the iteration dimension.
        """
        return History(
            self.ids[slc],
            self.xentropy[slc],
            self.target[slc],
            self.keep[slc],
            self.runtime[slc],
        )

    def _insert(self, new_ids, target, xentropy, keep, runtime):
        self.ids.append(new_ids.cpu().numpy())
        self.target.append(target.cpu().numpy())
        self.xentropy.append(xentropy.cpu().numpy())
        self.keep.append(keep.cpu().numpy())
        self.runtime.append(runtime)

    def _finalize(self):
        self.ids = np.stack(self.ids, axis=0)
        self.target = np.stack(self.target, axis=0)
        self.xentropy = np.stack(self.xentropy, axis=0)
        self.keep = np.stack(self.keep, axis=0)
        self.runtime = np.array(self.runtime)


class Selector:
    def __init__(
        self,
        model: torch.nn.Module,
        cache_run: Callable,
        X: torch.Tensor,
        batch_size: int,
    ):
        self.model = model
        self.cache_run = cache_run
        self.X = X
        self.batch_size = batch_size


class GradientSelector(Selector):
    uses_gradient = True

    def setup(self, input_ids: torch.Tensor):
        return token_grads(
            self.model,
            self.cache_run,
            input_ids,
            x_penalty=self.X,
            batch_size=self.batch_size,
        )

    def mutate(self, state, source_idx, input_ids, topk, window=None):
        # when just flipping, the current token gradient falls out of the
        # topk operation, so we can just use the negative new token grad
        topk_grad = (-state.token_grads).topk(k=topk, dim=-1)
        pos = torch.randint(
            low=0 if not window else window.start,
            high=input_ids.shape[1] if not window else window.stop,
            size=(input_ids.shape[0],),
            device=input_ids.device,
        )
        token_idx = torch.randint(
            low=0,
            high=topk,
            size=(input_ids.shape[0],),
            device=input_ids.device,
        )
        input_ids[torch.arange(input_ids.shape[0]), pos] = topk_grad.indices.to(
            input_ids.device
        )[source_idx, pos, token_idx]


def evaluate_fitness(
    model: torch.nn.Module,
    cache_run: Callable,
    input_ids: torch.Tensor,
    batch_size: int,
):
    target = torch.empty(input_ids.shape[0], dtype=torch.float, device=input_ids.device)
    xentropy = torch.empty(
        input_ids.shape[0], dtype=torch.float, device=input_ids.device
    )
    final_token = torch.empty(
        input_ids.shape[0], dtype=torch.long, device=input_ids.device
    )
    extra = dict()
    for i in range(0, input_ids.shape[0], batch_size):
        imax = min(i + batch_size, input_ids.shape[0])
        mini_batch = cache_run(input_ids=input_ids[i:imax])
        target[i:imax] = mini_batch["target"]
        xentropy[i:imax] = calc_xentropy(mini_batch["logits"], input_ids[i:imax])
        final_token[i:imax] = mini_batch["logits"][:, -1, :].argmax(dim=-1)

        for k in mini_batch:
            if k not in ["target", "logits"]:
                e = mini_batch[k]
                if k not in extra:
                    extra[k] = torch.empty(
                        (input_ids.shape[0], *e.shape[1:]),
                        dtype=e.dtype,
                        device=e.device,
                    )
                extra[k][i:imax] = e

    return State(input_ids, target, xentropy, final_token, None, extra)


def pareto_callback(
    cache_run: Callable,
    model: torch.nn.Module,
    tokenizer: transformers.PreTrainedTokenizer,
    x_penalty_min: float,
    x_penalty_max: float,
):
    def f(i, state, last_runtime, history, final=False):
        if last_runtime is not None:
            pass
            # print("runtime: {:.2f} seconds".format(last_runtime))
        # print(f"\nbeginning step {i}, current pareto frontier prompts:")
        last_idx = None
        output = []
        xentropy = []
        Xvs = torch.exp(
            torch.linspace(
                np.log(x_penalty_min / 10.0), np.log(x_penalty_max * 10.0), 200
            )
        ).to(model.device)
        loss = -state.target[None] + Xvs[:, None] * state.xentropy[None]
        idxs = loss.argmin(dim=1)
        for i in range(len(Xvs)):
            idx = idxs[i]
            if idx == last_idx:
                continue
            text = tokenizer.decode(state.ids[idx])
            last_token = tokenizer.decode(state.final_token[idx])
            # print(
            #    f"penalty={Xvs[i]:.2f} xentropy={state.xentropy[idx]:.2f} target={state.target[idx]:.2f} {repr(text + '[' + last_token + ']')}"
            # )
            last_idx = idx
            if final:
                output.append(text + "[" + last_token + "]")
                xentropy.append(f"{state.xentropy[idx]:.2f}")
        if final:
            return False, output, xentropy

    return f


@torch.no_grad
def epo(
    cache_run: Callable,
    model: torch.nn.Module,
    tokenizer: transformers.PreTrainedTokenizer,
    seq_len: int = 12,
    population_size: int = 8,
    iters: int = 300,
    explore_per_pop: int = 32,
    batch_size: int = 8,
    topk: int = 512,
    mutation_method: str = "gradient",
    x_penalty_min: float = 1.0 / 10.0,
    x_penalty_max: float = 10.0,
    restart_frequency: int = 50,
    restart_xentropy: float = 2.0,
    restart_xentropy_max_mult: float = 3.0,
    seed: int = 0,
    initial_ids: torch.Tensor = None,
    window: slice = None,
    history: History = None,
    catch_keyboard_interrupt: bool = False,
    callback: Union[Callable, bool] = None,
    always_recompute_gradients: bool = False,
) -> History:
    """
    Run the EPO algorithm. See the paper for details

    Parameters
    ----------
    cache_run
        A callable that accepts either input_ids or input_embeds and returns a
        dictionary containing the `target` and the logits for each token position
    model
    tokenizer
    population_size, optional
        The population to keep at each iteration, by default 8
    iters, optional
        Number of iterations to run EPO, by default 1000
    explore_per_pop, optional
        Number of children per population member per iteration, by default 32
    batch_size, optional
        GPU batch size, by default 8
    topk, optional
        When selecting token replacements, we select the `topk` tokens by
        gradient magnitude and choose uniformly at random between those, by
        default 32.
    mutation_method, optional
        research, ignore, by default "gradient"
     x_penalty_min, optional
        The minimum cross-entropy penalty, by default 1.0/16.0
    x_penalty_max, optional
        The maximum cross-entropy penalty, by default 16.0
    restart_frequency, optional
        How often do we reset the Pareto frontier, by default 50
    restart_xentropy, optional
        When we reset the Pareto frontier, we select a population member that
        is optimal according to a cross-entropy penalty that is selected
        uniformly at random in the domain
        [restart_xentropy / restart_xentropy_max_mult,
         restart_xentropy * restart_xentropy_max_mult],
        restart_xentropy is by default 2.0
    restart_xentropy_max_mult, optional
        See the explanation for restart_xentropy, by default 3.0
    seed, optional
        Random seed used for initialization, by default 0
    initial_ids, optional
        The initial token ids to begin optimizing from. If None, the initial
        token ids will be selected randomly, by default None
    window, optional
        Only allowed to change these window of tokens during every iteration,
        based on the mutation method
    history, optional
        The history of an EPO run that we want to continue, by default None
    catch_keyboard_interrupt, optional
        Should we catch keyboard interrupts and end the EPO loop?, by default False
    callback, optional
        A function called at the beginning of each iteration, by default None
    always_recompute_gradients, optional
        If a population member is retained across an iteration, we default to
        not recomputing that population member's token gradients. If your
        cache_run stores internal state that changes, you may want to override
        this behavior and recompute gradients every iteration.

    Returns
    -------
        A History object containing the full history of the EPO run.
    """
    start = time.time()
    explore_size = population_size * explore_per_pop
    device = model.device

    if seed is not None:
        torch.manual_seed(seed)

    if x_penalty_min is None or x_penalty_max is None:
        X = torch.zeros(population_size, device=model.device)
    else:
        X = torch.exp(
            torch.linspace(
                np.log(x_penalty_min), np.log(x_penalty_max), population_size
            )
        ).to(model.device)

    if callback is None:
        callback = pareto_callback(
            cache_run,
            model,
            tokenizer,
            X.min().item(),
            X.max().item(),
        )
    elif callback is False:
        callback = lambda *args: True

    ### history and initial_ids ###
    if history is not None:
        if initial_ids is not None:
            raise ValueError("Cannot specify both history and initial_ids")
        initial_ids = history.ids[-1, history.keep[-1]]
    elif initial_ids is not None:
        history = History()
        if initial_ids.shape[1] != seq_len:
            raise ValueError(f"initial_ids must have shape [*, {seq_len}]")
        elif initial_ids.shape[0] != population_size:
            initial_ids = initial_ids.repeat(population_size // initial_ids.shape[0], 1)
        input_ids = initial_ids.to(model.device)
    else:
        history = History()
        input_ids = torch.randint(
            low=0, high=tokenizer.vocab_size, size=(population_size, seq_len)
        ).to(model.device)

    ### choose an update selection method ###
    if mutation_method == "gradient":
        selector_type = GradientSelector
    else:
        raise ValueError(f"Unknown mutation_method: {mutation_method}")
    selector = selector_type(model, cache_run, X, batch_size)

    ### Run the EPO loop: ###
    if hasattr(cache_run, "setup"):
        cache_run.setup(input_ids)
    state = selector.setup(input_ids)

    # We use a try/except blok so that we can catch keyboard interrupts and
    # still return results. This is useful for interactive use when it's nice
    # to launch with a larger `iters` parameter and then just stop the run when
    # the results look good enough.
    for i in range(iters):
        # 1) Report
        terminate_flag = callback(i, state, time.time() - start, history)
        if (
            (isinstance(terminate_flag, str) and terminate_flag == "terminate")
            or (isinstance(terminate_flag, torch.Tensor) and terminate_flag.item())
            or (isinstance(terminate_flag, bool) and terminate_flag)
        ):
            if i == 0:
                history._insert(
                    state.ids,
                    state.target,
                    state.xentropy,
                    torch.arange(state.ids.shape[0]),
                    time.time() - start,
                )
            break
        else:
            start = time.time()
        recompute_gradients = always_recompute_gradients or (
            terminate_flag == "recompute_gradients"
        )

        # 2) Birth children from parents
        source_idx = torch.cat(
            (
                torch.arange(state.ids.shape[0], device=device).repeat(
                    explore_size // state.ids.shape[0]
                ),
                torch.arange(explore_size % state.ids.shape[0], device=device),
            )
        )
        assert source_idx.shape[0] == explore_size
        assert (source_idx < state.ids.shape[0]).all()

        new_ids = state.ids[source_idx, :].clone()

        # 3) Run the selector. This might be
        #    - random
        #    - gradient-based
        #    - cosine-similarity-guided
        selector.mutate(state, source_idx, new_ids, topk, window)

        # 4) Evaluate fitness
        new_state = evaluate_fitness(model, cache_run, new_ids, batch_size=batch_size)
        all_state = state.cat(new_state)

        # note that all_loss is a matrix with a row for each population
        # member because each population member slot uses a different
        # xentropy penalty.
        all_loss = -all_state.target[None, :] + X[:, None] * all_state.xentropy[None, :]

        # keep the population members with the lowest loss
        keep = (-all_loss).argmax(dim=1).to(torch.int)

        if i % restart_frequency == 0:
            min_mult = 1.0 / restart_xentropy_max_mult
            max_mult = restart_xentropy_max_mult
            mult = min_mult + (max_mult - min_mult) * torch.rand(1).item()
            restart_X = restart_xentropy * mult
            restart_loss = -all_state.target + restart_xentropy * all_state.xentropy
            print(f"restarting with xentropy penalty of {restart_X:.2f}")
            keep[:] = restart_loss.argmin()

        history._insert(
            all_state.ids,
            all_state.target,
            all_state.xentropy,
            keep,
            time.time() - start,
        )

        # 5) Calculate gradients for the next iteration.
        if i != iters - 1:
            if selector.uses_gradient:
                if recompute_gradients:
                    survived = torch.tensor([])
                    new = keep
                else:
                    survived = keep[keep < state.ids.shape[0]]
                    new = keep[keep >= state.ids.shape[0]]
                if new.shape[0] > 0:
                    state_new = selector.setup(all_state.ids[new])
                if survived.shape[0] > 0:
                    state_survived = state.subset(survived)
                    if new.shape[0] > 0:
                        state = state_survived.cat(state_new)
                    else:
                        state = state_survived
                else:
                    state = state_new
            else:
                state = all_state.subset(keep)

    terminate_flag, output, xentropy = callback(
        i, state, time.time() - start, history, final=True
    )

    history._finalize()

    return history, output, xentropy


@contextlib.contextmanager
def add_fwd_hooks(module_hooks: List[Tuple[torch.nn.Module, Callable]]):
    """
    Context manager for temporarily adding forward hooks to a model.

    Parameters
    ----------
    module_hooks
        A list of pairs: (module, fnc) The function will be registered as a
            forward hook on the module
    """
    try:
        handles = []
        for mod, hk in module_hooks:
            handles.append(mod.register_forward_hook(hk))
        yield
    finally:
        for h in handles:
            h.remove()


def does_retokenize(model, tokenizer, input_ids):
    """
    Ensures that the input_ids are the same after tokenization and detokenization
    to ensure target alignment (doesn't seem necessary however)
    """
    good = torch.empty(input_ids.shape[0], dtype=bool).to(model.device)
    input_strs = tokenizer.batch_decode(input_ids)
    for i, s in enumerate(input_strs):
        retokenized = tokenizer.encode(s, return_tensors="pt").to(model.device)
        if retokenized.shape[1] != input_ids.shape[1]:
            good[i] = False
        else:
            good[i] = (retokenized[0] == input_ids[i]).all()
        if not good[i]:
            print(f"bad input {i}: {s}")
    return good


def steering_base_runner(
    model,
    tokenizer,
    layer,
    context_start,
    vector,
    coeff=2.0,
    check_retokenization=False,
):
    def run(input_ids=None, inputs_embeds=None):
        if input_ids is not None:
            if check_retokenization:
                good = does_retokenize(model, tokenizer, input_ids)
            else:
                good = torch.ones(input_ids.shape[0], dtype=bool).to(model.device)
        else:
            good = torch.ones(input_ids.shape[0], dtype=bool).to(model.device)

        out = {}

        if input_ids is not None:
            output = model(input_ids)
        else:
            output = model(input_embeds=inputs_embeds.clone().detach())

        def steer_layer(module, input, output):
            v = vector.to(output[0].device)
            output[:, context_start:, :] += coeff * v[None, None, :]
            return output

        hooks = [(model.model.layers[layer], steer_layer)]

        with add_fwd_hooks(hooks):
            if input_ids is not None:
                steer_output = model(input_ids)
            else:
                steer_output = model(input_embeds=inputs_embeds)

        output_probs = F.log_softmax(output.logits, dim=-1)[:, context_start:]
        output_steer_probs = F.log_softmax(steer_output.logits, dim=-1)[
            :, context_start:
        ]
        out["logits"] = steer_output.logits
        out["target"] = torch.where(
            good,
            -torch.mean(
                output_probs.exp() * (output_probs - output_steer_probs), dim=1
            ),
            -torch.finfo(output.logits.dtype).max,
        )
        return out

    return run


def steering_logit_runner(
    model,
    tokenizer,
    token_id,
    layer,
    context_start,
    vector,
    coeff=2.0,
    check_retokenization=False,
):
    def run(input_ids=None, inputs_embeds=None):
        if input_ids is not None:
            if check_retokenization:
                good = does_retokenize(model, tokenizer, input_ids)
            else:
                good = torch.ones(input_ids.shape[0], dtype=bool).to(model.device)
        else:
            good = torch.ones(inputs_embeds.shape[0], dtype=bool).to(model.device)

        def steer_layer(module, input, output):
            v = vector.to(output[0].device)
            output[:, context_start:, :] += coeff * v[None, None, :]
            return output

        hooks = [(model.model.layers[layer], steer_layer)]

        with add_fwd_hooks(hooks):
            if input_ids is not None:
                steer_output = model(input_ids)
            else:
                steer_output = model(inputs_embeds=inputs_embeds)

        out = {}
        out["logits"] = steer_output.logits
        out["target"] = torch.where(
            good,
            steer_output.logits[:, -1, token_id]
            - torch.where(
                steer_output.logits[:, -1].argmax(dim=-1) == token_id,
                steer_output.logits[:, -1].topk(dim=-1, k=2).values[:, 1],
                steer_output.logits[:, -1].max(dim=-1).values,
            ),
            -torch.finfo(steer_output.logits.dtype).max,
        )
        return out

    return run


def normalize_choices(text: str):
    pattern = r"\([A-Z]\)"

    if not re.search(pattern, text):
        return text

    text = re.sub(r"\((A)\)", r"A.", text)
    text = re.sub(r"\((B)\)", r"B.", text)
    text = re.sub(r"\((C)\)", r"C.", text)
    text = re.sub(r"\((D)\)", r"D.", text)

    return text


def load_steering_vector(behavior: str, layer_idx: int):
    behavior_files = f"Llama-3.1-70B-Instruct-{behavior}-800.json"
    files = Path("caa_sv").glob(behavior_files)
    file_list = list(files)
    steering_vector = None
    for file in file_list:
        with open(file, "r") as f:
            vector = torch.tensor(json.load(f))[layer_idx]
            if steering_vector is None:
                steering_vector = vector
            else:
                steering_vector.add_(vector)
    return steering_vector / len(file_list)


def add_special_tokens(question, initial_prompt):
    return f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>{initial_prompt}"


def get_behavior_prompts(behavior: str, initial_prompt: str, tokenizer, invert):
    if behavior == "openness":
        ds = load_dataset("json", data_files=f"{behavior}.jsonl", split="train")
    else:
        if behavior in [
            "conscientiousness",
            "extraversion",
            "agreeableness",
            "neuroticism",
            "politically-liberal",
        ]:
            ds = load_dataset(
                "Anthropic/model-written-evals",
                data_dir="persona",
                data_files=f"{behavior}.jsonl",
                split="train",
            )
        else:
            ds = load_dataset(
                "Anthropic/model-written-evals",
                data_dir="advanced-ai-risk/lm_generated_evals",
                data_files=f"{behavior}.jsonl",
                split="train",
            )

    def extract_columns(example):
        question = normalize_choices(example["question"])
        if invert:
            ans_not_match = normalize_choices(example["answer_matching_behavior"])
            ans_match = normalize_choices(example["answer_not_matching_behavior"])
        else:
            ans_match = normalize_choices(example["answer_matching_behavior"])
            ans_not_match = normalize_choices(example["answer_not_matching_behavior"])
        formatted_prompt = add_special_tokens(question, initial_prompt)
        formatted_prompt_ids = tokenizer(
            formatted_prompt, return_tensors="pt", add_special_tokens=False
        )["input_ids"]
        context_start = torch.where(formatted_prompt_ids == 128007)[1][1].item() + 1
        return {
            "initial_prompt": formatted_prompt_ids,
            "answer_matching_behavior": tokenizer(ans_match, add_special_tokens=False)[
                "input_ids"
            ],
            "answer_not_matching_behavior": tokenizer(
                ans_not_match, add_special_tokens=False
            )["input_ids"],
            "context_start": context_start,
        }

    pair_dataset = ds.map(extract_columns, remove_columns=ds.column_names, num_proc=24)
    return pair_dataset


class EPOJob(Checkpointable):
    def __init__(self, behavior, coeff, layer_idx):
        self.behavior = behavior
        self.coeff = coeff
        self.layer_idx = layer_idx
        self.invert = coeff < 0

        self.save_dir = "histories"
        self.progress_path = os.path.join(
            self.save_dir, f"{self.behavior}_prog_{self.coeff}.json"
        )

    def load_progress(self):
        if os.path.exists(self.progress_path):
            with open(self.progress_path, "r") as f:
                data = json.load(f)
            return data["idx"], data["outputs"], data["xentropies"]
        return 0, [], []

    def save_progress(self, idx, outputs, xentropies):
        os.makedirs(self.save_dir, exist_ok=True)
        with open(self.progress_path, "w") as f:
            json.dump({"idx": idx, "outputs": outputs, "xentropies": xentropies}, f)

    def __call__(self):
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            "meta-llama/Llama-3.1-8B-Instruct"
        )

        model = transformers.AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-3.1-8B-Instruct",
            torch_dtype="bfloat16",
            device_map="auto",
        )

        vector = load_steering_vector(self.behavior, self.layer_idx)

        dataset = get_behavior_prompts(
            self.behavior,
            "Upon further reflection, I have finally decided ",
            tokenizer,
            self.invert,
        )

        start_idx, outputs, xentropies = self.load_progress()

        for i in range(start_idx, 50):
            row = dataset[i]

            runner = steering_logit_runner(
                model,
                tokenizer,
                row["answer_not_matching_behavior"][0],
                self.layer_idx,
                row["context_start"],
                vector,
                coeff=self.coeff,
            )

            initial_prompt = torch.tensor(row["initial_prompt"])

            history, output, xentropy = epo(
                runner,
                model,
                tokenizer,
                seq_len=initial_prompt.shape[-1],
                population_size=16,
                iters=20,
                explore_per_pop=128,
                batch_size=64,
                topk=2048,
                mutation_method="gradient",
                x_penalty_min=8.0 / 24.0,
                x_penalty_max=24.0,
                restart_frequency=50,
                restart_xentropy=2.0,
                restart_xentropy_max_mult=3.0,
                initial_ids=initial_prompt,
                window=slice(row["context_start"], initial_prompt.shape[-1]),
                seed=0,
                history=None,
                catch_keyboard_interrupt=False,
            )

            outputs.append(output)
            xentropies.append(xentropy)

            self.save_progress(i + 1, outputs, xentropies)

        dataset = dataset.select(range(len(outputs)))
        dataset = dataset.add_column("xentropy", xentropies)
        dataset = dataset.add_column("outputs", outputs)
        dataset = dataset.remove_columns(["initial_prompt", "context_start"])

        final_path = os.path.join(
            self.save_dir, f"{self.behavior}_run_{self.coeff}.jsonl"
        )

        dataset.to_json(final_path)

        if os.path.exists(self.progress_path):
            os.remove(self.progress_path)

        return final_path

    def checkpoint(self):
        return submitit.helpers.DelayedSubmission(self)


if __name__ == "__main__":
    BEHAVIORS = [
        "openness",
        "conscientiousness",
        "extraversion",
        "agreeableness",
        "neuroticism",
        "politically-liberal",
        "power-seeking-inclination",
        "self-awareness-general-ai",
        "corrigible-neutral-HHH",
    ]
    COEFFS = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]

    jobs = [(b, c) for b in BEHAVIORS for c in COEFFS]

    executor = submitit.AutoExecutor(folder="~/scratch/logs", slurm_max_num_timeout=3)

    executor.update_parameters(
        name="epo",
        timeout_min=180,
        cpus_per_task=16,
        slurm_gres="gpu:H200:1",
        slurm_mem="64G",
    )

    with executor.batch():
        for behavior, coeff in jobs:
            job = EPOJob(
                behavior=behavior,
                coeff=coeff,
                layer_idx=66,
            )
            executor.submit(job)
