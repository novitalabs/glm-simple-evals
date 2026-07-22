import json
import os
import sys

from functools import partial

from evals.aime_eval import AimeEval
from evals.gpqa_eval import GPQAEval
from evals.math_eval import MathEval
from evals.livecodebench_eval import LiveCodeBenchEval
from evals.scicode_eval import SciCodeEval
from evals.mmlu_pro_eval import MMLUProEval
from evals.hle_eval import HLEEval

from samplers.openai_sampler import OpenAISampler
from samplers.zai_sampler import ZaiSampler

import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save_dir",
        type=str,
        required=True,
        help="The save path of the evaluation results",
    )
    parser.add_argument(
        "--zai_api_key", type=str, default="", help="The api key of the zai-sdk"
    )
    parser.add_argument(
        "--openai_api_key", type=str, default=None, help="The api key of the openai-sdk"
    )
    parser.add_argument(
        "--openai_base_url",
        type=str,
        default=None,
        help="The base url of the openai-sdk",
    )
    parser.add_argument(
        "--extra-headers",
        type=str,
        default=None,
        help="Extra HTTP headers as a JSON object.",
    )
    parser.add_argument(
        "--checker_model_name",
        type=str,
        help="The name of the checker model",
    )
    parser.add_argument(
        "--checker_api_key",
        type=str,
        default=None,
        help="The api key of the checker model",
    )
    parser.add_argument(
        "--checker_url", type=str, default="", help="The url of the checker model"
    )
    parser.add_argument(
        "--data_dir", type=str, default="data", help="The data path of the evaluation"
    )
    parser.add_argument(
        "--proc_num",
        type=int,
        default=60,
        help="The number of processes to run the evaluation",
    )
    parser.add_argument(
        "--backbone", type=str, default="zai", help="The backbone of the evaluation"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="glm-4.5",
        help="The name of the model to evaluate",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        nargs="*",
        default=["aime2024"],
        help="The tasks to evaluate",
    )
    parser.add_argument(
        "--auto_extract_answer",
        action="store_true",
        default=False,
        help="Whether to automatically extract the answer from the model's response",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=2048,
        help="The max new tokens of the evaluation",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=131072,
        help="The max length of the evaluation",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.6,
        help="The temperature of the evaluation",
    )

    parser.add_argument(
        "--top_p", type=float, default=1.0, help="The top p of the evaluation"
    )
    parser.add_argument(
        "--top_k", type=int, default=-1, help="The top k of the evaluation"
    )
    parser.add_argument(
        "--worst_of_n",
        action="store_true",
        default=False,
        help="Whether to use the worst of n evaluation",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Whether to run the evaluation in debug mode",
    )
    parser.add_argument(
        "--lcb_date", type=str, default="latest", help="The date of the lcb evaluation"
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        default=False,
        help="Whether to use stream output mode",
    )
    parser.add_argument(
        "--sci_without_background",
        action="store_true",
        default=False,
        help="Whether to run the sci-code evaluation without background",
    )

    args = parser.parse_args()
    debug = args.debug

    extra_headers = {}
    if args.extra_headers:
        try:
            extra_headers = json.loads(args.extra_headers)
        except json.JSONDecodeError as e:
            print(f"Error: failed to parse --extra-headers JSON: {e}", file=sys.stderr)
            sys.exit(1)
        if not isinstance(extra_headers, dict) or not all(
            isinstance(k, str) and isinstance(v, str)
            for k, v in extra_headers.items()
        ):
            print(
                "Error: --extra-headers must be a JSON object with string keys and string values",
                file=sys.stderr,
            )
            sys.exit(1)

    os.makedirs(args.save_dir, exist_ok=True)

    extractor = None
    if not args.checker_model_name or not args.checker_url or not args.checker_api_key:
        raise ValueError(
            "--checker_model_name, --checker_url and --checker_api_key are all required"
        )
    equality_checker = OpenAISampler(
        api_key=args.checker_api_key,
        url=args.checker_url,
        model=args.checker_model_name,
        max_tokens=args.max_length,
        temperature=args.temperature,
        top_p=args.top_p,
        stream=args.stream,
        extra_headers=extra_headers,
    )

    if args.backbone == "zai":
        sampler = ZaiSampler(
            model=args.model_name,
            api_key=args.zai_api_key,
            max_tokens=args.max_new_tokens,
            stream=args.stream,
            temperature=args.temperature,
            top_p=args.top_p,
        )
    elif args.backbone == "openai":
        sampler = OpenAISampler(
            url=args.openai_base_url,
            api_key=args.openai_api_key,
            model=args.model_name,
            max_tokens=args.max_new_tokens,
            stream=args.stream,
            temperature=args.temperature,
            top_p=args.top_p,
            extra_headers=extra_headers,
        )
    else:
        raise ValueError(f"Unknown backbone {args.backbone}")

    # default num_examples:
    # lcb: 322, scicode: 80, gpqa: 198, aime24: 30, aime25: 30, math500: 500, mmlu_pro: 12032, hle: 2158
    eval_dict = {
        "lcb": partial(
            LiveCodeBenchEval,
            num_examples=1 if debug else 160,
            data_dir=args.data_dir,
            proc_num=args.proc_num,
            num_repeat=1 if debug else 2,
            date=args.lcb_date,
        ),
        "scicode": partial(
            SciCodeEval,
            num_examples=5 if debug else -1,
            data_dir=args.data_dir,
            proc_num=args.proc_num,
            num_repeat=2,
            save_dir=args.save_dir,
            with_background=not args.sci_without_background,
        ),
        "gpqa": partial(
            GPQAEval,
            equality_checker=equality_checker,
            n_repeats=1 if debug else 2,
            num_examples=5 if debug else None,
            data_dir=args.data_dir,
            proc_num=args.proc_num,
            auto_extract_answer=args.auto_extract_answer,
        ),
        "aime2024": partial(
            AimeEval,
            equality_checker=equality_checker,
            num_examples=3 if debug else -1,
            year=2024,
            data_dir=args.data_dir,
            proc_num=args.proc_num,
            n_repeats=1 if debug else 8,
            auto_extract_answer=args.auto_extract_answer,
            worst_of_n=args.worst_of_n,
            extractor=extractor,
        ),
        "aime2025": partial(
            AimeEval,
            equality_checker=equality_checker,
            num_examples=3 if debug else -1,
            year=2025,
            data_dir=args.data_dir,
            proc_num=args.proc_num,
            n_repeats=1 if debug else 8,
            auto_extract_answer=args.auto_extract_answer,
            worst_of_n=args.worst_of_n,
            extractor=extractor,
        ),
        "math500": partial(
            MathEval,
            equality_checker=equality_checker,
            num_examples=5 if debug else -1,
            data_dir=args.data_dir,
            proc_num=args.proc_num,
            n_repeats=2,
            extractor=extractor,
        ),
        "mmlu_pro": partial(
            MMLUProEval,
            num_examples=5 if debug else 2000,
            data_dir=args.data_dir,
            proc_num=args.proc_num,
        ),
        "hle": partial(
            HLEEval,
            equality_checker=equality_checker,
            num_examples=5 if debug else 1000,
            data_dir=args.data_dir,
            proc_num=args.proc_num,
        ),
    }

    eval_tasks = args.tasks
    if args.tasks[0] == "all":
        eval_tasks = list(eval_dict.keys())
    print(f"### eval_tasks: {eval_tasks}")

    os.makedirs(os.path.join(args.save_dir, "simple_evals"), exist_ok=True)

    for eval_name, eval_obj in eval_dict.items():
        if eval_name not in eval_tasks:
            continue
        print("Evaluating " + eval_name)

        result = eval_obj()(sampler)
        if isinstance(result, tuple):
            assert len(result) == 2
            if len(result) == 2:
                result, response_data = result
        else:
            response_data = None

        metrics = result.metrics | {"score": result.score}
        result_filename = os.path.join(
            args.save_dir, "simple_evals", f"{eval_name}.json"
        )
        print("metrics: ", metrics)
        with open(result_filename, "w") as f:
            json.dump(metrics, f, indent=4, ensure_ascii=False)

        if response_data:
            os.makedirs(
                os.path.join(args.save_dir, "simple_evals", eval_name), exist_ok=True
            )
            with open(
                os.path.join(args.save_dir, "simple_evals", eval_name, f"data.jsonl"),
                "w",
            ) as f:
                f.writelines(
                    [json.dumps(x, ensure_ascii=False) + "\n" for x in response_data]
                )
