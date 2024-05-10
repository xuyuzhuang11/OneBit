# Copyright 2020 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import ast
import collections
import functools
import json
import operator
import os
import re
import sys
import time
from typing import Dict, List, Optional, Union

import requests
from get_ci_error_statistics import get_job_links
from get_previous_daily_ci import get_last_daily_ci_reports
from slack_sdk import WebClient


client = WebClient(token=os.environ["CI_SLACK_BOT_TOKEN"])

NON_MODEL_TEST_MODULES = [
    "benchmark",
    "deepspeed",
    "extended",
    "fixtures",
    "generation",
    "onnx",
    "optimization",
    "pipelines",
    "sagemaker",
    "trainer",
    "utils",
]


def handle_test_results(test_results):
    expressions = test_results.split(" ")

    failed = 0
    success = 0

    # When the output is short enough, the output is surrounded by = signs: "== OUTPUT =="
    # When it is too long, those signs are not present.
    time_spent = expressions[-2] if "=" in expressions[-1] else expressions[-1]

    for i, expression in enumerate(expressions):
        if "failed" in expression:
            failed += int(expressions[i - 1])
        if "passed" in expression:
            success += int(expressions[i - 1])

    return failed, success, time_spent


def handle_stacktraces(test_results):
    # These files should follow the following architecture:
    # === FAILURES ===
    # <path>:<line>: Error ...
    # <path>:<line>: Error ...
    # <empty line>

    total_stacktraces = test_results.split("\n")[1:-1]
    stacktraces = []
    for stacktrace in total_stacktraces:
        try:
            line = stacktrace[: stacktrace.index(" ")].split(":")[-2]
            error_message = stacktrace[stacktrace.index(" ") :]

            stacktraces.append(f"(line {line}) {error_message}")
        except Exception:
            stacktraces.append("Cannot retrieve error message.")

    return stacktraces


def dicts_to_sum(objects: Union[Dict[str, Dict], List[dict]]):
    if isinstance(objects, dict):
        lists = objects.values()
    else:
        lists = objects

    # Convert each dictionary to counter
    counters = map(collections.Counter, lists)
    # Sum all the counters
    return functools.reduce(operator.add, counters)


class Message:
    def __init__(
        self, title: str, ci_title: str, model_results: Dict, additional_results: Dict, selected_warnings: List = None
    ):
        self.title = title
        self.ci_title = ci_title

        # Failures and success of the modeling tests
        self.n_model_success = sum(r["success"] for r in model_results.values())
        self.n_model_single_gpu_failures = sum(dicts_to_sum(r["failed"])["single"] for r in model_results.values())
        self.n_model_multi_gpu_failures = sum(dicts_to_sum(r["failed"])["multi"] for r in model_results.values())

        # Some suites do not have a distinction between single and multi GPU.
        self.n_model_unknown_failures = sum(dicts_to_sum(r["failed"])["unclassified"] for r in model_results.values())
        self.n_model_failures = (
            self.n_model_single_gpu_failures + self.n_model_multi_gpu_failures + self.n_model_unknown_failures
        )

        # Failures and success of the additional tests
        self.n_additional_success = sum(r["success"] for r in additional_results.values())

        all_additional_failures = dicts_to_sum([r["failed"] for r in additional_results.values()])
        self.n_additional_single_gpu_failures = all_additional_failures["single"]
        self.n_additional_multi_gpu_failures = all_additional_failures["multi"]
        self.n_additional_unknown_gpu_failures = all_additional_failures["unclassified"]
        self.n_additional_failures = (
            self.n_additional_single_gpu_failures
            + self.n_additional_multi_gpu_failures
            + self.n_additional_unknown_gpu_failures
        )

        # Results
        self.n_failures = self.n_model_failures + self.n_additional_failures
        self.n_success = self.n_model_success + self.n_additional_success
        self.n_tests = self.n_failures + self.n_success

        self.model_results = model_results
        self.additional_results = additional_results

        self.thread_ts = None

        if selected_warnings is None:
            selected_warnings = []
        self.selected_warnings = selected_warnings

    @property
    def time(self) -> str:
        all_results = [*self.model_results.values(), *self.additional_results.values()]
        time_spent = [r["time_spent"].split(", ")[0] for r in all_results if len(r["time_spent"])]
        total_secs = 0

        for time in time_spent:
            time_parts = time.split(":")

            # Time can be formatted as xx:xx:xx, as .xx, or as x.xx if the time spent was less than a minute.
            if len(time_parts) == 1:
                time_parts = [0, 0, time_parts[0]]

            hours, minutes, seconds = int(time_parts[0]), int(time_parts[1]), float(time_parts[2])
            total_secs += hours * 3600 + minutes * 60 + seconds

        hours, minutes, seconds = total_secs // 3600, (total_secs % 3600) // 60, total_secs % 60
        return f"{int(hours)}h{int(minutes)}m{int(seconds)}s"

    @property
    def header(self) -> Dict:
        return {"type": "header", "text": {"type": "plain_text", "text": self.title}}

    @property
    def ci_title_section(self) -> Dict:
        return {"type": "section", "text": {"type": "mrkdwn", "text": self.ci_title}}

    @property
    def no_failures(self) -> Dict:
        return {
            "type": "section",
            "text": {
                "type": "plain_text",
                "text": f"🌞 There were no failures: all {self.n_tests} tests passed. The suite ran in {self.time}.",
                "emoji": True,
            },
            "accessory": {
                "type": "button",
                "text": {"type": "plain_text", "text": "Check Action results", "emoji": True},
                "url": f"https://github.com/huggingface/transformers/actions/runs/{os.environ['GITHUB_RUN_ID']}",
            },
        }

    @property
    def failures(self) -> Dict:
        return {
            "type": "section",
            "text": {
                "type": "plain_text",
                "text": (
                    f"There were {self.n_failures} failures, out of {self.n_tests} tests.\n"
                    f"Number of model failures: {self.n_model_failures}.\n"
                    f"The suite ran in {self.time}."
                ),
                "emoji": True,
            },
            "accessory": {
                "type": "button",
                "text": {"type": "plain_text", "text": "Check Action results", "emoji": True},
                "url": f"https://github.com/huggingface/transformers/actions/runs/{os.environ['GITHUB_RUN_ID']}",
            },
        }

    @property
    def warnings(self) -> Dict:
        # If something goes wrong, let's avoid the CI report failing to be sent.
        button_text = "Check warnings (Link not found)"
        # Use the workflow run link
        job_link = f"https://github.com/huggingface/transformers/actions/runs/{os.environ['GITHUB_RUN_ID']}"
        if "Extract warnings in CI artifacts" in github_actions_job_links:
            button_text = "Check warnings"
            # Use the actual job link
            job_link = f"{github_actions_job_links['Extract warnings in CI artifacts']}"

        huggingface_hub_warnings = [x for x in self.selected_warnings if "huggingface_hub" in x]
        text = f"There are {len(self.selected_warnings)} warnings being selected."
        text += f"\n{len(huggingface_hub_warnings)} of them are from `huggingface_hub`."

        return {
            "type": "section",
            "text": {
                "type": "plain_text",
                "text": text,
                "emoji": True,
            },
            "accessory": {
                "type": "button",
                "text": {"type": "plain_text", "text": button_text, "emoji": True},
                "url": job_link,
            },
        }

    @staticmethod
    def get_device_report(report, rjust=6):
        if "single" in report and "multi" in report:
            return f"{str(report['single']).rjust(rjust)} | {str(report['multi']).rjust(rjust)} | "
        elif "single" in report:
            return f"{str(report['single']).rjust(rjust)} | {'0'.rjust(rjust)} | "
        elif "multi" in report:
            return f"{'0'.rjust(rjust)} | {str(report['multi']).rjust(rjust)} | "

    @property
    def category_failures(self) -> Dict:
        model_failures = [v["failed"] for v in self.model_results.values()]

        category_failures = {}

        for model_failure in model_failures:
            for key, value in model_failure.items():
                if key not in category_failures:
                    category_failures[key] = dict(value)
                else:
                    category_failures[key]["unclassified"] += value["unclassified"]
                    category_failures[key]["single"] += value["single"]
                    category_failures[key]["multi"] += value["multi"]

        individual_reports = []
        for key, value in category_failures.items():
            device_report = self.get_device_report(value)

            if sum(value.values()):
                if device_report:
                    individual_reports.append(f"{device_report}{key}")
                else:
                    individual_reports.append(key)

        header = "Single |  Multi | Category\n"
        category_failures_report = prepare_reports(
            title="The following modeling categories had failures", header=header, reports=individual_reports
        )

        return {"type": "section", "text": {"type": "mrkdwn", "text": category_failures_report}}

    def compute_diff_for_failure_reports(self, curr_failure_report, prev_failure_report):  # noqa
        # Remove the leading and training parts that don't contain failure count information.
        model_failures = curr_failure_report.split("\n")[3:-2]
        prev_model_failures = prev_failure_report.split("\n")[3:-2]
        entries_changed = set(model_failures).difference(prev_model_failures)

        prev_map = {}
        for f in prev_model_failures:
            items = [x.strip() for x in f.split("| ")]
            prev_map[items[-1]] = [int(x) for x in items[:-1]]

        curr_map = {}
        for f in entries_changed:
            items = [x.strip() for x in f.split("| ")]
            curr_map[items[-1]] = [int(x) for x in items[:-1]]

        diff_map = {}
        for k, v in curr_map.items():
            if k not in prev_map:
                diff_map[k] = v
            else:
                diff = [x - y for x, y in zip(v, prev_map[k])]
                if max(diff) > 0:
                    diff_map[k] = diff

        entries_changed = []
        for model_name, diff_values in diff_map.items():
            diff = [str(x) for x in diff_values]
            diff = [f"+{x}" if (x != "0" and not x.startswith("-")) else x for x in diff]
            diff = [x.rjust(9) for x in diff]
            device_report = " | ".join(diff) + " | "
            report = f"{device_report}{model_name}"
            entries_changed.append(report)
        entries_changed = sorted(entries_changed, key=lambda s: s.split("| ")[-1])

        return entries_changed

    @property
    def model_failures(self) -> List[Dict]:
        # Obtain per-model failures
        def per_model_sum(model_category_dict):
            return dicts_to_sum(model_category_dict["failed"].values())

        failures = {}
        non_model_failures = {
            k: per_model_sum(v) for k, v in self.model_results.items() if sum(per_model_sum(v).values())
        }

        for k, v in self.model_results.items():
            if k in NON_MODEL_TEST_MODULES:
                pass

            if sum(per_model_sum(v).values()):
                dict_failed = dict(v["failed"])
                pytorch_specific_failures = dict_failed.pop("PyTorch")
                tensorflow_specific_failures = dict_failed.pop("TensorFlow")
                other_failures = dicts_to_sum(dict_failed.values())

                failures[k] = {
                    "PyTorch": pytorch_specific_failures,
                    "TensorFlow": tensorflow_specific_failures,
                    "other": other_failures,
                }

        model_reports = []
        other_module_reports = []

        for key, value in non_model_failures.items():
            if key in NON_MODEL_TEST_MODULES:
                device_report = self.get_device_report(value)

                if sum(value.values()):
                    if device_report:
                        report = f"{device_report}{key}"
                    else:
                        report = key

                    other_module_reports.append(report)

        for key, value in failures.items():
            device_report_values = [
                value["PyTorch"]["single"],
                value["PyTorch"]["multi"],
                value["TensorFlow"]["single"],
                value["TensorFlow"]["multi"],
                sum(value["other"].values()),
            ]

            if sum(device_report_values):
                device_report = " | ".join([str(x).rjust(9) for x in device_report_values]) + " | "
                report = f"{device_report}{key}"

                model_reports.append(report)

        # (Possibly truncated) reports for the current workflow run - to be sent to Slack channels
        model_header = "Single PT |  Multi PT | Single TF |  Multi TF |     Other | Category\n"
        sorted_model_reports = sorted(model_reports, key=lambda s: s.split("| ")[-1])
        model_failures_report = prepare_reports(
            title="These following model modules had failures", header=model_header, reports=sorted_model_reports
        )

        module_header = "Single |  Multi | Category\n"
        sorted_module_reports = sorted(other_module_reports, key=lambda s: s.split("| ")[-1])
        module_failures_report = prepare_reports(
            title="The following non-model modules had failures", header=module_header, reports=sorted_module_reports
        )

        # To be sent to Slack channels
        model_failure_sections = [
            {"type": "section", "text": {"type": "mrkdwn", "text": model_failures_report}},
            {"type": "section", "text": {"type": "mrkdwn", "text": module_failures_report}},
        ]

        # Save the complete (i.e. no truncation) failure tables (of the current workflow run)
        # (to be uploaded as artifacts)
        if not os.path.isdir(os.path.join(os.getcwd(), "test_failure_tables")):
            os.makedirs(os.path.join(os.getcwd(), "test_failure_tables"))

        model_failures_report = prepare_reports(
            title="These following model modules had failures",
            header=model_header,
            reports=sorted_model_reports,
            to_truncate=False,
        )
        file_path = os.path.join(os.getcwd(), "test_failure_tables/model_failures_report.txt")
        with open(file_path, "w", encoding="UTF-8") as fp:
            fp.write(model_failures_report)

        module_failures_report = prepare_reports(
            title="The following non-model modules had failures",
            header=module_header,
            reports=sorted_module_reports,
            to_truncate=False,
        )
        file_path = os.path.join(os.getcwd(), "test_failure_tables/module_failures_report.txt")
        with open(file_path, "w", encoding="UTF-8") as fp:
            fp.write(module_failures_report)

        target_workflow = "huggingface/transformers/.github/workflows/self-scheduled.yml@refs/heads/main"
        if os.environ.get("CI_WORKFLOW_REF") == target_workflow:
            # Get the last previously completed CI's failure tables
            artifact_names = ["test_failure_tables"]
            output_dir = os.path.join(os.getcwd(), "previous_reports")
            os.makedirs(output_dir, exist_ok=True)
            prev_tables = get_last_daily_ci_reports(
                artifact_names=artifact_names, output_dir=output_dir, token=os.environ["ACCESS_REPO_INFO_TOKEN"]
            )

            # if the last run produces artifact named `test_failure_tables`
            if (
                "test_failure_tables" in prev_tables
                and "model_failures_report.txt" in prev_tables["test_failure_tables"]
            ):
                # Compute the difference of the previous/current (model failure) table
                prev_model_failures = prev_tables["test_failure_tables"]["model_failures_report.txt"]
                entries_changed = self.compute_diff_for_failure_reports(model_failures_report, prev_model_failures)
                if len(entries_changed) > 0:
                    # Save the complete difference
                    diff_report = prepare_reports(
                        title="Changed model modules failures",
                        header=model_header,
                        reports=entries_changed,
                        to_truncate=False,
                    )
                    file_path = os.path.join(os.getcwd(), "test_failure_tables/changed_model_failures_report.txt")
                    with open(file_path, "w", encoding="UTF-8") as fp:
                        fp.write(diff_report)

                    # To be sent to Slack channels
                    diff_report = prepare_reports(
                        title="*Changed model modules failures*",
                        header=model_header,
                        reports=entries_changed,
                    )
                    model_failure_sections.append(
                        {"type": "section", "text": {"type": "mrkdwn", "text": diff_report}},
                    )

        return model_failure_sections

    @property
    def additional_failures(self) -> Dict:
        failures = {k: v["failed"] for k, v in self.additional_results.items()}
        errors = {k: v["error"] for k, v in self.additional_results.items()}

        individual_reports = []
        for key, value in failures.items():
            device_report = self.get_device_report(value)

            if sum(value.values()) or errors[key]:
                report = f"{key}"
                if errors[key]:
                    report = f"[Errored out] {report}"
                if device_report:
                    report = f"{device_report}{report}"

                individual_reports.append(report)

        header = "Single |  Multi | Category\n"
        failures_report = prepare_reports(
            title="The following non-modeling tests had failures", header=header, reports=individual_reports
        )

        return {"type": "section", "text": {"type": "mrkdwn", "text": failures_report}}

    @property
    def payload(self) -> str:
        blocks = [self.header]

        if self.ci_title:
            blocks.append(self.ci_title_section)

        if self.n_model_failures > 0 or self.n_additional_failures > 0:
            blocks.append(self.failures)

        if self.n_model_failures > 0:
            blocks.append(self.category_failures)
            for block in self.model_failures:
                if block["text"]["text"]:
                    blocks.append(block)

        if self.n_additional_failures > 0:
            blocks.append(self.additional_failures)

        if self.n_model_failures == 0 and self.n_additional_failures == 0:
            blocks.append(self.no_failures)

        if len(self.selected_warnings) > 0:
            blocks.append(self.warnings)

        return json.dumps(blocks)

    @staticmethod
    def error_out(title, ci_title="", runner_not_available=False, runner_failed=False, setup_failed=False):
        blocks = []
        title_block = {"type": "header", "text": {"type": "plain_text", "text": title}}
        blocks.append(title_block)

        if ci_title:
            ci_title_block = {"type": "section", "text": {"type": "mrkdwn", "text": ci_title}}
            blocks.append(ci_title_block)

        offline_runners = []
        if runner_not_available:
            text = "💔 CI runners are not available! Tests are not run. 😭"
            result = os.environ.get("OFFLINE_RUNNERS")
            if result is not None:
                offline_runners = json.loads(result)
        elif runner_failed:
            text = "💔 CI runners have problems! Tests are not run. 😭"
        elif setup_failed:
            text = "💔 Setup job failed. Tests are not run. 😭"
        else:
            text = "💔 There was an issue running the tests. 😭"

        error_block_1 = {
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": text,
            },
        }

        text = ""
        if len(offline_runners) > 0:
            text = "\n  • " + "\n  • ".join(offline_runners)
            text = f"The following runners are offline:\n{text}\n\n"
        text += "🙏 Let's fix it ASAP! 🙏"

        error_block_2 = {
            "type": "section",
            "text": {
                "type": "plain_text",
                "text": text,
            },
            "accessory": {
                "type": "button",
                "text": {"type": "plain_text", "text": "Check Action results", "emoji": True},
                "url": f"https://github.com/huggingface/transformers/actions/runs/{os.environ['GITHUB_RUN_ID']}",
            },
        }
        blocks.extend([error_block_1, error_block_2])

        payload = json.dumps(blocks)

        print("Sending the following payload")
        print(json.dumps({"blocks": blocks}))

        client.chat_postMessage(
            channel=os.environ["CI_SLACK_REPORT_CHANNEL_ID"],
            text=text,
            blocks=payload,
        )

    def post(self):
        payload = self.payload
        print("Sending the following payload")
        print(json.dumps({"blocks": json.loads(payload)}))

        text = f"{self.n_failures} failures out of {self.n_tests} tests," if self.n_failures else "All tests passed."

        self.thread_ts = client.chat_postMessage(
            channel=os.environ["CI_SLACK_REPORT_CHANNEL_ID"],
            blocks=payload,
            text=text,
        )

    def get_reply_blocks(self, job_name, job_result, failures, device, text):
        """
        failures: A list with elements of the form {"line": full test name, "trace": error trace}
        """
        # `text` must be less than 3001 characters in Slack SDK
        # keep some room for adding "[Truncated]" when necessary
        MAX_ERROR_TEXT = 3000 - len("[Truncated]")

        failure_text = ""
        for idx, error in enumerate(failures):
            new_text = failure_text + f'*{error["line"]}*\n_{error["trace"]}_\n\n'
            if len(new_text) > MAX_ERROR_TEXT:
                # `failure_text` here has length <= 3000
                failure_text = failure_text + "[Truncated]"
                break
            # `failure_text` here has length <= MAX_ERROR_TEXT
            failure_text = new_text

        title = job_name
        if device is not None:
            title += f" ({device}-gpu)"

        content = {"type": "section", "text": {"type": "mrkdwn", "text": text}}

        # TODO: Make sure we always have a valid job link (or at least a way not to break the report sending)
        # Currently we get the device from a job's artifact name.
        # If a device is found, the job name should contain the device type, for example, `XXX (single-gpu)`.
        # This could be done by adding `machine_type` in a job's `strategy`.
        # (If `job_result["job_link"][device]` is `None`, we get an error: `... [ERROR] must provide a string ...`)
        if job_result["job_link"] is not None and job_result["job_link"][device] is not None:
            content["accessory"] = {
                "type": "button",
                "text": {"type": "plain_text", "text": "GitHub Action job", "emoji": True},
                "url": job_result["job_link"][device],
            }

        return [
            {"type": "header", "text": {"type": "plain_text", "text": title.upper(), "emoji": True}},
            content,
            {"type": "section", "text": {"type": "mrkdwn", "text": failure_text}},
        ]

    def post_reply(self):
        if self.thread_ts is None:
            raise ValueError("Can only post reply if a post has been made.")

        sorted_dict = sorted(self.model_results.items(), key=lambda t: t[0])
        for job, job_result in sorted_dict:
            if len(job_result["failures"]):
                for device, failures in job_result["failures"].items():
                    text = "\n".join(
                        sorted([f"*{k}*: {v[device]}" for k, v in job_result["failed"].items() if v[device]])
                    )

                    blocks = self.get_reply_blocks(job, job_result, failures, device, text=text)

                    print("Sending the following reply")
                    print(json.dumps({"blocks": blocks}))

                    client.chat_postMessage(
                        channel=os.environ["CI_SLACK_REPORT_CHANNEL_ID"],
                        text=f"Results for {job}",
                        blocks=blocks,
                        thread_ts=self.thread_ts["ts"],
                    )

                    time.sleep(1)

        for job, job_result in self.additional_results.items():
            if len(job_result["failures"]):
                for device, failures in job_result["failures"].items():
                    blocks = self.get_reply_blocks(
                        job,
                        job_result,
                        failures,
                        device,
                        text=f'Number of failures: {job_result["failed"][device]}',
                    )

                    print("Sending the following reply")
                    print(json.dumps({"blocks": blocks}))

                    client.chat_postMessage(
                        channel=os.environ["CI_SLACK_REPORT_CHANNEL_ID"],
                        text=f"Results for {job}",
                        blocks=blocks,
                        thread_ts=self.thread_ts["ts"],
                    )

                    time.sleep(1)


def retrieve_artifact(artifact_path: str, gpu: Optional[str]):
    if gpu not in [None, "single", "multi"]:
        raise ValueError(f"Invalid GPU for artifact. Passed GPU: `{gpu}`.")

    _artifact = {}

    if os.path.exists(artifact_path):
        files = os.listdir(artifact_path)
        for file in files:
            try:
                with open(os.path.join(artifact_path, file)) as f:
                    _artifact[file.split(".")[0]] = f.read()
            except UnicodeDecodeError as e:
                raise ValueError(f"Could not open {os.path.join(artifact_path, file)}.") from e

    return _artifact


def retrieve_available_artifacts():
    class Artifact:
        def __init__(self, name: str, single_gpu: bool = False, multi_gpu: bool = False):
            self.name = name
            self.single_gpu = single_gpu
            self.multi_gpu = multi_gpu
            self.paths = []

        def __str__(self):
            return self.name

        def add_path(self, path: str, gpu: str = None):
            self.paths.append({"name": self.name, "path": path, "gpu": gpu})

    _available_artifacts: Dict[str, Artifact] = {}

    directories = filter(os.path.isdir, os.listdir())
    for directory in directories:
        artifact_name = directory

        name_parts = artifact_name.split("_postfix_")
        if len(name_parts) > 1:
            artifact_name = name_parts[0]

        if artifact_name.startswith("single-gpu"):
            artifact_name = artifact_name[len("single-gpu") + 1 :]

            if artifact_name in _available_artifacts:
                _available_artifacts[artifact_name].single_gpu = True
            else:
                _available_artifacts[artifact_name] = Artifact(artifact_name, single_gpu=True)

            _available_artifacts[artifact_name].add_path(directory, gpu="single")

        elif artifact_name.startswith("multi-gpu"):
            artifact_name = artifact_name[len("multi-gpu") + 1 :]

            if artifact_name in _available_artifacts:
                _available_artifacts[artifact_name].multi_gpu = True
            else:
                _available_artifacts[artifact_name] = Artifact(artifact_name, multi_gpu=True)

            _available_artifacts[artifact_name].add_path(directory, gpu="multi")
        else:
            if artifact_name not in _available_artifacts:
                _available_artifacts[artifact_name] = Artifact(artifact_name)

            _available_artifacts[artifact_name].add_path(directory)

    return _available_artifacts


def prepare_reports(title, header, reports, to_truncate=True):
    report = ""

    MAX_ERROR_TEXT = 3000 - len("[Truncated]")
    if not to_truncate:
        MAX_ERROR_TEXT = float("inf")

    if len(reports) > 0:
        # `text` must be less than 3001 characters in Slack SDK
        # keep some room for adding "[Truncated]" when necessary

        for idx in range(len(reports)):
            _report = header + "\n".join(reports[: idx + 1])
            new_report = f"{title}:\n```\n{_report}\n```\n"
            if len(new_report) > MAX_ERROR_TEXT:
                # `report` here has length <= 3000
                report = report + "[Truncated]"
                break
            report = new_report

    return report


if __name__ == "__main__":
    runner_status = os.environ.get("RUNNER_STATUS")
    runner_env_status = os.environ.get("RUNNER_ENV_STATUS")
    setup_status = os.environ.get("SETUP_STATUS")

    runner_not_available = True if runner_status is not None and runner_status != "success" else False
    runner_failed = True if runner_env_status is not None and runner_env_status != "success" else False
    setup_failed = True if setup_status is not None and setup_status != "success" else False

    org = "huggingface"
    repo = "transformers"
    repository_full_name = f"{org}/{repo}"

    # This env. variable is set in workflow file (under the job `send_results`).
    ci_event = os.environ["CI_EVENT"]

    # To find the PR number in a commit title, for example, `Add AwesomeFormer model (#99999)`
    pr_number_re = re.compile(r"\(#(\d+)\)$")

    title = f"🤗 Results of the {ci_event} tests."
    # Add Commit/PR title with a link for push CI
    # (check the title in 2 env. variables - depending on the CI is triggered via `push` or `workflow_run` event)
    ci_title_push = os.environ.get("CI_TITLE_PUSH")
    ci_title_workflow_run = os.environ.get("CI_TITLE_WORKFLOW_RUN")
    ci_title = ci_title_push if ci_title_push else ci_title_workflow_run

    ci_sha = os.environ.get("CI_SHA")

    ci_url = None
    if ci_sha:
        ci_url = f"https://github.com/{repository_full_name}/commit/{ci_sha}"

    if ci_title is not None:
        if ci_url is None:
            raise ValueError(
                "When a title is found (`ci_title`), it means a `push` event or a `workflow_run` even (triggered by "
                "another `push` event), and the commit SHA has to be provided in order to create the URL to the "
                "commit page."
            )
        ci_title = ci_title.strip().split("\n")[0].strip()

        # Retrieve the PR title and author login to complete the report
        commit_number = ci_url.split("/")[-1]
        ci_detail_url = f"https://api.github.com/repos/{repository_full_name}/commits/{commit_number}"
        ci_details = requests.get(ci_detail_url).json()
        ci_author = ci_details["author"]["login"]

        merged_by = None
        # Find the PR number (if any) and change the url to the actual PR page.
        numbers = pr_number_re.findall(ci_title)
        if len(numbers) > 0:
            pr_number = numbers[0]
            ci_detail_url = f"https://api.github.com/repos/{repository_full_name}/pulls/{pr_number}"
            ci_details = requests.get(ci_detail_url).json()

            ci_author = ci_details["user"]["login"]
            ci_url = f"https://github.com/{repository_full_name}/pull/{pr_number}"

            merged_by = ci_details["merged_by"]["login"]

        if merged_by is None:
            ci_title = f"<{ci_url}|{ci_title}>\nAuthor: {ci_author}"
        else:
            ci_title = f"<{ci_url}|{ci_title}>\nAuthor: {ci_author} | Merged by: {merged_by}"

    elif ci_sha:
        ci_title = f"<{ci_url}|commit: {ci_sha}>"

    else:
        ci_title = ""

    if runner_not_available or runner_failed or setup_failed:
        Message.error_out(title, ci_title, runner_not_available, runner_failed, setup_failed)
        exit(0)

    arguments = sys.argv[1:][0]
    try:
        models = ast.literal_eval(arguments)
        # Need to change from elements like `models/bert` to `models_bert` (the ones used as artifact names).
        models = [x.replace("models/", "models_") for x in models]
    except SyntaxError:
        Message.error_out(title, ci_title)
        raise ValueError("Errored out.")

    github_actions_job_links = get_job_links(
        workflow_run_id=os.environ["GITHUB_RUN_ID"], token=os.environ["ACCESS_REPO_INFO_TOKEN"]
    )
    available_artifacts = retrieve_available_artifacts()

    modeling_categories = [
        "PyTorch",
        "TensorFlow",
        "Flax",
        "Tokenizers",
        "Pipelines",
        "Trainer",
        "ONNX",
        "Auto",
        "Unclassified",
    ]

    # This dict will contain all the information relative to each model:
    # - Failures: the total, as well as the number of failures per-category defined above
    # - Success: total
    # - Time spent: as a comma-separated list of elapsed time
    # - Failures: as a line-break separated list of errors
    model_results = {
        model: {
            "failed": {m: {"unclassified": 0, "single": 0, "multi": 0} for m in modeling_categories},
            "success": 0,
            "time_spent": "",
            "failures": {},
            "job_link": {},
        }
        for model in models
        if f"run_all_tests_gpu_{model}_test_reports" in available_artifacts
    }

    unclassified_model_failures = []

    # This prefix is used to get job links below. For past CI, we use `workflow_call`, which changes the job names from
    # `Model tests (...)` to `PyTorch 1.5 / Model tests (...)` for example.
    job_name_prefix = ""
    if ci_event.startswith("Past CI - "):
        framework, version = ci_event.replace("Past CI - ", "").split("-")
        framework = "PyTorch" if framework == "pytorch" else "TensorFlow"
        job_name_prefix = f"{framework} {version}"
    elif ci_event.startswith("Nightly CI"):
        job_name_prefix = "Nightly CI"
    elif ci_event.startswith("Push CI (AMD) - "):
        flavor = ci_event.replace("Push CI (AMD) - ", "")
        job_name_prefix = f"AMD {flavor}"

    for model in model_results.keys():
        for artifact_path in available_artifacts[f"run_all_tests_gpu_{model}_test_reports"].paths:
            artifact = retrieve_artifact(artifact_path["path"], artifact_path["gpu"])
            if "stats" in artifact:
                # Link to the GitHub Action job
                # The job names use `matrix.folder` which contain things like `models/bert` instead of `models_bert`
                job_name = f"Model tests ({model.replace('models_', 'models/')}, {artifact_path['gpu']}-gpu)"
                if job_name_prefix:
                    job_name = f"{job_name_prefix} / {job_name}"
                model_results[model]["job_link"][artifact_path["gpu"]] = github_actions_job_links.get(job_name)
                failed, success, time_spent = handle_test_results(artifact["stats"])
                model_results[model]["success"] += success
                model_results[model]["time_spent"] += time_spent[1:-1] + ", "

                stacktraces = handle_stacktraces(artifact["failures_line"])

                for line in artifact["summary_short"].split("\n"):
                    if line.startswith("FAILED "):
                        line = line[len("FAILED ") :]
                        line = line.split()[0].replace("\n", "")

                        if artifact_path["gpu"] not in model_results[model]["failures"]:
                            model_results[model]["failures"][artifact_path["gpu"]] = []

                        model_results[model]["failures"][artifact_path["gpu"]].append(
                            {"line": line, "trace": stacktraces.pop(0)}
                        )

                        if re.search("test_modeling_tf_", line):
                            model_results[model]["failed"]["TensorFlow"][artifact_path["gpu"]] += 1

                        elif re.search("test_modeling_flax_", line):
                            model_results[model]["failed"]["Flax"][artifact_path["gpu"]] += 1

                        elif re.search("test_modeling", line):
                            model_results[model]["failed"]["PyTorch"][artifact_path["gpu"]] += 1

                        elif re.search("test_tokenization", line):
                            model_results[model]["failed"]["Tokenizers"][artifact_path["gpu"]] += 1

                        elif re.search("test_pipelines", line):
                            model_results[model]["failed"]["Pipelines"][artifact_path["gpu"]] += 1

                        elif re.search("test_trainer", line):
                            model_results[model]["failed"]["Trainer"][artifact_path["gpu"]] += 1

                        elif re.search("onnx", line):
                            model_results[model]["failed"]["ONNX"][artifact_path["gpu"]] += 1

                        elif re.search("auto", line):
                            model_results[model]["failed"]["Auto"][artifact_path["gpu"]] += 1

                        else:
                            model_results[model]["failed"]["Unclassified"][artifact_path["gpu"]] += 1
                            unclassified_model_failures.append(line)

    # Additional runs
    additional_files = {
        "Examples directory": "run_examples_gpu",
        "PyTorch pipelines": "run_tests_torch_pipeline_gpu",
        "TensorFlow pipelines": "run_tests_tf_pipeline_gpu",
        "Torch CUDA extension tests": "run_tests_torch_cuda_extensions_gpu_test_reports",
    }

    if ci_event in ["push", "Nightly CI"] or ci_event.startswith("Past CI") or ci_event.startswith("Push CI (AMD)"):
        del additional_files["Examples directory"]
        del additional_files["PyTorch pipelines"]
        del additional_files["TensorFlow pipelines"]

    additional_results = {
        key: {
            "failed": {"unclassified": 0, "single": 0, "multi": 0},
            "success": 0,
            "time_spent": "",
            "error": False,
            "failures": {},
            "job_link": {},
        }
        for key in additional_files.keys()
    }

    for key in additional_results.keys():
        # If a whole suite of test fails, the artifact isn't available.
        if additional_files[key] not in available_artifacts:
            additional_results[key]["error"] = True
            continue

        for artifact_path in available_artifacts[additional_files[key]].paths:
            # Link to the GitHub Action job
            job_name = key
            if artifact_path["gpu"] is not None:
                job_name = f"{key} ({artifact_path['gpu']}-gpu)"
            if job_name_prefix:
                job_name = f"{job_name_prefix} / {job_name}"
            additional_results[key]["job_link"][artifact_path["gpu"]] = github_actions_job_links.get(job_name)

            artifact = retrieve_artifact(artifact_path["path"], artifact_path["gpu"])
            stacktraces = handle_stacktraces(artifact["failures_line"])

            failed, success, time_spent = handle_test_results(artifact["stats"])
            additional_results[key]["failed"][artifact_path["gpu"] or "unclassified"] += failed
            additional_results[key]["success"] += success
            additional_results[key]["time_spent"] += time_spent[1:-1] + ", "

            if len(artifact["errors"]):
                additional_results[key]["error"] = True

            if failed:
                for line in artifact["summary_short"].split("\n"):
                    if line.startswith("FAILED "):
                        line = line[len("FAILED ") :]
                        line = line.split()[0].replace("\n", "")

                        if artifact_path["gpu"] not in additional_results[key]["failures"]:
                            additional_results[key]["failures"][artifact_path["gpu"]] = []

                        additional_results[key]["failures"][artifact_path["gpu"]].append(
                            {"line": line, "trace": stacktraces.pop(0)}
                        )

    selected_warnings = []
    if "warnings_in_ci" in available_artifacts:
        directory = available_artifacts["warnings_in_ci"].paths[0]["path"]
        with open(os.path.join(directory, "selected_warnings.json")) as fp:
            selected_warnings = json.load(fp)

    message = Message(title, ci_title, model_results, additional_results, selected_warnings=selected_warnings)

    # send report only if there is any failure (for push CI)
    if message.n_failures or (ci_event != "push" and not ci_event.startswith("Push CI (AMD)")):
        message.post()
        message.post_reply()
