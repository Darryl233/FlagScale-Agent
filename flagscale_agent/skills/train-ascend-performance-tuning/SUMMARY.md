<!--
 Copyright 2026 FlagOS Contributors

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 -->

# Ascend Training Performance Tuning

Confirm the goal → establish a baseline → retain hypotheses with different mechanisms → choose experiments that distinguish them → continue, combine, or switch based on evidence → retest and deliver.
Consider known methods and new hypotheses across memory, compute, and communication. Aim for three different mechanisms by default, adjusting for evidence and budget. Record expected observations and supporting/opposing evidence in the existing experiment record. Choose by potential gain, information value, and cost, without a fixed search order or failure count.
The main workflow holds the decision and experiment loop. Reuse the Ascend path in `train-run` for launches. Methods cover batch size, parallelism and state sharding, recomputation and offload, communication scheduling, data and Host, graphs and compilation, and operators and backends; a method may serve several goals. Load methods and Knowledge only when selected or needed to resolve a question.
Keep the workload and quality requirements fixed. Preserve the best verified configuration and useful memory/speed trade-offs. Deliver reproducible results and state the limits of verification.
