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

# 昇腾训练调优

确认目标 → 基线 → 保留不同机制的假设 → 选择有区分价值的实验 → 按证据继续、组合或切换 → 复测交付。
从显存、计算、通信选择已知方法或新假设，默认争取 3 个不同机制的假设，按证据和预算增减；预期观察与支持/反对证据记入现有实验记录，按收益、信息价值和成本选择，不固定搜索顺序或失败次数。
主流程只保留决策与实验闭环；启动按需复用 `train-run` 的 Ascend 分支，方法按批量、并行与状态分片、重计算与卸载、通信调度、数据与 Host、图与编译、算子与后端组织，同一方法可服务多个目标；方法和 Knowledge 只在选中或有疑问时加载。
固定工作负载与质量要求，保留最佳已验证配置及有价值的显存/速度折中，交付可复现结果并注明验证范围。
