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

# 昇腾训练自动调优工作流

用于 FlagScale 三仓昇腾训练的自动调优、配置优化或接续已有实验。
准备比较 → 建立基线 → 生成一个候选 → 运行并比较 → 继续或复测交付。
一个主循环统一管理预算、实验记录和最终采纳；已有有效证据直接复用，纯分析不启动训练。
已有训练 SKILL 保持原样，仅按需参考适用内容；不依赖其具备新子任务接口，原有文档问题等实际测试后针对修订。
复用原生 plan_* 接续进度，read_file/write_file 保存证据，shell/shell_jobs 管理可追踪作业，
flagscale_train_monitor 读取本地可见日志，memory_* 保留有复用价值的已核实结论。
共用 [Agent 工具复用](references/agent-tools.md)，实验明细留在文件中，完整会话由 Agent 自身保存。

方法按已有证据选择：批量、并行布局、显存与重计算、通信与重叠、数据与执行路径、算子。
基础批量任务从 micro-batch 开始，不默认遍历全部方法。方法统一返回父配置、假设、变更和额外检查；
profiling 返回证据，不重复建立调优循环。新增方法按 methods/TEMPLATE.md 接入。
知识通过 Knowledge 索引按需读取。交付配置/补丁、复现命令、实验记录和报告；未通过验收则保留基线。
