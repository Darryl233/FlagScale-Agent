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

# 昇腾训练基础调优

当前只启用：确认条件 → `flagscale train -c ...` 建立基线 → 一个 MBS 候选 → 日志比较 → 复测交付。
固定工作负载、GBS 和并行布局，用同一份计划与实验记录推进，通过结果分析工具计算指标。
默认只加载主 SKILL；有具体阻塞时再查对应小节，进阶方法和其他技能的扩展入口暂时关闭。
扩展文件保留，后续按实际验证逐项接回。未通过本轮验收则保留原基线；短跑结论注明验证范围。
