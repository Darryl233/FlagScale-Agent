<!-- Copyright 2026 FlagOS Contributors. SPDX-License-Identifier: Apache-2.0 -->

# Tuning Method Template

Read this only when adding a method. Scope it by the object or mechanism being changed; extend an existing method when it already covers the work. The main workflow handles candidate records, patch preservation, launches, comparisons, and adoption. Do not repeat those steps here.

## When to use

State supporting evidence, version/backend requirements, and possible memory, compute, and communication costs. Link to Knowledge for principles and constraints as needed.

## Generate candidates

Specify fields, paths, or functions to change in the parent configuration/code, required coupled changes, and room for exploration. Do more than list parameter names or say "try a better configuration." Do not require every switch to improve performance independently.

## Additional checks

Include only method-specific validity, activation-path, and quality checks, plus any necessary result interpretation. For example, align real shards when comparing layouts; compare an operator's public forward and backward interfaces; separate graph preparation from replay cost.
