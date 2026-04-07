# OpenClaw area review pack

このフォルダは、country area metadata を OpenClaw にレビューさせるための最小構成です。

## 含まれているもの
- `AGENTS.md`
  - 全体ルール
- `skills/area-review/SKILL.md`
  - この作業専用のレビュー手順
- `schemas/area_patch.schema.md`
  - patch JSON の正式フォーマット
- `examples/area_patch.no_change.json`
  - 修正なしの例
- `examples/area_patch.needs_update.json`
  - 修正ありの例
- `prompts/review_one_country.prompt.md`
  - 1国レビュー時のテンプレート

## 想定する追加フォルダ
この pack を使う workspace には、以下も置いてください。
- `countries/`
- `review_packages/`
- `patches/`
- `reviews/`
- `70_地図仕様_map32.md`

## OpenClaw への考え方
毎回 patch 形式を説明するのではなく、
- `AGENTS.md`
- `SKILL.md`
- `schemas/area_patch.schema.md`
- `examples/*.json`
で固定ルールを読ませます。

## 最低限の実行イメージ
対象国が `BEL` の場合:
- `countries/BEL.json`
- `review_packages/BEL.json`
- `review_packages/BEL.svg`

を読ませて、
- `patches/BEL.patch.json`
- `reviews/BEL.md`
を書かせます。
