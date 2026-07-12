# テストカバレッジ分析 & テスト実装ガイド

このドキュメントは、opensearch-learning-to-rank-base プラグインのテストカバレッジ計測結果と、
このリポジトリで新しくテストを書くために必要な前提知識(リポジトリ構造・機械学習モデル・テスト基盤)を
初心者向けにまとめたものです。

- 計測日: 2026-07-12
- 計測対象: `src/main/java` に対する `src/test/java` のユニットテスト(`./gradlew test jacocoTestReport`)
- 注意: `src/javaRestTest`(統合テスト)はカバレッジ集計に **含まれていません**。
  そのため REST ハンドラや Transport アクションなど「統合テストでしか通らないコード」は
  実際の検証状況よりも低い数値になります(後述)。

---

## 1. このプラグインは何をするものか

**Learning to Rank (LTR、ランキング学習)** は、検索結果の並び順を機械学習モデルで最適化する手法です。
このプラグインは OpenSearch に以下を追加します。

1. **特徴量 (feature) の定義と保存**
   検索クエリと文書のペアから「シグナル」(例: タイトルの TF-IDF スコア、人気度) を取り出す定義。
   実体は Mustache テンプレート化された OpenSearch クエリ、Lucene expression、または script です。
2. **特徴量セット (feature set)**: 特徴量の順序付きリスト。モデルの入力ベクトルの並びを決めます。
3. **モデル (model)**: 外部で学習させた XGBoost / RankLib などのモデルを JSON/テキストで登録。
4. **`sltr` クエリ**: 保存済みモデルで文書をスコアリングする検索クエリ。
   通常は `rescore` フェーズで上位 N 件のみ再ランキングします。
5. **特徴量ロギング**: 検索時に各特徴量の値を返す機能(学習データ収集用)。

重要: **モデルの「学習」はこのプラグインの外(Python の XGBoost や RankLib CLI)で行い、
プラグインは「学習済みモデルの読み込み(パース)と推論(スコア計算)」だけを行います。**
テストで機械学習の学習処理を扱うことはありません。

### データの置き場所

特徴量・特徴量セット・モデルは `.ltrstore`(またはユーザー命名の `.ltrstore_<name>`)という
**システムインデックス**に JSON ドキュメントとして保存されます(`IndexFeatureStore`)。
ノードごとに `Caches` でキャッシュされます。

### 提供される検索クエリ (QueryBuilder)

| クエリ名 | クラス | 役割 |
|---|---|---|
| `sltr` | `StoredLtrQueryBuilder` | ストアのモデルでスコアリング(本番の主役) |
| `ltr` | `LtrQueryBuilder` | リクエスト内に直接モデルを渡してスコアリング(RankLib モデル) |
| `match_explorer` | `ExplorerQueryBuilder` | match クエリの term 統計 (min/max/avg など) を特徴量として取り出す |
| `term_stat` | `TermStatQueryBuilder` | Lucene expression で term 統計を自由に計算する |

### REST API(`/_plugins/_ltr/...`)

- `PUT /_plugins/_ltr` ストア作成(`RestStoreManager`)
- `PUT/GET/DELETE /_plugins/_ltr/_feature|_featureset|_model`(`RestFeatureManager`, `RestSearchStoreElements`)
- `POST /_plugins/_ltr/_featureset/{name}/_addfeatures`(`RestAddFeatureToSet`)
- `POST /_plugins/_ltr/_featureset/{name}/_createmodel`(`RestCreateModelFromSet`)
- `GET/POST /_plugins/_ltr/_cachestats`, `_clearcache`(`RestFeatureStoreCaches`)
- `GET /_plugins/_ltr/{store}/stats`(`RestStatsLTRAction`)

REST ハンドラは対応する Transport アクション(`com.o19s.es.ltr.action.*` の
`Transport*Action`)を呼び出す薄い層です。

---

## 2. 機械学習モデルの基礎知識(テストに必要な分だけ)

### 2.1 推論の抽象化: `LtrRanker`

すべてのモデルは `com.o19s.es.ltr.ranker.LtrRanker` インターフェースに正規化されます。

```java
FeatureVector v = ranker.newFeatureVector(null); // 特徴量ベクトルを確保
v.setFeatureScore(0, 2.0f);  // featureId(= featureset 内の順序) → 値
v.setFeatureScore(1, 3.5f);
float score = ranker.score(v); // モデルのスコア = 文書の新しい _score
```

- `DenseFeatureVector` は `float[] scores` を持つだけの単純な実装。
- マッチしなかった特徴量は 0 (デフォルト値) になります。
- つまり **「float 配列を入れると float が返る」純関数**であり、ユニットテストが非常に書きやすい部分です。

### 2.2 線形モデル: `LinearRanker`

`score = Σ weights[i] * features[i]`。重み配列を渡すだけです。

### 2.3 勾配ブースティング決定木 (GBDT): `NaiveAdditiveDecisionTree`

XGBoost / RankLib の LambdaMART・MART・Random Forest はすべて
「複数の回帰木の重み付き和」として実行されます。

```
score = normalizer( Σ_t weights[t] * tree_t.eval(features) )
```

- 各木は `Split`(内部ノード: `feature`, `threshold`, 左右の子)と `Leaf`(出力値)の木構造。
- 評価は根から `threshold > scores[feature] ? left : right` をたどり、到達した葉の値を返すだけ。
- **テストの発想**: 小さな木を手で組み立て、既知の入力に対する期待値を assert する
  (`NaiveAdditiveDecisionTreeTests` が良い見本)。

### 2.4 モデルパーサー (`ranker.parser`)

登録されたモデル文字列を `LtrRanker` に変換します。`type` フィールドで選択されます。

| type | パーサー | 入力形式 |
|---|---|---|
| `model/linear` | `LinearRankerParser` | `{"feature名": 重み, ...}` |
| `model/xgboost+json` | `XGBoostJsonParser` | XGBoost の `dump_model` 出力(木のJSON配列)。`{"objective": ..., "splits": [...]}` 形式のラッパーも可 |
| `model/xgboost+json+raw` | `XGBoostRawJsonParser` | XGBoost `save_model` のネイティブ JSON |
| `model/ranklib` | `RanklibModelParser` | RankLib のテキスト形式(`sample_models/*.txt` 参照) |

- 特徴量名 → featureId の解決に `FeatureSet` が必要(`set.featureOrdinal("title")`)。
- **テストの発想**: 正常な JSON で期待どおりの木/重みになるか、壊れた JSON で
  適切な `ParsingException` が出るか。`XGBoostJsonParserTests` が見本。
  実物のモデルは `src/test/resources/models/xgboost-wmf.json` と `sample_models/` にあります。

### 2.5 正規化 (2種類あるので混同注意)

1. **モデル出力の正規化** (`Normalizers`): `noop`(そのまま)と `sigmoid`(1/(1+e^-x))。
   XGBoost の `objective: binary:logistic` などで使われます。
2. **特徴量の正規化** (`FeatureNormalizingRanker` + `feature_norm`):
   モデル評価前に個々の特徴量値を変換します。
   - `MinMaxFeatureNormalizer`: `(v - min) / (max - min)`
   - `StandardFeatureNormalizer`: `(v - mean) / stdDeviation`
   学習時に正規化した特徴量を使った場合、推論時にも同じ変換が必要になるための機能です。

### 2.6 特徴量の3種類のテンプレート言語

`StoredFeature.template_language`:

- `mustache`(デフォルト): OpenSearch クエリの Mustache テンプレート → `PrecompiledTemplateFeature`
- `derived_expression`: 他の特徴量から計算する Lucene expression(例: `log(popularity) * title_score`)→ `PrecompiledExpressionFeature` / `DerivedExpressionQuery`
- `script_feature`: Painless などの script で特徴量を計算 → `ScriptFeature`

---

## 3. コードマップとテストの現状

パッケージごとの役割(テスト作成時にどこを見ればよいか):

| パッケージ | 役割 | テストしやすさ |
|---|---|---|
| `com.o19s.es.ltr.ranker(.dectree/.linear)` | モデル推論の中核。純粋 Java | ◎ 依存なし |
| `com.o19s.es.ltr.ranker.parser` | モデル JSON のパース | ◎ XContent だけ |
| `com.o19s.es.ltr.ranker.ranklib` | RankLib (RankyMcRankFace) ラッパ | ○ |
| `com.o19s.es.ltr.ranker.normalizer` | 正規化 | ◎ |
| `com.o19s.es.ltr.feature(.store)` | 特徴量・セット・モデルの定義とパース | ○ |
| `com.o19s.es.ltr.feature.store.index` | `.ltrstore` への保存・キャッシュ | △ ノードが必要な部分あり |
| `com.o19s.es.ltr.query` | Lucene クエリ実装 (RankerQuery ほか) | △ Lucene の index/searcher を組む |
| `com.o19s.es.explore` / `com.o19s.es.termstat` | match_explorer / term_stat | △ 同上 |
| `com.o19s.es.ltr.logging` | 特徴量ロギング | △ |
| `com.o19s.es.ltr.action` | Transport アクション(CRUD 等) | △ 統合テスト向き |
| `com.o19s.es.ltr.rest` | REST ハンドラ | △ 統合テスト向き |
| `com.o19s.es.template.mustache` | Mustache 拡張 | ○ |
| `org.opensearch.ltr.stats(.suppliers)` | 統計 API 用の値収集 | ◎ |
| `org.opensearch.ltr.breaker` | メモリサーキットブレーカー | ◎ |
| `org.opensearch.ltr.settings` | 動的設定 (プラグイン有効/無効) | ○ |
| `org.opensearch.ltr.transport` | stats 用 transport | △ |

---

## 4. テスト基盤の知識

### 4.1 テストの種類と場所

| 場所 | 種類 | 基盤 | 実行コマンド |
|---|---|---|---|
| `src/test/java` | ユニットテスト | LuceneTestCase / OpenSearchTestCase 系 | `./gradlew test` |
| `src/javaRestTest/java` | 統合テスト | `OpenSearchSingleNodeTestCase`(`BaseIntegrationTest`)+ YAML REST テスト | `./gradlew integTest`(実クラスタ起動)|
| `src/test/resources/rest-api-spec/test/fstore/*.yml` | YAML REST テスト | `LtrQueryClientYamlTestSuiteIT` から実行 | `integTest` に含まれる |

`./gradlew check` は `test` + `integTest` + `spotlessCheck` を実行します。

### 4.2 基底クラスの選び方

1. **純粋ロジック** (ranker, normalizer, breaker など)
   → `org.apache.lucene.tests.util.LuceneTestCase`。
   - JUnit3 スタイル: メソッド名を `testXxx` にする(`@Test` アノテーション不要)。
   - `random()` でシード管理された乱数が使える(randomized testing)。
   - 失敗時に表示される `-Dtests.seed=...` を付けて再実行すると同じ乱数で再現できる。
2. **OpenSearch のユーティリティ(XContent, Settings 等)を使うロジック**
   → `org.opensearch.test.OpenSearchTestCase`。`randomAlphaOfLength()` などの helper が使える。
3. **QueryBuilder(クエリの DSL 部分)**
   → `org.opensearch.test.AbstractQueryTestCase<T>`。
   `doCreateTestQueryBuilder()`(ランダムなクエリを生成)と
   `doAssertLuceneQuery()`(変換後の Lucene Query を検証)を実装すると、
   **シリアライズ往復・equals/hashCode・JSON パースの検証が自動で行われる**。
   `getPlugins()` で `LtrQueryParserPlugin` を返すのを忘れないこと
   (見本: `LtrQueryBuilderTests`, `TermStatQueryBuilderTests`)。
4. **Lucene クエリの実行レベルの検証** (RankerQuery, ExplorerQuery 等)
   → `LuceneTestCase` + 手動で `Directory`/`IndexWriter`/`IndexSearcher` を組む
   (見本: `LtrQueryTests`, `ExplorerQueryTests`, `TermStatQueryTests`)。
5. **ストア CRUD やモデル登録などプラグイン全体の動き**
   → `src/javaRestTest` の `BaseIntegrationTest` を継承
   (`createStore()`, `addElement()`, `getElement()` などの helper が既にある)。
6. **REST API の入出力仕様**
   → YAML REST テスト (`src/test/resources/rest-api-spec/test/fstore/`) に追記。
   HTTP レベルの記述だけで書けるので最も簡単。

### 4.3 便利なテストヘルパー(車輪の再発明をしない)

- `LtrTestUtils`: ランダムな `StoredFeature` / `StoredFeatureSet` / `CompiledLtrModel` の生成、
  `wrapMemStore()` / `nullLoader()` など。
- `MemStore`: `FeatureStore` のインメモリ実装。インデックス不要でストア依存コードをテストできる。
- `LinearRankerTests.generateRandomRanker()` / `NaiveAdditiveDecisionTreeTests` のランダム木生成。
- `StoredFeatureSetParserTests.buildRandomFeature()` 系。
- サンプルモデル: `src/test/resources/models/xgboost-wmf.json`, `sample_models/*.txt`(RankLib 各種)。

### 4.4 実行コマンド集

```bash
# ユニットテスト全部 + カバレッジ
./gradlew test jacocoTestReport
# → build/reports/jacoco/test/html/index.html をブラウザで開く

# 特定のテストクラス/メソッドだけ
./gradlew test --tests "com.o19s.es.ltr.ranker.parser.XGBoostJsonParserTests"
./gradlew test --tests "*XGBoost*.testBadStructure"

# 失敗したときのシード再現
./gradlew test --tests "..." -Dtests.seed=346D136EF48FDF8F

# 統合テスト(実 OpenSearch クラスタを起動するので重い)
./gradlew integTest

# コミット前(フォーマット + 全チェック)
./gradlew spotlessApply
./gradlew check
```

### 4.5 ハマりどころ

- **root ユーザーでは実行不可**: `OpenSearchTestCase` 系のテストは
  「can not run opensearch as root」で落ちます。一般ユーザーで実行してください。
- **クラス名の規約**: ユニットテストは `*Tests`、統合テストは `*IT` で終える
  (OpenSearch build-tools の命名規約。ヘルパーは `LtrTestUtils` のように回避)。
- **spotless**: フォーマット違反は `check` で失敗します。`./gradlew spotlessApply` で自動整形。
  ワイルドカード import は禁止です。
- **randomized testing のスレッドリーク検査**: テスト内で作ったスレッド/Directory を
  閉じないと失敗します。`try-with-resources` を徹底。
- **`assertEquals(float, float)` は delta 引数必須**: `Math.ulp(expected)` を使う慣習。
- **SNAPSHOT 依存**: `main` ブランチは `opensearch 3.7.0-SNAPSHOT` に依存します。
  snapshot リポジトリ(ci.opensearch.org)に届かない環境では
  `./gradlew test -Dopensearch.version=3.6.0 -Dbuild.snapshot=false` のように
  リリース版(Maven Central 取得可)へ差し替えて実行できます。

---

## 5. カバレッジ計測結果

計測方法(CI の `.github/workflows/CI.yml` と同じ方式):

```bash
./gradlew test jacocoTestReport
# HTML: build/reports/jacoco/test/html/index.html
# XML : build/reports/jacoco/test/jacocoTestReport.xml (codecov にアップロードされるもの)
```

結果: **ユニットテスト 290 件、全件成功**。

### 全体

| 指標 | カバー | 未カバー | カバレッジ |
|---|---:|---:|---:|
| 命令 (INSTRUCTION) | 13,545 | 11,196 | **54.7%** |
| 分岐 (BRANCH) | 740 | 934 | **44.2%** |
| 行 (LINE) | 2,942 | 2,370 | **55.4%** |
| メソッド | 885 | 684 | 56.4% |
| クラス | 151 | 89 | 62.9% |

### パッケージ別(行カバレッジの低い順)

| パッケージ | 行カバレッジ | 分岐 | 備考 |
|---|---:|---:|---|
| `org.opensearch.ltr.rest` | 0.0% (0/37) | 0% | 統合テストでのみ実行される層 ※ |
| `org.opensearch.ltr.exception` | 0.0% (0/2) | - | |
| `com.o19s.es.ltr.rest` | 3.4% (12/353) | 0% | 統合テストでのみ実行される層 ※ |
| `com.o19s.es.ltr.action` | 4.3% (38/891) | 0% | 統合テストでのみ実行される層 ※ |
| `com.o19s.es.template.mustache` | 22.3% (44/197) | 7.1% | **純ユニットで改善可能** |
| `org.opensearch.ltr.transport` | 40.0% (38/95) | - | シリアライズのテスト不足 |
| `com.o19s.es.ltr` | 50.5% (47/93) | 57.1% | プラグイン登録コード(浅い) |
| `com.o19s.es.ltr.ranker.normalizer` | 54.5% (48/88) | 17.6% | **純ユニットで改善可能** |
| `org.opensearch.ltr.settings` | 59.3% (16/27) | 50% | |
| `com.o19s.es.ltr.utils` | 60.8% (31/51) | 56.2% | |
| `com.o19s.es.ltr.logging` | 65.1% (151/232) | 53.6% | |
| `com.o19s.es.ltr.feature.store` | 67.6% (649/960) | 46.7% | ScriptFeature が 0% |
| `com.o19s.es.ltr.query` | 71.7% (439/612) | 53.2% | DerivedExpressionQuery が穴 |
| `com.o19s.es.explore` | 75.9% (271/357) | 57.4% | |
| `com.o19s.es.ltr.feature` | 78.2% (61/78) | 45.5% | |
| `com.o19s.es.ltr.feature.store.index` | 80.1% (214/267) | 68.9% | |
| `com.o19s.es.ltr.ranker.ranklib` | 81.4% (48/59) | 55.6% | |
| `org.opensearch.ltr.stats` | 85.4% (35/41) | 60% | |
| `com.o19s.es.termstat` | 87.2% (245/281) | 62% | |
| `com.o19s.es.ltr.ranker.linear` | 88.9% (16/18) | 45.5% | |
| `com.o19s.es.ltr.ranker` | 90.4% (66/73) | 65% | |
| `com.o19s.es.ltr.ranker.parser` | 93.1% (298/320) | 82.1% | |
| `org.opensearch.ltr.stats.suppliers` | 94.3% (50/53) | 83.3% | |
| `org.opensearch.ltr.stats.suppliers.utils` | 97.4% (38/39) | 64.3% | |
| `org.opensearch.ltr.breaker` | 97.6% (41/42) | 90% | |
| `com.o19s.es.ltr.ranker.dectree` | 100% (46/46) | 75% | |

※ **重要な注意**: JaCoCo は `src/test`(ユニットテスト)の実行分しか計測しません。
`src/javaRestTest` の統合テスト(`./gradlew integTest`)は別 JVM(実クラスタ)で動くため
数値に含まれず、CI の codecov も同様です。REST / action / transport 層は
`AddFeaturesToSetActionIT` や YAML REST テスト(`fstore/*.yml`)で機能的には検証されていますが、
**「カバレッジ数値としては見えない」** ことを踏まえて読む必要があります。

---

## 6. カバレッジ不足の分析と優先順位

「数値が低い場所」を 3 つのタイプに分けて考えるのが重要です。

### タイプ A: ユニットテストが本当に無く、純ユニットで書ける(最優先・初心者向き)

| 対象クラス | 現状 | 何をテストするか |
|---|---|---|
| `MinMaxFeatureNormalizer` | 28.6% | `(v-min)/(max-min)` の計算、`min >= max` で例外、`equals/hashCode` |
| `StandardFeatureNormalizer` | 44.4% | `(v-mean)/std` の計算、StreamInput/Output の往復 |
| `FeatureNormalizingRanker` | 58.8% (分岐21%) | 正規化された特徴量でモデルが呼ばれること、`ramBytesUsed` |
| `CustomMustacheFactory` の `ToJsonCode`/`JoinerCode`/`UrlEncoderCode` | 0% | `{{#toJson}}...{{/toJson}}` 等のテンプレート展開結果 |
| `CustomReflectionObjectHandler` の `ArrayMap`/`CollectionMap` | 0% | 配列/コレクションを Mustache から参照できること |
| `AutoDetectParser` (`com.o19s.es.ltr.rest`) | 0% | JSON/XContent からの feature/featureset 自動判別、不正入力で例外 |
| `Scripting` (`utils`) | 26.7% | expression のコンパイルとエラーケース |
| `LimitExceededException` | 0% | 2行だけ。ついでに |
| action 系 Request/Response のシリアライズ | 0% | 「レシピ2」参照。`writeTo` → `new Xxx(StreamInput)` の往復で等価になること |

action パッケージ(891行)は大半が Request/Response の boilerplate なので、
シリアライズ往復テストを足すだけで全体数値が大きく動きます。

### タイプ B: Lucene レベルのテストが必要(中級)

| 対象 | 現状 | 説明 |
|---|---|---|
| `DerivedExpressionQuery` + 内部クラス `FV*` | 29.4% / 0% | `derived_expression` 特徴量の実行パス。`LtrQueryTests` のように RAMディレクトリに文書を入れ、`sltr` 相当のクエリを実行して検証 |
| `RankerQuery$RankerScorer` | 47.5% (分岐31%) | `explain()`、feature cache 経路、`NoopScorer` |
| `PostingsExplorerQuery` | 37.5% | `match_explorer` の postings 統計 (tf/tp) 経路 |
| `ExplorerQuery` | 60.4% | 未カバーの統計タイプ分岐 |
| `PrecompiledExpressionFeature` | 59.2% | derived expression の doToQuery 経路 |

### タイプ C: 統合テスト (javaRestTest / YAML) で足すべきもの

| 対象 | 現状 | 説明 |
|---|---|---|
| `ScriptFeature` (計 ~170行) | 0% | `script_feature` テンプレート言語。`BaseIntegrationTest` に `NativeScriptPlugin`/`InjectionScriptPlugin` という専用モックが既にあるので、それを使う統合テストを追加 |
| `LoggingFetchSubPhase` 本体 | 1.5% | ログ抽出の本体は `LoggingIT` で検証済みだがユニット数値には出ない。ユニットで増やすなら `HitLogConsumer`(81.6%)の残り分岐 |
| REST ハンドラ (`RestFeatureManager` ほか) | 0% | YAML REST テスト (`fstore/*.yml`) への追加が最も簡単。バリデーションエラー系(不正な store 名、存在しない要素の GET/DELETE)が薄い |
| `TransportListStoresAction`, `TransportCreateModelFromSetAction` など | 0% | `ListStoresActionIT` 等はあるが、エラーパス(既存モデルと重複、壊れた定義)を追加 |

### おすすめの進め方(初心者向けロードマップ)

1. **STEP 1**: `MinMaxFeatureNormalizer` / `StandardFeatureNormalizer` のテスト
   (数式が明確で、既存の `NormalizersTests` を見本にほぼ写経で書ける)
2. **STEP 2**: action 系の Request/Response シリアライズテスト
   (`TransportLTRStatsActionTests` と「レシピ2」を見本に量産できる。数値インパクト大)
3. **STEP 3**: `CustomMustacheFactory` の拡張機能テスト(文字列 in/out で書きやすい)
4. **STEP 4**: `AbstractQueryTestCase` 系の既存テストに分岐を足す
   (`ValidatingLtrQueryBuilderTests` など、`doAssertLuceneQuery` の検証を厚くする)
5. **STEP 5**: Lucene レベル(タイプ B)や統合テスト(タイプ C)に挑戦

---

## 7. テスト実装レシピ(コード例)

### レシピ1: 純関数のユニットテスト(正規化を例に)

`src/test/java/com/o19s/es/ltr/ranker/normalizer/MinMaxFeatureNormalizerTests.java`:

```java
package com.o19s.es.ltr.ranker.normalizer;

import org.apache.lucene.tests.util.LuceneTestCase;

public class MinMaxFeatureNormalizerTests extends LuceneTestCase {

    public void testNormalize() {
        MinMaxFeatureNormalizer n = new MinMaxFeatureNormalizer(0.0f, 10.0f); // (min, max)
        float expected = 0.5f;
        assertEquals(expected, n.normalize(5.0f), Math.ulp(expected));
    }

    public void testInvalidRange() {
        // min >= max は IllegalArgumentException になるはず
        expectThrows(IllegalArgumentException.class, () -> new MinMaxFeatureNormalizer(1.0f, 1.0f));
    }
}
```

ポイント:
- クラス名は `*Tests`、メソッドは `public void testXxx()`(`@Test` 不要)。
- float 比較は必ず delta(`Math.ulp(expected)` など)を渡す。
- 例外は `expectThrows(...)` で。

### レシピ2: Request/Response のシリアライズ往復テスト

Transport 層の Request/Response は「ネットワークに書いて読み戻すと等価」が最重要の性質です。
`org.opensearch.ltr.transport` や `com.o19s.es.ltr.action` の 0% はこのパターンで埋められます。

```java
package com.o19s.es.ltr.action;

import org.opensearch.common.io.stream.BytesStreamOutput;
import org.opensearch.core.common.io.stream.StreamInput;
import org.opensearch.test.OpenSearchTestCase;

import com.o19s.es.ltr.action.AddFeaturesToSetAction.AddFeaturesToSetRequest;

public class AddFeaturesToSetRequestTests extends OpenSearchTestCase {

    public void testSerialization() throws Exception {
        AddFeaturesToSetRequest request = new AddFeaturesToSetRequest();
        request.setStore("my_store");
        request.setFeatureSet("my_set");
        request.setFeatureNameQuery("prefix*");
        // ... 他のフィールドも randomAlphaOfLength() 等でランダムに埋めると尚良い

        try (BytesStreamOutput out = new BytesStreamOutput()) {
            request.writeTo(out);
            try (StreamInput in = out.bytes().streamInput()) {
                AddFeaturesToSetRequest deserialized = new AddFeaturesToSetRequest(in);
                assertEquals(request.getStore(), deserialized.getStore());
                assertEquals(request.getFeatureSet(), deserialized.getFeatureSet());
                assertEquals(request.getFeatureNameQuery(), deserialized.getFeatureNameQuery());
            }
        }
    }

    public void testValidation() {
        AddFeaturesToSetRequest request = new AddFeaturesToSetRequest();
        // 必須フィールド未設定なら validate() が非 null を返すはず
        assertNotNull(request.validate());
    }
}
```

見本: `src/test/java/org/opensearch/ltr/transport/TransportLTRStatsActionTests.java`

### レシピ3: QueryBuilder のテスト(`AbstractQueryTestCase`)

```java
public class MyQueryBuilderTests extends AbstractQueryTestCase<StoredLtrQueryBuilder> {

    // プラグインを登録しないと "sltr" が unknown query になる
    @Override
    protected Collection<Class<? extends Plugin>> getPlugins() {
        return Arrays.asList(LtrQueryParserPlugin.class, TestGeoShapeFieldMapperPlugin.class);
    }

    @Override
    protected StoredLtrQueryBuilder doCreateTestQueryBuilder() {
        // ランダムなパラメータでクエリを作る。フレームワークが
        // JSON化 → パース → 再JSON化 / wire往復 / equals・hashCode を自動検証してくれる
    }

    @Override
    protected void doAssertLuceneQuery(StoredLtrQueryBuilder qb, Query query, SearchContext ctx) {
        // toQuery() の結果 (RankerQuery になっているか等) を検証
    }
}
```

見本: `LtrQueryBuilderTests`, `StoredLtrQueryBuilderTests`, `TermStatQueryBuilderTests`

### レシピ4: Lucene インデックスを使うクエリ実行テスト

```java
public class MyQueryTests extends LuceneTestCase {
    public void testScoring() throws IOException {
        try (Directory dir = newDirectory()) {
            try (RandomIndexWriter writer = new RandomIndexWriter(random(), dir)) {
                Document doc = new Document();
                doc.add(newTextField("field", "hello world", Field.Store.NO));
                writer.addDocument(doc);
                writer.commit();
            }
            try (IndexReader reader = DirectoryReader.open(dir)) {
                IndexSearcher searcher = newSearcher(reader);
                // ここで RankerQuery 等を組み立てて searcher.search(...) し、
                // TopDocs のスコアや explain() を検証する
            }
        }
    }
}
```

- `newDirectory()` / `newSearcher()` は LuceneTestCase の helper(リーク検査付き)。
- モデルは `LinearRankerTests.generateRandomRanker()` や手組みの
  `NaiveAdditiveDecisionTree` を使うと簡単。
- 見本: `LtrQueryTests`(モデル+特徴量+インデックスのフルセット)、`ExplorerQueryTests`, `TermStatQueryTests`

### レシピ5: 統合テスト(`BaseIntegrationTest`)

```java
public class MyActionIT extends BaseIntegrationTest {  // src/javaRestTest 側に置く。クラス名は *IT

    public void testAddFeature() throws Exception {
        // BaseIntegrationTest の setup() が .ltrstore を作成済み
        StoredFeature feature = LtrTestUtils.randomFeature("my_feature");
        addElement(feature);  // helper がバージョン・結果を assert してくれる

        StoredFeature stored = getElement(StoredFeature.class, StoredFeature.TYPE, "my_feature");
        assertEquals(feature.name(), stored.name());
    }
}
```

実行は `./gradlew integTest`(実クラスタが起動するため数分かかります)。

### レシピ6: YAML REST テスト

`src/test/resources/rest-api-spec/test/fstore/` に YAML を追加するだけで
`LtrQueryClientYamlTestSuiteIT` が自動で拾います。HTTP の入出力仕様のテストに最適:

```yaml
---
"Create feature with invalid body returns 400":
  - do:
      catch: bad_request
      ltr.create_feature:
        name: broken_feature
        body:
          feature: { not_a_valid_field: true }
```

既存の `10_manage.yml` 〜 `80_search_w_partial_models.yml` が豊富な見本です
(使える API 定義は `src/test/resources/rest-api-spec/api/*.json`)。

---

## 8. 付録: この計測を行ったときの環境メモ

- 実行コマンド(通常環境ならこれだけで OK):
  `./gradlew test jacocoTestReport`
- root ユーザーで実行すると OpenSearchTestCase 系 12 クラスが
  「can not run opensearch as root」で失敗します。必ず一般ユーザーで実行してください。
- `main` は `opensearch 3.7.0-SNAPSHOT` + Gradle 9.4.1 に依存します。SNAPSHOT リポジトリや
  Gradle 配布サイトに届かないネットワークでは、リリース版に差し替えて計測できます:
  `gradle test jacocoTestReport -Dopensearch.version=3.6.0 -Dbuild.snapshot=false`
  (本ドキュメントの数値はこの方法で計測。ソースは同一なのでカバレッジへの影響はありません)
- Spotless の Eclipse フォーマッタがダウンロードできない環境では
  CI と同様に `-x spotlessJava` で除外できます。
