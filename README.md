# AudioEnhancer
14種類のエフェクトをシグナルチェインとして統合した Python 製の音質向上ツール（プロトタイプ実装）。

## 概要
- ノイズゲート、残響低減（DeReverb）、位相補正、ディエッサー、パラメトリックEQ、トランジェントシェイパー、コンプレッサ、マルチバンドコンプ、ハーモニックエキサイター、テープ/チューブ風シミュレータ、超解像的高域補完、ステレオイメージャー、リミッター、ラウドネス正規化 を順番に適用します。
- 実験的なプロトタイプであり、実運用・商用目的で使う前にパラメータの調整と追加テストを推奨します。

## 機能（チェイン順）
1. NoiseGate — バックグラウンドノイズの抑制（指数的アタック/リリース）  
2. DeReverb — STFTベースの残響低減（ISTFTで再合成、長さ補正）  
3. PhaseAlignment — 参照チャネルに対するチャンネル間のズレ補正  
4. DeEsser — 高域帯域の包絡検出に基づくディエッサー（包絡平滑化あり）  
5. ParametricEQ — 低域/高域の簡易的ブースト/カット（近似実装）  
6. TransientShaper — アタックとサステインの強調/抑制  
7. DynamicCompressor — 時間定数で滑らかなコンプレッション  
8. MultibandCompressor — Low / High の分割コンプ（簡易クロスオーバー）  
9. HarmonicExciter — 倍音付加による“明るさ”の付与  
10. TapeTubeSimulator — 柔らかな非線形歪みとローパスでの整形  
11. AudioSuperResolution — 高域の倍音補完（簡易的）  
12. StereoImager — M/S的に幅の調整  
13. Limiter — ピーク抑制  
14. LoudnessNormalizer — pyloudnorm があれば LUFS、無ければ RMS ベースの正規化

## 依存ライブラリ
- Python 3.8+  
- numpy  
- scipy  
- soundfile  
- （任意）pyloudnorm — LUFS 正確測定を行いたい場合に推奨

## インストール
```bash
pip install numpy scipy soundfile
# LUFS を正確に合わせたい場合（推奨）
pip install pyloudnorm
```

## 使い方

CLI
```bash
python audio_enhancement.py input.wav output.wav
```

スクリプトから（API）
```python
from audio_enhancement import AudioEnhancementPipeline

pipeline = AudioEnhancementPipeline(sample_rate=44100)
pipeline.process_file("input.wav", "output_enhanced.wav")
```

## パラメータ調整
- 各エフェクトはクラスとして実装されているため、パイプラインのチェインを入れ替えたり、コンストラクタパラメータを変更して処理をカスタマイズできます。  
- 例：DeEsser の帯域や DynamicCompressor の threshold/ratio を調整することで音作りが変わります。

## 注意・既知の制約
- 本実装はプロトタイプです。特に以下に注意してください：  
  - SciPy の filtfilt は短い信号でエラーになりやすいため、短信号では内部で lfilter にフォールバックする仕組みを入れていますが、位相特性は変わります。  
  - ParametricEQ は簡易的な low/high の加算アプローチで、真正のシェルフ（biquad）実装とは挙動が異なります。  
  - マルチチャネル処理は N チャネル対応を目指していますが、複雑なマトリクスや異なるレイテンシを持つ入力には追加の調整が必要です。  
  - LoudnessNormalizer は pyloudnorm が無い場合 RMS ベースで近似します。LUFS に厳密に合わせたい場合は pyloudnorm を入れてください。  
  - 一部の処理（STFT/ISTFT、フィルタ）は出力長が入力と完全一致しないことがあるため、再構成時にパディング／トリムで整合させています。  
  - 高品質な商用用途では位相補償クロスオーバー、オーバーラップ処理、遅延補償などの追加実装が必要です。

## パフォーマンス
- サンプルごとの Python ループ（例：コンプレッサの逐次更新）は長いファイルでは遅くなるため、必要なら NumPy のベクトル化や Numba / Cython への置換を検討してください。  
- STFT 等は SciPy 側で C 実装が使われているので比較的高速です。

## テスト
- 推奨されるテストケース：  
  - 短い信号（数百サンプル）で各種フィルタが例外を出さないか  
  - 1ch/2ch/4ch といった複数チャネルでの整合性  
  - サンプリングレートの異なるファイル（44.1k/48k/96k）  
  - 目標 LUFS（pyloudnorm 使用時）に対する収束

## トラブルシューティング
- 「ValueError: The length of the input vector x must be greater than padlen」  
  → 信号が短すぎます。短信号では安全に lfilter にフォールバックする設計になっていますが、もし処理の品質が気になるなら入力を適切にパディングしてから処理してください。  
- 出力がクリップしている  
  → 最後にクリップ（-1..1）を行っています。必要に応じてリミッターの閾値やラウドネス目標を下げてください。  
- LUFS を正確に合わせたい  
  → pyloudnorm をインストールしてください。

## ライセンス
[GPLv3（GNU General Public License version 3）](LICENSE)

## 開発・実行環境
- Python 3.8+  
- 実行環境により SciPy, NumPy のバージョン差で振る舞いが変わる可能性があります。主要パッケージは最新の安定版を推奨します。
