# AudioEnhancer
14種類のエフェクト・アルゴリズムをシグナルチェインとして統合したPython製の音質向上ツールです。

## 機能一覧 (シグナルチェイン順)
1. **NoiseGate**: バックグラウンドノイズのカット
2. **DeReverb**: STFT処理を用いた部屋鳴り・残響成分の低減
3. **PhaseAlignment**: チャンネル間の位相ズレの自動補正
4. **DeEsser**: 高域の歯擦音（サ行など）の低減
5. **ParametricEQ**: 低域・高域の基本周波数調整
6. **TransientShaper**: アタック感と余韻のダイナミクス調整
7. **DynamicCompressor**: 全体音量の均一化・ダイナミックレンジ制御
8. **MultibandCompressor**: 周波数帯域別（Low / High）コンプレッション
9. **HarmonicExciter**: 高次倍音付加による輪郭強調
10. **TapeTubeSimulator**: テープ/真空管特有の緩やかな歪みと質感の付加
11. **AudioSuperResolution**: 圧縮音源の高域倍音補元
12. **StereoImager**: M/S処理によるステレオ空間の拡張
13. **Limiter**: ピーク超過防止と音割れカット
14. **LoudnessNormalizer**: 目標RMSレベルへの音量自動正規化

## 依存ライブラリ
```bash
pip install numpy scipy soundfile
```

## 使用方法
### CLIでの実行
```bash
python audio_enhancer.py input.wav output.wav
```

### スクリプト内での実行
```python
from audio_enhancer import AudioEnhancementPipeline

# パイプラインの初期化
pipeline = AudioEnhancementPipeline(sample_rate=44100)

# ファイルの処理
pipeline.process_file("input.wav", "output_enhanced.wav")
