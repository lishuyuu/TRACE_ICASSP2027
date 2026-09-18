# Upstream assets

Obtain upstream source and model files separately under their respective terms.
Code licenses, model licenses, and dataset licenses are independent.

| Component | Source / identifier |
|---|---|
| CosyVoice3 | https://github.com/FunAudioLLM/CosyVoice ; `Fun-CosyVoice3-0.5B-2512` |
| F5-TTS | https://github.com/SWivid/F5-TTS ; `F5TTS_v1_Base` |
| dots.tts | https://github.com/studio-dots-ai/dots.tts ; `dots-studio/dots.tts-soar` |
| CtrlSpeech | https://github.com/zszheng147/ctrlspeech |
| WordVoice | https://github.com/XXH333/WordVoice-main |
| WhiStress | https://github.com/slp-rl/WhiStress ; https://huggingface.co/slprl/WhiStress |
| emphases | https://github.com/interactiveaudiolab/emphases ; `maxrmorrison/emphases` |
| UTMOS22 | https://github.com/sarulab-speech/UTMOS22 |

Select the upstream revision and checkpoint in the host configuration. The
acoustic interface must use that host's native state representation and native
decoder/vocoder. Model-name strings do not substitute for host bindings.

The emphases wrapper uses distribution version `0.0.2`; provide the model
checkpoint locally. WhiStress and UTMOS wrappers likewise accept local source and
model locations. Fix the evaluator checkpoint, MFA acoustic model and dictionary,
word normalization, and scoring sample rate before processing the evaluation set.

Dataset partitions and speaker prompt identifiers are in `data/`. Resolve those
identifiers against the upstream source; recordings and model weights are not
included in the source distribution.
