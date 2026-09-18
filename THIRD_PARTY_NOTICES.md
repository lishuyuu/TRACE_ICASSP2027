# Attribution and scope of licensing

Project-authored TRACE source is under the root MIT license. This does not
relicense third-party implementations, datasets or pretrained weights.

- `control_embeddings.py::CtrlSpeechPhoneEmbedding` adapts the control embedding
  mechanism referenced from `zszheng147/ctrlspeech`, local source revision
  prefix `b2ec05230`. Preserve `licenses/CtrlSpeech_LICENSE` (MIT).
- `control_embeddings.py::WordVoiceFrameModulation` adapts word-attribute
  modulation referenced from `XXH333/WordVoice-main`, local source revision
  prefix `2d7e53c14`. Preserve `licenses/WordVoice_LICENSE` (Apache-2.0).
  Changes: destination-frame duration representation, explicit shape/coverage
  checks, configurable insertion width. This is not an unmodified original system.
- `emphases_evaluator.py` loads selected definitions from the separately installed
  upstream distribution. The upstream model implementation is not copied here.
- F5, CosyVoice3, dots.tts, WhiStress and UTMOS model code/weights are obtained
  separately. Refer to their licenses in the downloaded distributions.

The package intentionally keeps method-source names and limitations explicit.
Review institutional/coauthor ownership authorization before public publication.
