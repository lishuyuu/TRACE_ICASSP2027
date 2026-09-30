# Audio samples


This directory contains the 20 WAV files used by the listening page. The page
loads them directly; no sample files are generated at page-load time.


## Main comparison and backbone examples


The first four files compare WordVoice and TRACE on CosyVoice3 for the two
targets in the paper's main example. The next four show TRACE on F5-TTS and
dots.tts-soar using the same target pair.


```text
cv3_wordvoice_moral.wav
cv3_wordvoice_theory.wav
cv3_trace_moral.wav
cv3_trace_theory.wav
f5_trace_moral.wav
f5_trace_theory.wav
dots_trace_moral.wav
dots_trace_theory.wav
```


## Additional illustrative examples


Three further CosyVoice3 pairs compare WordVoice and TRACE. They are labelled
as illustrative examples on the listening page; they are not additional
quantitative evaluation results.


```text
blue_vase_cv3_wordvoice_blue.wav
blue_vase_cv3_trace_blue.wav
blue_vase_cv3_wordvoice_vase.wav
blue_vase_cv3_trace_vase.wav
doctor_family_cv3_wordvoice_doctor.wav
doctor_family_cv3_trace_doctor.wav
doctor_family_cv3_wordvoice_family.wav
doctor_family_cv3_trace_family.wav
friday_morning_cv3_wordvoice_friday.wav
friday_morning_cv3_trace_friday.wav
friday_morning_cv3_wordvoice_morning.wav
friday_morning_cv3_trace_morning.wav
```


Within each comparison, the text, speaker prompt, generation seed, and control
gain are held fixed. The published waveforms are not normalized separately and
have no pitch or time post-processing. Upstream model and prompt sources are described in [ASSETS.md](../../ASSETS.md)
