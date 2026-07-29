# Data Card: Financial Research Corpus

Status: corpus must be regenerated with
`scripts/prepare_financial_research_data.py`.

## Source and intended use

Text is a controlled collection of financial terms embedded in generic
earnings-call-style templates plus common-language control utterances. Audio is
generated with Microsoft Edge-TTS voices. It is intended for controlled ASR
adaptation experiments, not deployment evaluation.

## Partitions

Train, validation, and test are disjoint by both TTS voice and template family.
The generator aborts if either property overlaps. Each split includes
domain-containing utterances and common controls. The test set is reserved for
post-training evaluation.

Every row records:

- stable ID;
- repository-relative audio path;
- transcript and target term;
- domain/control indicator;
- TTS voice and template family;
- split;
- quality diagnostics; and
- SHA-256 hashes of audio and normalized transcript.

Exact counts and assignments are generated in
`data/financial_research/dataset_report.json`; they are intentionally not
hand-entered here.

## Limitations

All audio is synthetic. The same TTS system contributes training and test
audio, so measured WER is expected to be optimistic. TTS voice labels are not
human demographic observations. Results do not establish performance on real
earnings calls, spontaneous speech, overlapping speech, telephony channels, or
unseen financial vocabulary.

No real-audio WER estimate will be published until a licensed, manually
verified, speaker-disjoint evaluation set is available.

## Licensing

Repository code is MIT licensed. Edge-TTS service and generated-output usage
remain subject to the applicable Microsoft terms; users must review those terms
before redistributing audio.
