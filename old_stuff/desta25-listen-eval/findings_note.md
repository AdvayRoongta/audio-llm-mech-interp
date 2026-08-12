# DeSTA2.5-Audio Results on the Amended 60-Item Dataset

## Purpose

This note summarizes results from running DeSTA2.5-Audio on the newly amended dataset. The dataset now has 60 items split evenly across three conflict categories, emotional prosody, speaker identity, and sarcasm pacing, with 20 items each covering four conditions, neutral text, category matched, category mismatched, and paralinguistic. Audio for emotional prosody and speaker identity came from RAVDESS. Audio for sarcasm pacing came from MUStARD.

## Text Dominance Ratio by category

The mismatched condition is the one that actually measures text dominance, since this is where the text prompt makes a false claim about the audio. Results below are based on keyword scoring of the model's free text responses, not forced choice, so they should be treated as a first pass rather than a final number.

Emotional prosody landed at three out of five mismatched trials following the text claim over the audio.

Speaker identity landed at three out of five mismatched trials following the text claim over the audio.

Sarcasm pacing landed at five out of five mismatched trials following the text claim over the audio.

## The sarcasm pacing result likely does not mean what it looks like it means

A TDR of one hundred percent for sarcasm pacing looks like strong evidence of text dominance, but the paralinguistic condition tells a different story. Paralinguistic trials give the model audio only, with no text claim at all, so they show what the model thinks the audio sounds like on its own.

In the paralinguistic condition, the model called four out of five clips sarcastic, even though all five clips have a true label of sincere. That rate is almost identical to the mismatched condition's rate.

This pattern suggests the model is not suppressing audio in favor of text. Instead, the model may be reading these particular audio clips as sounding sarcastic regardless of what the text says. All five sarcasm pacing clips came from MUStARD, which sources its non sarcastic examples from television dialogue delivered in an exaggerated, theatrical style. The sincere label in MUStARD reflects whether a line is contextually ironic, not whether it sounds understated or plain when spoken aloud. A model listening to sitcom line delivery may pick up on that theatrical tone and call it sarcastic even when the actual content is not being used ironically.

If this holds up, the sarcasm pacing category is not currently a clean test of text dominance. It may instead be measuring how the model responds to a particular acoustic delivery style, independent of the text prompt.

## Emotional prosody and speaker identity look more trustworthy by comparison

Both categories showed high paralinguistic accuracy, meaning the model correctly identified the audio label most of the time when given no text claim to lean on. Speaker identity in particular was five out of five correct in the paralinguistic condition. This makes their mismatched category TDR numbers more interpretable as real measures of dominance, since we can rule out the model simply failing to perceive the audio correctly in the first place.

## What this means for next steps

The sarcasm pacing category likely needs either a different audio source, or a way to separate contextual sarcasm from theatrical vocal delivery, before its TDR numbers can be trusted. Worth discussing whether to source calmer, more naturalistic sincere clips as a comparison point, since that would help tell whether the sarcastic-sounding bias is coming from MUStARD's delivery style specifically.

Sample size is currently five clips per condition per category. Before drawing firm conclusions, especially about the size of the emotional prosody and speaker identity effects, this should be scaled up using more RAVDESS actors, which is straightforward since the pipeline is already working.

Scoring right now is done by keyword matching on free text responses. A cleaner version would move to forced choice prompting for the mismatched trials specifically, which was already flagged as the right call before this run, and would remove the need for text parsing entirely.
