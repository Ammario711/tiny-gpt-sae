# Results: SAE sweep on tiny-gpt-sae (layer 3 residual stream)

Run date: 2026-10-10, on a Kaggle Tesla T4. Raw numbers are in
[`sae_sweep_results.json`](sae_sweep_results.json); everything below comes from that file
and the feature inspector output for the `16x_k32` SAE.

## Setup

- Base model: the ~20M-parameter tiny-gpt-sae GPT trained on TinyStories (15,000 steps,
  val loss 1.58).
- SAEs: TopK sparse autoencoders on the layer-3 residual stream (d_model = 384), trained
  for 5,000 steps each.
- Sweep: expansion factor {8, 16, 32} x k (active features per token) {16, 32, 64}.
- Metrics, on held-out validation tokens:
  - **FVE**: fraction of variance of the activations explained by the reconstruction.
  - **CE loss recovered**: `(CE_zero - CE_sae) / (CE_zero - CE_clean)`, where the SAE
    reconstruction (or a zero vector) is spliced back into the forward pass.
    CE_clean = 1.582, CE_zero = 7.856.
  - **Dead**: features that never fired on the eval set. **Dense**: features firing on
    more than 10% of tokens.

## Sparsity vs. fidelity

![sparsity vs fidelity](sparsity_vs_fidelity.png)

| SAE | features | L0 | FVE | CE with SAE | CE loss recovered | dead | dense | train time |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 8x_k16  | 3,072  | 16 | 0.841 | 1.794 | 96.6% | 5     | 6  | 184 s |
| 8x_k32  | 3,072  | 32 | 0.872 | 1.718 | 97.8% | 1     | 17 | 172 s |
| 8x_k64  | 3,072  | 64 | 0.898 | 1.671 | 98.6% | 0     | 56 | 176 s |
| 16x_k16 | 6,144  | 16 | 0.860 | 1.755 | 97.2% | 100   | 6  | 298 s |
| 16x_k32 | 6,144  | 32 | 0.888 | 1.693 | 98.2% | 6     | 15 | 302 s |
| 16x_k64 | 6,144  | 64 | 0.910 | 1.658 | 98.8% | 3     | 40 | 307 s |
| 32x_k16 | 12,288 | 16 | 0.872 | 1.732 | 97.6% | 1,112 | 5  | 539 s |
| 32x_k32 | 12,288 | 32 | 0.900 | 1.677 | 98.5% | 30    | 16 | 539 s |
| 32x_k64 | 12,288 | 64 | 0.918 | 1.649 | 98.9% | 20    | 38 | 539 s |

Observations:

- Both fidelity metrics improve monotonically with width and with k; no configuration
  is off the frontier.
- Every SAE recovers at least 96.6% of the CE loss gap, yet FVE tops out at 0.918. The
  unexplained ~8–16% of variance matters little to next-token loss, a gap worth
  investigating (e.g. whether it lives in a few high-norm directions).
- Dead features grow sharply with width at low k (1,112 of 12,288, i.e. 9%, for
  32x_k16), and dense features grow with k (56 for 8x_k64). 16x_k32 is a reasonable
  middle point: 98.2% loss recovered with 6 dead and 15 dense features, and is the SAE
  used for feature review.

## Feature review (16x_k32)

Method: the inspector dumped the top-activating validation contexts for features 0–299
(the first 300 by index, not cherry-picked). I read the top ~12 contexts for features
0–33, every one of which was alive and non-dense (firing frequency 0.08%–3.2% of
tokens). A label is given only when the top contexts clearly agree; otherwise the
feature is marked unclear. These are labels from top activations only — they say what
the feature responds to most strongly, not that it is monosemantic across its whole
activation range, and no causal (ablation/steering) checks have been done yet.

Example notation: the activating token is in `[[ ]]`.

| feature | fire freq | label | example |
|---|---:|---|---|
| 0  | 0.16% | " go" after "let's" | `Now, let's[[ go]] home and` |
| 1  | 0.43% | " to" after "the X thing" (moral judgments) | `That was the wise thing[[ to]] do.` |
| 2  | 0.93% | period ending "showed it to her mom/friends" | `showed the pearl to her mommy[[.]] Her mom` |
| 3  | 0.30% | " his" after "loved to ride" | `Timmy loved to ride[[ his]] bike` |
| 4  | 0.14% | " to" after "use it" (instrumental purpose) | `You can use it[[ to]] fight dragons` |
| 5  | 0.14% | " you" in "What do you ...?" | `What do[[ you]] think he` |
| 6  | 0.23% | " and" after "felt/got scared" | `Timmy got scared[[ and]] ran inside` |
| 7  | 0.17% | comma after "looked everywhere" (sets up "but") | `He looked everywhere[[,]] but he` |
| 8  | 0.35% | " heard" after "Suddenly, they/she" | `Suddenly, they[[ heard]] a loud` |
| 9  | 0.41% | possessive after "in" (in her head/journal/hands) | `write in[[ her]] journal` |
| 10 | 2.02% | location preposition after "X and Y liked to play" at story start | `Tom and Lily liked to play[[ in]] the park` |
| 11 | 0.16% | preposition after "cared/proud" | `we are proud[[ of]] you` |
| 12 | 0.89% | sentence-initial He/She after a sentence about liking or thinking | `very cool and exciting.[[ She]] wanted to` |
| 13 | 0.74% | sentence-initial He/She after "wanted ... too" | `wanted to try the big berry too.[[ She]] saw` |
| 14 | 0.30% | " to"/" of" after "way" | `the best way[[ to]] make friends` |
| 15 | 3.19% | " was" after "it" in story transitions ("But then it was") | `But then it[[ was]] time for` |
| 16 | 1.48% | " they" after "became (best) friends" | `became best friends.[[ They]] played` |
| 17 | 0.49% | sentence-initial "And" | `made a new friend.[[ And]] so,` |
| 18 | 0.09% | the word "rest" | `stopped to take a[[ rest]] in the` |
| 19 | 0.30% | " even" after "learned that" (moral of the story) | `He learned that[[ even]] when things` |
| 20 | 0.29% | possessive after "brush" | `brush[[ your]] teeth` |
| 21 | 0.08% | " town" in "in a small town" | `Once upon a time, in a small[[ town]], there` |
| 22 | 0.54% | opening quote after an angry character "says," | `She was very angry. She said,[[ "]]Lily` |
| 23 | 0.18% | " what" after "wondered" | `She wondered[[ what]] it was` |
| 24 | 0.93% | " not" after "do/did" in repetitive negation | `They did not laugh. They did[[ not]] smile.` |
| 25 | 0.20% | " and" after "nodded" | `Lily nodded[[ and]] smiled.` |
| 26 | 0.08% | " breath" in "took a deep breath" | `Lily took a deep[[ breath]] and tried` |
| 27 | 0.33% | "Who" opening a question ("Who's there?") | `"[[Who]]'s there` |
| 28 | 0.20% | " pieces" after "into (small)" | `It broke into[[ pieces]].` |
| 29 | 0.32% | "Once" right after `<\|endoftext\|>` (new story) | `<\|endoftext\|>[[Once]] upon a` |
| 30 | 0.60% | mostly the name " Jack" (one top example is " Jane") | `[[ Jack]] was scared` |
| 31 | 2.67% | unclear: tokens at the start of a reply to an exclamation ("Wow", quote, comma) | `"Wow[[,]] that's` |
| 32 | 0.57% | " the" after "near"/"in" in location phrases | `a big rock near[[ the]] fence` |
| 33 | 1.08% | unclear: weak (max act 2.8) verb after a He/She subject (told/showed/tells) | `She[[ showed]] her daughter` |

Takeaways:

- 32 of 34 reviewed features have a clear top-activation pattern; 2 are unclear.
- Most clear features are **context-dependent token features**: the same surface token
  (" to", " and", " his", " the") is split into separate features by what precedes it
  — e.g. " to" in "the wise thing to do" (F1), "use it to" (F4) and "best way to" (F14);
  " and" after "scared" (F6) vs. after "nodded" (F25).
- A few track **story structure** rather than local syntax: the "Once" that starts a new
  story (F29), "Suddenly, they heard" plot turns (F8), "learned that even..." morals
  (F19), and "became best friends. They..." endings (F16).

## Next steps

- Causal checks: ablate/steer a few of the clear features and measure the effect on
  next-token predictions.
- Automated labeling over all 6,144 features, spot-checked against the hand labels here.
- Use the saved early checkpoints to study when these features emerge during training.
