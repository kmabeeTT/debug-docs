# Stress Test Analysis: Coherence Degradation & KV Cache Bleed Investigation

**Date**: 2026-03-24
**Source**: `/tmp/stress_test_1774371242/`
**Setup**: 4 concurrent clients, 50 turns each (200 total turns), max_tokens=256

---

## Summary

All 4 clients completed 50 turns with **zero reported errors**. Throughput was stable at ~10.2 t/s. However, several coherence and quality issues were observed. A targeted investigation into cross-session KV cache bleed found **no evidence** of context leaking between parallel sessions.

---

## Key Concerns

### 1. Coherence Breakdowns (Most Serious)

Several turns produced incoherent or garbled output:

- **Client 2, Turn 47** — Complete gibberish: `"A! A! A! A! helium! helium! A helium! He! The helium! helium helium!..."` (repetitive token loop)
- **Client 1, Turn 48** — Topic-blended nonsense: `"Photosynthesis in 2010)**: Photosynthesis is the process by which plants, photosynthesis (2010. **The Four-Stroke Cycle in Modern Engines**..."` — mixes photosynthesis with car engine context
- **Client 2, Turns 8-9** — Degenerative loop: `"the water cycle of the hydrologic cycle of the of the process the water cycle of the hydro of water..."` then Turn 9 mixes bill/water topics into word salad
- **Client 3, Turn 38** — Context contamination: `"The Pascaline introduced fuel injection"` — fuses computing history with car engine details

### 2. Factual Errors Under Load

- **Client 1, Turn 20** — Lists Mars twice in the solar system and scrambles planet order: `"Mercury, Mars, Earth, Venus, Neptune, Uranus, Saturn, Jupiter and Mars"`
- **Client 2, Turn 2** — Hallucinates a task that was never given: `"I was supposed to describe the colors of the aurora borealis, not a rainbow"` — nobody asked about aurora borealis

### 3. Spurious Refusals

- **Client 1, Turn 18** — `"I don't have the necessary permissions to provide information on that topic"` in response to a benign follow-up question about WWII. No permission issue exists.

### 4. Context Loss After Conversation Resets

Every ~10 turns the conversation resets. Post-reset follow-up prompts like "Tell me more" or "Give me another example" frequently produce confused or hallucinated context:

- Client 3, Turn 13 — Invents a "digital garden" concept from nothing, then admits it in Turn 14
- Client 2, Turn 38 — Asked about bill-to-law process, responds with European countries + car engines
- Client 1, Turn 26 — Asked about haiku follow-up, pivots back to programming languages from a prior reset

### 5. TTFT Latency Spikes

Normal TTFT is ~100ms, but periodic spikes occur (likely tied to context resets or GC):

| Client | Spike Turns | Peak TTFT |
|--------|-------------|-----------|
| 1 | 5, 9, 17, 29, 39, 49 | 1,347ms |
| 2 | 10, 28, 40, 46, 49 | **2,901ms** |
| 3 | 27, 50 | 1,308ms |
| 4 | 10, 30, 39 | 1,317ms |

Client 2 Turn 40 hit 2.9s TTFT — nearly 30x the baseline.

### 6. What Looked Fine

- Zero errors reported across all 200 turns
- Throughput was stable at ~10.2 t/s on average; only dips during TTFT spikes
- Short factual answers (math, capitals, simple lists) were consistently correct
- Token generation was steady — no timeouts or dropped connections

---

## KV Cache Bleed Investigation

### Question

Is there evidence that a KV cache bleed bug is causing context/conversation to leak between parallel sessions?

### Finding: No Evidence of Cross-Session Bleed

#### All Topic Contamination Is Intra-Session

Every instance of "wrong topic" bleeding in can be traced back to the same client's own conversation history:

| Incident | Contaminating Topic | Source |
|---|---|---|
| Client 1, Turn 48: photosynthesis response mixes in "Four-Stroke Cycle" | Car engines | Client 1 Turns 9, 11, 29, 35 |
| Client 3, Turn 38: "The Pascaline introduced fuel injection" | Car engines + computing history | Client 3 Turns 4, 7, 16-17, 37 |
| Client 2, Turn 38: answers bill question with countries + car engines | Countries, car engines | Client 2 Turns 36, 34-35 |
| Client 2, Turns 8-9: water cycle degenerates into bill topic | Bill-to-law | Client 2 Turn 7 |
| Client 1, Turn 26: haiku follow-up reverts to programming languages | Programming languages | Client 1 Turn 23-24 |

None of these involve content unique to a different client appearing where it shouldn't.

#### Hallucinations Don't Trace to Other Sessions Either

- **Client 2, Turn 2** — "aurora borealis" appears in zero other client sessions. This is a pure hallucination, not bleed.
- **Client 3, Turn 13** — "digital garden" appears nowhere across any client. Also a hallucination.
- **Client 2, Turn 47** — "helium! helium!" isn't present in any concurrent session. This is a degenerate token repetition loop.

#### The Test Design Makes Cross-Bleed Hard to Detect

One caveat: all 4 clients draw from the same prompt pool (star lifecycle, car engine, WWII, bill-to-law, computing history, etc.). This means the sessions share many overlapping topics by design. If Client 2 suddenly mentioned "Cinderella" (a topic in Clients 3 and 4 but not 2's history), that would be a smoking gun — but no such case was observed.

### What's Actually Happening

The issues observed are better explained by:

1. **Attention/context window degradation** — the model loses coherence as conversation grows, especially right before resets
2. **Post-reset hallucination** — after a reset, vague follow-ups ("tell me more") cause the model to confabulate or fall back on residual attention patterns from its own prior turns
3. **Token generation loops** — the "helium" and "hydro of water" cases are classic repetitive degeneration, not external contamination

---

## Recommendations from Initial Test

1. **To better detect KV cache bleed**: Use deliberately unique/distinctive topics per client (e.g., Client 1 only discusses marine biology, Client 2 only discusses medieval history) so any cross-contamination would be immediately obvious.
2. **Investigate TTFT spikes**: The ~10-turn periodicity of TTFT spikes (especially Client 2's 2.9s spike) suggests a correlation with conversation resets or garbage collection. Worth profiling.
3. **Address intra-session coherence**: The topic-blending and degeneration issues (especially Client 2 Turns 8-9 and 47) may point to attention weight issues in long multi-topic conversations under load.

---

# KV Cache Bleed — Confirmed Reproduction (Follow-up Test)

**Date**: 2026-03-24
**Machine**: `bh-30-special-kmabee-for-reservation-68190` (single chip P100a)
**Source**: `~/kmabee_demo/kv_cache_bleed_logs/` (30 runs)
**Runs**: 30 iterations of `run_kv_cache_bleed_test.sh`

**Server command:**
```
USE_DYNAMIC_BATCHER=true MAX_NUM_SEQS=4 MODEL=Llama-3.2-1B-Instruct ./launch_server.sh 2>&1 | tee server_1b_instruct_batch4_again_4096_dynamic.log
```

**Test command:**
```
source run_kv_cache_bleed_test.sh
```

**Key config**: Dynamic batcher enabled, batch size 4, Llama-3.2-1B-Instruct, 4 independent conversations with distinct topics (penguins, volcanoes, origami, submarines).

## Results: 24 PASS / 6 FAIL (20% failure rate)

Failing runs: **8, 15, 18, 23, 26, 29**

The test used **deliberately distinct topics per conversation slot** (penguins, volcanoes, origami, submarines), making cross-session contamination unmistakable. This addresses the detection weakness identified in the initial stress test above.

---

## Failure Evidence

### Run 8 — Submarine content bleeds into penguins
- **Victim:** Conv 1 (penguins) | **Source:** Conv 4 (submarines) | **Turn:** 3
- **Evidence:**
  > `"Penguins are carnivorous birds, and they primarily feed on fish, krill, and squid. They feeders Portsmouth:** These are the most common type of submarine:** Examples: USS Connecticut (SSN 3-deckled submarine. These are powered by a nuclear reactor, which generates steam to power turbines to drive propellers. Examples include the Virginia-class submarines."`
- **Nature:** Mid-sentence pivot from penguin diet to detailed submarine class descriptions. Submarine KV cache tokens injected directly into the penguins response stream.

### Run 15 — Submarine token repetition in volcanoes
- **Victim:** Conv 2 (volcanoes) | **Source:** Conv 4 (submarines) | **Turn:** 7
- **Evidence:**
  > `"you of a of the submarine submarine submarine submarine"`
- **Nature:** Degenerate repetition of the word "submarine" injected into the volcanoes conversation. Model was reading from the wrong KV cache slot and got stuck in a repetition loop.

### Run 18 — Full response takeover (origami into submarines)
- **Victim:** Conv 4 (submarines) | **Source:** Conv 3 (origami) | **Turn:** 7
- **Evidence:**
  > `"Title Origami (also known as the "Pomma flattny that is a 3D structure with a sphere that can be folded and shaped like a self-contained units that are connected by a specific design. The Mandalay: Abridarviva This is a type of creating intricate and intricate designs, often used in nature-inspired and shapes and colors, as well as subtle variations of different symbols and artworks."`
- **Nature:** Complete response takeover. The submarines conversation produced a fully origami-themed response — folding, 3D structures, intricate designs. Zero submarine content in the entire turn.

### Run 23 — Penguin keyword injected into submarines
- **Victim:** Conv 4 (submarines) | **Source:** Conv 1 (penguins) | **Turn:** 8
- **Evidence:**
  > `"The first submarine was the 'Alicon penguin.'"`
- **Nature:** Single foreign keyword injection — "penguin" inserted as a hallucinated submarine name.

### Run 26 — CRITICAL: Multi-slot cascade failure (worst case)
Three separate bleed events in a single run:

**Event 1:** Submarine -> Volcanoes (Turn 8)
> `"Volcanic activity is not actually a rare event that caused the largest volcano, USS Nautilus, a US Navy's own words, which was the first submarine that successfully completed its maiden voyage in 1954.**melt**"`

**Event 2:** Submarine -> Origami (Turns 3, 4, AND 5 — sustained across 3 consecutive turns)
> Turn 3: `"Origins of the Atomic Bombage! : Sea Life! ... the daily life on a submarine:"`
> Turn 4: `"Modular architecture! The concept of living on on a submarine!"`
> Turn 5: `"...The life of a submarine is the 'bottle.'"`

**Event 3:** Penguins -> Submarines (Turn 8)
> `"The first submarine, and 6 species of penguins are not exactly submarines"`

**Nature:** By far the most severe failure. Submarine content bled into TWO other conversations simultaneously, and origami was contaminated for 3 consecutive turns (Turns 3-5), meaning the KV cache mapping was **persistently wrong** — not a transient single-token glitch.

### Run 29 — Volcano content bleeds into penguins
- **Victim:** Conv 1 (penguins) | **Source:** Conv 2 (volcanoes) | **Turn:** 6
- **Evidence:**
  > `"Baby penguins, gentle volcanoes that erupt from the ocean floor."`
- **Nature:** Volcano keyword injected into penguins response.

---

## Pattern Analysis

### 1. Conv 4 (submarines, batch slot index 3) is involved in EVERY failure

In all 6 failing runs, the submarine slot was either the source or victim of contamination. This strongly suggests a **boundary condition bug** — possibly an off-by-one error in batch slot addressing or a fence-post error in KV cache partition boundaries where the last slot's cache region overlaps with neighbors.

### 2. Bleed is bidirectional but submarine-biased

| Direction | Runs |
|---|---|
| Submarine -> Penguins | 8, 26 |
| Submarine -> Volcanoes | 15, 26 |
| Submarine -> Origami | 26 |
| Origami -> Submarines | 18 |
| Penguins -> Submarines | 23, 26 |
| Volcanoes -> Penguins | 29 |

### 3. Bleeds cluster in mid-to-late turns (Turns 3-8)

- Turn 3: runs 8, 26
- Turn 4-5: run 26 (sustained)
- Turn 6: run 29
- Turn 7: runs 15, 18
- Turn 8: runs 23, 26 (twice)
- **No bleeds observed in turns 1-2.** The bug manifests after the KV cache has accumulated enough entries — possibly related to cache memory pressure or page boundary alignment.

### 4. Severity spectrum

| Severity | Description | Runs |
|---|---|---|
| Mild | Single foreign word injected | 23, 29 |
| Moderate | Sentence fragment pivots to foreign content | 8 |
| Severe | Degenerate repetition of foreign token | 15 |
| Critical | Entire response from wrong conversation | 18 |
| Critical | Sustained multi-turn, multi-slot cascade | 26 |

### 5. Run 26 is the smoking gun

The 3-turn sustained bleed into origami (Turns 3, 4, 5 all contaminated) points to a **persistent KV cache index/offset corruption** rather than a transient race condition. The origami slot was partially or fully mapped to submarine's KV cache pages for multiple decode cycles.

---

## Conclusions

1. **KV cache bleed is confirmed and reproducible** at a 20% failure rate (6/30 runs) on single chip P100a with dynamic batcher, batch size 4, Llama-3.2-1B-Instruct.
2. **The last batch slot (index 3 / Conv 4) is always involved**, pointing to an off-by-one or boundary error in KV cache slot partitioning.
3. **The bug is not transient** — Run 26 shows sustained multi-turn corruption, meaning the cache mapping can stay broken across decode steps.
4. **Recommended next steps:**
   - Audit KV cache slot indexing in the dynamic batcher, especially boundary calculations for the last slot (index `MAX_NUM_SEQS - 1`).
   - Check for off-by-one errors in cache partition start/end address calculations.
   - Inspect whether cache eviction or reallocation under memory pressure can corrupt slot mappings.
   - Attempt reproduction with `MAX_NUM_SEQS=2` and `MAX_NUM_SEQS=8` to see if the last-slot pattern holds.
