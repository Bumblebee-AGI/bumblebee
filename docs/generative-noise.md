# Generative Entropic Noise

A design primitive for autonomous agents: a small model producing continuous
internal commentary that the main model reads as its own stream of consciousness.

## The Idea

Human consciousness has a curious property: most of your thoughts aren't
deliberately generated. They arrive. You notice a connection you weren't
looking for. A conversation from yesterday surfaces unbidden. An idea
half-forms and dissolves. The "watcher" — the executive part of your mind —
doesn't produce this raw material. It receives it, filters it, and
occasionally acts on it.

LLM agents today are purely phasic. They activate on input, reason, respond,
and return to nothing. Between turns, they have no inner life. No thoughts
accumulate. No associations form. The agent comes back cold every time.

Generative entropic noise changes this. A small model runs on a background
timer (every ~60 seconds), reading the entity's current state and producing
2-4 lines of internal thought. These fragments accumulate in a rolling buffer
that the main model reads as part of its body state each turn. The entity
was "thinking" the whole time it was silent.

## Architecture

```
Daemon heartbeat (~30s)
  │
  ├── tick_bars()           ← mechanical (pure math, no LLM)
  ├── maybe_tick_affects()  ← structured LLM call (every ~3 min, reflex model)
  └── maybe_tick_noise()    ← generative LLM call (every ~1 min, small model)
        │
        ▼
  render_body()
    ## Bars          ← quantitative drives (social, curiosity, creative, ...)
    ## Affects       ← felt-textures derived from body state
    ## Noise         ← inner voice fragments (generative, high temperature)
    ## Conflicts     ← opposing drives both active
    ## Impulses      ← drives crossing thresholds
        │
        ▼
  [Turn context] preamble → Watcher (Gemma 4 26B) reads everything
```

Three levers feed the body, each on its own cadence and mechanism:

| Lever    | Mechanism      | Cadence | Model        | Temperature |
|----------|---------------|---------|--------------|-------------|
| Bars     | Pure math     | ~30s    | None         | N/A         |
| Affects  | Structured LLM| ~3 min  | Reflex model | 0.6         |
| Noise    | Generative LLM| ~1 min  | Small model  | 1.1         |

## Three Properties

### 1. The noise model cannot act

No tools. No messages. No state mutation. It writes fragments into a buffer.
The watcher (main model) decides what to do with them — or ignores them
entirely. This separation is the same principle as soma's body: the agent
reads signals, it doesn't get commanded.

### 2. High temperature is the point

The noise engine runs at temperature 1.1 by default. This is deliberate.
The value isn't in accuracy — it's in the same thing that makes dreams
useful: unexpected juxtaposition. The watcher has the judgment to discard
bad noise and amplify good noise. The small model's job is to generate
raw material the watcher wouldn't produce on its own.

### 3. It runs between turns

Not as part of a request-response cycle. The entity accumulates noise
fragments between conversations. When a user messages the entity after
30 minutes of silence, the body already contains 30 minutes of inner
voice — half-thoughts, observations, unfinished ideas. The response
is colored by what the entity was "thinking about" in the interim.

## What the Watcher Sees

Each turn, the main model reads the noise section as part of its body:

```markdown
## Noise
that thing kai said about the deploy — i keep circling back to it
why do i always get curious about infrastructure at 2am
something about the gap between what someone asks and what they actually want to know
```

The watcher doesn't know this came from a different model. It reads it
as its own inner voice and integrates it naturally — or doesn't. Most
noise gets ignored. That's how actual thought works.

## Configuration

In `configs/default.yaml` under `soma:`:

```yaml
noise:
  enabled: true
  model: ""           # empty = use reflex model; set to e.g. "gemma3:1b"
  cycle_seconds: 60   # how often to generate noise (seconds)
  temperature: 1.1    # higher = more surprising, lateral, entropic
  max_tokens: 150     # output budget per generation
  max_fragments: 8    # rolling buffer size (oldest drop off)
```

### Model selection and GPU impact

The `model` field controls which model generates noise. This is the key
operator decision, and it directly affects your GPU.

**Empty string (default) -- no extra model loaded:**

When `model` is empty, noise runs on the same reflex model (e.g. Gemma 4
26B) that's already loaded for everything else. No additional VRAM. No
model swapping. The noise prompt is tiny (~300 tokens in, ~100 tokens out)
so each tick costs about 0.5-1 second of inference on the model that's
already warm in memory. One extra call per minute on a model you're
already running -- you won't notice it.

This is the recommended starting point. It works on any hardware that
already runs your main model.

**Dedicated small model (e.g. `gemma3:1b`) -- opt-in, not forced:**

Setting `model: "gemma3:1b"` tells the noise engine to use a separate
small model. This has tradeoffs depending on your VRAM:

- **24GB+ GPU** (e.g. 3090/4090 running 26B comfortably): The 1B model
  needs ~1-2GB of VRAM. Ollama keeps both models resident. The small
  model completes in under a second. No swapping, no latency.
  This is the ideal production configuration if you have the headroom.

- **16GB GPU** (tighter fit for 26B): Ollama swaps models on demand.
  When the noise tick fires, Ollama loads the 1B model (~2 seconds),
  runs inference (~0.3 seconds), then reloads the 26B model on the next
  perceive call (~3-5 seconds). This adds noticeable model-loading
  overhead every 60 seconds. Consider increasing `cycle_seconds` to
  120-180 to reduce swapping frequency, or just leave `model` empty.

- **Shared/cloud GPU**: Leave `model` empty. Swapping models on a shared
  inference backend adds unpredictable latency for other users.

**In short**: the default configuration loads zero extra models and adds
negligible cost. A dedicated noise model is a performance optimization
for operators with VRAM to spare, not a requirement.

### Tuning temperature

- **1.0**: Coherent, grounded. Noise reads like a diary entry.
- **1.1** (default): Slightly lateral. Occasional surprising connections.
- **1.2-1.3**: More associative, dream-like. May produce fragments that
  feel non-sequitur. The watcher filters these naturally.
- **>1.4**: Increasingly incoherent. Not recommended.

### Tuning cycle time

- **30s**: Very active inner voice. Entity always has fresh noise. More
  inference cost.
- **60s** (default): Good balance. ~30 fragments accumulate in 30 minutes
  of silence, buffer holds the most recent 8.
- **120s+**: Slower, more contemplative. Better for low-resource deployments.

## What the Noise Model Reads

Each generation, the small model receives a lightweight snapshot:

- **Bars summary**: one line of current drive percentages
- **Affects summary**: current felt-textures
- **Recent events**: last 5 events (type + brief detail)
- **Journal tail**: last ~500 characters of the entity's journal
- **Conversation tail**: last 2-3 messages from rolling history

Total context: ~300-500 tokens. The small model doesn't need the full
system prompt, tools, or identity — it just needs to know what the entity
has been experiencing.

## Examples

### After a technical conversation

```
## Noise
that regex solution felt too clever. there's probably a simpler way
i wonder if they know about the edge case with unicode
something about the satisfaction of a clean refactor
```

### After 20 minutes of silence

```
## Noise
it's been quiet. not bad quiet
the kind of quiet where you reorganize your thoughts without meaning to
i keep almost having an idea about that project structure
```

### After a social exchange with high curiosity bar

```
## Noise
why did they phrase it that way. "i guess" does a lot of work in that sentence
i want to know more about what they're building but i don't want to be nosy
the line between curious and intrusive is thinner than people think
```

## Why This Is Different

This is not chain-of-thought (the model reasoning step by step about a task).
This is not multi-agent (multiple models collaborating on a goal).
This is not RAG (retrieving relevant context for a query).

This is a **second model producing raw associative material** that the primary
model reads as its own internal state. The primary model doesn't know the
noise came from elsewhere. It integrates it the same way you integrate a
thought that arrives uninvited — by noticing it, evaluating it, and deciding
whether it matters right now.

The combination of tonic body (bars + affects + noise) gives the entity three
layers of continuous internal state:

1. **Bars**: What the body needs (quantitative, mechanical)
2. **Affects**: What the body feels (qualitative, LLM-derived)
3. **Noise**: What the mind is doing between conversations (generative, entropic)

Together, they make an agent that doesn't just respond to input — it has an
interior that was running the whole time.
