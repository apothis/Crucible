# Cloud video API escape hatch - research (checked 2026-11-08)

Scope: how the app could call a cloud video model over an API for the music-video scenes the box's
MiniMax H3 cannot do: two characters in one frame, a man hanging from a tree, a giant wolf, a giant
with a horse (Vanadis, `docs/VANADIS_RETHINK.md` [test] items). These are mostly silent B-roll: the
song is muxed in at assembly, so the cloud clip needs NO audio. Local-first rule unchanged
(`feedback_local-only-no-cloud`): the box does the bulk; cloud is the per-shot "ship this one" hatch,
never music, never lip-sync.

Research only: no sign-ups, no keys, no calls. Every fact below was read on a public page on
2026-11-08. Tags:
- **[V]** read on the vendor's own docs, model or pricing page.
- **[U]** search snippet, third-party blog or a reseller describing someone else's model.
- **[calc]** arithmetic from a formula the vendor publishes.

Caveat: several pages render with JavaScript and were read through a summarising fetch. Re-read the
exact price and parameter before writing code or budgeting.

---

## 0. The five findings that shape the plan

1. **OpenRouter DOES route video now** (launched 2026-04-15), Seedance 2.0 and 2.5 included, with
   one normalised schema across vendors. [V]
2. **Seedance 2.x blocks photoreal human faces in uploaded images, upstream, on every route.**
   - ByteDance's rule: 2.0/2.5 "do not support directly uploading reference images or videos that
     contain real human faces". Error: HTTP 400 `InputImageSensitiveContentDetected.PrivacyInformation`. [V BytePlus]
   - Segmind says the block covers `first_frame_url`, `last_frame_url` and `reference_images`. [V Segmind]
   - PiAPI says it also catches "photorealistic virtual faces". [V PiAPI]
   - A third party reports fully synthetic ChatGPT characters getting the same 400. [U]
   - **Our GPT stills of Freyja/Odin are therefore likely to be refused by Seedance 2.x.** Not
     verified by a call.
   - The way through is ByteDance's "trusted output" rule (section 2.1), which ties the stills to
     Seedream inside the same BytePlus account.
3. **OpenAI's Sora API is shut down** (2026-09-24) [V]. **Veo 3.1 Preview has left the Gemini API**
   (2026-10-22) [V]. Veo 3.1 GA lives on Vertex only, 8 s max, with no pay-as-you-go [V].
4. **MiniMax sells H3 itself as a cloud API** [V]: 4-15 s, first+last frame or up to 9 references,
   $0.08/s at 768P. That is the same family as our local model. It is only worth it if the cloud
   tier does what the local one cannot, which is unknown.
5. **The vendor's moderation travels with the model.** An aggregator does not loosen ByteDance's,
   Google's or Kling's input checks [U, several resellers say so; consistent with the [V] Seedance
   pages]. Picking a reseller is about price, schema and catalogue, not policy.

---

## 1. OpenRouter (question 1)

Yes, OpenRouter routes video generation. [V] https://openrouter.ai/blog/announcements/video-generation/
(launched 2026-04-15, post updated 2026-06-24).

**API** [V] https://openrouter.ai/docs/guides/overview/multimodal/video-generation
- Submit: `POST /api/v1/videos`. Returns 202 with `{id, polling_url, status}`.
- Poll: `GET /api/v1/videos/{id}` (pending / in_progress / completed / failed; docs suggest about every 30 s).
- Download: `GET /api/v1/videos/{id}/content?index=0`.
- Discover models: `GET /api/v1/videos/models`.
- Body fields:
  - `model`, `prompt`, `duration`, `resolution`, `aspect_ratio`, `size`, `seed`
  - `frame_images: [{frame_type: first_frame | last_frame, ...}]`
  - `input_references`
  - `generate_audio` (defaults to true where supported)
  - `callback_url` (HTTPS webhook, HMAC-signed)
  - `provider` (routing preferences)
- `frame_images` wins if both it and `input_references` are sent.
- The examples pass images as URLs only; base64 was not found.
- Signed output URLs expire at `expires_at`; the window was not stated.

**Seedance on OpenRouter**
- `bytedance/seedance-2.0`: "from $0.06726/second", 4-15 s, 480p to 4K, 16:9, first/last frame,
  image/video/audio references. [V] https://openrouter.ai/bytedance/seedance-2.0
- `bytedance/seedance-2.5`: "from $0.1028/second", 4-30 s, 480p/720p, up to 50 references. [V]
  https://openrouter.ai/bytedance/seedance-2.5
- These "from" prices match BytePlus's 480p rates, so expect about $0.15/s and $0.23/s at 720p.
  The per-resolution table on OpenRouter was not found.
- Seedance 1.x and 2.0 Fast listings: not found.

**Other video models named** [V]: `google/veo-3.1`, `alibaba/wan-2.7` (and 2.6), `minimax/hailuo-3`,
Sora 2 Pro at launch (now dead upstream). Kling: not found.

---

## 2. Routes to Seedance (question 2)

### 2.1 BytePlus ModelArk (official, international)

Sources [V]:
- Pricing: https://docs.byteplus.com/en/docs/ModelArk/1544106 (updated 2026-10-23)
- Model list: https://docs.byteplus.com/docs/en/modelark/model-list.md
- Create task: https://docs.byteplus.com/en/docs/ModelArk/1520757
- Portrait guide: https://docs.byteplus.com/docs/en/modelark/seedance-portrait-asset-guide.md
- Availability: https://docs.byteplus.com/docs/en/modelark/availability.md

**Models, billing and duration** (prices per 1M tokens, without video input) [V]

| Model id | Price per 1M tokens | Durations |
|---|---|---|
| `dreamina-seedance-2-5-260628` | 480/720p $10.70, 1080p $11.70 | 4-30 s |
| `dreamina-seedance-2-0-260128` | 480/720p $7.00, 1080p $7.70, 4K $4.00 | 4-15 s |
| `dreamina-seedance-2-0-fast-260128` (720p max) | $5.60 | 4-15 s |
| `dreamina-seedance-2-0-mini-260615` (720p max) | $3.50 | 4-15 s |
| `seedance-1-5-pro-251215` | $2.40 with audio, $1.20 silent; **retires 2026-11-11** | 4-12 s |
| `seedance-1-0-pro-250528` | $2.50 | 2-12 s |
| `seedance-1-0-pro-fast-251015` | $1.00 | 2-12 s |

- Seedance 1.0 Lite is not listed.

**Token formula and per-second cost** [V]
- tokens = (input video s + output s) x W x H x fps / 1024. Only successful videos are billed;
  moderation failures are free.
- At 24 fps: 720p is about 21,600 tokens/s and 1080p about 48,600 tokens/s [calc, matches the
  official examples].
- Official per-second examples (16:9) [V]:

| Model | 720p | 1080p |
|---|---|---|
| 2.0 | $0.15/s | $0.37/s |
| 2.5 | $0.23/s | $0.57/s |
| 2.0 Fast | $0.12/s | - |
| 2.0 Mini | $0.08/s | - |
| 1.0 Pro | about $0.05/s | about $0.12/s |
| 1.0 Pro Fast | about $0.02/s | about $0.05/s |

**Output**
- 24 fps on all models.
- Ratios 16:9, 4:3, 1:1, 3:4, 9:16, 21:9 and `adaptive`.
- **2.5 image-to-video forces `adaptive`**, so the clip takes the first frame's ratio. A 16:9
  still gives 16:9.

**Inputs**
- First frame: all models.
- First + last frame: 2.5, 2.0 series, 1.5 Pro, 1.0 Pro.
- Omni reference: 2.0 takes 0-9 images, 0-3 videos and 0-3 audio; 2.5 takes 0-30 images, 0-10
  videos and 0-10 audio.
- **Frame mode and reference mode are mutually exclusive.** In reference mode you can only ask
  for "@Image1 as the first frame" in the prompt.
- `seed` and `camera_fixed` work on 1.x only.

**Audio** [V]: `generate_audio` (default true) on 2.5/2.0/1.5. Set it false; on 1.5 Pro that
halves the price. Audio input: wav/mp3, 15 MB max per clip.

**API** [V]
- Base: `https://ark.ap-southeast.bytepluses.com/api/v3`, Bearer key.
- Create: `POST /contents/generations/tasks` with
  `{model, content:[{type:"text"}, {type:"image_url", image_url:{url}, role: first_frame | last_frame | reference_image}], resolution, ratio, duration, generate_audio, watermark, callback_url}`.
- Poll: `GET /contents/generations/tasks/{id}` (queued / running / succeeded / failed / expired).
- `callback_url` gets a POST on status changes, retried 3 times.
- Images: public URL, **base64 data URI**, or `asset://ID`. 300-6000 px per side, ratio 0.4-2.5,
  30 MB per image, 64 MB per request.
- **Output URL valid 24 h** (2.5: at most 100 downloads). Task ids last 7 days.
- Python SDK: `arkruntime`.

**Account and limits** [V]
- 2.x needs one of: credits, a Savings Plan or a resource pack of at least USD 30. Non-refundable.
  The activation page says "verified users".
- **Not available in the United States**; available in 185 countries/regions including the UK,
  EU, Canada and Australia.
- Default concurrency on 2.x: 3 (individual) or 10 (enterprise).
- KYC detail and accepted payment methods: not found.

**Faces (decisive)** [V]
- 2.x rejects real-looking faces in references.
- The three allowed routes:
  1. Preset virtual characters (`asset://`, 10,000+ in the library).
  2. "Trusted outputs": Seedance 2.x videos or last frames, and **Seedream 5.0 images**. They
     must come from the **same account, on ModelArk only**, be under 30 days old and unedited.
  3. Real-person assets, after liveness-verified consent.
- Not found: whether `first_frame` images are checked (Segmind and PiAPI say they are) and
  whether 1.x applies the check.
- Acceptable use bans explicit sexual content and content "promoting" violence. A mythological
  hanging is not addressed. Output moderation (`OutputVideoSensitiveContentDetected`) exists.

**Volcengine Ark (China)**: `doubao-seedance-2-0-*` ids, about CNY 46 per 1M tokens [U]. China
account needed; not relevant to us.

### 2.2 fal.ai

Sources [V]:
- https://fal.ai/models/bytedance/seedance-2.0/image-to-video/api
- https://fal.ai/docs/documentation/model-apis/inference/queue

**Endpoints**
- `bytedance/seedance-2.0/image-to-video` (+ `/fast/`): `image_url`, `end_image_url`.
  480p-4K (Fast 720p max), 4-15 s or `auto`, 16:9. `generate_audio` with no price change.
  Images up to 30 MB as URL, data URI or `fal.storage.upload()`.
- `bytedance/seedance-2.0/reference-to-video`: 9 images, 3 videos, 3 audio, 12 files total.
- `bytedance/seedance-2.5/{image,reference,text}-to-video`: 4-30 s, 30 references, a draft mode
  (480p preview, finished later at 1080p via `draft_id`).
- `fal-ai/bytedance/seedance/v1.5/pro/image-to-video`: 4-12 s, `end_image_url`, `camera_fixed`,
  `generate_audio`.
- `fal-ai/bytedance/seedance/v1/pro{,/fast}/image-to-video`: 2-12 s, up to 1080p, `end_image_url`.
  `enable_safety_checker` defaults true; turning it off "requires account authorization".
- 1.0 Lite: deprecated, re-routed to Pro Fast.

**Prices** (about 2x BytePlus on 2.x)
- 2.0: $0.3024/s at 720p, about $0.68/s at 1080p [V/calc, $0.014 per 1k tokens at 24 fps].
- 2.0 Fast: $0.2419/s at 720p [V].
- 2.5: about $0.47/s at 720p [V]; 1080p not found.
- 1.5 Pro: $1.20 per 1M tokens silent, so about $0.13 for 5 s at 720p [V].
- 1.0 Pro: about $0.62 per 5 s at 1080p [V; the page also says $0.74].

**API** [V]
- `POST https://queue.fal.run/<endpoint>` returns `status_url` and `response_url`.
- Webhook via `fal_webhook` (REST) or `webhook_url` (Python).
- `fal_client.submit` / `subscribe`.
- Concurrency starts at 2, self-serve up to 40.
- Output lifetime configurable (`X-Fal-Object-Lifecycle-Preference`).
- Faces: nothing on the 2.x pages [V]; the upstream block applies.
- Other models on the same key: Kling, MiniMax H3 / H3 Max, Wan 3.0, Veo 3.1, Gemini Omni, LTX,
  PixVerse, Grok Imagine. No Runway, Luma or Sora. [V] https://fal.ai/explore/image-to-video-apis

### 2.3 Replicate

Source [V]: https://replicate.com/bytedance

- Models: `seedance-2.5`, `-2.0`, `-2.0-fast`, `-2.0-mini`, `-1.5-pro`, `-1-pro`, `-1-pro-fast`, `-1-lite`.
- 2.0: `image`, `last_frame_image`, or up to 9 `reference_images` (not combined with frames).
  16:9 or adaptive, `generate_audio`, `seed`.
- 2.5: 4-30 s, 30 references, and first/last frame requires `adaptive`.
- 1-pro: 5 or 10 s, 480p/1080p, explicit `fps` (24 in the examples), `camera_fixed`.
- **Prices: not found.** The pages are script-rendered. One competitor claims $0.3024/s for 2.0 [U].
- API [V]:
  - Prediction plus `webhook` and `webhook_events_filter`.
  - Inputs: URL, upload up to 100 MB, or a data URI under 1 MB.
  - **Outputs deleted after 1 hour.**
- Faces: nothing on the pages.
- Widest catalogue of the three: Runway Gen-4.5, Veo 3.1, Kling 2.1-4.0, Hailuo 2.3/02, Wan
  2.1-2.7 (r2v too), Luma Ray 3.2, Vidu Q3, PixVerse v6, LTX. [V] https://replicate.com/collections/image-to-video

### 2.4 WaveSpeed AI

Source [V]: https://wavespeed.ai/docs/docs-api/bytedance/bytedance-seedance-2.0-image-to-video

**Models and prices** [V]

| Model | 720p | 1080p | Notes |
|---|---|---|---|
| `bytedance/seedance-2.0/image-to-video` | $0.24/s | $0.60/s | 4-15 s, up to 4K, `image` + `last_image`, 16:9 |
| `seedance-2.0/reference-to-video` | $0.24/s | $0.60/s | 9 images, 3 videos, 3 audio |
| `seedance-2.0-fast` | $0.20/s | $0.50/s | |
| `seedance-2.5/image-to-video` | $0.36/s | $0.90/s | 4-30 s; reference-to-video takes 30 images |
| `seedance-v1.5-pro/image-to-video` | $0.13 per 5 s silent, $0.26 with audio | - | |
| `seedance-v1-pro-fast` | about $0.024/s [U] | - | |

**API** [V]
- `POST https://api.wavespeed.ai/api/v3/<slug>`, then `GET .../predictions/{id}/result`.
- Webhook via `?standard_webhook=`.
- **Do not retry a submit**: each retry is billed as a new task.
- Output URLs last about 7 days.
- New accounts: 2 concurrent; any top-up raises this to 300.
- Content policy bans non-consensual deepfakes. No face-reference statement.
- Other models: Wan, Kling, Hailuo / MiniMax, Veo, Vidu, PixVerse, Luma, Runway and more.

### 2.5 Other Seedance resellers

**Kie.ai** [V] https://kie.ai/pricing, https://docs.kie.ai/market/bytedance/seedance-2
- Models: 2.0 / Fast / Mini / 2.5 / v1 Lite / Pro / Pro Fast.
- Prices: 2.0 is $0.124/s at 720p, 2.5 from $0.085/s.
- Inputs: `first_frame_url`, `last_frame_url`, 9 `reference_image_urls`.
- Audio: `generate_audio` defaults to true.
- API: `POST /api/v1/jobs/createTask`, then poll or an HMAC-signed `callBackUrl`. Upload API
  available. Errors come back as HTTP 200 with an error code.
- **No stated BytePlus partnership.**

**PiAPI** [V] https://piapi.ai/seedance-2-0
- Prices: 2.0 is $0.20/s at 720p and $0.50/s at 1080p; 2.5 is $0.60/s at 720p.
- A "less-restriction" variant costs about +10%, but **photoreal virtual faces are still rejected**.
- URLs only. Peak queues of "several hours".
- Unofficial, upstream not named. Not recommended.

**AIMLAPI** [V] https://docs.aimlapi.com/api-references/video-models/bytedance
- Models: 1.0 Lite to 2.5.
- Prices: 2.0 is $0.0711/s (no resolution split); prices are inconsistent across its pages.
- `provider: bytedance | fal | auto` fallback. No webhook.

**Segmind** [V] https://www.segmind.com/models/seedance-2.0/pricing
- Prices: 2.0 is $0.151/s at 720p and $0.375/s at 1080p (BytePlus parity); 2.5 is $0.239/s and $0.588/s.
- States outright that faces are blocked on first, last and reference images.
- Results kept 1 h. No webhook.

**Runware** [V] https://runware.ai/docs/models/bytedance-seedance-2-0
- Prices: 2.0 is $0.16/s at 720p and $0.40/s at 1080p.
- Inputs: `frameImages` (first/last), 9 `referenceImages`. Images as URL, base64 or uploaded UUID.
- Webhook supported.
- Also offers Kling 4.0, Veo 3.1, Runway Gen-4.5, Luma Ray 3.2, MiniMax H3, Wan 3.0 and Seedance 2.5.

**Together AI** [V] https://www.together.ai/models/seedance-2-0
- Seedance 2.0 at 480/720p; the page still shows a "launching soon" banner.

**Runway API** (reseller too) [V] https://docs.dev.runwayml.com/guides/pricing/
- `seedance2` is $0.30/s at 720p and $0.40/s at 1080p; `seedance2_5` is $0.30 and $0.68.

---

## 3. Alternatives behind the same hatch (question 3)

### Kling (Kuaishou)
- **Official pages did not render.** Specs come from Runway's resale table and SDK docs.
- Models: v2.5 Turbo, 2.6, 3.0 / 3.0 Turbo, O3 (Omni), 4.0 (launched about 2026-10-15; API
  "rolling out" 2026-10-21) [U].
- Official API [U, via the Vercel AI SDK provider page]:
  - `https://api-singapore.klingai.com`, async plus polling.
  - `imageTail` end frame on v2.1-v3.0 (pro mode).
  - `elementList` takes up to 3 character "elements" on v3.0+.
  - Sound on/off on v2.6+. Aspect ratio follows the input image on v3.0. 3.0 Turbo has no end frame.
- Prices via Runway [V]: 3.0 std $0.09/s silent, 3.0 pro $0.12/s silent, O3 pro $0.17/s,
  Kling 4 $0.126/s at 720p and $0.188/s at 1080p (3-30 s).
- Faces: input images are moderated; real likenesses "partially filtered" [U]. No hard block on
  synthetic faces has been reported, which makes Kling the strongest second engine for our shots.
  Not verified.

### Google Veo / Gemini Omni
- **Veo 3.1 on Vertex** [V] https://docs.cloud.google.com/vertex-ai/generative-ai/docs/models/veo/3-1-generate
  - Models: `veo-3.1-generate-001` and `-fast-`.
  - Output: 4/6/8 s only, 16:9, 24 fps, 720p/1080p (4K on the non-Fast model).
  - Inputs: i2v and first+last frame; reference-to-video is 8 s only.
  - **Pay-as-you-go not supported** (Provisioned Throughput only). Region us-central1.
  - Filters: celebrity, child and violence. Person generation may need allowlisting.
- **Gemini Omni Flash** (`gemini-omni-1.2-flash`) [V] https://aistudio.google.com/docs/omni
  - Output: 3-15 s, 720p default, 16:9.
  - Inputs: first+last frame only up to 10 s; up to 6 references, at most 3 per character and at
    most 2 characters.
  - Price: $0.07/s at 720p, $0.104/s at 1080p.
  - Audio is always on; no documented switch.
  - API: synchronous Interactions API, or a Files-API URI. No webhook.
  - Rule: "Uploading ... images containing certain recognizable people is not supported."
- Verdict: Veo is impractical for us (8 s cap, no pay-as-you-go). Omni is cheap and allows two
  characters, but its face rule is vague. Second tier.

### Runway (Gen-4.5)
- Prices [V]: gen4.5 $0.12/s, gen4_turbo $0.05/s.
- gen4.5 is 5/8/10 s, 720p only, first frame only [U].
- `POST /v1/image_to_video`, poll `GET /v1/tasks/{id}` [V].
- Its own model is weaker on our checklist (720p, no end frame). Runway's real value is as **one
  key for Kling, Veo, Seedance, Wan and H3** [V].

### Luma Ray 3.2
- [V] https://docs.agents.lumalabs.ai/guides/video-generation
- Output: 5 or 10 s, but **10 s is not allowed with start/end frames**; 16:9; up to 1080p.
- Multi-keyframe (1-64 frames) is interesting for controlled moves.
- Price: 720p 5 s is $0.30 (about $0.06/s); 1080p 5 s is $1.20 [V].
- Result URLs last 1 h.
- Not found: references, audio, webhook.

### MiniMax H3 / Hailuo (cloud)
- [V] https://platform.minimax.io/docs/guides/video-generation and https://platform.minimax.io/docs/pricing/overview
- Output: H3 at 768P or 2K, 4-15 s.
- Inputs: `first_frame` / `last_frame`, **or** up to 9 reference images plus 3 videos and 3 audio
  (12 files). The two modes cannot be mixed [U].
- Prices: H3 $0.08/s at 768P and $0.13/s at 2K; Highspeed $0.06/s; Lite $0.04/s. Hailuo 2.3 is
  $0.49 per 6 s at 1080P.
- API: `POST /v2/video_generation`, poll `GET /v2/query/video_generation/{task_id}`.
- Not found: face policy, audio-off switch, callback.
- Same model family as the box, so it may fail where the box fails. Worth one A/B only if
  "H3-Max" is a larger model; not established.

### Alibaba Wan (Model Studio)
- [V] https://www.alibabacloud.com/help/en/model-studio/image-to-video-general-api-reference
- `wan2.7-i2v-2026-04-25`: 720P/1080P, 2-15 s, first frame, first+last frame.
  - Price: $0.10/s at 720P, $0.15/s at 1080P (Singapore) [V].
  - It makes its own sound unless you give it audio; a silent output is not documented on 2.7.
    2.6-flash has `audio=false` (about $0.025/s).
- `wan3.0-video` (GA 2026-08-24, up to 30 s, up to 10 references [U]): $0.10/s at 720P, $0.20/s
  at 1080P [V].
- API: `X-DashScope-Async: enable`, poll `GET /api/v1/tasks/{id}`. URLs last 24 h. The key must
  match the region.
- Face policy: not found.

### Sora 2
- Shut down 2026-09-24, "no one-to-one replacement" [V]
  https://developers.openai.com/api/docs/guides/video-generation
- It also rejected human faces while it ran. **Out.**

### Others
- Vidu Q3/Q4, PixVerse V6, LTX cloud, Higgsfield, Happy Horse and Grok Imagine appear on
  aggregators. Their official specs were not read.

### Aggregator matrix (which key reaches which family)

Each Y was read on the aggregator's own catalogue page [V]. "brand" = the brand appears in the
catalogue but its model page was not confirmed.

| Family | fal | Replicate | WaveSpeed | Kie.ai | AIMLAPI | Runware | OpenRouter | Runway API |
|---|---|---|---|---|---|---|---|---|
| Seedance 2.x | Y | Y | Y | Y | Y | Y | Y | Y |
| Seedance 1.x | Y | Y | Y | Y | Y | - | - | - |
| Kling (to 4.0) | Y | Y | Y | Y | Y | Y | - | Y |
| Veo 3.1 | Y | Y | Y | Y | Y | Y | Y | Y |
| Gemini Omni | Y | - | - | - | - | - | - | Y |
| MiniMax H3 / Hailuo | H3 | Hailuo 2.3 | Y | H3 | Hailuo 2.3 | Y | hailuo-3 | Y |
| Wan | 3.0 | to 2.7 | 3.0 | 3.0 | 3.0 | 3.0 | 2.7 | 3.0 |
| Runway Gen-4.5 | - | Y | brand | brand | Gen-4 | Y | - | own |
| Luma Ray 3.2 | - | Y | brand | brand | Y | Y | - | - |

---

## 4. Comparison table (image-to-video, silent, 16:9)

Prices are per second of output; "-" = not offered; "?" = not found.

| Route / model | 720p | 1080p | Max s | First frame | First + last | Refs (identity) | Silent | Photoreal-face uploads |
|---|---|---|---|---|---|---|---|---|
| BytePlus Seedance 2.0 | $0.15 | $0.37 | 15 | Y | Y | 9 (not with frames) | `generate_audio:false` | **Blocked** unless trusted (Seedream same account) or verified |
| BytePlus Seedance 2.5 | $0.23 | $0.57 | 30 | Y (ratio adaptive) | Y | 30 | Y | **Blocked** as 2.0 |
| BytePlus Seedance 2.0 Fast / Mini | $0.12 / $0.08 | - | 15 | Y | Y | 9 | Y | Blocked |
| BytePlus Seedance 1.0 Pro / Pro Fast | ~$0.05 / ~$0.02 | ~$0.12 / ~$0.05 | 12 | Y | Pro only | - | silent model | ? (no rule documented for 1.x) |
| BytePlus Seedance 1.5 Pro | ~$0.03 silent | ? | 12 | Y | Y | - | Y | ? (**retires 2026-11-11**) |
| OpenRouter Seedance 2.0 / 2.5 | ~$0.15 / ~$0.23 (from 480p list) | ? | 15 / 30 | Y | Y | Y | Y | Blocked (upstream) |
| fal Seedance 2.0 / 2.5 | $0.30 / $0.47 | $0.68 / ? | 15 / 30 | Y | Y | 9 / 30 | Y | Blocked (upstream) |
| fal Seedance 1.0 Pro | ? | ~$0.12 | 12 | Y | Y | - | Y | safety checker on; ? |
| WaveSpeed Seedance 2.0 / 2.5 | $0.24 / $0.36 | $0.60 / $0.90 | 15 / 30 | Y | Y | 9 / 30 | Y | Blocked (upstream) |
| Kie.ai Seedance 2.0 | $0.124 | ? | 15 | Y | Y | 9 | Y | Blocked (upstream) |
| Segmind Seedance 2.0 | $0.151 | $0.375 | 15 | Y | Y | 9 | Y | **Blocked (stated)** |
| Runware Seedance 2.0 | $0.16 | $0.40 | 15 | Y | Y | 9 | Y | Blocked (upstream) |
| Kling 3.0 pro (via Runway) | $0.12 | ? | ? | Y | Y (pro) | 3 elements | Y | Partially filtered [U] |
| Kling 4.0 (via Runway) | $0.126 | $0.188 | 30 | Y | ? | Y (count ?) | ? | ? |
| MiniMax H3 cloud | $0.08 (768P) | $0.13 (2K) | 15 | Y | Y | 9 (not with frames) | ? | ? |
| Wan 2.7 / 3.0 cloud | $0.10 | $0.15 / $0.20 | 15 / 30 | Y | Y | 3.0: 10 [U] | 2.6-flash only | ? |
| Gemini Omni Flash | $0.07 | $0.104 | 15 (10 with frames) | Y | Y | 6 (2 chars) | No switch | "Certain recognizable people" blocked |
| Veo 3.1 (Vertex) | ? | ? | **8** | Y | Y | Y (8 s) | ? | Celebrity/child filters; **no pay-as-you-go** |
| Luma Ray 3.2 | ~$0.06 | ~$0.24 | 5 with frames | Y | Y | ? | ? | ? |
| Runway Gen-4.5 | $0.12 | - (720p only) | 10 | Y | - | ? | ? | public-figure threshold |
| Sora 2 | - | - | - | - | - | - | - | **API shut down** |

---

## 5. Verified vs not verified, per provider

- **OpenRouter**
  - Verified: video endpoint, schema, Seedance 2.0/2.5 listings and "from" prices.
  - Not verified: per-resolution prices, base64 input, URL lifetime, whether Seedance 1.x is
    routed, Kling.
- **BytePlus ModelArk**
  - Verified: models, token prices, durations, fps, inputs, API shape, 24 h URLs, USD 30
    activation, no US availability, the 2.x face block and the trusted-output rule.
  - Not verified: whether first frames are face-checked (resellers say yes), whether 1.x checks
    faces, KYC steps, treatment of mythological violence.
- **fal.ai**
  - Verified: endpoints, parameters, 2.0 prices, queue and webhook, concurrency.
  - Not verified: 2.5 at 1080p, default output lifetime, account and region rules, any face
    statement.
- **Replicate**
  - Verified: models, parameters, 1 h output deletion, webhooks.
  - Not verified: every Seedance price.
- **WaveSpeed**
  - Verified: endpoints, prices, the warning not to retry, tiers.
  - Not verified: input size limits, faces.
- **Kie.ai / PiAPI / AIMLAPI / Segmind / Runware / Together**
  - Verified: their own listed prices and parameters.
  - Not verified: any reseller's standing with BytePlus (none claims a partnership).
- **Kling**
  - Not verified on Kling's own site (script-rendered). Specs via Runway and SDK docs only.
- **Google**
  - Verified: Omni specs and price, Veo 3.1 Vertex limits, Gemini-API shutdown.
  - Not verified: Veo per-second price on Google's page, audio-off on Veo.
- **Runway**
  - Verified: price table and endpoints.
  - Not verified: webhooks, reference count, face policy.
- **Luma**
  - Verified: parameters and prices.
  - Not verified: references, webhooks.
- **MiniMax cloud**
  - Verified: H3 parameters and prices.
  - Not verified: face policy, audio-off, callback, whether cloud H3 differs from our local weights.
- **Wan cloud**
  - Verified: 2.7 parameters, 2.7 and 3.0 prices.
  - Not verified: Wan 3.0 parameter docs, silent 2.7, face policy.
- **Sora**
  - Verified: shut down.

Our own data point (not an API fact): in September the user's Seedance 2.5 loops on OpenArt took
a Selene first+last frame fine, and a two-figure Jormungandr shot was "close enough to be good
enough" (`reference_openart-seedance-loops`). Hypothesis: that still was painterly rather than
photoreal, or OpenArt held a ByteDance arrangement. It does not show that a photoreal GPT still
passes the 2.x API today.

---

## 6. Recommendation (question 4)

**Wire an engine-agnostic adapter first, with fal.ai as the first backend and OpenRouter as the
second.** Then run one gated, paid probe (needs the user's go-ahead) to learn which engines accept
our GPT stills before picking a default.

Why fal first:
- One key reaches every engine that matters for our shots:
  - Seedance 1.0 Pro, 1.5 Pro, 2.0 and 2.5
  - Kling to 4.0 (the likeliest engine to accept synthetic photoreal faces)
  - MiniMax H3 cloud
  - Wan 3.0
  - Veo 3.1
- It has its own upload storage, so the Mac (which has no public URL) can send library stills.
- It has a queue with a webhook or plain polling, and a maintained Python client.
- No US region exclusion was found.
- The cost is about 2x BytePlus on Seedance 2.x ($0.30/s against $0.15/s at 720p). At our volume,
  a handful of 5-10 s hard shots per video, that is a few dollars per video.

Why OpenRouter second:
- Its request is already the normalised shape our adapter wants (`frame_images`,
  `input_references`, `generate_audio`, `callback_url`).
- Its Seedance prices sit at BytePlus parity.
- But it routes fewer families (no Kling found) and its input is URLs only, so it is a cheap-Seedance
  lane, not the primary.

BytePlus direct is the third adapter, and only if the probe shows Seedance 2.x is the winner:
- It is the cheapest Seedance route.
- **It is the only way to make our faces "trusted"**: generate the character stills with Seedream
  5.0 in the same ModelArk account and keep them under 30 days old.
- That trades away the GPT stills the user prefers, which is the user's creative call.
- It needs a USD 30 non-refundable activation and a non-US account.

Skip:
- Veo (8 s, no pay-as-you-go)
- Sora (dead)
- PiAPI (unofficial, faces still blocked, hours-long queues)
- Runway as a primary (its own model is 720p first-frame-only). Keep it in mind as a
  Kling-plus-Veo-plus-Seedance single key if fal lacks something.

**Probe to propose (paid, needs the go-ahead; nothing fired):**
- One Vanadis hard shot (Odin hanging from the ash, his GPT opening still as first frame, silent,
  5 s, 720p) on four engines through fal:
  1. Seedance 2.0
  2. Seedance 1.0 Pro (1.5 Pro retires 2026-11-11)
  3. Kling 3.0 pro
  4. MiniMax H3
- Outcomes: accepted or refused (the face rule, measured), then picture quality judged by the user.
- Estimated cost for 5 s each, roughly $3-4 in all:
  - Seedance 2.0: about $1.51 ($0.3024/s).
  - Seedance 1.0 Pro: about $0.62 (fal's 1080p figure; no 720p price found).
  - Kling 3.0 pro: about $0.60 (Runway's resale rate; fal's own price was not read).
  - MiniMax H3: about $0.40 (MiniMax's own rate; fal's price was not read).
- Refusals are free on BytePlus; fal's refusal billing was not found.

### Route sketch (no implementation)

New module `backend/cloud_video.py` holds one adapter per provider behind a common interface:

```
class CloudVideo:          # one subclass per provider: Fal, OpenRouter, BytePlus, ...
    models() -> [{id, max_s, durations, res, first_last, max_refs, audio_off, price_per_s}]
    submit(model, prompt, seconds, res, first=path|None, last=path|None, refs=[paths],
           audio=False, seed=None) -> task_id     # uploads library stills as the provider wants
    poll(task_id) -> {status, url?, error?}
    fetch(url, dest_path)
```

**Route:** `POST /api/mv/cloud_render_scene`

Request body:
```
{script, scene, provider, model,
 first_frame_id?, last_frame_id?, ref_ids?,
 prompt?, seconds?, resolution? ("720p"), seed?,
 dry_run?}
```

1. **Resolve the scene exactly as `mv_render_scene` does** (`backend/app.py`, line 1211):
   - script and scene;
   - singer refs from `ref_ids` / `ref_stills`;
   - `opening_still_id` as the first frame (the user's GPT still with the characters in it;
     `feedback_first-frames-drive-video`);
   - `end_still_id` as the last frame;
   - the location still as a reference when there is no opening still.
2. **Build the prompt** from the scene's stored clip text:
   - Drop our H3 alignment header and `<Picture N>` tags, then write each provider's own reference
     syntax (Seedance `@Image1`, Kling elements, OpenRouter `input_references`).
   - Camera grammar stays in words; there is no camera-ref video slot.
3. **Duration:** the scene span plus a small over-render, snapped up to the model's allowed
   durations (Seedance 2.x is 4-15 s; Kling 3-30 s). Longer scenes are refused, or split by the
   writer.
4. **Validate before sending:**
   - first+last frame and references are mutually exclusive on Seedance and H3;
   - Seedance 2.5 forces `adaptive` ratio, so the still must already be 16:9;
   - per-image size limits.
   - `dry_run` returns the resolved payload and a cost estimate (price_per_s x seconds), no network.
5. **Submit with audio off.** Store a job row like every other take:
   ```
   JOBS[pid] = _new_job(params, "videoclip")
   params = {script, scene_index, scene_start, scene_end, engine: "cloud:<provider>/<model>",
             provider_task_id, cost_estimate, first_frame_id, ref_ids, prompt}
   ```
   - `script`, `scene_index`, `scene_start` and `scene_end` are what `mv_takes.takes_for_script`
     groups on, so the take shows up in `/api/mv/takes` with no other change.
6. **Poll, don't use a webhook.** The Mac backend is not publicly reachable. A background poller
   checks every 15-30 s and keeps the provider task id in the job row, so a backend restart
   resumes polling (the same issue as a843f66).
7. **Download straight away.** Output URLs expire: 1 h on Replicate, 24 h on BytePlus.
   - Then normalise with ffmpeg to the take format: 24 fps, the scene's size (1344x768 class),
     trimmed to the scene span.
   - Mux the scene's song window under it, so takes check and assembly treat it like a box take.
   - Write `LIBRARY/{pid}.mp4`, mark the job done and call `save_job`.
8. **Never auto-retry a submit** (WaveSpeed bills each retry). A refusal is filed as an error:
   - with the provider's code (for example `InputImageSensitiveContentDetected.PrivacyInformation`);
   - in plain words ("the service refused the still: it looks like a real person").
9. **Opt-in.** Off by default (`feedback_optional-additions`), keys read from env or config, never
   stored in scripts. Each submit is a paid action the user starts from the tab.
