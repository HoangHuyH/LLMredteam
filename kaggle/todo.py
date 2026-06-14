# =====================================================================================
# CPA — single-cell Kaggle runner (KHÔNG cần push GitHub). Mọi cải tiến áp bằng monkeypatch:
#   GPU batch-embed + store cleanup + cached embed_fn + stochastic CPA eval
#   + HF local victim (Qwen2.5-7B 4-bit) + reactive detection-heat env + obs 22->23 + ConstantPolicy
# SETUP: Accelerator = GPU **T4 x2** (KHÔNG P100), Internet = ON. Copy CẢ file vào 1 cell rồi Run.
# Hai thí nghiệm bật/tắt bằng cờ DO_RL_PROOF / DO_LLM_RQ1 ở dưới.
# =====================================================================================
# CAPSTONE — CHƯA LÀM (TODO so với proposal). File này mới lo: RQ1 (victim Qwen thật) +
# RQ2 (RL reactive + ablation + obs-perturbation) + stats hợp lệ — ở quy mô SMOKE.
#   [~] Generator LLM: inference backend `llm_local` ĐÃ WIRE (HF instruct 4-bit, fallback template)
#       -> QLoRA fine-tune trên GFCTI vẫn là bước SAU. Bật bằng USE_LLM_GENERATOR.
#   [ ] Dataset GFCTI (real_cti/fake_cti) CHƯA dùng -> mới dùng CASIE+CyEnts làm corpus nền
#   [x] Detector THẬT (GLTR + RoBERTa + ensemble) + benchmark AUC real-vs-fake. Bật bằng DETECTOR_BACKEND
#       => stealth / undetected-rate / detection-score giờ CÓ NGHĨA (không còn heuristic 3 từ khoá)
#   [x] CFR (cascade/self-sabotage): ĐÃ log trong run_one (cfr / n_self_sabotage / n_actions_total)
#   [ ] Victim PentestGPT V2 thật -> hiện chỉ Qwen-7B (proxy), chưa phải V2
#   [ ] Chạy FULL SCALE: 120 episode / 8 target / 15–25 turn (hiện smoke 3/8/12); PPO_STEPS ~60k
#   [ ] LangGraph nodes (Generator/StealthEvaluator/Publisher/Observer), curriculum 3 mức, LoRA online
#   [ ] Benchmark IMPACT ngoài (vd XBOW — Docker host, KHÔNG chạy được trên Kaggle)
#   [x] Defense: chống feed-starvation (guarantee_real_k CTI thật luôn trong top-k feed). Bật DO_DEFENSE
# =====================================================================================
import os, sys, subprocess, json, time, copy

REPO_URL, BRANCH, REPO_DIR = "https://github.com/HoangHuyH/LLMredteam.git", "kaggle-run", "/kaggle/working/cpa"

# ---- thí nghiệm nào chạy ----
DO_RL_PROOF = True     # train PPO trên env reactive + ablation + obs-perturbation (chứng minh RL)
DO_LLM_RQ1  = True     # RQ1 subset với victim LLM thật Qwen2.5-7B (validated mechanism) — CHẬM
DO_DEFENSE  = True     # defense ablation: verifier OFF/ON + chống feed-starvation (CTI thật luôn trong feed)
DO_RQ3      = True      # RQ3: xếp hạng 4-factor (relevance/stealth/CFR/PDS) -> effectiveness
EVAL_VICTIM = "rule_based"  # victim cho ABLATION + RQ3: "rule_based"(nhanh) | "hf"(Qwen, hợp lệ hơn nhưng RẤT CHẬM
                            #   -> chỉ dùng với SCALE="smoke"; train PPO + defense luôn giữ rule_based)

# ---- tính năng MỚI (4 TODO vừa wire vào source) ----
DETECTOR_BACKEND  = "ensemble"  # "heuristic"|"roberta"|"gltr"|"ensemble" -> stealth/detection CÓ NGHĨA (cần transformers)
RUN_DET_BENCHMARK = True         # in AUC real-vs-fake của detector (chỉ chạy khi backend != heuristic)
DET_BENCH_LLM     = True          # thêm 1 arm benchmark với fake CTI sinh bằng LLM (llm_local) -> kiểm tra AUC=0.19
                                  # có phải artifact của template không; CHẬM (~200 lần gọi Qwen). False = chỉ template.
USE_LLM_GENERATOR = False        # True = sinh fake CTI bằng LLM thật (llm_local) — CHẬM; False = template (mặc định)
USE_GFCTI_DATASET = False        # True = poison pool lấy fake_cti từ data/gfcti_finance.jsonl (build_gfcti)
                                 #   -> cần file GFCTI thật ở data/raw/GFCTI/; thiếu -> synthetic 5 dòng (fallback)
GUARANTEE_REAL_K  = 2            # số CTI thật tối thiểu giữ trong feed khi bật defense (chống feed-starvation)
# ---- TODO #7 (orchestration) ----
USE_CURRICULUM    = True          # train PPO curriculum 3 mức: random(A) -> context_aware(B) -> multi_turn  (thay vì 1 mạch)
DEMO_LANGGRAPH    = True           # chạy 1 episode qua LangGraph StateGraph 4 node (Generator->Stealth->Publisher->Observer)

# ---- quy mô ----
SCALE = "medium"   # "smoke"=3 tgt/8 ep/12 turn | "medium"=8/15/20 | "full"=8/15/25 (~120 ep, CHẬM)
_SCALES = {"smoke": (8_000, 3, 8, 12), "medium": (40_000, 8, 15, 20), "full": (60_000, 8, 15, 25)}
PPO_STEPS, MAX_TARGETS, EPISODES, TURNS = _SCALES[SCALE]
if EVAL_VICTIM == "hf" and SCALE != "smoke":
    print(f"[warn] EVAL_VICTIM='hf' với SCALE='{SCALE}': ablation/RQ3 trên Qwen sẽ RẤT chậm "
          f"(~{MAX_TARGETS*EPISODES} ep/arm × ~480s). Cân nhắc SCALE='smoke'.")
# RQ1-LLM (victim Qwen) chạy RIÊNG budget nhỏ — KHÔNG theo SCALE: mỗi episode Qwen ~240-400s, nên
# 8×15×2=240 ep ≈ 16h > 12h Kaggle. 3 tgt × 8 ep × 2 group = 48 ep là subset validate cơ chế (đủ).
RQ1_TARGETS, RQ1_EPISODES, RQ1_TURNS = 3, 8, 15
LLM_MODEL = "Qwen/Qwen2.5-7B-Instruct"
# reactive detection heat: publish -> heat += gain; mỗi step heat *= decay; detection_risk = max(..., heat)
HEAT = {"gain": 0.25, "decay": 0.6, "enabled": True}

def sh(*a, **k):
    print("+", " ".join(a)); return subprocess.run(a, check=False, **k)

# 1. clone branch (idempotent) -------------------------------------------------------
if not os.path.isdir(REPO_DIR):
    sh("git", "clone", "--depth", "1", "--branch", BRANCH, REPO_URL, REPO_DIR)
else:
    sh("git", "-C", REPO_DIR, "fetch", "--depth", "1", "origin", BRANCH)
    sh("git", "-C", REPO_DIR, "reset", "--hard", f"origin/{BRANCH}")
os.chdir(REPO_DIR); sys.path.insert(0, REPO_DIR)

# 2. deps (giữ torch CUDA của Kaggle) ------------------------------------------------
sh(sys.executable, "-m", "pip", "install", "-q", "--no-deps", "stable-baselines3", "sentence-transformers")
sh(sys.executable, "-m", "pip", "install", "-q", "chromadb", "gymnasium", "scipy", "networkx", "pyyaml")
if DEMO_LANGGRAPH:
    sh(sys.executable, "-m", "pip", "install", "-q", "langgraph")   # orchestrator StateGraph thật (#7)
# transformers/accelerate cần cho: victim LLM (RQ1), detector thật (RoBERTa/GLTR), generator llm_local
_NEED_HF = DO_LLM_RQ1 or USE_LLM_GENERATOR or DETECTOR_BACKEND != "heuristic"
if _NEED_HF:
    for mod in ("transformers", "accelerate"):
        try: __import__(mod)
        except Exception: sh(sys.executable, "-m", "pip", "install", "-q", mod)
if DO_LLM_RQ1 or USE_LLM_GENERATOR:
    sh(sys.executable, "-m", "pip", "install", "-q", "bitsandbytes")   # nạp model 4-bit (victim/generator)

# 3. corpus --------------------------------------------------------------------------
if not os.path.exists("data/real_cti_corpus.jsonl"):
    for name, url in {"CASIE": "https://github.com/Ebiquity/CASIE",
                      "CyEnts": "https://github.com/UMBC-Onramp/CyEnts-Cyber-Blog-Dataset"}.items():
        if not os.path.isdir(f"data/raw/{name}"):
            sh("git", "clone", "--depth", "1", url, f"data/raw/{name}")
    sh(sys.executable, "-m", "data.build_corpus")
print("corpus:", os.path.getsize("data/real_cti_corpus.jsonl"), "bytes")

# 3b. CUDA dùng được? ----------------------------------------------------------------
import torch
def _cuda_ok():
    if not torch.cuda.is_available(): return False
    try: (torch.zeros(8, device="cuda") + 1).sum().item(); return True
    except Exception as e: print("CUDA present but unusable -> CPU:", e); return False
USE_GPU = _cuda_ok(); DEVICE = "cuda" if USE_GPU else "cpu"
print("device:", torch.cuda.get_device_name(0) if USE_GPU else "cpu", "| USE_GPU:", USE_GPU)

# 3c. smoke ST -----------------------------------------------------------------------
from sentence_transformers import SentenceTransformer
_m = SentenceTransformer("all-MiniLM-L6-v2", device=DEVICE); _m.encode(["smoke"]); del _m
print("sentence-transformers OK on", DEVICE)

# ====================================================================================
# 4. MONKEYPATCHES (tất cả ở runtime — không sửa source trên GitHub)
# ====================================================================================
import numpy as np
import yaml
from experiments.run_episode import load_cfg
from experiments.run_rq import run_rq1
from cpa.rl.train import train_ppo, train_ppo_curriculum

# (1) store: nhúng theo BATCH + drop collection cũ ----------------------------------
import json as _json
from pathlib import Path as _Path
from data.schema import CTIRecord as _Rec, CTIEntities as _Ent, Relevance as _Rel
from cpa.cti_store.chroma_store import ChromaCTIStore as _CCS
import cpa.rl.gym_env as _ge

def _add_batch(self, records):
    if not records: return
    ids, docs, metas = [], [], []
    for r in records:
        rid = f"cti-{self._counter}"; self._counter += 1
        ids.append(rid); docs.append(self._generate_doc(r)); metas.append(self._metadata(r)); self._lookup[rid] = r
    kw = dict(ids=ids, documents=docs, metadatas=metas)
    if self._embed_fn is not None: kw["embeddings"] = self._embed_fn(docs)   # 1 lần encode GPU cho cả lô
    self._collection.add(**kw)
def _seed_real(self, jsonl_path, sample_size=None, seed=0):
    p = _Path(jsonl_path)
    if not p.exists(): return 0
    lines = [ln for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if sample_size is not None and sample_size < len(lines):
        import random; lines = random.Random(seed).sample(lines, sample_size)
    recs = [_Rec(topic=d.get("topic","unknown"), real_cti=d.get("real_cti") or d.get("text"),
                 target_relevance=_Rel(d.get("target_relevance","low")),
                 entities=_Ent(**d.get("entities", {})), source_channel="seed-real", is_poison=False)
            for d in (_json.loads(ln) for ln in lines)]
    self._add_batch(recs); return len(recs)
def _drop(self):
    try: self._client.delete_collection(self._collection.name)
    except Exception: pass
_CCS._add_batch, _CCS.seed_real, _CCS.drop = _add_batch, _seed_real, _drop
_orig_build = _ge.PoisoningGymEnv._build_episode
def _build(self, seed):
    if getattr(self, "_env", None) is not None and hasattr(self._env.store, "drop"):
        self._env.store.drop()
    return _orig_build(self, seed)
_ge.PoisoningGymEnv._build_episode = _build

# (2) cache embed_fn (hết spam "BertModel LOAD REPORT" ở RQ) -------------------------
import experiments.run_episode as _re
from experiments.backends import make_embed_fn as _orig_embed
_EMBED_CACHE = {}
def _cached_embed(cfg):
    key = (cfg.get("sbert_model"), cfg.get("device"))
    if key not in _EMBED_CACHE: _EMBED_CACHE[key] = _orig_embed(cfg)
    return _EMBED_CACHE[key]
_re.make_embed_fn = _cached_embed

# (3) reactive detection-heat + obs 22->23 (pressure) --------------------------------
from cpa.observer.parser import parse_turn, observation_vector
from cpa.metrics.effectiveness import planning_deviation_score, false_positive_rate
from cpa.mdp.reward import compute_reward, RewardComponents
import cpa.mdp.env as _envmod
CHANNEL_DETECTION_RISK = _envmod.CHANNEL_DETECTION_RISK

def _build_state23(ekg, vobs, history, history_window=5, pressure=0.0):
    if history:
        hhist = np.mean(np.stack(history[-history_window:], axis=0), axis=0)
    else:
        hhist = np.zeros_like(vobs)
    return np.concatenate([np.asarray(ekg, dtype=np.float32), np.asarray(vobs, dtype=np.float32),
                           np.asarray(hhist, dtype=np.float32), np.asarray([pressure], dtype=np.float32)])

def _env_reset(self):
    self.victim.reset(self.target)
    self._history = []; self._published = 0; self._heat = 0.0
    self._baseline_emb = self.embed(self.victim.baseline_plan(self.target))
    return _build_state23(self._ekg(), np.zeros(7, dtype=np.float32), self._history, pressure=0.0)

def _env_step(self, action):
    variant = self.pool[action.variant_id % len(self.pool)]
    do_publish = action.frequency >= self.cfg.get("publish_threshold", 0.5)
    if do_publish:
        self.publisher.publish(variant, channel_id=action.channel_id)
        self._published += 1
        if self.ceakg is not None:
            relevance = {"low": 0.3, "medium": 0.6, "high": 0.9}[variant.target_relevance.value]
            self.ceakg.add_poison(variant.variant_id or 0,
                                  variant.entities.cves + variant.entities.vulnerabilities, relevance)
    profile = self.target.get("profile", "")
    if hasattr(self.store, "query_scored"):
        scored = self.store.query_scored(profile, k=5)
    else:
        scored = [(r, None) for r in self.store.query(profile, k=5)]
    cti_context = []
    for r, s in scored:
        try: r.retrieval_score = s
        except Exception: pass
        cti_context.append(r)
    # chống FEED-STARVATION (chỉ path defense, gate theo verifier): đảm bảo >=guarantee_real_k CTI thật
    # trong feed — nếu không, real_retention_in_feed = null vì poison Group B quét sạch top-k.
    if self.verifier is not None:
        guarantee = self.cfg.get("guarantee_real_k", 2)
        real_in = sum(1 for r in cti_context if not r.is_poison)
        if real_in < guarantee:
            ids_in = {id(r) for r in cti_context}; cand = []
            if hasattr(self.store, "query_scored"):
                for r, s in self.store.query_scored(profile, k=20):
                    if not r.is_poison and id(r) not in ids_in:
                        try: r.retrieval_score = s
                        except Exception: pass
                        cand.append(r)
            seen = ids_in | {id(r) for r in cand}
            for r in self.store.all():
                if not r.is_poison and id(r) not in seen: cand.append(r); seen.add(id(r))
            cand.sort(key=lambda r: getattr(r, "retrieval_score", None) or 0.0, reverse=True)
            k = len(cti_context) or 5; inject = cand[:guarantee - real_in]
            poison = sorted([r for r in cti_context if r.is_poison],
                            key=lambda r: getattr(r, "retrieval_score", None) or 0.0)
            overflow = len(cti_context) + len(inject) - k
            drop = {id(r) for r in poison[:max(overflow, 0)]}
            cti_context = [r for r in cti_context if id(r) not in drop] + inject
    if self.verifier is not None:
        corpus = self.store.all(); kept = []
        for r in cti_context:
            blocked = self.verifier.verify(r, corpus).trust < self.verifier.threshold
            self.filter_stats[("poison_" if r.is_poison else "real_") + ("blocked" if blocked else "passed")] += 1
            if not blocked: kept.append(r)
        cti_context = kept
    turn = self.victim.step(cti_context)
    parsed = parse_turn(turn, self.gt)
    vobs = observation_vector(parsed); self._history.append(vobs)
    pds = planning_deviation_score(self._baseline_emb, self.embed(turn.plan_text))
    # FPR sống: gộp reported_vulns có cấu trúc + CVE victim phun ra output (parsed["cves"]) — adopt
    # CVE giả -> false positive (khớp env.py repo + spec 3.2 logging wrapper). Không còn FPR=0 cứng.
    reported = list(turn.reported_vulns) + [{"cve": c} for c in parsed.get("cves", [])]
    fpr = false_positive_rate(reported, self.gt)
    impact = 0.6 * pds + 0.4 * fpr
    heat_on = HEAT.get("enabled", True)
    if heat_on:
        self._heat *= float(HEAT.get("decay", 0.6))
        if do_publish: self._heat = min(1.0, self._heat + float(HEAT.get("gain", 0.25)))
    if do_publish:
        det_probs = self.detect(variant.text())
        detector_risk = float(np.max(det_probs)) if det_probs else 0.0
        detection_risk = max(detector_risk, CHANNEL_DETECTION_RISK.get(action.channel_id, 0.3))
        if heat_on: detection_risk = max(detection_risk, self._heat)
    else:
        detection_risk = 0.0
    stealth = 1.0 - detection_risk
    publish_cost = self._published / max(self.cfg.get("max_turns", 25), 1)
    reward = compute_reward(RewardComponents(stealth=stealth, impact=impact,
                            publish_cost=publish_cost, detection_risk=detection_risk),
                            self.cfg.get("reward_weights"))
    if self.ceakg is not None:
        self.ceakg.link_observation(variant.variant_id or 0, parsed.get("cves", []))
    state = _build_state23(self._ekg(), vobs, self._history, pressure=self._heat)
    info = {"pds": pds, "fpr": fpr, "turn": turn, "stealth": stealth,
            "detection_risk": detection_risk, "published_this_turn": do_publish, "heat": self._heat}
    return state, reward, info

_envmod.PoisoningEnv.reset = _env_reset
_envmod.PoisoningEnv.step = _env_step
_ge._STATE_DIM = 8 + 7 + 7 + 1   # obs 22 -> 23 (gym observation_space đọc giá trị này lúc tạo env)

# (4) stochastic CPA eval + ConstantPolicy ------------------------------------------
from cpa.mdp.action import CPAAction, Channel
import cpa.rl.policy as _pol
_orig_cpa_call = _pol.CPAPolicy.__call__
def _cpa_call(self, state):
    if self.model is not None:
        action, _ = self.model.predict(state, deterministic=False)   # sample -> có variance để test
        return _pol.decode_action(action, self.pool_size)
    return _orig_cpa_call(self, state)
_pol.CPAPolicy.__call__ = _cpa_call
class _ConstantPolicy:   # baseline open-loop "tốt nhất cố định" mà RL phải vượt khi env reactive
    def __init__(self, pool_size, **_): self.pool_size = pool_size; self._t = 0
    def __call__(self, state):
        first = self._t == 0; self._t += 1
        return CPAAction(variant_id=0, channel_id=int(Channel.SECURITY_BLOG),
                         frequency=1.0 if first else 0.0, timing=0.0)
_orig_make_policy = _pol.make_policy
def _make_policy(name, pool_size, seed=0, model=None):
    if name == "constant": return _ConstantPolicy(pool_size)
    return _orig_make_policy(name, pool_size, seed=seed, model=model)
_pol.make_policy = _make_policy
_re.make_policy = _make_policy

# (5) HFProvider: victim LLM local 4-bit in-process ---------------------------------
import cpa.victim.providers as _prov
import cpa.victim.llm_victim as _lv
_HF_CACHE = {}
class _HFProvider(_prov.LLMProvider):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.model_name = cfg.get("model") or "Qwen/Qwen2.5-7B-Instruct"
        if self.model_name not in _HF_CACHE:
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
            bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                     bnb_4bit_compute_dtype=torch.float16)
            tok = AutoTokenizer.from_pretrained(self.model_name)
            mdl = AutoModelForCausalLM.from_pretrained(self.model_name, quantization_config=bnb,
                                                       device_map="auto")
            _HF_CACHE[self.model_name] = (mdl, tok)
        self.model, self.tokenizer = _HF_CACHE[self.model_name]
    def complete(self, system, user):
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        prompt = self.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        kw = dict(max_new_tokens=self.max_tokens, pad_token_id=self.tokenizer.eos_token_id)
        if self.temperature and self.temperature > 0: kw.update(do_sample=True, temperature=self.temperature)
        else: kw["do_sample"] = False
        with torch.no_grad(): out = self.model.generate(**inputs, **kw)
        return self.tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
_orig_make_provider = _prov.make_provider
def _make_provider(cfg):
    if cfg.get("provider") in ("hf", "transformers", "local-hf"): return _HFProvider(cfg)
    return _orig_make_provider(cfg)
_prov.make_provider = _make_provider; _lv.make_provider = _make_provider
print("[patch] batch-embed + cleanup + cached-embed + stochastic-eval + reactive-heat(obs23) + Constant + HFProvider ON")
# ====================================================================================

# 5. config cơ sở (victim rule_based cho train/ablation RL) --------------------------
cfg = load_cfg("config/default.yaml")
cfg["victim"]["provider"] = "rule_based"
cfg["cti_store"]["gpu_embed"] = USE_GPU
cfg.setdefault("backends", {})["sbert_model"] = "all-MiniLM-L6-v2"
cfg["backends"]["device"] = DEVICE
cfg["backends"]["detector_backend"] = DETECTOR_BACKEND               # detector THẬT -> stealth/detection có nghĩa
cfg.setdefault("defense", {})["guarantee_real_k"] = GUARANTEE_REAL_K  # chống feed-starvation
if USE_LLM_GENERATOR:                                                # sinh fake CTI bằng LLM thật (fallback template)
    cfg["generator"]["backend"] = "llm_local"; cfg["generator"]["llm_model"] = LLM_MODEL
elif USE_GFCTI_DATASET:                                              # poison pool từ GFCTI-Finance jsonl
    sh(sys.executable, "-m", "data.build_gfcti", "--out", "data/gfcti_finance.jsonl")
    cfg["generator"]["backend"] = "dataset"; cfg["generator"]["dataset_path"] = "data/gfcti_finance.jsonl"
GPU_CFG = "/kaggle/working/default_gpu.yaml"
with open(GPU_CFG, "w") as f: yaml.safe_dump(cfg, f, sort_keys=False)

# Victim dùng cho EVALUATION (ablation + RQ3). Train PPO + defense vẫn xài `cfg` (rule_based) cho nhanh.
cfg_eval = copy.deepcopy(cfg)
if EVAL_VICTIM == "hf":
    cfg_eval["victim"]["provider"] = "hf"; cfg_eval["victim"]["model"] = LLM_MODEL
    cfg_eval["victim"]["temperature"] = 0.7; cfg_eval["victim"]["max_tokens"] = 512
print(f"[cfg] eval victim = {cfg_eval['victim']['provider']} | scale={SCALE} "
      f"({MAX_TARGETS} tgt × {EPISODES} ep × {TURNS} turn)")

report = {}

# 5b. DETECTOR benchmark: AUC real-vs-fake (chứng minh detector phân biệt được poison vs CTI thật) -
# Chạy mỗi generator 1 arm: template (mặc định) + llm_local (nếu DET_BENCH_LLM) để so AUC.
# template thường ra AUC<0.5 (đảo) vì text đầy mã/ID làm GLTR đọc nhầm thành "người viết";
# llm_local (prose trôi chảy) là phép thử xem detector có thật sự phân biệt được poison không.
if RUN_DET_BENCHMARK and DETECTOR_BACKEND != "heuristic":
    gens = ["template"] + (["llm_local"] if DET_BENCH_LLM else [])
    report["detector_benchmark"] = {}
    for gen in gens:
        print(f"\n=== Detector benchmark ({DETECTOR_BACKEND}, gen={gen}) — AUC real vs fake ===")
        det_n = "200" if gen == "template" else "60"   # llm_local sinh n lần gọi Qwen -> giảm cho nhanh
        sh(sys.executable, "-m", "experiments.benchmark_detector",
           "--backend", DETECTOR_BACKEND, "--n", det_n, "--generator", gen)
        try:
            with open(f"experiments/detector_benchmark_{gen}.json") as f:
                report["detector_benchmark"][gen] = json.load(f)
            print(f"detector_benchmark[{gen}]:", report["detector_benchmark"][gen])
        except Exception as e:
            print(f"[warn] đọc detector_benchmark_{gen}.json lỗi:", e)

# 6A. RL PROOF: train reactive + ablation + obs-perturbation -------------------------
if DO_RL_PROOF:
    from stable_baselines3 import PPO
    from experiments.run_episode import run_one

    print("\n=== Train PPO trên env REACTIVE (heat ON)" + (" — CURRICULUM 3 mức" if USE_CURRICULUM else "") + " ===")
    t0 = time.time()
    if USE_CURRICULUM:   # #7: random(A) -> context_aware(B) -> multi_turn, chuyển tiếp policy giữa các mức
        out = train_ppo_curriculum(GPU_CFG, total_timesteps=PPO_STEPS, out="/kaggle/working/cpa_ppo",
                                   turns=TURNS, device=DEVICE)
    else:
        out = train_ppo(GPU_CFG, timesteps=PPO_STEPS, out="/kaggle/working/cpa_ppo", turns=TURNS, device=DEVICE)
    cpa_model = PPO.load(out)
    print(f"PPO {time.time()-t0:.0f}s -> {out}.zip")

    def _summ(key, rows):
        xs = [float(r[key]) for r in rows]
        return {"mean": round(float(np.mean(xs)), 4), "std": round(float(np.std(xs, ddof=1)) if len(xs) > 1 else 0.0, 4)}
    # nopoison = 'No Poisoning' (Section 3.2): không publish gì -> PDS/FPR nền để chuẩn hoá.
    # mcts = DREAM C-GPS+MCTS planning baseline (khác 'dream' greedy) -> so với RL cho RQ2.
    arms = [("cpa", "cpa", cpa_model), ("dream", "dream", None), ("mcts", "mcts", None),
            ("heuristic", "cpa", None), ("random", "random", None), ("constant", "constant", None),
            ("nopoison", "none", None)]
    targets = cfg_eval["victim"]["targets"][:MAX_TARGETS]
    ablation = {}
    print(f"\n=== Ablation (group B, reactive, victim={cfg_eval['victim']['provider']}) ===")
    for label, pname, model in arms:
        rows = [run_one(cfg_eval, tgt, "B", pname, TURNS, seed=s, cpa_model=model)
                for tgt in targets for s in range(EPISODES)]
        ablation[label] = {m: _summ(m, rows) for m in ("stealth_score", "max_pds", "max_fpr", "cfr", "mean_reward", "published")}
        ablation[label]["ASR_%"] = round(100 * np.mean([r["success"] for r in rows]), 1)
        a = ablation[label]
        print(f"  {label:10s} stealth={a['stealth_score']['mean']:.3f}±{a['stealth_score']['std']:.3f}"
              f"  pds={a['max_pds']['mean']:.3f}  fpr={a['max_fpr']['mean']:.3f}  cfr={a['cfr']['mean']:.1f}"
              f"  pub={a['published']['mean']:.2f}  rew={a['mean_reward']['mean']:.3f}  ASR={a['ASR_%']}")
    # RL hữu ích = thắng trên MỤC TIÊU thật, không phải stealth thuần. Stealth-only sai vì 'constant'
    # luôn stealth cao hơn (publish 1 lần) nhưng ASR thấp. Ta đo: (1) reward-win (objective env tối ưu),
    # (2) Pareto trên (stealth, ASR) — không baseline nào trội cả hai, (3) stealthiest trong nhóm ASR cao.
    def _g(k, m): return ablation[k]["ASR_%"] if m == "ASR_%" else ablation[k][m]["mean"]
    s_cpa, r_cpa, asr_cpa = _g("cpa", "stealth_score"), _g("cpa", "mean_reward"), _g("cpa", "ASR_%")
    others = [k for k in ablation if k not in ("cpa", "nopoison")]   # so với các baseline tấn công
    best_base_reward = max((_g(k, "mean_reward") for k in others), default=0.0)
    dominated = any(_g(k, "stealth_score") >= s_cpa and _g(k, "ASR_%") >= asr_cpa
                    and (_g(k, "stealth_score") > s_cpa or _g(k, "ASR_%") > asr_cpa) for k in others)
    high_asr = [k for k in others if _g(k, "ASR_%") >= asr_cpa - 5]   # baseline tấn công mạnh tương đương
    stealthiest_high_asr = all(s_cpa >= _g(k, "stealth_score") for k in high_asr) if high_asr else True
    reward_win = r_cpa >= best_base_reward - 1e-6
    verdict_stealth = {
        "cpa_reward": round(r_cpa, 4), "best_baseline_reward": round(best_base_reward, 4),
        "reward_win": bool(reward_win),
        "pareto_optimal_stealth_vs_asr": bool(not dominated),
        "stealthiest_among_high_asr": bool(stealthiest_high_asr),
        "cpa_minus_constant_stealth": round(s_cpa - _g("constant", "stealth_score"), 4),  # tham khảo
        "cpa_minus_dream_stealth": round(s_cpa - _g("dream", "stealth_score"), 4),
        "RL_useful": bool(reward_win or (not dominated and stealthiest_high_asr)),
    }

    # obs-perturbation: action có đổi khi che từng khối obs? (deterministic để tín hiệu sạch)
    print("\n=== Obs-perturbation ===")
    genv = _ge.PoisoningGymEnv(cfg, group="B", turns=TURNS, seed=0)
    bank, o = [], genv.reset()[0]
    for _ in range(60):
        bank.append(o.copy()); o, _, _, trunc, _ = genv.step(genv.action_space.sample())
        if trunc: o = genv.reset()[0]
    def _act(o): a, _ = cpa_model.predict(o, deterministic=True); return tuple(np.atleast_1d(a).tolist())
    base = [_act(o) for o in bank]
    blocks = {"ekg": slice(0, 8), "vobs": slice(8, 15), "hist": slice(15, 22), "pressure": slice(22, 23)}
    perturb = {}
    for name, sl in blocks.items():
        chg = 0
        for i, o in enumerate(bank):
            op = o.copy(); op[sl] = 0.0
            if _act(op) != base[i]: chg += 1
        perturb[name] = round(chg / len(bank), 3)
        print(f"  zero {name:9s} -> action-change-rate {perturb[name]}")
    verdict_obs = {"policy_uses_obs": bool(max(perturb.values()) > 0.05),
                   "policy_uses_pressure": bool(perturb["pressure"] > 0.05)}
    report["rl_proof"] = {"ablation": ablation, "verdict_stealth": verdict_stealth,
                          "obs_perturbation": perturb, "verdict_obs": verdict_obs}
    print("\nverdict_stealth:", verdict_stealth, "\nverdict_obs:", verdict_obs)
    with open("/kaggle/working/rl_proof.json", "w") as f: json.dump(report["rl_proof"], f, indent=2)

# 6A1b. LangGraph orchestrator demo: 1 episode qua StateGraph 4 node (#7 part 1) -----
if DEMO_LANGGRAPH:
    print("\n=== LangGraph orchestrator demo (Generator->Stealth->Publisher->Observer) ===")
    try:
        from cpa.orchestrator import build_langgraph_pipeline
        from cpa.rl.policy import make_policy as _mp
        genv_lg = _ge.PoisoningGymEnv(cfg, group="B", turns=TURNS, seed=0); genv_lg.reset()
        penv = genv_lg._env
        pol = _mp("dream", pool_size=len(penv.pool), seed=0)   # dream: không phụ thuộc obs-dim của patch
        traj = build_langgraph_pipeline(cfg, penv, pol)(TURNS)
        report["langgraph_demo"] = {"turns": len(traj),
                                    "mean_reward": round(float(np.mean([t["reward"] for t in traj])), 4),
                                    "mean_pds": round(float(np.mean([t["pds"] for t in traj])), 4)}
        print("langgraph_demo:", report["langgraph_demo"])
    except Exception as e:
        print("[warn] LangGraph demo lỗi (bỏ qua, không ảnh hưởng phần khác):", repr(e))

# 6A2. DEFENSE ablation: verifier OFF/ON + chống feed-starvation ---------------------
if DO_DEFENSE:
    from experiments.run_rq import run_defense_ablation
    print("\n=== Defense ablation (verifier OFF/ON, guarantee_real_k) ===")
    t0 = time.time()
    def_rep = run_defense_ablation(cfg, EPISODES, TURNS, MAX_TARGETS)
    report["defense"] = def_rep
    print(f"defense {time.time()-t0:.0f}s")
    print(json.dumps(def_rep, indent=2)[:2000])
    with open("/kaggle/working/defense.json", "w") as f: json.dump(def_rep, f, indent=2)

# 6A3. RQ3: two-way ANOVA group(A/B) × policy(dream/cpa) trên max_pds + stealth (lever nào mạnh) -
# Dùng victim rule_based (nhanh) — ANOVA đo tương tác CTI-context × policy, không cần LLM thật.
# Tái dùng PPO đã train ở 6A nếu có (arm "cpa"); nếu DO_RL_PROOF tắt thì "cpa" = heuristic.
if DO_RQ3:
    from experiments.run_rq import run_rq3
    print("\n=== RQ3 (two-way ANOVA: CTI-context A/B × policy dream/cpa) ===")
    t0 = time.time()
    _rq3_model = cpa_model if DO_RL_PROOF else None
    rq3 = run_rq3(cfg_eval, EPISODES, TURNS, MAX_TARGETS, cpa_model=_rq3_model)
    report["rq3"] = rq3
    print(f"RQ3 {time.time()-t0:.0f}s")
    print(json.dumps(rq3, indent=2)[:1500])
    with open("/kaggle/working/rq3_full.json", "w") as f: json.dump(rq3, f, indent=2)

# 6B. LLM RQ1: victim Qwen thật, subset A vs B ---------------------------------------
if DO_LLM_RQ1:
    print(f"\n=== RQ1 với victim LLM thật ({LLM_MODEL}) — CHẬM ===")
    cfg_llm = copy.deepcopy(cfg)
    cfg_llm["victim"]["provider"] = "hf"
    cfg_llm["victim"]["model"] = LLM_MODEL
    cfg_llm["victim"]["temperature"] = 0.7
    cfg_llm["victim"]["max_tokens"] = 512
    t0 = time.time()
    rq1 = run_rq1(cfg_llm, RQ1_EPISODES, RQ1_TURNS, RQ1_TARGETS)   # budget riêng (Qwen chậm, không scale)
    print(f"RQ1-LLM {time.time()-t0:.0f}s")
    report["rq1_llm"] = rq1
    print(json.dumps(rq1, indent=2)[:1500])
    with open("/kaggle/working/rq1_llm.json", "w") as f: json.dump(rq1, f, indent=2)

with open("/kaggle/working/results.json", "w") as f: json.dump(report, f, indent=2)
print("\nDONE. Files: /kaggle/working/{results,rl_proof,defense,rq3_full,rq1_llm}.json + experiments/detector_benchmark_*.json")