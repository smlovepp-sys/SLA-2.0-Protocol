# physics_extractor.py
# 最终生产版 v11：弹性与撕裂使用加权embedding轴，形容词权重大
import re
import hashlib
import torch
import numpy as np
from collections import OrderedDict
from typing import Dict, List, Tuple, Optional

class PhysicsExtractor:
    PARAM_BOUNDS = {
        "stiffness": (0.0, 1.0),
        "friction": (0.0, 1.0),
        "elasticity": (0.0, 1.0),
        "wetness": (0.0, 1.0),
        "tear_intensity": (0.0, 1.0),
        "tension": (-1.0, 1.0),
        "body_type": (0.0, 1.0),
    }
    PARAM_NAMES = ["stiffness", "friction", "elasticity", "wetness", "tear_intensity", "tension", "body_type"]

    AXIS_PAIRS = {
        "stiffness": ("highly stiff solid unyielding", "deformable soft compliant squishy"),
        "friction": ("rough abrasive sandpaper coarse", "smooth slippery polished lubricated"),
        "elasticity": ("elastic springy bouncy resilient", "brittle rigid inelastic stiff"),
        "wetness": ("soaking wet drenched saturated", "bone dry arid dehydrated"),
        "tear_intensity": ("fragile flimsy tearing shredding", "tough durable indestructible intact"),
        "tension": ("stretched taut tensioned pulled", "slack loose relaxed draped"),
        "body_type": ("tight muscular fleshy", "loose flabby skinny"),
    }

    MATERIAL_PRIORS = {
        "steel":    {"stiffness": 0.95, "elasticity": 0.10, "tear_intensity": 0.05, "wetness": 0.05, "friction": 0.40},
        "metal":    {"stiffness": 0.90, "elasticity": 0.10, "tear_intensity": 0.05, "wetness": 0.05, "friction": 0.45},
        "stone":    {"stiffness": 0.85, "elasticity": 0.05, "tear_intensity": 0.10, "wetness": 0.05, "friction": 0.50},
        "glass":    {"stiffness": 0.80, "elasticity": 0.05, "tear_intensity": 0.15, "wetness": 0.05, "friction": 0.20},
        "ice":      {"stiffness": 0.55, "elasticity": 0.10, "tear_intensity": 0.20, "wetness": 0.30, "friction": 0.15},
        "ceramic":  {"stiffness": 0.72, "elasticity": 0.05, "tear_intensity": 0.25, "wetness": 0.05, "friction": 0.35},
        "concrete": {"stiffness": 0.70, "elasticity": 0.05, "tear_intensity": 0.10, "wetness": 0.10, "friction": 0.40},
        "wood":     {"stiffness": 0.65, "elasticity": 0.15, "tear_intensity": 0.30, "wetness": 0.15, "friction": 0.45},
        "plastic":  {"stiffness": 0.55, "elasticity": 0.30, "tear_intensity": 0.35, "wetness": 0.05, "friction": 0.35},
        "rubber":   {"stiffness": 0.40, "elasticity": 0.85, "tear_intensity": 0.40, "wetness": 0.10, "friction": 0.70},
        "leather":  {"stiffness": 0.35, "elasticity": 0.20, "tear_intensity": 0.50, "wetness": 0.20, "friction": 0.55},
        "paper":    {"stiffness": 0.25, "elasticity": 0.10, "tear_intensity": 0.70, "wetness": 0.40, "friction": 0.35},
        "cotton":   {"stiffness": 0.15, "elasticity": 0.10, "tear_intensity": 0.60, "wetness": 0.50, "friction": 0.40},
        "silk":     {"stiffness": 0.10, "elasticity": 0.15, "tear_intensity": 0.55, "wetness": 0.45, "friction": 0.30},
        "foam":     {"stiffness": 0.05, "elasticity": 0.70, "tear_intensity": 0.80, "wetness": 0.30, "friction": 0.40},
        "sand":     {"stiffness": 0.30, "elasticity": 0.05, "tear_intensity": 0.05, "wetness": 0.05, "friction": 0.50},
        "skin":     {"stiffness": 0.20, "elasticity": 0.50, "tear_intensity": 0.50, "wetness": 0.30, "friction": 0.40},
    }

    POS_WEIGHTS = {
        "ADJ": 3.0,
        "NOUN": 1.0,
        "OTHER": 0.5,
    }

    def __init__(self, model_id: str = "v11_prod", cache_size: int = 1024, alpha: float = 0.2, top_k_similar: int = 5):
        self.model_id = model_id
        self.cache_size = cache_size
        self.alpha = alpha
        self.top_k_similar = top_k_similar
        self._fingerprint = None
        
        self._classic_axes = {}
        self._classic_biases = {}
        self._ridge_models = {}
        self._bias_correction = {}
        self._text_cache = OrderedDict()
        self._similar_words_cache = {}

    # ---------- 辅助函数 ----------
    def _get_embedding_weight(self, clip) -> torch.Tensor:
        if hasattr(clip, "model") and hasattr(clip.model, "shared"):
            return clip.model.shared.weight
        if hasattr(clip, "shared"):
            return clip.shared.weight
        raise AttributeError("无法从当前模型中提取 token embedding 权重")

    def _compute_fingerprint(self, clip) -> str:
        w = self._get_embedding_weight(clip)
        sample = w.data.flatten()[:1000].detach().cpu().numpy().tobytes()
        return hashlib.sha256(sample).hexdigest()

    def _get_token_avg(self, text: str, clip, w: torch.Tensor, device: torch.device) -> torch.Tensor:
        tokenizer = clip.tokenizer
        ids = tokenizer.encode(text, add_special_tokens=False)
        if not ids:
            ids = [tokenizer.pad_token_id or 0]
        emb = w[torch.tensor(ids, device=device)].mean(dim=0)
        return emb / (emb.norm() + 1e-6)

    def _get_similar_words(self, seed_words: List[str], clip, w: torch.Tensor, top_k: int = 15) -> List[str]:
        device = w.device
        seed_vecs = []
        for wd in seed_words:
            ids = clip.tokenizer.encode(wd, add_special_tokens=False)
            if ids:
                v = w[torch.tensor(ids, device=device)].mean(dim=0)
                seed_vecs.append(v / (v.norm() + 1e-6))
        if not seed_vecs:
            return seed_words
            
        mean_seed = torch.stack(seed_vecs).mean(dim=0)
        mean_seed = mean_seed / (mean_seed.norm() + 1e-6)
        
        w_norm = w / (w.norm(dim=1, keepdim=True) + 1e-6)
        sims = torch.matmul(w_norm, mean_seed)
        vals, indices = torch.topk(sims, k=min(top_k * 3, w.size(0)))
        
        expanded = list(seed_words)
        seen_words = set(seed_words)
        for idx in indices.detach().cpu().numpy():
            word = clip.tokenizer.decode([int(idx)]).strip().lower()
            if len(word) >= 3 and word.isalpha() and word not in seen_words:
                expanded.append(word)
                seen_words.add(word)
            if len(expanded) >= top_k:
                break
        return expanded

    # ---------- 加权短语 embedding ----------
    def _get_weighted_phrase_embedding(self, phrase: str, clip) -> torch.Tensor:
        w_emb = self._get_embedding_weight(clip)
        device = w_emb.device
        tokenizer = clip.tokenizer
        tokens = phrase.lower().split()
        
        phys_adjs = {
            'hard', 'soft', 'rough', 'smooth', 'stiff', 'rigid', 'flexible', 'spongy', 'brittle', 'firm',
            'solid', 'limp', 'floppy', 'bendable', 'tough', 'plush', 'fluffy', 'wet', 'dry', 'slippery',
            'oily', 'coarse', 'taut', 'slack', 'damp', 'elastic', 'fragile', 'abrasive', 'springy',
            'resilient', 'bouncy', 'stretchy', 'flimsy', 'shredding', 'indestructible', 'durable',
            'muscular', 'fleshy', 'skinny', 'flabby', 'draped', 'pulled', 'rebounding'
        }
        
        weighted_sum = None
        total_weight = 0
        for token in tokens:
            if token in phys_adjs:
                weight = self.POS_WEIGHTS["ADJ"]
            elif token in self.MATERIAL_PRIORS:
                weight = self.POS_WEIGHTS["NOUN"]
            else:
                weight = self.POS_WEIGHTS["OTHER"]
                
            ids = tokenizer.encode(token, add_special_tokens=False)
            if not ids:
                continue
            
            token_emb = w_emb[torch.tensor(ids, device=device)].mean(dim=0)
            token_emb = token_emb / (token_emb.norm() + 1e-6)
            
            if weighted_sum is None:
                weighted_sum = weight * token_emb
            else:
                weighted_sum += weight * token_emb
            total_weight += weight
            
        if weighted_sum is None:
            return self._get_token_avg(phrase, clip, w_emb, device)
            
        final_emb = weighted_sum / total_weight
        return final_emb / (final_emb.norm() + 1e-6)

    # ---------- 经典轴构建（使用加权 embedding）----------
    def _build_classic_axis(self, param: str, clip, w: torch.Tensor, device: torch.device):
        pos_desc, neg_desc = self.AXIS_PAIRS[param]
        pos_vec = self._get_weighted_phrase_embedding(pos_desc, clip).to(device)
        neg_vec = self._get_weighted_phrase_embedding(neg_desc, clip).to(device)
        
        raw_axis = pos_vec - neg_vec
        
        for prev_param in self.PARAM_NAMES[:self.PARAM_NAMES.index(param)]:
            if prev_param in self._classic_axes:
                prev_axis = self._classic_axes[prev_param].to(device)
                proj = torch.dot(raw_axis, prev_axis) / torch.dot(prev_axis, prev_axis)
                raw_axis = raw_axis - proj * prev_axis
        
        axis = raw_axis / (raw_axis.norm() + 1e-6)
        
        neutral_words = ["the", "is", "a", "of", "in", "it", "and", "with", "an", "at"]
        neutral_projs = []
        for wd in neutral_words:
            n_vec = self._get_weighted_phrase_embedding(wd, clip).to(device)
            neutral_projs.append(torch.dot(n_vec, axis).item())
            
        self._classic_axes[param] = axis.detach()
        self._classic_biases[param] = float(np.median(neutral_projs))

    # ---------- Ridge 模型 ----------
    def _build_ridge_model(self, param: str, clip, w: torch.Tensor, device: torch.device):
        pos_desc, neg_desc = self.AXIS_PAIRS[param]
        pos_seeds = pos_desc.split()
        neg_seeds = neg_desc.split()
        
        pos_expanded = self._get_similar_words(pos_seeds, clip, w, top_k=15)
        neg_expanded = self._get_similar_words(neg_seeds, clip, w, top_k=15)
        
        X, y = [], []
        for wd in pos_expanded:
            v = self._get_token_avg(wd, clip, w, device).detach().cpu().numpy()
            X.append(v)
            y.append(1.0)
        for wd in neg_expanded:
            v = self._get_token_avg(wd, clip, w, device).detach().cpu().numpy()
            X.append(v)
            y.append(-1.0)
            
        X = np.array(X)
        y = np.array(y)
        
        alpha = 0.5
        XtX = np.dot(X.T, X)
        XtY = np.dot(X.T, y)
        coef = np.linalg.solve(XtX + alpha * np.eye(X.shape[1]), XtY)
        
        pos_projs = np.dot(X[:len(pos_expanded)], coef)
        neg_projs = np.dot(X[len(pos_expanded):], coef)
        center = (np.mean(pos_projs) + np.mean(neg_projs)) / 2.0
        
        self._ridge_models[param] = {"coef": coef, "center": center}

    # ---------- 基准线校准 ----------
    def _compute_bias_correction(self, clip):
        neutral_set = ["the", "this", "it", "thing", "object", "item"]
        for param in self.PARAM_NAMES:
            biases = []
            for word in neutral_set:
                emb = self._encode_text(word, clip).detach().cpu().numpy()
                rm = self._ridge_models[param]
                ridge_proj = np.dot(emb, rm["coef"]) - rm["center"]
                ridge_proj = np.clip(ridge_proj, -3.0, 3.0)
                pred = np.tanh(ridge_proj)
                biases.append(0.0 - pred)
            self._bias_correction[param] = float(np.mean(biases))
        print(f"[PhysicsExtractor] 基准线校准完成: { {k: round(v, 4) for k, v in self._bias_correction.items()} }")

    # ---------- 自动重建 ----------
    def _check_and_rebuild(self, clip):
        fp = self._compute_fingerprint(clip)
        if self._fingerprint != fp or not self._classic_axes:
            w = self._get_embedding_weight(clip)
            device = w.device
            self._fingerprint = fp
            self._classic_axes.clear()
            self._classic_biases.clear()
            self._ridge_models.clear()
            self._bias_correction.clear()
            self._text_cache.clear()
            
            for param in self.PARAM_NAMES:
                self._build_classic_axis(param, clip, w, device)
                self._build_ridge_model(param, clip, w, device)
                
            self._compute_bias_correction(clip)
            print("[PhysicsExtractor] 重建完成。")

    # ---------- 文本编码（使用加权 embedding）----------
    def _encode_text(self, text: str, clip) -> torch.Tensor:
        self._text_cache["last_phrase"] = text.lower()
        return self._get_weighted_phrase_embedding(text, clip)

    # ---------- 短语分割 ----------
    def _extract_phrases(self, text: str) -> List[str]:
        clean = re.sub(r'\(.*?:\d+\.?\d*\)', '', text)
        clean = re.sub(r'[()]', '', clean).strip().lower()
        clauses = re.split(r'[,.;!?]+', clean)
        stop_words = {'a','an','the','and','or','of','in','on','at','with','without','to','for','by','is','are','was','were','be','been','being','have','has','had','do','does','did','but','so','very','just','only','not','no','nor','too','can','will','would','should','could','may','might','must','this','that','these','those','it','they','we','you','he','she','it','there','their','some','any','such','both','each','few','more','most','other','some','such','no','nor','not','only','own','same','so','than','too','very','just','but','do','does','did','doing'}
        phrases = []
        for clause in clauses:
            words = clause.split()
            if not words:
                continue
            for n in range(1, min(4, len(words) + 1)):
                for i in range(len(words) - n + 1):
                    candidate = ' '.join(words[i:i+n])
                    if all(w in stop_words for w in candidate.split()):
                        continue
                    if any(w not in stop_words for w in candidate.split()):
                        phrases.append(candidate)
        return list(dict.fromkeys(phrases))

    # ---------- 融合预测 + 材质先验 ----------
    def _get_hybrid_prediction(self, emb_np: np.ndarray, param: str, phrase: str) -> float:
        lo, hi = self.PARAM_BOUNDS[param]
        
        axis = self._classic_axes[param].cpu().numpy()
        bias = self._classic_biases[param]
        proj_val = np.dot(emb_np, axis) - bias
        classic_val = np.tanh(4.0 * proj_val)
        
        rm = self._ridge_models[param]
        ridge_proj = np.dot(emb_np, rm["coef"]) - rm["center"]
        ridge_proj = np.clip(ridge_proj, -3.0, 3.0)
        ridge_val_raw = np.tanh(ridge_proj)
        ridge_val = ridge_val_raw + self._bias_correction.get(param, 0.0)
        
        confidence = min(1.0, abs(ridge_proj) / 2.0)
        if confidence > 0.65:
            raw_val = ridge_val
        else:
            raw_val = 0.6 * classic_val + 0.4 * ridge_val
        
        if lo == 0.0 and hi == 1.0:
            val = (raw_val + 1.0) / 2.0
        else:
            val = raw_val
        val = np.clip(val, lo, hi)
        
        for mat, prior_dict in self.MATERIAL_PRIORS.items():
            if mat in phrase and param in prior_dict:
                prior = prior_dict[param]
                diff = abs(val - prior)
                if diff > 0.15:
                    val = 0.6 * val + 0.4 * prior
                break
        
        return float(np.clip(val, lo, hi))

    # ---------- 对外接口 ----------
    def extract_full_physics_vectors(self, prompt: str, clip) -> Dict[str, Dict[str, float]]:
        self._check_and_rebuild(clip)
        phrases = self._extract_phrases(prompt)
        result = {}
        for ph in phrases:
            emb = self._encode_text(ph, clip).detach().cpu().numpy()
            params = {}
            for param in self.PARAM_NAMES:
                val = self._get_hybrid_prediction(emb, param, ph)
                params[param] = round(val, 4)
            result[ph] = params
        return result

    def extract_phrases_from_text(self, prompt: str, clip) -> Dict[str, Dict[str, float]]:
        return self.extract_full_physics_vectors(prompt, clip)

    def extract_from_text(self, prompt: str, clip) -> Dict[str, float]:
        vectors = self.extract_full_physics_vectors(prompt, clip)
        if not vectors:
            return {k: (lo+hi)/2 for k, (lo, hi) in self.PARAM_BOUNDS.items()}
            
        phys_adjs = {
            'hard', 'soft', 'rough', 'smooth', 'stiff', 'rigid', 'flexible', 'spongy', 'brittle', 'firm',
            'solid', 'limp', 'floppy', 'bendable', 'tough', 'coarse', 'plush', 'fluffy', 'wet', 'dry', 
            'taut', 'slack', 'damp', 'elastic', 'fragile', 'abrasive'
        }
        weight_sum = {k: 0.0 for k in self.PARAM_NAMES}
        total_weights = {k: 0.0 for k in self.PARAM_NAMES}
        for ph, p_vec in vectors.items():
            tokens = ph.split()
            has_mat = any(mat in ph for mat in self.MATERIAL_PRIORS)
            has_adj = any(tk in phys_adjs for tk in tokens)
            if has_mat and has_adj:
                ph_weight = 5.0
            elif has_mat:
                ph_weight = 3.0
            elif has_adj:
                ph_weight = 2.0
            else:
                ph_weight = 0.4
            for param in self.PARAM_NAMES:
                weight_sum[param] += p_vec[param] * ph_weight
                total_weights[param] += ph_weight
        avg = {}
        for param in self.PARAM_NAMES:
            avg[param] = round(weight_sum[param] / max(total_weights[param], 1e-6), 4)
        return avg

    def clear_cache(self):
        self._text_cache.clear()
        self._classic_axes.clear()
        self._classic_biases.clear()
        self._ridge_models.clear()
        self._bias_correction.clear()
        self._similar_words_cache.clear()
        self._fingerprint = None