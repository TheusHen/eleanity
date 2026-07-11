from __future__ import annotations

from typing import Any

from eleanity.adapters.base import BackendAdapter, CapabilitySet, HealthcheckResult
from eleanity.fingerprints import collect_environment_fingerprint
from eleanity.models.schemas import (
    ArtifactFingerprint,
    LayerObservation,
    LayerState,
    ModelSpec,
    PromptObservation,
    Scenario,
    TokenObservation,
)
from eleanity.utils.hashing import text_sha256
from eleanity.utils.security import sanitize_path


class TransformersAdapter(BackendAdapter):
    """Hugging Face Transformers adapter with CPU/GPU and large-model loading policy."""

    name = "transformers"
    version = "0.2"
    _tokenizer_cache: dict[str, object] = {}
    _model_cache: dict[str, object] = {}

    def __init__(self, model_ref: str, model_spec: ModelSpec | None = None, *, tokenizer_only: bool | None = None):
        self.model_ref = model_ref
        self.model_spec = model_spec or ModelSpec(id=model_ref)
        if self.model_spec.id is None:
            self.model_spec = self.model_spec.model_copy(update={"id": model_ref})
        if tokenizer_only is not None:
            self.model_spec = self.model_spec.model_copy(update={"tokenizer_only": tokenizer_only})
        self.tokenizer_only = bool(self.model_spec.tokenizer_only)
        try:
            import transformers  # noqa: F401

            self.version = getattr(transformers, "__version__", self.version)
            # Tokenizer-only mode: cheap CI path — no weights, no logits/generation.
            can_weights = not self.tokenizer_only
            self.capabilities = CapabilitySet(
                render=True,
                tokenize=True,
                logits=can_weights,
                stream=False,
                tools=True,
                artifact=True,
                template=True,
                rendered_prompt=True,
                tokenization=True,
                special_tokens=True,
                logprobs=can_weights,
                generation=can_weights,
                structured_output=can_weights,
                streaming=False,
                usage=can_weights,
                errors=True,
                healthcheck=True,
                notes=({"mode": "tokenizer_only"} if self.tokenizer_only else {}),
            )
        except ImportError:
            self.capabilities = CapabilitySet(artifact=True, healthcheck=True)

    # ------------------------------------------------------------------ helpers
    def _cache_key(self) -> str:
        spec = self.model_spec
        return "|".join(
            [
                self.model_ref,
                str(spec.revision or ""),
                str(spec.dtype or "auto"),
                str(spec.device_map or ""),
                str(spec.quantization or ""),
                str(spec.trust_remote_code),
                str(spec.load_in_4bit),
                str(spec.load_in_8bit),
                str(spec.attn_implementation or ""),
            ]
        )

    def _resolve_torch_dtype(self):
        import torch

        requested = (self.model_spec.dtype or "auto").lower()
        mapping = {
            "float32": torch.float32,
            "fp32": torch.float32,
            "float16": torch.float16,
            "fp16": torch.float16,
            "half": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
        }
        if requested in mapping:
            return mapping[requested]
        # auto
        if torch.cuda.is_available():
            if getattr(torch.cuda, "is_bf16_supported", lambda: False)():
                return torch.bfloat16
            return torch.float16
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.float16
        return torch.float32

    def _dtype_label(self) -> str:
        dtype = self._resolve_torch_dtype()
        return str(dtype).replace("torch.", "")

    def _load_tokenizer(self):
        key = f"tok:{self._cache_key()}"
        if key not in self._tokenizer_cache:
            from transformers import AutoTokenizer

            self._tokenizer_cache[key] = AutoTokenizer.from_pretrained(
                self.model_ref,
                revision=self.model_spec.revision,
                trust_remote_code=self.model_spec.trust_remote_code,
            )
        return self._tokenizer_cache[key]

    def _load_model(self):
        if self.tokenizer_only:
            raise RuntimeError("tokenizer_only mode: model weights are not loaded")
        key = f"mdl:{self._cache_key()}"
        if key not in self._model_cache:
            import torch
            from transformers import AutoModelForCausalLM

            dtype = self._resolve_torch_dtype()
            kwargs: dict[str, Any] = {
                "revision": self.model_spec.revision,
                "trust_remote_code": self.model_spec.trust_remote_code,
                "low_cpu_mem_usage": True,
            }
            # transformers>=4.56 / 5.x prefer `dtype=` over deprecated `torch_dtype=`
            try:
                import transformers
                from packaging.version import Version

                if Version(transformers.__version__.split("+")[0]) >= Version("4.56.0"):
                    kwargs["dtype"] = dtype
                else:
                    kwargs["torch_dtype"] = dtype
            except Exception:
                kwargs["torch_dtype"] = dtype
            if self.model_spec.attn_implementation:
                kwargs["attn_implementation"] = self.model_spec.attn_implementation
            if self.model_spec.max_memory:
                kwargs["max_memory"] = self.model_spec.max_memory

            # Quantization / device placement for large models
            if self.model_spec.load_in_4bit or self.model_spec.quantization in {"4bit", "nf4", "fp4"}:
                kwargs["load_in_4bit"] = True
                kwargs["device_map"] = self.model_spec.device_map or "auto"
            elif self.model_spec.load_in_8bit or self.model_spec.quantization in {"8bit", "int8"}:
                kwargs["load_in_8bit"] = True
                kwargs["device_map"] = self.model_spec.device_map or "auto"
            else:
                device_map = self.model_spec.device_map
                if device_map and device_map != "none":
                    # Prefer accelerate device_map for multi-GPU / large checkpoints.
                    # Fall back to single-device placement when accelerate is missing.
                    try:
                        import accelerate  # noqa: F401

                        kwargs["device_map"] = device_map
                    except ImportError:
                        pass

            model = AutoModelForCausalLM.from_pretrained(self.model_ref, **kwargs)
            if "device_map" not in kwargs:
                if torch.cuda.is_available():
                    model = model.to("cuda")
                elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                    model = model.to("mps")
            model.eval()
            self._model_cache[key] = model
        return self._model_cache[key]

    def _input_device(self, model):
        try:
            return next(model.parameters()).device
        except StopIteration:  # pragma: no cover
            import torch

            return torch.device("cpu")

    def _generation_kwargs(self, scenario: Scenario, tokenizer) -> dict[str, Any]:
        params = scenario.parameters
        kwargs: dict[str, Any] = {
            "max_new_tokens": int(params.get("max_tokens", 64)),
            "pad_token_id": tokenizer.eos_token_id or tokenizer.pad_token_id,
        }
        temperature = float(params.get("temperature", 0))
        if temperature > 0:
            kwargs.update({"do_sample": True, "temperature": temperature})
            if params.get("top_p") is not None:
                kwargs["top_p"] = float(params["top_p"])
            if params.get("top_k") is not None:
                kwargs["top_k"] = int(params["top_k"])
        else:
            kwargs["do_sample"] = False
        if params.get("repetition_penalty") is not None:
            kwargs["repetition_penalty"] = float(params["repetition_penalty"])
        if params.get("stop"):
            # stop sequences are handled post-hoc for broader backend parity
            pass
        return kwargs

    # ----------------------------------------------------------------- protocol
    def healthcheck(self) -> HealthcheckResult:
        if not self.capabilities.tokenize:
            return HealthcheckResult(ok=False, detail="install eleanity[transformers]")
        try:
            self._load_tokenizer()
            return HealthcheckResult(ok=True, detail="tokenizer loadable")
        except Exception as error:  # pragma: no cover - depends on network/model cache
            return HealthcheckResult(ok=False, detail=str(error))

    def fingerprint(self, model_ref: str) -> ArtifactFingerprint:
        env = collect_environment_fingerprint()
        if not self.capabilities.tokenize:
            return ArtifactFingerprint(
                model_ref=model_ref,
                python_version=env.python_version,
                os=env.platform,
                cpu_arch=env.machine,
                gpu=env.gpu_name,
                cuda_or_rocm=env.cuda_version,
                library_versions=env.packages,
                backend_flags={"dependency": "transformers unavailable"},
            )
        tokenizer = self._load_tokenizer()
        template = getattr(tokenizer, "chat_template", None)
        quant = self.model_spec.quantization
        if self.model_spec.load_in_4bit:
            quant = quant or "4bit"
        elif self.model_spec.load_in_8bit:
            quant = quant or "8bit"
        model_type = None
        architecture = None
        config_hash = None
        try:
            from transformers import AutoConfig

            config = AutoConfig.from_pretrained(
                self.model_ref,
                revision=self.model_spec.revision,
                trust_remote_code=self.model_spec.trust_remote_code,
            )
            model_type = getattr(config, "model_type", None)
            architectures = getattr(config, "architectures", None) or []
            architecture = architectures[0] if architectures else None
            config_hash = text_sha256(str(config.to_dict()))
        except Exception:
            pass
        runtime_version = None
        try:
            import transformers

            runtime_version = getattr(transformers, "__version__", None)
        except ImportError:
            pass
        special = {
            "bos_token_id": getattr(tokenizer, "bos_token_id", None),
            "eos_token_id": getattr(tokenizer, "eos_token_id", None),
            "pad_token_id": getattr(tokenizer, "pad_token_id", None),
            "unk_token_id": getattr(tokenizer, "unk_token_id", None),
        }
        return ArtifactFingerprint(
            model_ref=model_ref,
            revision=self.model_spec.revision,
            local_path=sanitize_path(self.model_spec.local_path),
            tokenizer=sanitize_path(getattr(tokenizer, "name_or_path", None))
            if getattr(tokenizer, "name_or_path", None)
            and (
                "/" in str(getattr(tokenizer, "name_or_path", ""))
                or "\\" in str(getattr(tokenizer, "name_or_path", ""))
            )
            and str(getattr(tokenizer, "name_or_path", "")).count("/") != 1
            else getattr(tokenizer, "name_or_path", None),
            tokenizer_hash=text_sha256(str(getattr(tokenizer, "vocab_size", ""))),
            chat_template_hash=ArtifactFingerprint.text_hash(template if isinstance(template, str) else None),
            model_type=model_type,
            architecture=architecture,
            config_hash=config_hash,
            quantization=quant,
            dtype=self._dtype_label(),
            lora_adapters=list(self.model_spec.lora_adapters),
            special_tokens=special,
            runtime_version=runtime_version,
            library_versions=env.packages,
            python_version=env.python_version,
            os=env.platform,
            cpu_arch=env.machine,
            gpu=env.gpu_name,
            cuda_or_rocm=env.cuda_version,
            backend_flags={
                "runtime": "transformers",
                "trust_remote_code": self.model_spec.trust_remote_code,
                "device_map": self.model_spec.device_map,
                "attn_implementation": self.model_spec.attn_implementation,
                "tokenizer_only": self.tokenizer_only,
            },
        )

    def structured(self, scenario: Scenario) -> LayerObservation:
        if self.tokenizer_only or not self.capabilities.generation:
            return LayerObservation(
                state=LayerState.NOT_OBSERVABLE,
                note="structured requires generation weights",
            )
        generation = self.generate(scenario)
        if generation.state != LayerState.OBSERVED:
            return generation
        import json

        text = str(generation.data.get("text") or "").strip()
        parsed = None
        parse_error = None
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as error:
            parse_error = str(error)
        return LayerObservation(
            state=LayerState.OBSERVED,
            data={
                "raw_text": text,
                "parsed": parsed,
                "is_json": parsed is not None,
                "parse_error": parse_error,
                "tool_calls": None,
                "stop_reason": generation.data.get("stop_reason"),
                "schema_valid": parsed is not None,
            },
        )

    def special_tokens(self) -> LayerObservation:
        if not self.capabilities.special_tokens:
            return LayerObservation(state=LayerState.NOT_OBSERVABLE, note="install eleanity[transformers]")
        tokenizer = self._load_tokenizer()
        additional = list(getattr(tokenizer, "additional_special_tokens", []) or [])
        data = {
            "bos_token": getattr(tokenizer, "bos_token", None),
            "bos_token_id": getattr(tokenizer, "bos_token_id", None),
            "eos_token": getattr(tokenizer, "eos_token", None),
            "eos_token_id": getattr(tokenizer, "eos_token_id", None),
            "pad_token": getattr(tokenizer, "pad_token", None),
            "pad_token_id": getattr(tokenizer, "pad_token_id", None),
            "unk_token": getattr(tokenizer, "unk_token", None),
            "unk_token_id": getattr(tokenizer, "unk_token_id", None),
            "sep_token_id": getattr(tokenizer, "sep_token_id", None),
            "cls_token_id": getattr(tokenizer, "cls_token_id", None),
            "mask_token_id": getattr(tokenizer, "mask_token_id", None),
            "additional_special_tokens": additional,
            "additional_special_tokens_ids": [tokenizer.convert_tokens_to_ids(token) for token in additional],
            "vocab_size": getattr(tokenizer, "vocab_size", None),
            "model_max_length": getattr(tokenizer, "model_max_length", None),
            "chat_template_hash": ArtifactFingerprint.text_hash(
                getattr(tokenizer, "chat_template", None)
                if isinstance(getattr(tokenizer, "chat_template", None), str)
                else None
            ),
        }
        return LayerObservation(state=LayerState.OBSERVED, data=data)

    def render(self, scenario: Scenario) -> LayerObservation:
        if not self.capabilities.render:
            return LayerObservation(state=LayerState.NOT_OBSERVABLE, note="install eleanity[transformers]")
        tokenizer = self._load_tokenizer()
        add_generation_prompt = scenario.generation.add_generation_prompt
        continue_final = scenario.generation.continue_final_message
        template = getattr(tokenizer, "chat_template", None)
        kwargs: dict[str, Any] = {
            "tokenize": False,
            "add_generation_prompt": add_generation_prompt,
        }
        try:
            text = tokenizer.apply_chat_template(
                [message.model_dump() for message in scenario.messages],
                continue_final_message=continue_final,
                **kwargs,
            )
        except TypeError:
            text = tokenizer.apply_chat_template(
                [message.model_dump() for message in scenario.messages],
                **kwargs,
            )
        markers = []
        for marker in ("<|im_start|>", "<|im_end|>", "<|endoftext|>", "<|assistant|>"):
            if marker in text:
                markers.append(marker)
        obs = PromptObservation(
            chat_template_source="tokenizer.chat_template" if template else None,
            chat_template_hash=text_sha256(template) if isinstance(template, str) else None,
            add_generation_prompt=add_generation_prompt,
            continue_final_message=continue_final,
            rendered_text=text,
            rendered_utf8_hex=text.encode("utf-8").hex(),
            rendered_byte_length=len(text.encode("utf-8")),
            rendered_char_length=len(text),
            roles=[message.role for message in scenario.messages],
            tools_included=any(message.role == "tool" for message in scenario.messages),
            special_markers=markers,
            text=text,
        )
        return LayerObservation(state=LayerState.OBSERVED, data=obs.to_layer_data())

    def tokenize(self, rendered: str) -> LayerObservation:
        if not self.capabilities.tokenize:
            return LayerObservation(state=LayerState.NOT_OBSERVABLE, note="install eleanity[transformers]")
        tokenizer = self._load_tokenizer()
        encoded = tokenizer(rendered, add_special_tokens=False)
        ids = list(encoded["input_ids"])
        attention = list(encoded["attention_mask"]) if "attention_mask" in encoded else None
        special_ids: set[int] = set()
        for attr in ("bos_token_id", "eos_token_id", "pad_token_id", "unk_token_id"):
            value = getattr(tokenizer, attr, None)
            if isinstance(value, int):
                special_ids.add(value)
            elif isinstance(value, (list, tuple)):
                special_ids.update(int(item) for item in value if item is not None)
        added: list[int] = []
        for token in getattr(tokenizer, "additional_special_tokens", []) or []:
            token_id = tokenizer.convert_tokens_to_ids(token)
            if isinstance(token_id, int) and token_id >= 0:
                special_ids.add(token_id)
                added.append(token_id)
        special_positions = [index for index, token_id in enumerate(ids) if token_id in special_ids]
        try:
            token_strings = tokenizer.convert_ids_to_tokens(ids)
        except Exception:
            token_strings = None
        obs = TokenObservation(
            token_ids=ids,
            token_strings=list(token_strings) if token_strings is not None else None,
            decoded_text=tokenizer.decode(ids) if ids else "",
            bos_token_id=getattr(tokenizer, "bos_token_id", None)
            if isinstance(getattr(tokenizer, "bos_token_id", None), int)
            else None,
            eos_token_id=getattr(tokenizer, "eos_token_id", None)
            if isinstance(getattr(tokenizer, "eos_token_id", None), int)
            else None,
            pad_token_id=getattr(tokenizer, "pad_token_id", None)
            if isinstance(getattr(tokenizer, "pad_token_id", None), int)
            else None,
            unk_token_id=getattr(tokenizer, "unk_token_id", None)
            if isinstance(getattr(tokenizer, "unk_token_id", None), int)
            else None,
            added_special_tokens=added,
            attention_mask=attention,
            original_length=len(ids),
            final_length=len(ids),
            truncated=False,
            padding_side=getattr(tokenizer, "padding_side", None),
            truncation_side=getattr(tokenizer, "truncation_side", None),
            special_token_positions=special_positions,
            special_token_count=len(special_positions),
            add_special_tokens=False,
        )
        return LayerObservation(state=LayerState.OBSERVED, data=obs.to_layer_data())

    def forward(self, tokens: LayerObservation) -> LayerObservation:
        if self.tokenizer_only:
            return LayerObservation(
                state=LayerState.NOT_OBSERVABLE,
                note="tokenizer_only mode: logits not available without weights",
            )
        if not self.capabilities.logits:
            return LayerObservation(state=LayerState.NOT_OBSERVABLE, note="install eleanity[transformers]")
        if tokens.state != LayerState.OBSERVED or "ids" not in tokens.data:
            return LayerObservation(state=LayerState.NOT_OBSERVABLE, note="tokens unavailable for forward pass")
        import torch

        model = self._load_model()
        device = self._input_device(model)
        input_ids = torch.tensor([tokens.data["ids"]], device=device)
        with torch.inference_mode():
            outputs = model(input_ids=input_ids)
            logits = outputs.logits[0, -1].float().cpu()
        top_k = min(10, logits.shape[-1])
        values, indices = torch.topk(logits, top_k)
        return LayerObservation(
            state=LayerState.OBSERVED,
            data={
                "top_ids": indices.tolist(),
                "top_logits": values.tolist(),
                "vocab_slice": top_k,
                "device": str(device),
            },
        )

    def generate(self, scenario: Scenario) -> LayerObservation:
        if self.tokenizer_only:
            return LayerObservation(
                state=LayerState.NOT_OBSERVABLE,
                note="tokenizer_only mode: generation not available without weights",
            )
        if not self.capabilities.generation:
            return LayerObservation(state=LayerState.NOT_OBSERVABLE, note="install eleanity[transformers]")
        import torch

        seed = int(scenario.parameters.get("seed", 42))
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        rendered = self.render(scenario)
        if rendered.state != LayerState.OBSERVED:
            return rendered
        tokenizer = self._load_tokenizer()
        model = self._load_model()
        device = self._input_device(model)
        inputs = tokenizer(rendered.data["text"], return_tensors="pt", add_special_tokens=False)
        inputs = {key: value.to(device) for key, value in inputs.items()}
        kwargs = self._generation_kwargs(scenario, tokenizer)
        with torch.inference_mode():
            generated = model.generate(**inputs, **kwargs)
        prompt_len = inputs["input_ids"].shape[1]
        new_ids = generated[0, prompt_len:].detach().cpu().tolist()

        stop_sequences = scenario.parameters.get("stop") or []
        if isinstance(stop_sequences, str):
            stop_sequences = [stop_sequences]
        text = tokenizer.decode(new_ids, skip_special_tokens=True)
        stop_reason = "max_tokens"
        eos_ids: set[int] = set()
        if isinstance(tokenizer.eos_token_id, int):
            eos_ids.add(tokenizer.eos_token_id)
        elif isinstance(tokenizer.eos_token_id, (list, tuple)):
            eos_ids.update(int(item) for item in tokenizer.eos_token_id if item is not None)
        if new_ids and new_ids[-1] in eos_ids:
            stop_reason = "eos"
        for stop in stop_sequences:
            if stop and stop in text:
                stop_reason = "stop_sequence"
                break

        return LayerObservation(
            state=LayerState.OBSERVED,
            data={
                "text": text,
                "ids": new_ids,
                "stop_reason": stop_reason,
                "prompt_token_count": prompt_len,
                "completion_token_count": len(new_ids),
                "seed": seed,
            },
        )
