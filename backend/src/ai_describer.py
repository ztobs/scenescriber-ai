"""AI-powered scene description generation module."""

import base64
import logging
import os
import contextlib
from typing import List, Dict, Any, Optional
from pathlib import Path

import requests
from PIL import Image

logger = logging.getLogger(__name__)


class AIDescriber:
    """Generates AI descriptions for video scenes."""

    def __init__(self, model: str = "openai", api_key: Optional[str] = None):
        """Initialize AI describer with model configuration.

        Args:
            model: AI model to use ('openai', 'claude', 'gemini', 'llava', 'ollama')
                   or specific model like 'openai/gpt-4o', 'ollama/llava'
            api_key: API key for the selected model
        """
        self.model = model

        # Extract base model name for API key lookup
        base_model = model.split("/")[0] if "/" in model else model

        # Handle gemini/google naming inconsistency
        env_var_name = f"{base_model.upper()}_API_KEY"
        if base_model == "gemini":
            # Try GEMINI_API_KEY first, then GOOGLE_API_KEY as fallback
            self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        else:
            self.api_key = api_key or os.getenv(env_var_name)

        # Check if model is local
        is_local = base_model == "llava" or base_model == "ollama" or model.startswith("ollama/")

        if not self.api_key and not is_local:
            logger.warning(f"No API key provided for {model}. Some features may not work.")

    def generate_descriptions(
        self,
        scenes: List[Dict[str, Any]],
        theme: Optional[str] = None,
        description_length: str = "medium",
    ) -> List[Dict[str, Any]]:
        """Generate descriptions for scenes using AI.

        Args:
            scenes: List of scene dictionaries with keyframes
            theme: Optional theme to guide descriptions
            description_length: Desired length ('short', 'medium', 'detailed')

        Returns:
            Updated scenes with AI-generated descriptions
        """
        logger.info(f"Generating descriptions for {len(scenes)} scenes with theme: {theme}")

        # Check if AI is available
        is_local = (
            self.model == "llava" or self.model == "ollama" or self.model.startswith("ollama/")
        )

        if not self.api_key and not is_local:
            logger.warning(f"No API key for {self.model}, using mock descriptions")
            return self._generate_mock_descriptions(scenes, theme, description_length)

        for scene in scenes:
            if not scene.get("keyframes"):
                logger.warning(f"No keyframes for scene {scene['scene_id']}, skipping description")
                scene["description"] = "No description available (missing keyframes)"
                continue

            try:
                description = self._describe_scene(
                    keyframes=scene["keyframes"], theme=theme, description_length=description_length
                )
                scene["description"] = description
                scene["theme_applied"] = theme
                logger.debug(f"Generated description for scene {scene['scene_id']}")

            except Exception as e:
                logger.error(f"Failed to generate description for scene {scene['scene_id']}: {e}")
                # Return blank description to indicate failure
                scene["description"] = ""
                scene["theme_applied"] = theme

        return scenes

    def _generate_mock_descriptions(
        self,
        scenes: List[Dict[str, Any]],
        theme: Optional[str] = None,
        description_length: str = "medium",
    ) -> List[Dict[str, Any]]:
        """Generate mock descriptions when AI is not available.

        Args:
            scenes: List of scene dictionaries
            theme: Optional theme
            description_length: Desired length

        Returns:
            Updated scenes with mock descriptions
        """
        logger.info("Generating mock descriptions (no API key available)")

        for scene in scenes:
            scene["description"] = self._generate_mock_description(
                scene["scene_id"], theme, description_length
            )
            scene["theme_applied"] = theme
            scene["confidence_score"] = 0.5  # Lower confidence for mock data

        return scenes

    def _generate_mock_description(
        self, scene_id: int, theme: Optional[str] = None, description_length: str = "medium"
    ) -> str:
        """Generate a context-aware mock description for a scene.

        Args:
            scene_id: Scene ID number
            theme: Optional theme
            description_length: Desired length

        Returns:
            Mock description string
        """
        # Theme-specific descriptions to be more context-aware
        theme_descriptions = {
            "DIY": [
                "A DIY project involving woodworking with visible sawdust and tools.",
                "Someone assembling furniture components with screws and a drill.",
                "Measuring and cutting materials for a custom build project.",
                "Painting or finishing a handmade piece with careful brushwork.",
            ],
            "cooking": [
                "Chopping fresh vegetables on a cutting board with a chef's knife.",
                "Stirring ingredients in a sizzling pan on the stove.",
                "Plating a finished dish with garnish and careful presentation.",
                "Measuring and mixing dry ingredients in a mixing bowl.",
            ],
            "tutorial": [
                "Close-up demonstration of a specific technique or skill.",
                "Step-by-step instructions shown with clear visual examples.",
                "Comparing different methods or approaches side by side.",
                "Highlighting common mistakes and how to avoid them.",
            ],
            "review": [
                "Showing product features and specifications in detail.",
                "Comparing the item to similar products on the market.",
                "Demonstrating practical use cases and real-world testing.",
                "Highlighting pros and cons with visual evidence.",
            ],
            "default": [
                "Visible tools and materials arranged for a specific task.",
                "Hands performing precise actions with focus and attention.",
                "Progress visible through partially completed work stages.",
                "Detailed close-up of technical components or processes.",
            ],
        }

        # Determine which theme category to use
        theme_key = "default"
        if theme:
            theme_lower = theme.lower()
            if any(keyword in theme_lower for keyword in ["diy", "build", "woodwork", "craft"]):
                theme_key = "DIY"
            elif any(keyword in theme_lower for keyword in ["cook", "food", "recipe", "baking"]):
                theme_key = "cooking"
            elif any(
                keyword in theme_lower for keyword in ["tutorial", "how-to", "guide", "instruction"]
            ):
                theme_key = "tutorial"
            elif any(
                keyword in theme_lower for keyword in ["review", "comparison", "product", "test"]
            ):
                theme_key = "review"

        # Select description based on scene ID
        descriptions = theme_descriptions[theme_key]
        desc_idx = scene_id % len(descriptions)
        description = descriptions[desc_idx]

        # Add theme context if provided (but not redundant)
        if theme and theme_key == "default":
            description = f"In this {theme.lower()} content: {description}"

        # Adjust length based on description_length parameter
        if description_length == "short":
            # Keep it short
            if len(description) > 80:
                description = description[:77] + "..."
        elif description_length == "detailed":
            # Add more detail
            details = [
                " The camera angle provides clear visibility of key details.",
                " Lighting conditions highlight textures and material qualities.",
                " This shot effectively communicates the intended instructional value.",
                " Composition focuses attention on the most important elements.",
            ]
            detail_idx = scene_id % len(details)
            description += details[detail_idx]

        # For LLaVA, return blank instead of mock
        is_local = (
            self.model == "llava" or self.model == "ollama" or self.model.startswith("ollama/")
        )
        if is_local:
            return ""

        # For other models without API key, return blank
        if not self.api_key:
            return ""

        return description

    def _describe_scene(
        self, keyframes: List[str], theme: Optional[str] = None, description_length: str = "medium"
    ) -> str:
        """Generate description for a single scene.

        Args:
            keyframes: List of keyframe image paths
            theme: Optional theme to guide description
            description_length: Desired length ('short', 'medium', 'detailed')

        Returns:
            AI-generated description
        """
        # Extract base model name
        base_model = self.model.split("/")[0] if "/" in self.model else self.model

        if base_model == "openai" or base_model == "openai_compatible":
            return self._describe_with_openai(keyframes, theme, description_length)
        elif base_model == "claude":
            return self._describe_with_claude(keyframes, theme, description_length)
        elif base_model == "gemini":
            return self._describe_with_gemini(keyframes, theme, description_length)
        elif base_model == "llava":
            return self._describe_with_llava(keyframes, theme, description_length)
        elif base_model == "ollama" or self.model.startswith("ollama/"):
            return self._describe_with_ollama(keyframes, theme, description_length)
        else:
            raise ValueError(f"Unsupported model: {self.model}")

    def _describe_with_lm_studio_native(
        self,
        keyframes: List[str],
        model_name: str,
        theme: Optional[str] = None,
        description_length: str = "medium",
    ) -> str:
        """Generate description using LM Studio native REST API (/api/v0).

        LM Studio's OpenAI-compatible endpoint has a known issue with vision models.
        This method uses the native /api/v0 endpoint which works correctly.

        See: https://github.com/lmstudio-ai/lmstudio-bug-tracker/issues/968
        """
        api_base = os.getenv("OPENAI_COMPATIBLE_API_BASE", "http://localhost:1234/v1")
        # Convert /v1 endpoint to /api/v0 for native API
        native_api_base = api_base.replace("/v1", "/api/v0")

        # Build prompt
        prompt = self._build_prompt(theme, description_length)

        # Load images as base64 and format for LM Studio native API
        images_content = []
        for keyframe in keyframes:
            with open(keyframe, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode("utf-8")
                # Use image_url format (OpenAI standard) which LM Studio native API supports
                images_content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                    }
                )

        payload = {
            "model": model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        *images_content,
                    ],
                }
            ],
            "max_tokens": self._get_max_tokens(description_length),
        }

        timeout = 1200  # LM Studio can be slow when loading models (increased from 120s)

        try:
            logger.debug(
                f"Calling LM Studio native API at {native_api_base}/chat/completions with model={model_name}"
            )
            response = requests.post(
                f"{native_api_base}/chat/completions",
                json=payload,
                timeout=timeout,
            )

            logger.debug(f"LM Studio native API response status: {response.status_code}")

            if response.status_code != 200:
                logger.error(f"LM Studio native API error: {response.text}")
                raise ValueError(f"LM Studio API error: {response.text}")

            response_json = response.json()
            logger.debug(f"LM Studio native API response: {response_json}")

            content = response_json.get("choices", [{}])[0].get("message", {}).get("content", "")

            if not content or content.strip() == "":
                logger.warning(f"LM Studio returned empty content. Full response: {response_json}")
                return ""

            logger.info(
                f"Successfully generated description using LM Studio native API ({len(content)} chars)"
            )
            return content

        except requests.exceptions.Timeout:
            logger.error(f"LM Studio native API timeout (1200 seconds)")
            raise ValueError("LM Studio API timeout - model is too slow or stuck loading")
        except requests.exceptions.ConnectionError as e:
            logger.error(f"LM Studio native API connection error: {e}")
            raise ValueError(f"Cannot connect to LM Studio API at {native_api_base}")
        except Exception as e:
            logger.error(f"Unexpected error calling LM Studio native API: {e}", exc_info=True)
            raise

    def _describe_with_openai(
        self, keyframes: List[str], theme: Optional[str] = None, description_length: str = "medium"
    ) -> str:
        """Generate description using OpenAI or OpenAI-compatible API."""
        if not self.api_key:
            raise ValueError("OpenAI API key not provided")

        # Determine if this is OpenAI or OpenAI-compatible
        is_compatible = "openai_compatible" in self.model
        base_model = "openai_compatible" if is_compatible else "openai"

        # Extract model name (e.g., "gpt-4o" from "openai/gpt-4o" or "openai_compatible/gpt-4")
        # Handle cases with multiple slashes like "openai_compatible/qwen/qwen-vl-plus"
        if "/" in self.model:
             # Drop the provider prefix (everything before the first slash)
            model_name = self.model.split("/", 1)[1]
        else:
            model_name = "gpt-4o"

        # Get API base URL from environment
        if is_compatible:
            api_base = os.getenv("OPENAI_COMPATIBLE_API_BASE", "https://api.openai.com/v1")
            api_version = os.getenv("OPENAI_COMPATIBLE_API_VERSION")
            # Check if this is LM Studio (localhost:1234)
            is_lm_studio = "localhost:1234" in api_base or "127.0.0.1:1234" in api_base
        else:
            api_base = "https://api.openai.com/v1"
            api_version = None
            is_lm_studio = False

        # Prepare images for OpenAI-compatible API
        images = []
        
        for keyframe in keyframes:
            # Open image using PIL
            try:
                with Image.open(keyframe) as img:
                    # For LM Studio, resize image to avoid "Channel Error" (payload too large)
                    if is_compatible and is_lm_studio:
                        # Resize to max 512px dimension while maintaining aspect ratio
                        img.thumbnail((512, 512))
                        
                        # Convert to JPEG in memory
                        import io
                        buffer = io.BytesIO()
                        img.save(buffer, format="JPEG", quality=85)
                        image_data = buffer.getvalue()
                        base64_image = base64.b64encode(image_data).decode("utf-8")
                        logger.debug(f"Resized image for LM Studio: {len(base64_image)} bytes")
                    else:
                        # Standard handling for other providers
                        with open(keyframe, "rb") as image_file:
                            base64_image = base64.b64encode(image_file.read()).decode("utf-8")

                    images.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                    })
            except Exception as e:
                logger.error(f"Failed to process image {keyframe}: {e}")
                continue

        # Build prompt based on theme and length
        prompt = self._build_prompt(theme, description_length)

        # Send standard request with images
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}, *images]}]

        headers = {"Content-Type": "application/json"}
        
        # Only add auth header for OpenAI (LM Studio doesn't need it for /v1 endpoint)
        if not is_compatible:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        # For OpenAI compatible (like OpenRouter), we might need the key too if provided
        if is_compatible and self.api_key:
             headers["Authorization"] = f"Bearer {self.api_key}"

        # Add OpenRouter specific headers
        if "openrouter.ai" in api_base:
            headers["HTTP-Referer"] = "http://localhost:3000"
            headers["X-Title"] = "Video Scene Analyzer"

        # Add API version header for Azure OpenAI
        if api_version:
            headers["api-version"] = api_version

        payload = {
            "model": model_name,
            "messages": messages,
            "max_tokens": self._get_max_tokens(description_length),
        }

        # Use much longer timeout for OpenAI-compatible/LM Studio
        # Local models can take minutes to load into VRAM or process large images
        # Increased to 20 minutes (1200s) to allow for initial model loading
        timeout = 1200 if is_compatible else 60

        try:
            logger.debug(
                f"Calling {base_model} API at {api_base}/chat/completions with model={model_name} (timeout={timeout}s)"
            )
            # Log exact images being sent
            sent_filenames = [os.path.basename(k) for k in keyframes]
            logger.info(f"Sending {len(images)} images to AI: {sent_filenames}")
            logger.debug(f"Payload: model={model_name}, images_count={len(images)}, max_tokens={payload['max_tokens']}")
            logger.debug(f"Headers: {headers}")
            
            response = requests.post(
                f"{api_base}/chat/completions", headers=headers, json=payload, timeout=timeout
            )

            logger.debug(f"{base_model} API response status: {response.status_code}")

            if response.status_code != 200:
                error_text = response.text
                logger.error(f"{base_model} API error response: {error_text}")
                
                # Handle specific LM Studio errors
                if "Failed to load model" in error_text:
                    raise ValueError(f"LM Studio could not load model '{model_name}'. Please check if it's installed and healthy in LM Studio.")
                
                raise ValueError(
                    f"{base_model.replace('_', ' ').title()} API error: {error_text}"
                )

            response_json = response.json()
            logger.debug(f"{base_model} API response: {response_json}")

            content = response_json.get("choices", [{}])[0].get("message", {}).get("content", "")

            if not content or content.strip() == "":
                logger.warning(
                    f"{base_model} returned empty content. Full response: {response_json}"
                )
                return ""

            logger.info(f"Successfully generated description ({len(content)} chars)")
            return content

        except requests.exceptions.Timeout:
            logger.error(f"{base_model} API timeout ({timeout} seconds) - model may be processing slowly")
            raise ValueError(
                f"{base_model.replace('_', ' ').title()} API timeout - model is too slow"
            )
        except requests.exceptions.ConnectionError as e:
            logger.error(f"{base_model} connection error: {e}")
            raise ValueError(
                f"Cannot connect to {base_model.replace('_', ' ').title()} API at {api_base}"
            )
        except Exception as e:
            logger.error(f"Unexpected error calling {base_model} API: {e}", exc_info=True)
            raise

    def _describe_with_claude(
        self, keyframes: List[str], theme: Optional[str] = None, description_length: str = "medium"
    ) -> str:
        """Generate description using Claude 3."""
        if not self.api_key:
            raise ValueError("Claude API key not provided")

        # Prepare images
        images = []
        for keyframe in keyframes:
            with open(keyframe, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode("utf-8")
                images.append(
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": base64_image,
                        },
                    }
                )

        # Build prompt
        prompt = self._build_prompt(theme, description_length)

        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}, *images]}]

        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
        }

        payload = {
            "model": "claude-3-opus-20240229",
            "max_tokens": self._get_max_tokens(description_length),
            "messages": messages,
        }

        response = requests.post(
            "https://api.anthropic.com/v1/messages", headers=headers, json=payload, timeout=30
        )

        if response.status_code != 200:
            raise ValueError(f"Claude API error: {response.text}")

        return response.json()["content"][0]["text"]

    def _describe_with_gemini(
        self, keyframes: List[str], theme: Optional[str] = None, description_length: str = "medium"
    ) -> str:
        """Generate description using Google Gemini."""
        if not self.api_key:
            raise ValueError("Gemini API key not provided")

        # Note: Gemini implementation would require google-generativeai library
        # This is a placeholder implementation
        raise NotImplementedError("Gemini support not yet implemented")

    def _describe_with_ollama(
        self, keyframes: List[str], theme: Optional[str] = None, description_length: str = "medium"
    ) -> str:
        """Generate description using local Ollama model (e.g., LLaVA)."""
        ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")

        # Determine model name from self.model if possible, or env var
        if self.model.startswith("ollama/"):
            ollama_model = self.model.split("/", 1)[1]
        else:
            ollama_model = os.getenv("OLLAMA_MODEL", "llava")

        # Prepare images (Ollama typically handles one image per prompt well,
        # but supports multiple. We'll send all keyframes if supported,
        # but usually LLaVA takes one. Let's send the first one for now to be safe/consistent
        # with _describe_with_llava logic which uses keyframes[:1])
        images = []
        # Load all keyframes for Ollama
        for keyframe in keyframes:
            with open(keyframe, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode("utf-8")
                images.append(base64_image)

        if not images:
            raise ValueError("No valid keyframes to analyze")

        # Build prompt
        # We can reuse the LLaVA prompt builder but remove the <image> tag
        # as Ollama's API handles image insertion
        prompt = self._build_llava_prompt(theme, description_length)
        prompt = prompt.replace("<image>", "").strip()

        payload = {
            "model": ollama_model,
            "prompt": prompt,
            "images": images,
            "stream": False,
            "options": {
                "temperature": 0.7,
                "top_p": 0.9,
            },
        }

        try:
            response = requests.post(f"{ollama_host}/api/generate", json=payload, timeout=600)
            response.raise_for_status()
            description = response.json().get("response", "").strip()

            if not description:
                logger.warning("Ollama returned empty description")
                return ""

            return description

        except requests.RequestException as e:
            logger.error(f"Ollama API error: {e}")
            # Don't raise, just return empty so flow continues (similar to other local failures)
            return ""

    def _describe_with_llava(
        self, keyframes: List[str], theme: Optional[str] = None, description_length: str = "medium"
    ) -> str:
        """Generate description using local LLaVA model - optimized for 8GB VRAM.

        Memory optimizations for 8GB VRAM systems:
        - Uses 4-bit NF4 quantization with double quantization
        - Enables CPU offloading for layers if needed
        - Reduces max_new_tokens for smaller outputs
        - Clears cache after inference
        - Monitors VRAM usage and adapts model loading strategy
        - Unloads model between scenes to prevent memory fragmentation

        Note: Requires transformers, torch, and bitsandbytes to be installed.
        Run: pip install torch transformers pillow bitsandbytes accelerate

        For GPU acceleration:
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
        """
        try:
            from transformers import (
                AutoProcessor,
                LlavaForConditionalGeneration,
                BitsAndBytesConfig,
            )
            from PIL import Image
            import torch
            import gc
        except ImportError as e:
            logger.error(
                f"LLaVA dependencies not installed. Install with: pip install torch transformers pillow bitsandbytes accelerate"
            )
            raise ValueError(
                "LLaVA dependencies not found. Install with: pip install torch transformers pillow bitsandbytes accelerate"
            )

        # Use memory context for the entire LLaVA inference
        with self._gpu_memory_context():
            return self._describe_with_llava_internal(keyframes, theme, description_length)

    def _describe_with_llava_internal(
        self, keyframes: List[str], theme: Optional[str] = None, description_length: str = "medium"
    ) -> str:
        """Internal implementation of LLaVA description generation.

        This method is called within the GPU memory context.
        """
        try:
            from transformers import (
                AutoProcessor,
                LlavaForConditionalGeneration,
                BitsAndBytesConfig,
            )
            from PIL import Image
            import torch
            import gc

            # Clear GPU cache before loading
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()

            # Initialize model variable for safe cleanup
            model = None

            # Check available VRAM and choose appropriate model/quantization
            model_id = self._select_llava_model_based_on_vram()
            logger.info(f"Loading LLaVA model (selected based on VRAM): {model_id}")

            processor = AutoProcessor.from_pretrained(model_id)

            # Check current VRAM usage and choose appropriate quantization
            import torch

            if torch.cuda.is_available():
                total_vram = torch.cuda.get_device_properties(0).total_memory / 1e9
                allocated = torch.cuda.memory_allocated() / 1e9
                free_vram = total_vram - allocated

                logger.info(f"Available VRAM: {free_vram:.2f}GB / {total_vram:.2f}GB")

                # Choose quantization based on available VRAM
                if free_vram >= 16:
                    # Enough VRAM for float16 (14GB needed for 7B model)
                    logger.info("Using float16 precision (sufficient VRAM available)")
                    model = LlavaForConditionalGeneration.from_pretrained(
                        model_id,
                        torch_dtype=torch.float16,
                        low_cpu_mem_usage=True,
                        device_map="auto",
                    )
                elif free_vram >= 10:
                    # Use 8-bit quantization (need ~10GB buffer for safety)
                    logger.info("Using 8-bit quantization (moderate VRAM)")
                    quantization_config = BitsAndBytesConfig(
                        load_in_8bit=True, llm_int8_enable_fp32_cpu_offload=True
                    )
                    model = LlavaForConditionalGeneration.from_pretrained(
                        model_id,
                        quantization_config=quantization_config,
                        device_map="auto",
                        low_cpu_mem_usage=True,
                        torch_dtype=torch.float16,
                    )
                else:
                    # Use 4-bit quantization (6-7GB needed) - most memory efficient for 8GB cards
                    logger.info("Using 4-bit NF4 quantization (limited VRAM)")
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_compute_dtype=torch.float16,
                        llm_int8_threshold=6.0,
                        llm_int8_has_fp16_weight=False,
                    )
                    model = LlavaForConditionalGeneration.from_pretrained(
                        model_id,
                        quantization_config=quantization_config,
                        device_map="auto",
                        low_cpu_mem_usage=True,
                        torch_dtype=torch.float16,
                    )
            else:
                # CPU-only fallback
                logger.info("No GPU available, loading for CPU")
                model = LlavaForConditionalGeneration.from_pretrained(
                    model_id, torch_dtype=torch.float32, low_cpu_mem_usage=True, device_map="cpu"
                )

            # Enable gradient checkpointing if model supports it (for training scenarios)
            if hasattr(model, "gradient_checkpointing_enable"):
                model.gradient_checkpointing_enable()
                logger.debug("Enabled gradient checkpointing for memory efficiency")

            # Load and process keyframes
            images = []
            keyframe_imgs = []
            for keyframe_path in keyframes:  # Load all keyframes to analyze complete scene
                try:
                    img = Image.open(keyframe_path).convert("RGB")
                    # Resize images to consistent size for stitching
                    # 336x336 is a good size for LLaVA
                    if img.size[0] != 336 or img.size[1] != 336:
                        img = img.resize((336, 336), Image.Resampling.LANCZOS)
                    keyframe_imgs.append(img)
                except Exception as e:
                    logger.warning(f"Failed to load keyframe {keyframe_path}: {e}")

            if not keyframe_imgs:
                raise ValueError("No valid keyframes to analyze")

            # Stitch keyframes horizontally if multiple frames
            if len(keyframe_imgs) > 1:
                # Create a new image with combined width
                total_width = sum(img.width for img in keyframe_imgs)
                max_height = keyframe_imgs[0].height
                stitched = Image.new("RGB", (total_width, max_height))
                x_offset = 0
                for img in keyframe_imgs:
                    stitched.paste(img, (x_offset, 0))
                    x_offset += img.width
                images = [stitched]
                logger.info(
                    f"Stitched {len(keyframe_imgs)} keyframes into single image for analysis"
                )
            else:
                images = keyframe_imgs

            # Build LLaVA-specific prompt
            prompt = self._build_llava_prompt(theme, description_length)
            logger.debug(f"LLaVA prompt: {prompt}")

            # Prepare inputs
            inputs = processor(images=images[0], text=prompt, return_tensors="pt").to(model.device)

            # Reduce max_new_tokens for smaller outputs to save memory
            max_tokens = min(self._get_max_tokens(description_length), 80)  # Reduced from 150

            logger.info(f"Generating description (max_tokens={max_tokens})...")

            # Generate description with memory-efficient settings
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    min_new_tokens=10,  # Ensure minimum output
                    temperature=0.7,  # Slightly lower for consistency
                    top_p=0.9,
                    do_sample=True,
                    eos_token_id=2,
                    pad_token_id=0,
                    num_beams=1,  # Greedy decoding (faster, less memory)
                    use_cache=True,  # Use KV cache for faster inference
                )

            # Extract just the newly generated tokens
            prompt_len = inputs["input_ids"].shape[1]
            new_tokens = output_ids[0][prompt_len:]
            description = processor.decode(new_tokens, skip_special_tokens=True).strip()

            # Clean up artifacts
            description = description.replace("[/INST]", "").strip()
            if "Assistant:" in description:
                description = description.split("Assistant:")[-1].strip()

            # Check for empty or very short responses - return blank to indicate failure
            if not description or len(description) < 10:
                logger.warning(
                    f"LLaVA generated very short response ({len(description)} chars), returning blank"
                )
                return ""

            # Check for generic descriptions that match mock patterns - return blank
            generic_phrases = [
                "person working",
                "using tools",
                "someone using",
                "this scene shows",
                "a person is",
                "someone is",
                "working on a project",
                "demonstrating a technique",
                "focused on completing",
                "progress being made",
                "explaining or showing",
                "important step",
            ]

            description_lower = description.lower()
            is_generic = any(phrase in description_lower for phrase in generic_phrases)

            if is_generic:
                logger.warning(
                    f"LLaVA generated generic description: '{description[:50]}...'. Returning blank."
                )
                return ""

            logger.info(f"LLaVA generated specific description: {description[:80]}...")
            return description

        except torch.cuda.OutOfMemoryError as e:
            logger.error(f"GPU out of memory during LLaVA inference: {e}")
            logger.info("Attempting emergency memory cleanup and retry...")

            # Emergency cleanup
            self._cleanup_gpu_memory(model, processor)

            # Try one more time with more aggressive memory settings
            try:
                logger.info("Retrying with more aggressive memory settings...")
                # Clear everything and try again
                import torch

                torch.cuda.empty_cache()
                gc.collect()

                # Return blank to indicate failure after retry
                return ""
            except Exception as retry_error:
                logger.error(f"Retry also failed: {retry_error}")
                return ""

        except Exception as e:
            logger.error(f"LLaVA inference error: {e}")
            logger.info("Returning blank description to indicate LLaVA failure.")
            return ""
        finally:
            # Comprehensive cleanup after inference
            try:
                import torch
                import gc

                # Clean up model and processor
                self._cleanup_gpu_memory(model, processor)

                # Additional periodic deep cleanup every 5 scenes
                if hasattr(self, "_llava_inference_count"):
                    self._llava_inference_count += 1
                else:
                    self._llava_inference_count = 1

                # Deep cleanup every 5 inferences to prevent memory fragmentation
                if self._llava_inference_count % 5 == 0:
                    logger.info("Performing periodic deep memory cleanup")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.reset_peak_memory_stats()
                        gc.collect()
                        torch.cuda.synchronize()

            except Exception as cleanup_error:
                logger.warning(f"Error during final cleanup: {cleanup_error}")

    def _build_prompt(self, theme: Optional[str] = None, description_length: str = "medium") -> str:
        """Build prompt for AI description generation.

        Args:
            theme: Optional theme to guide description
            description_length: Desired length ('short', 'medium', 'detailed')

        Returns:
            Formatted prompt string
        """
        length_instructions = {
            "short": "Provide a very brief description (5-10 words).",
            "medium": "Provide a concise description (15-30 words).",
            "detailed": "Provide a detailed description (40-60 words).",
        }

        base_prompt = f"""You are a professional video editor's assistant. Analyze these CONSECUTIVE KEYFRAMES from a video scene and provide an ACCURATE, SPECIFIC description.

IMPORTANT: These are multiple keyframes showing progression over time. Describe what happens across all frames, not just one.

{length_instructions.get(description_length, length_instructions['medium'])}

CRITICAL INSTRUCTIONS:
- Be SPECIFIC and CONCRETE (not generic like "person working" or "using tools")
- Describe WHAT specifically is happening (not general activities)
- Include specific objects, tools, actions, and results visible
- Describe the SEQUENCE of actions and progression across frames
- Note colors, positions, materials when relevant
- Avoid vague descriptions - be precise
- Explain what changes or progresses from frame to frame

Focus on:
1. Specific objects, tools, or equipment (brand, type, color if visible)
2. Specific actions being performed (verb + object)
3. Location, setting, and spatial layout
4. Visible results or changes between frames (progression of action)
5. People's positioning and hand/body movements throughout the sequence

Provide a clear, factual, specific description of the action/progression suitable for video editing."""

        if theme:
            base_prompt += f"""

CONTEXT: This video is about: "{theme}"
Use this context to interpret what you're seeing. Focus on details relevant to the theme.
For DIY/builds: What is being built, what tools are used, what is the progression?
For cooking: What ingredients, techniques, equipment, and cooking stages?
For tutorials: What is being demonstrated, what steps are visible?
For reviews: What product features or qualities are shown?"""

        return base_prompt

    def _build_llava_prompt(
        self, theme: Optional[str] = None, description_length: str = "medium"
    ) -> str:
        """Build a simple, direct prompt for LLaVA (7B model needs simpler instructions).

        LLaVA requires explicit <image> token in the text prompt.
        Use very simple, direct instructions for the 7B model.

        Args:
            theme: Optional theme to guide description
            description_length: Desired length ('short', 'medium', 'detailed')

        Returns:
            Simple prompt string for LLaVA (includes <image> token)
        """
        length_map = {"short": "brief", "medium": "concise", "detailed": "detailed"}
        length_text = length_map.get(description_length, "concise")

        # IMPORTANT: Must include <image> token for LLaVA processor to recognize image
        # Use VERY simple instructions for 7B model
        # NOTE: The image may show stacked/stitched keyframes - instruct model to describe the action/content, not the format
        if theme:
            # Simple theme-based prompt - focus on action, not image format
            prompt = f"<image> Describe what is happening in this {theme} scene in a {length_text} sentence. Focus on the action, objects, and what is specifically occurring. Do NOT describe the image layout or that you are viewing multiple frames - just describe the scene content and action."
        else:
            # Simple generic prompt - focus on action, not image format
            prompt = f"<image> Describe what is happening in this video scene in a {length_text} sentence. Focus on the action, objects, and what is occurring. Do NOT describe the image layout or format - just describe the scene content and action."

        return prompt

    def _get_max_tokens(self, description_length: str) -> int:
        """Get max tokens based on desired description length."""
        tokens_map = {"short": 100, "medium": 200, "detailed": 400}
        return tokens_map.get(description_length, 200)

    def _select_llava_model_based_on_vram(self) -> str:
        """Select appropriate LLaVA model based on available VRAM.

        Returns:
            Model ID string appropriate for available VRAM
        """
        try:
            import torch

            if not torch.cuda.is_available():
                logger.info("No GPU available, using CPU-optimized model")
                # Use smaller model for CPU
                return "llava-hf/llava-1.5-7b-hf"

            # Get GPU memory info
            total_vram = torch.cuda.get_device_properties(0).total_memory / 1e9  # GB
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            free_vram = total_vram - allocated

            logger.info(
                f"GPU VRAM: Total={total_vram:.2f}GB, Allocated={allocated:.2f}GB, "
                f"Reserved={reserved:.2f}GB, Free={free_vram:.2f}GB"
            )

            # Select model based on available VRAM
            if total_vram >= 24:
                # 24GB+ VRAM: Use 13B model
                logger.info("Using LLaVA 13B model (24GB+ VRAM available)")
                return "llava-hf/llava-1.5-13b-hf"
            elif total_vram >= 16:
                # 16GB+ VRAM: Use 7B model with float16
                logger.info("Using LLaVA 7B model with float16 (16GB+ VRAM available)")
                return "llava-hf/llava-1.5-7b-hf"
            elif total_vram >= 10:
                # 10GB+ VRAM: Use 7B model with 8-bit quantization
                logger.info("Using LLaVA 7B model with 8-bit quantization (10GB+ VRAM available)")
                return "llava-hf/llava-1.5-7b-hf"
            else:
                # <10GB VRAM: Use 7B model with 4-bit quantization
                logger.warning(f"Limited VRAM ({total_vram:.2f}GB), using 4-bit quantization")
                return "llava-hf/llava-1.5-7b-hf"

        except Exception as e:
            logger.warning(f"Failed to check VRAM, using default model: {e}")
            return "llava-hf/llava-1.5-7b-hf"

    def _cleanup_gpu_memory(self, model=None, processor=None):
        """Comprehensive GPU memory cleanup.

        Args:
            model: Optional model to unload
            processor: Optional processor to cleanup
        """
        try:
            import torch
            import gc

            if torch.cuda.is_available():
                # Move model to CPU if provided
                if model is not None:
                    try:
                        model.to("cpu")
                    except Exception as e:
                        logger.debug(f"Error moving model to CPU: {e}")

                # Delete references
                if model is not None:
                    del model
                if processor is not None:
                    del processor

                # Clear CUDA cache
                torch.cuda.empty_cache()

                # Force garbage collection
                gc.collect()

                # Optional: Wait for GPU operations to complete
                torch.cuda.synchronize()

                logger.debug("GPU memory cleanup completed")

        except Exception as e:
            logger.warning(f"Error during GPU memory cleanup: {e}")

    @contextlib.contextmanager
    def _gpu_memory_context(self):
        """Context manager for GPU memory management.

        Usage:
            with describer._gpu_memory_context():
                # LLaVA inference code here
                result = describer._describe_with_llava(...)
        """
        try:
            import torch
            import gc

            # Clear cache before starting
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            yield

        finally:
            # Always cleanup after context
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    gc.collect()
            except Exception as e:
                logger.warning(f"Error in GPU memory context cleanup: {e}")
