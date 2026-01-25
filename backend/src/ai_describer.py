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
            model: AI model to use ('openai', 'claude', 'gemini', 'llava')
            api_key: API key for the selected model
        """
        self.model = model
        
        # Handle gemini/google naming inconsistency
        env_var_name = f"{model.upper()}_API_KEY"
        if model == "gemini":
            # Try GEMINI_API_KEY first, then GOOGLE_API_KEY as fallback
            self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        else:
            self.api_key = api_key or os.getenv(env_var_name)
        
        if not self.api_key and model != "llava":
            logger.warning(f"No API key provided for {model}. Some features may not work.")

    def generate_descriptions(
        self, 
        scenes: List[Dict[str, Any]], 
        theme: Optional[str] = None,
        description_length: str = "medium"
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
        if not self.api_key and self.model not in ['llava']:
            logger.warning(f"No API key for {self.model}, using mock descriptions")
            return self._generate_mock_descriptions(scenes, theme, description_length)
        
        for scene in scenes:
            if not scene.get("keyframes"):
                logger.warning(f"No keyframes for scene {scene['scene_id']}, skipping description")
                scene["description"] = "No description available (missing keyframes)"
                continue

            try:
                description = self._describe_scene(
                    keyframes=scene["keyframes"],
                    theme=theme,
                    description_length=description_length
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
        description_length: str = "medium"
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
        self, 
        scene_id: int, 
        theme: Optional[str] = None,
        description_length: str = "medium"
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
            ]
        }
        
        # Determine which theme category to use
        theme_key = "default"
        if theme:
            theme_lower = theme.lower()
            if any(keyword in theme_lower for keyword in ["diy", "build", "woodwork", "craft"]):
                theme_key = "DIY"
            elif any(keyword in theme_lower for keyword in ["cook", "food", "recipe", "baking"]):
                theme_key = "cooking"
            elif any(keyword in theme_lower for keyword in ["tutorial", "how-to", "guide", "instruction"]):
                theme_key = "tutorial"
            elif any(keyword in theme_lower for keyword in ["review", "comparison", "product", "test"]):
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
        if self.model == 'llava':
            return ""
        
        # For other models without API key, return blank
        if not self.api_key:
            return ""
        
        return description

    def _describe_scene(
        self, 
        keyframes: List[str], 
        theme: Optional[str] = None,
        description_length: str = "medium"
    ) -> str:
        """Generate description for a single scene.
        
        Args:
            keyframes: List of keyframe image paths
            theme: Optional theme to guide description
            description_length: Desired length ('short', 'medium', 'detailed')
            
        Returns:
            AI-generated description
        """
        if self.model == "openai":
            return self._describe_with_openai(keyframes, theme, description_length)
        elif self.model == "claude":
            return self._describe_with_claude(keyframes, theme, description_length)
        elif self.model == "gemini":
            return self._describe_with_gemini(keyframes, theme, description_length)
        elif self.model == "llava":
            return self._describe_with_llava(keyframes, theme, description_length)
        else:
            raise ValueError(f"Unsupported model: {self.model}")

    def _describe_with_openai(
        self, 
        keyframes: List[str], 
        theme: Optional[str] = None,
        description_length: str = "medium"
    ) -> str:
        """Generate description using OpenAI GPT-4 Vision."""
        if not self.api_key:
            raise ValueError("OpenAI API key not provided")

        # Prepare images
        images = []
        for keyframe in keyframes:
            with open(keyframe, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode("utf-8")
                images.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                })

        # Build prompt based on theme and length
        prompt = self._build_prompt(theme, description_length)
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    *images
                ]
            }
        ]

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        payload = {
            "model": "gpt-4o",
            "messages": messages,
            "max_tokens": self._get_max_tokens(description_length)
        }

        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=30
        )
        
        if response.status_code != 200:
            raise ValueError(f"OpenAI API error: {response.text}")
        
        return response.json()["choices"][0]["message"]["content"]

    def _describe_with_claude(
        self, 
        keyframes: List[str], 
        theme: Optional[str] = None,
        description_length: str = "medium"
    ) -> str:
        """Generate description using Claude 3."""
        if not self.api_key:
            raise ValueError("Claude API key not provided")

        # Prepare images
        images = []
        for keyframe in keyframes:
            with open(keyframe, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode("utf-8")
                images.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": base64_image
                    }
                })

        # Build prompt
        prompt = self._build_prompt(theme, description_length)
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    *images
                ]
            }
        ]

        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01"
        }

        payload = {
            "model": "claude-3-opus-20240229",
            "max_tokens": self._get_max_tokens(description_length),
            "messages": messages
        }

        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers=headers,
            json=payload,
            timeout=30
        )
        
        if response.status_code != 200:
            raise ValueError(f"Claude API error: {response.text}")
        
        return response.json()["content"][0]["text"]

    def _describe_with_gemini(
        self, 
        keyframes: List[str], 
        theme: Optional[str] = None,
        description_length: str = "medium"
    ) -> str:
        """Generate description using Google Gemini."""
        if not self.api_key:
            raise ValueError("Gemini API key not provided")

        # Note: Gemini implementation would require google-generativeai library
        # This is a placeholder implementation
        raise NotImplementedError("Gemini support not yet implemented")

    def _describe_with_llava(
        self, 
        keyframes: List[str], 
        theme: Optional[str] = None,
        description_length: str = "medium"
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
            from transformers import AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig
            from PIL import Image
            import torch
            import gc
        except ImportError as e:
            logger.error(f"LLaVA dependencies not installed. Install with: pip install torch transformers pillow bitsandbytes accelerate")
            raise ValueError(
                "LLaVA dependencies not found. Install with: pip install torch transformers pillow bitsandbytes accelerate"
            )
        
        # Use memory context for the entire LLaVA inference
        with self._gpu_memory_context():
            return self._describe_with_llava_internal(
                keyframes, theme, description_length
            )
    
    def _describe_with_llava_internal(
        self, 
        keyframes: List[str], 
        theme: Optional[str] = None,
        description_length: str = "medium"
    ) -> str:
        """Internal implementation of LLaVA description generation.
        
        This method is called within the GPU memory context.
        """
        try:
            from transformers import AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig
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
                        device_map="auto"
                    )
                elif free_vram >= 10:
                    # Use 8-bit quantization (need ~10GB buffer for safety)
                    logger.info("Using 8-bit quantization (moderate VRAM)")
                    quantization_config = BitsAndBytesConfig(
                        load_in_8bit=True,
                        llm_int8_enable_fp32_cpu_offload=True
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
                        llm_int8_has_fp16_weight=False
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
                    model_id,
                    torch_dtype=torch.float32,
                    low_cpu_mem_usage=True,
                    device_map="cpu"
                )
            
            # Enable gradient checkpointing if model supports it (for training scenarios)
            if hasattr(model, "gradient_checkpointing_enable"):
                model.gradient_checkpointing_enable()
                logger.debug("Enabled gradient checkpointing for memory efficiency")
            
            # Load and process keyframes
            images = []
            for keyframe_path in keyframes[:1]:  # Use first keyframe for speed
                try:
                    img = Image.open(keyframe_path).convert('RGB')
                    # Resize large images to reduce memory usage
                    # LLaVA works with 336x336, but smaller images still work
                    if img.size[0] > 336 or img.size[1] > 336:
                        img.thumbnail((336, 336), Image.Resampling.LANCZOS)
                    images.append(img)
                except Exception as e:
                    logger.warning(f"Failed to load keyframe {keyframe_path}: {e}")
            
            if not images:
                raise ValueError("No valid keyframes to analyze")
            
            # Build LLaVA-specific prompt
            prompt = self._build_llava_prompt(theme, description_length)
            logger.debug(f"LLaVA prompt: {prompt}")
            
            # Prepare inputs
            inputs = processor(images=images[0], text=prompt, return_tensors='pt').to(model.device)
            
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
            prompt_len = inputs['input_ids'].shape[1]
            new_tokens = output_ids[0][prompt_len:]
            description = processor.decode(new_tokens, skip_special_tokens=True).strip()
            
            # Clean up artifacts
            description = description.replace("[/INST]", "").strip()
            if "Assistant:" in description:
                description = description.split("Assistant:")[-1].strip()
            
            # Check for empty or very short responses - return blank to indicate failure
            if not description or len(description) < 10:
                logger.warning(f"LLaVA generated very short response ({len(description)} chars), returning blank")
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
                "important step"
            ]
            
            description_lower = description.lower()
            is_generic = any(phrase in description_lower for phrase in generic_phrases)
            
            if is_generic:
                logger.warning(f"LLaVA generated generic description: '{description[:50]}...'. Returning blank.")
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
                if hasattr(self, '_llava_inference_count'):
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
            "detailed": "Provide a detailed description (40-60 words)."
        }
        
        base_prompt = f"""You are a professional video editor's assistant. Analyze these keyframes from a video scene and provide an ACCURATE, SPECIFIC description.

{length_instructions.get(description_length, length_instructions['medium'])}

CRITICAL INSTRUCTIONS:
- Be SPECIFIC and CONCRETE (not generic like "person working" or "using tools")
- Describe WHAT specifically is happening (not general activities)
- Include specific objects, tools, actions, and results visible
- Describe the SEQUENCE of actions if multiple frames show progression
- Note colors, positions, materials when relevant
- Avoid vague descriptions - be precise

Focus on:
1. Specific objects, tools, or equipment (brand, type, color if visible)
2. Specific actions being performed (verb + object)
3. Location, setting, and spatial layout
4. Any visible results or changes between frames
5. People's positioning and hand/body movements

Provide a clear, factual, specific description suitable for video editing."""
        
        if theme:
            base_prompt += f"""

CONTEXT: This video is about: "{theme}"
Use this context to interpret what you're seeing. Focus on details relevant to the theme.
For DIY/builds: What is being built, what tools are used, what is the progression?
For cooking: What ingredients, techniques, equipment, and cooking stages?
For tutorials: What is being demonstrated, what steps are visible?
For reviews: What product features or qualities are shown?"""
        
        return base_prompt

    def _build_llava_prompt(self, theme: Optional[str] = None, description_length: str = "medium") -> str:
        """Build a simple, direct prompt for LLaVA (7B model needs simpler instructions).
        
        LLaVA requires explicit <image> token in the text prompt.
        Use very simple, direct instructions for the 7B model.
        
        Args:
            theme: Optional theme to guide description
            description_length: Desired length ('short', 'medium', 'detailed')
            
        Returns:
            Simple prompt string for LLaVA (includes <image> token)
        """
        length_map = {
            "short": "brief",
            "medium": "concise", 
            "detailed": "detailed"
        }
        length_text = length_map.get(description_length, "concise")
        
        # IMPORTANT: Must include <image> token for LLaVA processor to recognize image
        # Use VERY simple instructions for 7B model
        if theme:
            # Simple theme-based prompt
            prompt = f"<image> Describe this {theme} scene in a {length_text} sentence. Be specific about what you see."
        else:
            # Simple generic prompt
            prompt = f"<image> Describe what you see in this video scene in a {length_text} sentence. Be specific."
        
        return prompt

    def _get_max_tokens(self, description_length: str) -> int:
        """Get max tokens based on desired description length."""
        tokens_map = {
            "short": 100,
            "medium": 200,
            "detailed": 400
        }
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
            
            logger.info(f"GPU VRAM: Total={total_vram:.2f}GB, Allocated={allocated:.2f}GB, "
                       f"Reserved={reserved:.2f}GB, Free={free_vram:.2f}GB")
            
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
                        model.to('cpu')
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