"""AI-powered scene description generation module."""

import base64
import logging
import os
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
        self.api_key = api_key or os.getenv(f"{model.upper()}_API_KEY")
        
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
                scene["description"] = f"Failed to generate description: {str(e)}"

        return scenes

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
            "model": "gpt-4-vision-preview",
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
        """Generate description using local LLaVA model."""
        # Note: LLaVA requires local model installation
        # This is a placeholder implementation
        raise NotImplementedError("LLaVA support not yet implemented")

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
        
        base_prompt = f"""Analyze these video frames from a single scene and describe what is happening.
        
        {length_instructions.get(description_length, length_instructions['medium'])}
        
        Focus on:
        1. Main objects and people visible
        2. Actions taking place
        3. Setting and environment
        4. Notable visual details
        
        Provide a clear, factual description suitable for video editing."""
        
        if theme:
            base_prompt += f"""
            
            Important context: This video is about "{theme}". 
            Please tailor your description to focus on elements relevant to this theme.
            For example, if this is a DIY/build video, focus on tools, assembly steps, and progress indicators.
            If this is a cooking video, focus on ingredients, cooking techniques, and preparation steps."""
        
        return base_prompt

    def _get_max_tokens(self, description_length: str) -> int:
        """Get max tokens based on desired description length."""
        tokens_map = {
            "short": 100,
            "medium": 200,
            "detailed": 400
        }
        return tokens_map.get(description_length, 200)