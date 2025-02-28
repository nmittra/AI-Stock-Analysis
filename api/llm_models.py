"""Module for handling open-source LLM models integration."""

from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch
import streamlit as st
from typing import Optional, Dict

class OpenSourceLLMClient:
    """Client for handling open-source LLM models."""

    # Add this import at the top of the file
    import hashlib
    import json
    import os
    from functools import lru_cache
    
    # Add these class variables
    _CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cache")
    
    def __init__(self, model_name: str, device: Optional[str] = None, quantize: bool = False):
        """Initialize the LLM model.

        Args:
            model_name: Name of the model to use (e.g., 'falcon-7b', 'llama-2-13b')
            device: Device to use ('cuda', 'cpu', or None for auto-detection)
            quantize: Whether to use quantization to reduce memory usage
        """
        self.model_name = model_name
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        self.quantize = quantize
        # Use the class-level model registry
        self._model_registry = {
            'falcon-7b': 'tiiuae/falcon-7b',
            'llama-2-13b': 'meta-llama/Llama-2-13b-hf',
            # Add more models here
        }
        
        # Ensure cache directory exists
        os.makedirs(self._CACHE_DIR, exist_ok=True)

    def load_model(self):
        """Load the selected model and tokenizer."""
        try:
            if self.model_name not in self._model_registry:
                raise ValueError(f"Unsupported model: {self.model_name}")
                
            model_path = self._model_registry[self.model_name]
            
            # Show loading indicator in Streamlit
            with st.spinner(f"Loading {self.model_name}..."):
                try:
                    self.tokenizer = AutoTokenizer.from_pretrained(model_path)
                    
                    # Configure model loading options based on quantization setting
                    model_kwargs = {
                        'device_map': 'auto',
                        'low_cpu_mem_usage': True
                    }
                    
                    if self.quantize:
                        # Use 8-bit quantization if requested
                        model_kwargs.update({
                            'load_in_8bit': True,
                        })
                        st.info("Loading quantized model (8-bit) to reduce memory usage")
                    else:
                        # Use standard precision based on device
                        model_kwargs.update({
                            'torch_dtype': torch.float16 if self.device == 'cuda' else torch.float32
                        })
                    
                    self.model = AutoModelForCausalLM.from_pretrained(
                        model_path,
                        **model_kwargs
                    )

                    # Create text generation pipeline
                    self.pipeline = pipeline(
                        'text-generation',
                        model=self.model,
                        tokenizer=self.tokenizer,
                        device=self.device
                    )
                    
                    return True
                    
                except ImportError as e:
                    if "bitsandbytes" in str(e) and self.quantize:
                        st.error("Quantization requires the 'bitsandbytes' package. Install with: pip install bitsandbytes")
                        # Fall back to non-quantized model
                        self.quantize = False
                        return self.load_model()
                    else:
                        raise
                        
        except Exception as e:
            st.error(f"Error loading {self.model_name}: {str(e)}")
            return False

    def _get_cache_key(self, prompt: str, params: Dict) -> str:
        """Generate a unique cache key for a prompt and parameters.
        
        Args:
            prompt: The input prompt
            params: Generation parameters
            
        Returns:
            str: Cache key
        """
        # Create a string representation of the parameters
        param_str = json.dumps(params, sort_keys=True)
        
        # Create a hash of the prompt and parameters
        key = f"{self.model_name}_{hashlib.md5((prompt + param_str).encode()).hexdigest()}"
        
        return key
    
    def _get_from_cache(self, cache_key: str) -> Optional[Dict]:
        """Retrieve a cached response if available.
        
        Args:
            cache_key: The cache key
            
        Returns:
            Optional[Dict]: Cached response or None
        """
        cache_file = os.path.join(self._CACHE_DIR, f"{cache_key}.json")
        
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                st.warning(f"Failed to load cached response: {str(e)}")
                
        return None
    
    def _save_to_cache(self, cache_key: str, response: Dict) -> None:
        """Save a response to the cache.
        
        Args:
            cache_key: The cache key
            response: The response to cache
        """
        cache_file = os.path.join(self._CACHE_DIR, f"{cache_key}.json")
        
        try:
            with open(cache_file, 'w') as f:
                json.dump(response, f)
        except Exception as e:
            st.warning(f"Failed to cache response: {str(e)}")

    def generate_analysis(
        self, 
        prompt: str, 
        max_length: int = 500,
        temperature: float = 0.7,
        top_p: float = 0.9,
        num_return_sequences: int = 1,
        use_cache: bool = True
    ) -> Optional[Dict]:
        """Generate stock analysis using the loaded model.

        Args:
            prompt: The analysis prompt
            max_length: Maximum length of generated text
            temperature: Sampling temperature (higher = more creative)
            top_p: Nucleus sampling parameter
            num_return_sequences: Number of responses to generate
            use_cache: Whether to use cached responses

        Returns:
            dict: Analysis results containing action and justification
        """
        # Create parameters dictionary for cache key
        params = {
            'max_length': max_length,
            'temperature': temperature,
            'top_p': top_p,
            'num_return_sequences': num_return_sequences
        }
        
        # Check cache first if enabled
        if use_cache:
            cache_key = self._get_cache_key(prompt, params)
            cached_response = self._get_from_cache(cache_key)
            
            if cached_response:
                st.info("Using cached response")
                return cached_response
        
        try:
            if not self.model or not self.tokenizer:
                if not self.load_model():
                    return {
                        'action': 'Hold',
                        'justification': 'Failed to load model. Please check logs for details.'
                    }

            # Generate response
            with st.spinner("Generating analysis..."):
                try:
                    response = self.pipeline(
                        prompt,
                        max_length=max_length,
                        temperature=temperature,
                        top_p=top_p,
                        num_return_sequences=num_return_sequences,
                        do_sample=True
                    )[0]['generated_text']
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        st.error("GPU out of memory. Try using a quantized model or reducing model size.")
                        # Try to recover by clearing cache
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        return {
                            'action': 'Hold',
                            'justification': 'Model ran out of memory. Try using a smaller model or enabling quantization.'
                        }
                    else:
                        raise

            # Extract relevant parts
            result = self._parse_response(response)
            
            # Cache the result if caching is enabled
            if use_cache and result:
                cache_key = self._get_cache_key(prompt, params)
                self._save_to_cache(cache_key, result)
                
            return result

        except Exception as e:
            st.error(f"Error generating analysis with {self.model_name}: {str(e)}")
            # Return a graceful fallback response
            return {
                'action': 'Hold',
                'justification': f'Error during analysis: {str(e)}. Please try again or select a different model.'
            }

    def generate_analysis_with_streaming(
        self, 
        prompt: str,
        max_tokens: int = 500,
        temperature: float = 0.7,
        use_cache: bool = True
    ) -> Optional[Dict]:
        """Generate analysis with streaming output for better UX.
        
        Args:
            prompt: The analysis prompt
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            use_cache: Whether to use cached responses
            
        Returns:
            dict: Analysis results
        """
        # Create parameters dictionary for cache key
        params = {
            'max_tokens': max_tokens,
            'temperature': temperature,
            'streaming': True
        }
        
        # Check cache first if enabled
        if use_cache:
            cache_key = self._get_cache_key(prompt, params)
            cached_response = self._get_from_cache(cache_key)
            
            if cached_response:
                st.info("Using cached response")
                return cached_response
        
        try:
            if not self.model or not self.tokenizer:
                if not self.load_model():
                    return {
                        'action': 'Hold',
                        'justification': 'Failed to load model. Please check logs for details.'
                    }
                    
            # Create placeholder for streaming output
            output_placeholder = st.empty()
            
            # Setup streaming
            generated_text = ""
            
            try:
                # Generate tokens one by one
                input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
                
                with st.spinner("Generating analysis..."):
                    for i in range(max_tokens):  # Max tokens
                        try:
                            with torch.no_grad():
                                outputs = self.model.generate(
                                    input_ids,
                                    max_length=input_ids.shape[1] + 1,
                                    do_sample=True,
                                    temperature=temperature,
                                )
                            
                            next_token = outputs[0][-1]
                            input_ids = torch.cat([input_ids, next_token.unsqueeze(0).unsqueeze(0)], dim=1)
                            
                            # Decode the new token
                            next_word = self.tokenizer.decode(next_token)
                            generated_text += next_word
                            
                            # Update display
                            output_placeholder.markdown(f"**Generating:** {generated_text}")
                            
                            # Check for stopping condition (e.g., EOS token)
                            if next_token.item() == self.tokenizer.eos_token_id:
                                break
                        except RuntimeError as e:
                            if "out of memory" in str(e).lower():
                                output_placeholder.error("GPU out of memory. Try using a quantized model.")
                                if torch.cuda.is_available():
                                    torch.cuda.empty_cache()
                                return {
                                    'action': 'Hold',
                                    'justification': 'Model ran out of memory during generation.'
                                }
                            else:
                                raise
            except Exception as e:
                output_placeholder.error(f"Error during streaming generation: {str(e)}")
                return {
                    'action': 'Hold',
                    'justification': f'Error during streaming generation: {str(e)}'
                }
            
            # Final result
            result = self._parse_response(generated_text)
            
            # Cache the result if caching is enabled
            if use_cache and result:
                cache_key = self._get_cache_key(prompt, params)
                self._save_to_cache(cache_key, result)
                
            return result
            
        except Exception as e:
            st.error(f"Error generating analysis with {self.model_name}: {str(e)}")
            return {
                'action': 'Hold',
                'justification': f'Error during analysis: {str(e)}. Please try again or select a different model.'
            }

    @staticmethod
    def is_available(model_name: str) -> bool:
        """Check if the model can be loaded.

        Args:
            model_name: Name of the model to check

        Returns:
            bool: True if model can be loaded, False otherwise
        """
        # Create a temporary instance to access the model registry
        temp_client = OpenSourceLLMClient(model_name)
        
        try:
            if model_name not in temp_client._model_registry:
                return False
                
            model_path = temp_client._model_registry[model_name]

            # Just try to load the tokenizer as a quick check
            AutoTokenizer.from_pretrained(model_path)
            return True
        except Exception:
            return False

    def unload_model(self):
        """Unload model from memory."""
        if self.model is not None:
            del self.model
            del self.tokenizer
            del self.pipeline
            self.model = None
            self.tokenizer = None
            self.pipeline = None
            
            # Force garbage collection
            import gc
            gc.collect()
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            return True
        return False

    @classmethod
    def available_models(cls) -> Dict[str, str]:
        """Return a dictionary of available models and their descriptions.
        
        Returns:
            Dict mapping model IDs to human-readable descriptions
        """
        return {
            'falcon-7b': 'Falcon 7B - Balanced performance (7B parameters)',
            'llama-2-13b': 'LLaMA 2 13B - Higher quality but slower (13B parameters)',
            # Add more models with descriptions
        }
    
    @staticmethod
    def create_model_selector_ui():
        """Create a Streamlit UI element for model selection.
        
        Returns:
            Selected model name
        """
        models = OpenSourceLLMClient.available_models()
        
        # Create a radio button for model selection
        selected_model = st.sidebar.radio(
            "Select LLM Model",
            list(models.keys()),
            format_func=lambda x: models[x]
        )
        
        # Show model details
        with st.sidebar.expander("Model Details"):
            if selected_model == 'falcon-7b':
                st.write("A 7B parameter model by TII UAE. Good balance of speed and quality.")
                st.write("Memory required: ~14GB GPU RAM or ~28GB CPU RAM")
            elif selected_model == 'llama-2-13b':
                st.write("A 13B parameter model by Meta. Higher quality but more resource intensive.")
                st.write("Memory required: ~26GB GPU RAM or ~52GB CPU RAM")
        
        return selected_model

    def _parse_response(self, response: str) -> Dict:
        """Parse the model response into a structured format.
        
        Args:
            response: Raw text response from the model
            
        Returns:
            Dict containing parsed response
        """
        # Create parameters dictionary for cache key
        params = {
            'max_length': max_length,
            'temperature': temperature,
            'top_p': top_p,
            'num_return_sequences': num_return_sequences
        }
        
        # Check cache first if enabled
        if use_cache:
            cache_key = self._get_cache_key(prompt, params)
            cached_response = self._get_from_cache(cache_key)
            
            if cached_response:
                st.info("Using cached response")
                return cached_response
        
        try:
            if not self.model or not self.tokenizer:
                if not self.load_model():
                    return {
                        'action': 'Hold',
                        'justification': 'Failed to load model. Please check logs for details.'
                    }

            # Generate response
            with st.spinner("Generating analysis..."):
                try:
                    response = self.pipeline(
                        prompt,
                        max_length=max_length,
                        temperature=temperature,
                        top_p=top_p,
                        num_return_sequences=num_return_sequences,
                        do_sample=True
                    )[0]['generated_text']
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        st.error("GPU out of memory. Try using a quantized model or reducing model size.")
                        # Try to recover by clearing cache
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        return {
                            'action': 'Hold',
                            'justification': 'Model ran out of memory. Try using a smaller model or enabling quantization.'
                        }
                    else:
                        raise

            # Extract relevant parts
            result = self._parse_response(response)
            
            # Cache the result if caching is enabled
            if use_cache and result:
                cache_key = self._get_cache_key(prompt, params)
                self._save_to_cache(cache_key, result)
                
            return result

        except Exception as e:
            st.error(f"Error generating analysis with {self.model_name}: {str(e)}")
            # Return a graceful fallback response
            return {
                'action': 'Hold',
                'justification': f'Error during analysis: {str(e)}. Please try again or select a different model.'
            }

    def generate_analysis_with_streaming(
        self, 
        prompt: str,
        max_tokens: int = 500,
        temperature: float = 0.7,
        use_cache: bool = True
    ) -> Optional[Dict]:
        """Generate analysis with streaming output for better UX.
        
        Args:
            prompt: The analysis prompt
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            use_cache: Whether to use cached responses
            
        Returns:
            dict: Analysis results
        """
        # Create parameters dictionary for cache key
        params = {
            'max_tokens': max_tokens,
            'temperature': temperature,
            'streaming': True
        }
        
        # Check cache first if enabled
        if use_cache:
            cache_key = self._get_cache_key(prompt, params)
            cached_response = self._get_from_cache(cache_key)
            
            if cached_response:
                st.info("Using cached response")
                return cached_response
        
        try:
            if not self.model or not self.tokenizer:
                if not self.load_model():
                    return {
                        'action': 'Hold',
                        'justification': 'Failed to load model. Please check logs for details.'
                    }
                    
            # Create placeholder for streaming output
            output_placeholder = st.empty()
            
            # Setup streaming
            generated_text = ""
            
            try:
                # Generate tokens one by one
                input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
                
                with st.spinner("Generating analysis..."):
                    for i in range(max_tokens):  # Max tokens
                        try:
                            with torch.no_grad():
                                outputs = self.model.generate(
                                    input_ids,
                                    max_length=input_ids.shape[1] + 1,
                                    do_sample=True,
                                    temperature=temperature,
                                )
                            
                            next_token = outputs[0][-1]
                            input_ids = torch.cat([input_ids, next_token.unsqueeze(0).unsqueeze(0)], dim=1)
                            
                            # Decode the new token
                            next_word = self.tokenizer.decode(next_token)
                            generated_text += next_word
                            
                            # Update display
                            output_placeholder.markdown(f"**Generating:** {generated_text}")
                            
                            # Check for stopping condition (e.g., EOS token)
                            if next_token.item() == self.tokenizer.eos_token_id:
                                break
                        except RuntimeError as e:
                            if "out of memory" in str(e).lower():
                                output_placeholder.error("GPU out of memory. Try using a quantized model.")
                                if torch.cuda.is_available():
                                    torch.cuda.empty_cache()
                                return {
                                    'action': 'Hold',
                                    'justification': 'Model ran out of memory during generation.'
                                }
                            else:
                                raise
            except Exception as e:
                output_placeholder.error(f"Error during streaming generation: {str(e)}")
                return {
                    'action': 'Hold',
                    'justification': f'Error during streaming generation: {str(e)}'
                }
            
            # Final result
            result = self._parse_response(generated_text)
            
            # Cache the result if caching is enabled
            if use_cache and result:
                cache_key = self._get_cache_key(prompt, params)
                self._save_to_cache(cache_key, result)
                
            return result
            
        except Exception as e:
            st.error(f"Error generating analysis with {self.model_name}: {str(e)}")
            return {
                'action': 'Hold',
                'justification': f'Error during analysis: {str(e)}. Please try again or select a different model.'
            }

    @staticmethod
    def is_available(model_name: str) -> bool:
        """Check if the model can be loaded.

        Args:
            model_name: Name of the model to check

        Returns:
            bool: True if model can be loaded, False otherwise
        """
        # Create a temporary instance to access the model registry
        temp_client = OpenSourceLLMClient(model_name)
        
        try:
            if model_name not in temp_client._model_registry:
                return False
                
            model_path = temp_client._model_registry[model_name]

            # Just try to load the tokenizer as a quick check
            AutoTokenizer.from_pretrained(model_path)
            return True
        except Exception:
            return False

    def unload_model(self):
        """Unload model from memory."""
        if self.model is not None:
            del self.model
            del self.tokenizer
            del self.pipeline
            self.model = None
            self.tokenizer = None
            self.pipeline = None
            
            # Force garbage collection
            import gc
            gc.collect()
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            return True
        return False

    @classmethod
    def available_models(cls) -> Dict[str, str]:
        """Return a dictionary of available models and their descriptions.
        
        Returns:
            Dict mapping model IDs to human-readable descriptions
        """
        return {
            'falcon-7b': 'Falcon 7B - Balanced performance (7B parameters)',
            'llama-2-13b': 'LLaMA 2 13B - Higher quality but slower (13B parameters)',
            # Add more models with descriptions
        }
    
    @staticmethod
    def create_model_selector_ui():
        """Create a Streamlit UI element for model selection.
        
        Returns:
            Selected model name
        """
        models = OpenSourceLLMClient.available_models()
        
        # Create a radio button for model selection
        selected_model = st.sidebar.radio(
            "Select LLM Model",
            list(models.keys()),
            format_func=lambda x: models[x]
        )
        
        # Show model details
        with st.sidebar.expander("Model Details"):
            if selected_model == 'falcon-7b':
                st.write("A 7B parameter model by TII UAE. Good balance of speed and quality.")
                st.write("Memory required: ~14GB GPU RAM or ~28GB CPU RAM")
            elif selected_model == 'llama-2-13b':
                st.write("A 13B parameter model by Meta. Higher quality but more resource intensive.")
                st.write("Memory required: ~26GB GPU RAM or ~52GB CPU RAM")
        
        return selected_model

    def _parse_response(self, response: str) -> Dict:
        """Parse the model response into a structured format.
        
        Args:
            response: Raw text response from the model
            
        Returns:
            Dict containing parsed response
        """
        # Create parameters dictionary for cache key
        params = {
            'max_length': max_length,
            'temperature': temperature,
            'top_p': top_p,
            'num_return_sequences': num_return_sequences
        }
        
        # Check cache first if enabled
        if use_cache:
            cache_key = self._get_cache_key(prompt, params)
            cached_response = self._get_from_cache(cache_key)
            
            if cached_response:
                st.info("Using cached response")
                return cached_response
        
        try:
            if not self.model or not self.tokenizer:
                if not self.load_model():
                    return {
                        'action': 'Hold',
                        'justification': 'Failed to load model. Please check logs for details.'
                    }

            # Generate response
            with st.spinner("Generating analysis..."):
                try:
                    response = self.pipeline(
                        prompt,
                        max_length=max_length,
                        temperature=temperature,
                        top_p=top_p,
                        num_return_sequences=num_return_sequences,
                        do_sample=True
                    )[0]['generated_text']
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        st.error("GPU out of memory. Try using a quantized model or reducing model size.")
                        # Try to recover by clearing cache
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        return {
                            'action': 'Hold',
                            'justification': 'Model ran out of memory. Try using a smaller model or enabling quantization.'
                        }
                    else:
                        raise

            # Extract relevant parts
            result = self._parse_response(response)
            
            # Cache the result if caching is enabled
            if use_cache and result:
                cache_key = self._get_cache_key(prompt, params)
                self._save_to_cache(cache_key, result)
                
            return result

        except Exception as e:
            st.error(f"Error generating analysis with {self.model_name}: {str(e)}")
            # Return a graceful fallback response
            return {
                'action': 'Hold',
                'justification': f'Error during analysis: {str(e)}. Please try again or select a different model.'
            }

    def generate_analysis_with_streaming(
        self, 
        prompt: str,
        max_tokens: int = 500,
        temperature: float = 0.7,
        use_cache: bool = True
    ) -> Optional[Dict]:
        """Generate analysis with streaming output for better UX.
        
        Args:
            prompt: The analysis prompt
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            use_cache: Whether to use cached responses
            
        Returns:
            dict: Analysis results
        """
        # Create parameters dictionary for cache key
        params = {
            'max_tokens': max_tokens,
            'temperature': temperature,
            'streaming': True
        }
        
        # Check cache first if enabled
        if use_cache:
            cache_key = self._get_cache_key(prompt, params)
            cached_response = self._get_from_cache(cache_key)
            
            if cached_response:
                st.info("Using cached response")
                return cached_response
        
        try:
            if not self.model or not self.tokenizer:
                if not self.load_model():
                    return {
                        'action': 'Hold',
                        'justification': 'Failed to load model. Please check logs for details.'
                    }
                    
            # Create placeholder for streaming output
            output_placeholder = st.empty()
            
            # Setup streaming
            generated_text = ""
            
            try:
                # Generate tokens one by one
                input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
                
                with st.spinner("Generating analysis..."):
                    for i in range(max_tokens):  # Max tokens
                        try:
                            with torch.no_grad():
                                outputs = self.model.generate(
                                    input_ids,
                                    max_length=input_ids.shape[1] + 1,
                                    do_sample=True,
                                    temperature=temperature,
                                )
                            
                            next_token = outputs[0][-1]
                            input_ids = torch.cat([input_ids, next_token.unsqueeze(0).unsqueeze(0)], dim=1)
                            
                            # Decode the new token
                            next_word = self.tokenizer.decode(next_token)
                            generated_text += next_word
                            
                            # Update display
                            output_placeholder.markdown(f"**Generating:** {generated_text}")
                            
                            # Check for stopping condition (e.g., EOS token)
                            if next_token.item() == self.tokenizer.eos_token_id:
                                break
                        except RuntimeError as e:
                            if "out of memory" in str(e).lower():
                                output_placeholder.error("GPU out of memory. Try using a quantized model.")
                                if torch.cuda.is_available():
                                    torch.cuda.empty_cache()
                                return {
                                    'action': 'Hold',
                                    'justification': 'Model ran out of memory during generation.'
                                }
                            else:
                                raise
            except Exception as e:
                output_placeholder.error(f"Error during streaming generation: {str(e)}")
                return {
                    'action': 'Hold',
                    'justification': f'Error during streaming generation: {str(e)}'
                }
            
            # Final result
            result = self._parse_response(generated_text)
            
            # Cache the result if caching is enabled
            if use_cache and result:
                cache_key = self._get_cache_key(prompt, params)
                self._save_to_cache(cache_key, result)
                
            return result
            
        except Exception as e:
            st.error(f"Error generating analysis with {self.model_name}: {str(e)}")
            return {
                'action': 'Hold',
                'justification': f'Error during analysis: {str(e)}. Please try again or select a different model.'
            }

    @staticmethod
    def is_available(model_name: str) -> bool:
        """Check if the model can be loaded.

        Args:
            model_name: Name of the model to check

        Returns:
            bool: True if model can be loaded, False otherwise
        """
        # Create a temporary instance to access the model registry
        temp_client = OpenSourceLLMClient(model_name)
        
        try:
            if model_name not in temp_client._model_registry:
                return False
                
            model_path = temp_client._model_registry[model_name]

            # Just try to load the tokenizer as a quick check
            AutoTokenizer.from_pretrained(model_path)
            return True
        except Exception:
            return False

    def unload_model(self):
        """Unload model from memory."""
        if self.model is not None:
            del self.model
            del self.tokenizer
            del self.pipeline
            self.model = None
            self.tokenizer = None
            self.pipeline = None
            
            # Force garbage collection
            import gc
            gc.collect()
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            return True
        return False

    @classmethod
    def available_models(cls) -> Dict[str, str]:
        """Return a dictionary of available models and their descriptions.
        
        Returns:
            Dict mapping model IDs to human-readable descriptions
        """
        return {
            'falcon-7b': 'Falcon 7B - Balanced performance (7B parameters)',
            'llama-2-13b': 'LLaMA 2 13B - Higher quality but slower (13B parameters)',
            # Add more models with descriptions
        }
    
    @staticmethod
    def create_model_selector_ui():
        """Create a Streamlit UI element for model selection.
        
        Returns:
            Selected model name
        """
        models = OpenSourceLLMClient.available_models()
        
        # Create a radio button for model selection
        selected_model = st.sidebar.radio(
            "Select LLM Model",
            list(models.keys()),
            format_func=lambda x: models[x]
        )
        
        # Show model details
        with st.sidebar.expander("Model Details"):
            if selected_model == 'falcon-7b':
                st.write("A 7B parameter model by TII UAE. Good balance of speed and quality.")
                st.write("Memory required: ~14GB GPU RAM or ~28GB CPU RAM")
            elif selected_model == 'llama-2-13b':
                st.write("A 13B parameter model by Meta. Higher quality but more resource intensive.")
                st.write("Memory required: ~26GB GPU RAM or ~52GB CPU RAM")
        
        return selected_model

    def _parse_response(self, response: str) -> Dict:
        """Parse the model response into a structured format.
        
        Args:
            response: Raw text response from the model
            
        Returns:
            Dict containing parsed response
        """
        # Create parameters dictionary for cache key
        params = {
            'max_length': max_length,
            'temperature': temperature,
            'top_p': top_p,
            'num_return_sequences': num_return_sequences
        }
        
        # Check cache first if enabled
        if use_cache:
            cache_key = self._get_cache_key(prompt, params)
            cached_response = self._get_from_cache(cache_key)
            
            if cached_response:
                st.info("Using cached response")
                return cached_response
        
        try:
            if not self.model or not self.tokenizer:
                if not self.load_model():
                    return {
                        'action': 'Hold',
                        'justification': 'Failed to load model. Please check logs for details.'
                    }

            # Generate response
            with st.spinner("Generating analysis..."):
                try:
                    response = self.pipeline(
                        prompt,
                        max_length=max_length,
                        temperature=temperature,
                        top_p=top_p,
                        num_return_sequences=num_return_sequences,
                        do_sample=True
                    )[0]['generated_text']
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        st.error("GPU out of memory. Try using a quantized model or reducing model size.")
                        # Try to recover by clearing cache
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        return {
                            'action': 'Hold',
                            'justification': 'Model ran out of memory. Try using a smaller model or enabling quantization.'
                        }
                    else:
                        raise

            # Extract relevant parts
            result = self._parse_response(response)
            
            # Cache the result if caching is enabled
            if use_cache and result:
                cache_key = self._get_cache_key(prompt, params)
                self._save_to_cache(cache_key, result)
                
            return result

        except Exception as e:
            st.error(f"Error generating analysis with {self.model_name}: {str(e)}")
            # Return a graceful fallback response
            return {
                'action': 'Hold',
                'justification': f'Error during analysis: {str(e)}. Please try again or select a different model.'
            }

    def generate_analysis_with_streaming(
        self, 
        prompt: str,
        max_tokens: int = 500,
        temperature: float = 0.7,
        use_cache: bool = True
    ) -> Optional[Dict]:
        """Generate analysis with streaming output for better UX.
        
        Args:
            prompt: The analysis prompt
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            use_cache: Whether to use cached responses
            
        Returns:
            dict: Analysis results
        """
        # Create parameters dictionary for cache key
        params = {
            'max_tokens': max_tokens,
            'temperature': temperature,
            'streaming': True
        }
        
        # Check cache first if enabled
        if use_cache:
            cache_key = self._get_cache_key(prompt, params)
            cached_response = self._get_from_cache(cache_key)
            
            if cached_response:
                st.info("Using cached response")
                return cached_response
        
        try:
            if not self.model or not self.tokenizer:
                if not self.load_model():
                    return {
                        'action': 'Hold',
                        'justification': 'Failed to load model. Please check logs for details.'
                    }
                    
            # Create placeholder for streaming output
            output_placeholder = st.empty()
            
            # Setup streaming
            generated_text = ""
            
            try:
                # Generate tokens one by one
                input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
                
                with st.spinner("Generating analysis..."):
                    for i in range(max_tokens):  # Max tokens
                        try:
                            with torch.no_grad():
                                outputs = self.model.generate(
                                    input_ids,
                                    max_length=input_ids.shape[1] + 1,
                                    do_sample=True,
                                    temperature=temperature,
                                )
                            
                            next_token = outputs[0][-1]
                            input_ids = torch.cat([input_ids, next_token.unsqueeze(0).unsqueeze(0)], dim=1)
                            
                            # Decode the new token
                            next_word = self.tokenizer.decode(next_token)
                            generated_text += next_word
                            
                            # Update display
                            output_placeholder.markdown(f"**Generating:** {generated_text}")
                            
                            # Check for stopping condition (e.g., EOS token)
                            if next_token.item() == self.tokenizer.eos_token_id:
                                break
                        except RuntimeError as e:
                            if "out of memory" in str(e).lower():
                                output_placeholder.error("GPU out of memory. Try using a quantized model.")
                                if torch.cuda.is_available():
                                    torch.cuda.empty_cache()
                                return {
                                    'action': 'Hold',
                                    'justification': 'Model ran out of memory during generation.'
                                }
                            else:
                                raise
            except Exception as e:
                output_placeholder.error(f"Error during streaming generation: {str(e)}")
                return {
                    'action': 'Hold',
                    'justification': f'Error during streaming generation: {str(e)}'
                }
            
            # Final result
            result = self._parse_response(generated_text)
            
            # Cache the result if caching is enabled
            if use_cache and result:
                cache_key = self._get_cache_key(prompt, params)
                self._save_to_cache(cache_key, result)
                
            return result
            
        except Exception as e:
            st.error(f"Error generating analysis with {self.model_name}: {str(e)}")
            return {
                'action': 'Hold',
                'justification': f'Error during analysis: {str(e)}. Please try again or select a different model.'
            }

    @staticmethod
    def is_available(model_name: str) -> bool:
        """Check if the model can be loaded.

        Args:
            model_name: Name of the model to check

        Returns:
            bool: True if model can be loaded, False otherwise
        """
        # Create a temporary instance to access the model registry
        temp_client = OpenSourceLLMClient(model_name)
        
        try:
            if model_name not in temp_client._model_registry:
                return False
                
            model_path = temp_client._model_registry[model_name]

            # Just try to load the tokenizer as a quick check
            AutoTokenizer.from_pretrained(model_path)
            return True
        except Exception:
            return False

    def unload_model(self):
        """Unload model from memory."""
        if self.model is not None:
            del self.model
            del self.tokenizer
            del self.pipeline
            self.model = None
            self.tokenizer = None
            self.pipeline = None
            
            # Force garbage collection
            import gc
            gc.collect()
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            return True
        return False

    @classmethod
    def available_models(cls) -> Dict[str, str]:
        """Return a dictionary of available models and their descriptions.
        
        Returns:
            Dict mapping model IDs to human-readable descriptions
        """
        return {
            'falcon-7b': 'Falcon 7B - Balanced performance (7B parameters)',
            'llama-2-13b': 'LLaMA 2 13B - Higher quality but slower (13B parameters)',
            # Add more models with descriptions
        }
    
    @staticmethod
    def create_model_selector_ui():
        """Create a Streamlit UI element for model selection.
        
        Returns:
            Selected model name
        """
        models = OpenSourceLLMClient.available_models()
        
        # Create a radio button for model selection
        selected_model = st.sidebar.radio(
            "Select LLM Model",
            list(models.keys()),
            format_func=lambda x: models[x]
        )
        
        # Show model details
        with st.sidebar.expander("Model Details"):
            if selected_model == 'falcon-7b':
                st.write("A 7B parameter model by TII UAE. Good balance of speed and quality.")
                st.write("Memory required: ~14GB GPU RAM or ~28GB CPU RAM")
            elif selected_model == 'llama-2-13b':
                st.write("A 13B parameter model by Meta. Higher quality but more resource intensive.")
                st.write("Memory required: ~26GB GPU RAM or ~52GB CPU RAM")
        
        return selected_model

    def _parse_response(self, response: str) -> Dict:
        """Parse the model response into a structured format.
        
        Args:
            response: Raw text response from the model
            
        Returns:
            Dict containing parsed response
        """
        # Create parameters dictionary for cache key
        params = {
            'max_length': max_length,
            'temperature': temperature,
            'top_p': top_p,
            'num_return_sequences': num_return_sequences
        }
        
        # Check cache first if enabled
        if use_cache:
            cache_key = self._get_cache_key(prompt, params)
            cached_response = self._get_from_cache(cache_key)
            
            if cached_response:
                st.info("Using cached response")
                return cached_response
        
        try:
            if not self.model or not self.tokenizer:
                if not self.load_model():
                    return {
                        'action': 'Hold',
                        'justification': 'Failed to load model. Please check logs for details.'
                    }

            # Generate response
            with st.spinner("Generating analysis..."):
                try:
                    response = self.pipeline(
                        prompt,
                        max_length=max_length,
                        temperature=temperature,
                        top_p=top_p,
                        num_return_sequences=num_return_sequences,
                        do_sample=True
                    )[0]['generated_text']
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        st.error("GPU out of memory. Try using a quantized model or reducing model size.")
                        # Try to recover by clearing cache
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        return {
                            'action': 'Hold',
                            'justification': 'Model ran out of memory. Try using a smaller model or enabling quantization.'
                        }
                    else:
                        raise

            # Extract relevant parts
            result = self._parse_response(response)
            
            # Cache the result if caching is enabled
            if use_cache and result:
                cache_key = self._get_cache_key(prompt, params)
                self._save_to_cache(cache_key, result)
                
            return result

        except Exception as e:
            st.error(f"Error generating analysis with {self.model_name}: {str(e)}")
            # Return a graceful fallback response
            return {
                'action': 'Hold',
                'justification': f'Error during analysis: {str(e)}. Please try again or select a different model.'
            }

    def generate_analysis_with_streaming(
        self, 
        prompt: str,
        max_tokens: int = 500,
        temperature: float = 0.7,
        use_cache: bool = True
    ) -> Optional[Dict]:
        """Generate analysis with streaming output for better UX.
        
        Args:
            prompt: The analysis prompt
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            use_cache: Whether to use cached responses
            
        Returns:
            dict: Analysis results
        """
        # Create parameters dictionary for cache key
        params = {
            'max_tokens': max_tokens,
            'temperature': temperature,
            'streaming': True
        }
        
        # Check cache first if enabled
        if use_cache:
            cache_key = self._get_cache_key(prompt, params)
            cached_response = self._get_from_cache(cache_key)
            
            if cached_response:
                st.info("Using cached response")
                return cached_response
        
        try:
            if not self.model or not self.tokenizer:
                if not self.load_model():
                    return {
                        'action': 'Hold',
                        'justification': 'Failed to load model. Please check logs for details.'
                    }
                    
            # Create placeholder for streaming output
            output_placeholder = st.empty()
            
            # Setup streaming
            generated_text = ""
            
            try:
                # Generate tokens one by one
                input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
                
                with st.spinner("Generating analysis..."):
                    for i in range(max_tokens):  # Max tokens
                        try:
                            with torch.no_grad():
                                outputs = self.model.generate(
                                    input_ids,
                                    max_length=input_ids.shape[1] + 1,
                                    do_sample=True,
                                    temperature=temperature,
                                )
                            
                            next_token = outputs[0][-1]
                            input_ids = torch.cat([input_ids, next_token.unsqueeze(0).unsqueeze(0)], dim=1)
                            
                            # Decode the new token
                            next_word = self.tokenizer.decode(next_token)
                            generated_text += next_word
                            
                            # Update display
                            output_placeholder.markdown(f"**Generating:** {generated_text}")
                            
                            # Check for stopping condition (e.g., EOS token)
                            if next_token.item() == self.tokenizer.eos_token_id:
                                break
                        except RuntimeError as e:
                            if "out of memory" in str(e).lower():
                                output_placeholder.error("GPU out of memory. Try using a quantized model.")
                                if torch.cuda.is_available():
                                    torch.cuda.empty_cache()
                                return {
                                    'action': 'Hold',
                                    'justification': 'Model ran out of memory during generation.'
                                }
                            else:
                                raise
            except Exception as e:
                output_placeholder.error(f"Error during streaming generation: {str(e)}")
                return {
                    'action': 'Hold',
                    'justification': f'Error during streaming generation: {str(e)}'
                }
            
            # Final result
            result = self._parse_response(generated_text)
            
            # Cache the result if caching is enabled
            if use_cache and result:
                cache_key = self._get_cache_key(prompt, params)
                self._save_to_cache(cache_key, result)
                
            return result
            
        except Exception as e:
            st.error(f"Error generating analysis with {self.model_name}: {str(e)}")
            return {
                'action': 'Hold',
                'justification': f'Error during analysis: {str(e)}. Please try again or select a different model.'
            }

    @staticmethod
    def is_available(model_name: str) -> bool:
        """Check if the model can be loaded.

        Args:
            model_name: Name of the model to check

        Returns:
            bool: True if model can be loaded, False otherwise
        """
        # Create a temporary instance to access the model registry
        temp_client = OpenSourceLLMClient(model_name)
        
        try:
            if model_name not in temp_client._model_registry:
                return False
                
            model_path = temp_client._model_registry[model_name]

            # Just try to load the tokenizer as a quick check
            AutoTokenizer.from_pretrained(model_path)
            return True
        except Exception:
            return False

    def unload_model(self):
        """Unload model from memory."""
        if self.model is not None:
            del self.model
            del self.tokenizer
            del self.pipeline
            self.model = None
            self.tokenizer = None
            self.pipeline = None
            
            # Force garbage collection
            import gc
            gc.collect()
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            return True
        return False

    @classmethod
    def available_models(cls) -> Dict[str, str]:
        """Return a dictionary of available models and their descriptions.
        
        Returns:
            Dict mapping model IDs to human-readable descriptions
        """
        return {
            'falcon-7b': 'Falcon 7B - Balanced performance (7B parameters)',
            'llama-2-13b': 'LLaMA 2 13B - Higher quality but slower (13B parameters)',
            # Add more models with descriptions
        }
    
    @staticmethod
    def create_model_selector_ui():
        """Create a Streamlit UI element for model selection.
        
        Returns:
            Selected model name
        """
        models = OpenSourceLLMClient.available_models()
        
        # Create a radio button for model selection
        selected_model = st.sidebar.radio(
            "Select LLM Model",
            list(models.keys()),
            format_func=lambda x: models[x]
        )
        
        # Show model details
        with st.sidebar.expander("Model Details"):
            if selected_model == 'falcon-7b':
                st.write("A 7B parameter model by TII UAE. Good balance of speed and quality.")
                st.write("Memory required: ~14GB GPU RAM or ~28GB CPU RAM")
            elif selected_model == 'llama-2-13b':
                st.write("A 13B parameter model by Meta. Higher quality but more resource intensive.")
                st.write("Memory required: ~26GB GPU RAM or ~52GB CPU RAM")
        
        return selected_model

    def get_prompt_template(template_name, **kwargs):
        """Get a formatted prompt template.
        
        Args:
            template_name: Name of the template to use
            **kwargs: Variables to substitute in the template
            
        Returns:
            str: Formatted prompt
        """
        templates = {
            "stock_analysis": "Analyze the following stock: {ticker}. Consider recent news: {news}. Provide investment advice.",
            "technical_analysis": "Perform technical analysis on {ticker}. Key metrics: Price: {price}, Volume: {volume}, Moving Averages: {ma}.",
            "market_sentiment": "Evaluate market sentiment for {ticker} based on the following news: {news}. How might this affect stock performance?",
            "risk_assessment": "Assess the risk level for {ticker} considering: Market cap: {market_cap}, Beta: {beta}, Industry: {industry}.",
            "comparison": "Compare stocks {ticker1} and {ticker2} based on: P/E ratio, growth potential, and recent performance."
        }
        
        if template_name not in templates:
            raise ValueError(f"Unknown template: {template_name}. Available templates: {', '.join(templates.keys())}")
            
        return templates[template_name].format(**kwargs)
    
    async def generate_analysis_async(
        self, 
        prompt: str, 
        max_length: int = 500,
        temperature: float = 0.7,
        top_p: float = 0.9,
        num_return_sequences: int = 1,
        use_cache: bool = True
    ) -> Dict:
        """Generate stock analysis asynchronously.

        Args:
            prompt: The analysis prompt
            max_length: Maximum length of generated text
            temperature: Sampling temperature (higher = more creative)
            top_p: Nucleus sampling parameter
            num_return_sequences: Number of responses to generate
            use_cache: Whether to use cached responses

        Returns:
            dict: Analysis results containing action and justification
        """
        import asyncio
        
        # Create parameters dictionary for cache key
        params = {
            'max_length': max_length,
            'temperature': temperature,
            'top_p': top_p,
            'num_return_sequences': num_return_sequences
        }
        
        # Check cache first if enabled
        if use_cache:
            cache_key = self._get_cache_key(prompt, params)
            cached_response = self._get_from_cache(cache_key)
            
            if cached_response:
                return cached_response
        
        # Run the model in a separate thread to avoid blocking
        loop = asyncio.get_event_loop()
        
        def _generate():
            try:
                if not self.model or not self.tokenizer:
                    if not self.load_model():
                        return {
                            'action': 'Hold',
                            'justification': 'Failed to load model. Please check logs for details.'
                        }

                # Generate response
                try:
                    response = self.pipeline(
                        prompt,
                        max_length=max_length,
                        temperature=temperature,
                        top_p=top_p,
                        num_return_sequences=num_return_sequences,
                        do_sample=True
                    )[0]['generated_text']
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        # Try to recover by clearing cache
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        return {
                            'action': 'Hold',
                            'justification': 'Model ran out of memory. Try using a smaller model or enabling quantization.'
                        }
                    else:
                        raise

                # Extract relevant parts
                result = self._parse_response(response)
                
                # Cache the result if caching is enabled
                if use_cache and result:
                    cache_key = self._get_cache_key(prompt, params)
                    self._save_to_cache(cache_key, result)
                    
                return result

            except Exception as e:
                # Return a graceful fallback response
                return {
                    'action': 'Hold',
                    'justification': f'Error during analysis: {str(e)}. Please try again or select a different model.'
                }
        
        # Run in thread pool and return result
        return await loop.run_in_executor(None, _generate)
    
    async def analyze_stock_async(self, ticker: str, news: str = "", **kwargs) -> Dict:
        """Analyze a stock asynchronously using a template.
        
        Args:
            ticker: Stock ticker symbol
            news: Recent news about the stock
            **kwargs: Additional parameters for the template
            
        Returns:
            Dict: Analysis results
        """
        # Prepare the prompt using the template
        prompt = self.get_prompt_template("stock_analysis", ticker=ticker, news=news)
        
        # Generate the analysis asynchronously
        return await self.generate_analysis_async(prompt, **kwargs)