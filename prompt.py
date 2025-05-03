import google.generativeai as genai
import spacy
import json
import textwrap
import re
import os
import logging
import tiktoken
import argparse
import time
import sys
import colorama
from colorama import Fore, Style
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt, Confirm
from collections import defaultdict
import traceback
import asyncio
from dotenv import load_dotenv

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("gil.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('GIL')

# Load environment variables
load_dotenv()

# Initialize colorama
colorama.init()

# Initialize rich console
console = Console()

class TokenCounter:
    """
    Utility class to count tokens in prompts and estimate costs
    """
    def __init__(self):
        try:
            self.encoder = tiktoken.get_encoding("cl100k_base")  # OpenAI's encoder which is widely compatible
        except Exception as e:
            logger.warning(f"Could not initialize token counter: {e}")
            self.encoder = None
    
    def count_tokens(self, text):
        """Count the number of tokens in a text string"""
        if not self.encoder:
            return len(text.split()) * 1.3  # Rough estimate if tiktoken not available
        try:
            return len(self.encoder.encode(text))
        except Exception as e:
            logger.warning(f"Error counting tokens: {e}")
            return len(text.split()) * 1.3  # Fallback to rough estimate

    def estimate_cost(self, prompt_tokens, response_tokens=0, model="gemini-2.0-flash"):
        """Estimate the cost in USD based on token count"""
        # Approximate costs (subject to change)
        rates = {
            "gemini-2.0-flash": {"input": 0.000125, "output": 0.000375},
            "gemini-2.0-pro": {"input": 0.000375, "output": 0.001125},
            "gemini-2.0-ultra": {"input": 0.001875, "output": 0.005625}
        }
        
        if model not in rates:
            model = "gemini-2.0-flash"  # Default model
            
        input_cost = prompt_tokens * rates[model]["input"] / 1000
        output_cost = response_tokens * rates[model]["output"] / 1000
        
        return input_cost + output_cost


class AIModelClient:
    """
    Class to handle interactions with AI models (Gemini and potentially others)
    """
    def __init__(self, api_key=None, model_name="gemini-2.0-flash"):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            logger.warning("No API key provided. Using demo mode with limited functionality.")
            self.demo_mode = True
        else:
            self.demo_mode = False
            try:
                genai.configure(api_key=self.api_key)
                self.model = genai.GenerativeModel(model_name)
                logger.info(f"Initialized {model_name} model.")
            except Exception as e:
                logger.error(f"Failed to initialize Gemini model: {e}")
                self.demo_mode = True
        
        self.model_name = model_name
        self.token_counter = TokenCounter()
        self.request_history = []
    
    async def generate_content_async(self, prompt, temperature=0.7, max_retries=3, retry_delay=2):
        """
        Asynchronous method to generate content with Gemini API, with error handling and retries
        """
        if self.demo_mode:
            return {"text": "Demo mode: API response would appear here.", "status": "demo"}
        
        # Log and count tokens
        prompt_tokens = self.token_counter.count_tokens(prompt)
        logger.info(f"Sending prompt to {self.model_name} with {prompt_tokens} tokens")
        
        # Record request for analytics
        request_time = time.time()
        self.request_history.append({
            "timestamp": request_time,
            "model": self.model_name,
            "prompt_tokens": prompt_tokens,
            "temperature": temperature
        })
        
        attempts = 0
        while attempts < max_retries:
            try:
                # Wait for 2 seconds between retries to avoid rate limits
                if attempts > 0:
                    await asyncio.sleep(retry_delay)
                
                generation_config = genai.types.GenerationConfig(
                    temperature=temperature,
                    top_p=0.95,
                    top_k=40,
                    max_output_tokens=8192,
                )
                
                safety_settings = [
                    {
                        "category": "HARM_CATEGORY_HARASSMENT",
                        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                    },
                    {
                        "category": "HARM_CATEGORY_HATE_SPEECH",
                        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                    },
                    {
                        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                    },
                    {
                        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                    }
                ]
                
                response = self.model.generate_content(
                    prompt,
                    generation_config=generation_config,
                    safety_settings=safety_settings
                )
                
                # Process the response
                if response.parts:
                    text = response.parts[0].text
                    response_tokens = self.token_counter.count_tokens(text)
                    cost = self.token_counter.estimate_cost(prompt_tokens, response_tokens, self.model_name)
                    
                    # Update the request history with response information
                    self.request_history[-1].update({
                        "response_time": time.time() - request_time,
                        "response_tokens": response_tokens,
                        "estimated_cost": cost,
                        "status": "success"
                    })
                    
                    logger.info(f"Received response with {response_tokens} tokens. Estimated cost: ${cost:.6f}")
                    return {"text": text, "tokens": response_tokens, "cost": cost, "status": "success"}
                else:
                    logger.warning("Empty response from Gemini API")
                    self.request_history[-1].update({"status": "empty_response"})
                    return {"text": "", "error": "Empty response from API", "status": "error"}
            
            except Exception as e:
                attempts += 1
                error_msg = f"API error (attempt {attempts}/{max_retries}): {str(e)}"
                logger.error(f"{error_msg}\n{traceback.format_exc()}")
                
                if attempts >= max_retries:
                    self.request_history[-1].update({"status": "failed", "error": str(e)})
                    return {"text": "", "error": error_msg, "status": "error"}
    
    def generate_content(self, prompt, temperature=0.7, max_retries=3):
        """
        Synchronous wrapper for the asynchronous generate_content method
        """
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            response = loop.run_until_complete(
                self.generate_content_async(prompt, temperature, max_retries)
            )
            return response
        finally:
            loop.close()
    
    def get_usage_stats(self):
        """Return usage statistics"""
        if not self.request_history:
            return {"requests": 0, "tokens": {"prompt": 0, "response": 0}, "estimated_cost": 0}
        
        total_prompt_tokens = sum(req.get("prompt_tokens", 0) for req in self.request_history)
        total_response_tokens = sum(req.get("response_tokens", 0) for req in self.request_history if "response_tokens" in req)
        total_cost = sum(req.get("estimated_cost", 0) for req in self.request_history if "estimated_cost" in req)
        
        return {
            "requests": len(self.request_history),
            "tokens": {
                "prompt": total_prompt_tokens,
                "response": total_response_tokens,
                "total": total_prompt_tokens + total_response_tokens
            },
            "estimated_cost": total_cost
        }
    
    def change_model(self, model_name):
        """Change the model being used"""
        try:
            self.model = genai.GenerativeModel(model_name)
            self.model_name = model_name
            logger.info(f"Changed model to {model_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to change model: {e}")
            return False


class NLPProcessor:
    """
    Enhanced NLP processor using spaCy with more advanced techniques
    """
    def __init__(self, model_name="en_core_web_sm"):
        self.model_name = model_name
        self.nlp = None
        self.load_model()
    
    def load_model(self):
        """Load the spaCy model with proper error handling"""
        try:
            with console.status(f"Loading spaCy model ({self.model_name})..."):
                self.nlp = spacy.load(self.model_name)
                logger.info(f"Successfully loaded spaCy model: {self.model_name}")
        except OSError:
            console.print(f"[yellow]Downloading spaCy model ({self.model_name})...[/]")
            try:
                spacy.cli.download(self.model_name)
                self.nlp = spacy.load(self.model_name)
                logger.info(f"Successfully downloaded and loaded spaCy model: {self.model_name}")
            except Exception as e:
                logger.error(f"Error downloading spaCy model: {e}")
                console.print(f"[red]Failed to download spaCy model: {e}[/]")
                self.nlp = None
        except Exception as e:
            logger.error(f"Error loading spaCy model: {e}")
            console.print(f"[red]Failed to load spaCy model: {e}[/]")
            self.nlp = None
    
    def analyze_text(self, text):
        """Perform comprehensive text analysis with spaCy"""
        if not self.nlp:
            logger.warning("NLP model not loaded. Skipping text analysis.")
            return {"success": False, "error": "NLP model not loaded"}
        
        try:
            doc = self.nlp(text)
            
            # Extract key information
            entities = [(ent.text, ent.label_) for ent in doc.ents]
            noun_chunks = [chunk.text for chunk in doc.noun_chunks]
            key_verbs = [token.lemma_ for token in doc if token.pos_ == "VERB"]
            
            # Sentiment analysis (basic, using rule-based methods)
            sentiment_words = []
            for token in doc:
                if token.pos_ in ["ADJ", "ADV"] and token.is_alpha:
                    sentiment_words.append(token.text)
            
            # Get main subjects and objects
            subjects = []
            objects = []
            for token in doc:
                if token.dep_ in ["nsubj", "nsubjpass"]:
                    subjects.append(token.text)
                elif token.dep_ in ["dobj", "pobj"]:
                    objects.append(token.text)
            
            # Document level categories
            categories = []
            if any(token.text.lower() in ["create", "make", "build", "develop", "generate"] for token in doc):
                categories.append("creation")
            if any(token.text.lower() in ["analyze", "examine", "investigate", "review", "evaluate"] for token in doc):
                categories.append("analysis")
            if any(token.text.lower() in ["explain", "describe", "elaborate", "detail", "clarify"] for token in doc):
                categories.append("explanation")
            
            return {
                "success": True,
                "entities": entities,
                "noun_chunks": noun_chunks,
                "key_verbs": key_verbs,
                "sentiment_words": sentiment_words,
                "subjects": subjects,
                "objects": objects,
                "categories": categories
            }
            
        except Exception as e:
            logger.error(f"Error in text analysis: {e}")
            return {"success": False, "error": f"Analysis error: {str(e)}"}
    
    def extract_keywords(self, text, n=5):
        """Extract the most important keywords from text"""
        if not self.nlp:
            return []
        
        try:
            doc = self.nlp(text)
            keywords = {}
            
            # Count lemmas of non-stopword tokens
            for token in doc:
                if not token.is_stop and not token.is_punct and token.is_alpha and len(token.text) > 2:
                    lemma = token.lemma_.lower()
                    keywords[lemma] = keywords.get(lemma, 0) + 1
            
            # Sort by frequency and return top n
            sorted_keywords = sorted(keywords.items(), key=lambda x: x[1], reverse=True)
            return [k for k, v in sorted_keywords[:n]]
            
        except Exception as e:
            logger.error(f"Error extracting keywords: {e}")
            return []
    
    def identify_ambiguities(self, text):
        """Identify potential ambiguities in the text"""
        if not self.nlp:
            return []
        
        ambiguities = []
        try:
            doc = self.nlp(text)
            
            # Check for pronouns without clear antecedents
            for token in doc:
                if token.pos_ == "PRON" and token.i > 0:
                    ambiguities.append(f"Ambiguous pronoun: '{token.text}'")
            
            # Check for vague quantifiers
            vague_quantifiers = ["some", "many", "several", "few", "various", "different"]
            for token in doc:
                if token.text.lower() in vague_quantifiers:
                    ambiguities.append(f"Vague quantifier: '{token.text}'")
            
            # Check for potentially ambiguous modifiers
            for token in doc:
                if token.pos_ == "ADJ" and token.head.pos_ == "NOUN":
                    if any(child.dep_ == "conj" for child in token.head.children):
                        ambiguities.append(f"Potentially ambiguous modifier: '{token.text} {token.head.text}'")
            
            return ambiguities
            
        except Exception as e:
            logger.error(f"Error identifying ambiguities: {e}")
            return []


class PromptTemplate:
    """
    Class to handle prompt templates, patterns, and examples
    """
    def __init__(self):
        self.templates = {
            "general": {
                "description": "General purpose template",
                "template": """
You are an expert {role}. 
Task: {task}
Context: {context}
Additional information: {additional_info}
Output format: {output_format}
Style and tone: {style_and_tone}
Constraints: {constraints}

Provide a detailed and comprehensive response.
                """
            },
            "creative": {
                "description": "Template for creative writing",
                "template": """
You are a creative {role} with extensive experience.
Create a {content_type} with the following requirements:
- Topic/Theme: {topic}
- Style: {style_and_tone}
- Length: {length}
- Audience: {audience}
- Special requirements: {requirements}

Make it original, engaging, and compelling.
                """
            },
            "analysis": {
                "description": "Template for analytical tasks",
                "template": """
You are an expert {role} specializing in {domain}.
Analyze the following:
{context}

Consider these aspects:
{aspects}

Requirements:
- Depth: {depth}
- Focus areas: {focus}
- Output format: {output_format}
- Special instructions: {special_instructions}

Provide a thorough, evidence-based analysis.
                """
            },
            "technical": {
                "description": "Template for technical content",
                "template": """
You are an expert in {technical_domain} with {experience_level} experience.
Task: {task}
Technical context: {context}
Requirements:
- Complexity level: {complexity}
- Target audience: {audience}
- Include code examples: {include_code}
- Format: {output_format}
- Technical constraints: {constraints}

Provide detailed technical guidance with accuracy and clarity.
                """
            },
            "instruction": {
                "description": "Template for step-by-step instructions",
                "template": """
You are an expert instructor in {domain}.
Create a step-by-step guide for:
{task}

The guide should:
- Be appropriate for {audience_level} audience
- Include {detail_level} level of detail
- Cover {scope}
- Format: {output_format}
- Additional requirements: {requirements}

Make the instructions clear, concise, and easy to follow.
                """
            }
        }
        
        # Examples of effective prompts for different scenarios
        self.examples = {
            "detailed explanation": "Explain the concept of quantum entanglement at a college physics level. Include analogies to make it easier to understand, key mathematical principles without complex equations, historical context of the discovery, and current applications in quantum computing.",
            "code review": "Review this Python function that calculates Fibonacci numbers. Identify any efficiency issues, potential bugs, readability concerns, and suggest improvements. Explain why each suggested change would improve the code.",
            "creative writing": "Write a short story (800 words) about someone who discovers they can hear the thoughts of plants. The story should have a surprising twist ending, be written in third-person limited perspective, and include themes of environmental consciousness without being preachy.",
            "technical tutorial": "Create a step-by-step tutorial for deploying a Flask application to AWS Lambda with API Gateway. Target audience is intermediate developers familiar with Python but new to AWS. Include code examples, configuration steps, common pitfalls to avoid, and best practices for production deployments."
        }
    
    def get_template(self, template_name):
        """Get a specific template by name"""
        return self.templates.get(template_name, self.templates["general"])
    
    def list_templates(self):
        """List all available templates with descriptions"""
        return [(name, details["description"]) for name, details in self.templates.items()]
    
    def get_example(self, example_name):
        """Get a specific example by name"""
        return self.examples.get(example_name, "Example not found")
    
    def list_examples(self):
        """List all available examples"""
        return list(self.examples.keys())
    
    def fill_template(self, template_name, **kwargs):
        """Fill a template with provided values"""
        template = self.get_template(template_name)["template"]
        
        # Fill in provided values, using empty strings for missing values
        for key in re.findall(r'\{(\w+)\}', template):
            if key not in kwargs:
                kwargs[key] = ""
        
        return template.format(**kwargs)
    
    def add_template(self, name, description, template_text):
        """Add a new template"""
        if name in self.templates:
            return False, "Template name already exists"
        
        self.templates[name] = {
            "description": description,
            "template": template_text
        }
        return True, f"Template '{name}' added successfully"
    
    def add_example(self, name, example_text):
        """Add a new example"""
        if name in self.examples:
            return False, "Example name already exists"
        
        self.examples[name] = example_text
        return True, f"Example '{name}' added successfully"


class PromptEngineer:
    """
    Enhanced PromptEngineer class with more advanced functionality
    """
    def __init__(self, ai_client, nlp_processor, template_manager):
        self.ai_client = ai_client
        self.nlp = nlp_processor
        self.templates = template_manager
        self.history = []  # Store previous prompt engineering sessions
    
    async def analyze_intent(self, prompt, temperature=0.7):
        """
        Enhanced intent analysis with more structured output and better error handling
        """
        # First, use local NLP for basic analysis
        local_analysis = self.nlp.analyze_text(prompt)
        keywords = self.nlp.extract_keywords(prompt)
        ambiguities = self.nlp.identify_ambiguities(prompt)
        
        # Prepare the prompt for the AI model
        prompt_for_gemini = f"""
You are an expert prompt analyzer with advanced natural language processing capabilities. Your task is to understand the user's intent from their prompt, considering potential ambiguities and implicit requests.

Here's the user's prompt:
"{prompt}"

Keywords identified: {', '.join(keywords) if keywords else 'None'}
Potential ambiguities: {', '.join(ambiguities) if ambiguities else 'None'}

Analyze the prompt and provide a structured analysis in valid JSON format with the following fields. If you cannot determine a value with confidence, use "unknown" for string values or empty arrays for lists. DO NOT include any text outside of the JSON structure.

{{
    "task": "The main task or goal the user wants to achieve",
    "main_topics": ["Primary subject matters or domains"],
    "output_format": "Desired format for the response (e.g., essay, list, code)",
    "style_and_tone": "Desired style and tone for the response",
    "audience": "The intended audience for the output",
    "complexity_level": "Desired complexity level (e.g., beginner, intermediate, expert)",
    "constraints": ["Any limitations or requirements"],
    "examples_provided": ["Any examples mentioned in the prompt"],
    "implicit_assumptions": ["Any unstated assumptions that might be important"],
    "recommended_template": "Which prompt template would work best for this",
    "reasoning": "Brief explanation of your analysis"
}}
"""
        
        try:
            with console.status("Analyzing intent..."):
                response = await self.ai_client.generate_content_async(prompt_for_gemini, temperature)
            
            if response["status"] != "success":
                logger.error(f"Intent analysis failed: {response.get('error', 'Unknown error')}")
                return {
                    "error": f"Intent analysis failed: {response.get('error', 'Unknown error')}",
                    "local_analysis": local_analysis
                }
            
            # Parse the response
            result_text = response["text"].strip()
            # Remove any markdown code blocks if present
            result_text = re.sub(r'```json\s*|\s*```', '', result_text)
            
            try:
                analysis = json.loads(result_text)
                analysis["local_analysis"] = local_analysis
                return analysis
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse intent analysis JSON: {e}")
                logger.debug(f"Raw response: {result_text}")
                return {
                    "error": f"Invalid JSON response: {str(e)}",
                    "raw_response": result_text,
                    "local_analysis": local_analysis
                }
                
        except Exception as e:
            logger.error(f"Exception in intent analysis: {e}\n{traceback.format_exc()}")
            return {"error": f"Analysis error: {str(e)}", "local_analysis": local_analysis}
    
    async def get_missing_info(self, prompt, analysis):
        """
        Enhanced method to identify missing information based on the intent analysis
        """
        # Extract relevant fields from the analysis
        task = analysis.get("task", "unknown")
        topics = analysis.get("main_topics", [])
        style = analysis.get("style_and_tone", "unknown")
        audience = analysis.get("audience", "unknown")
        constraints = analysis.get("constraints", [])
        
        # Get potential ambiguities from local NLP
        ambiguities = self.nlp.identify_ambiguities(prompt)
        
        prompt_for_gemini = f"""
You are an expert at identifying missing information in user prompts for AI models. Your goal is to ensure the prompt is complete and unambiguous.

Here's the context:
Original prompt: "{prompt}"
Task: {task}
Main Topics: {', '.join(topics) if topics else 'unknown'}
Style and Tone: {style}
Audience: {audience}
Constraints: {', '.join(constraints) if constraints else 'None'}
Identified ambiguities: {', '.join(ambiguities) if ambiguities else 'None'}

Based on this, identify ONLY the most crucial missing details that would significantly improve the prompt. For each missing detail:
1. Provide a clear, concise question
2. Explain briefly why this information is important
3. Give an example of a good answer

Format your response as valid JSON with an array of objects, each with "question", "importance", and "example" fields. Limit to 3 most important questions maximum.

Example format:
{{
  "missing_information": [
    {{
      "question": "What is the target age range for your audience?",
      "importance": "This helps adjust language complexity and examples",
      "example": "12-15 year old middle school students"
    }}
  ]
}}

Only include questions for truly important missing information. If the prompt is already clear and complete, return an empty array.
"""

        try:
            with console.status("Identifying missing information..."):
                response = await self.ai_client.generate_content_async(prompt_for_gemini, temperature=0.7)
            
            if response["status"] != "success":
                logger.error(f"Missing info identification failed: {response.get('error', 'Unknown error')}")
                return {"error": response.get("error", "Unknown error")}
            
            # Parse the response
            result_text = response["text"].strip()
            # Remove any markdown code blocks if present
            result_text = re.sub(r'```json\s*|\s*```', '', result_text)
            
            try:
                missing_info_data = json.loads(result_text)
                return missing_info_data
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse missing info JSON: {e}")
                logger.debug(f"Raw response: {result_text}")
                return {
                    "error": f"Invalid JSON response: {str(e)}",
                    "raw_response": result_text
                }
                
        except Exception as e:
            logger.error(f"Exception in missing info identification: {e}\n{traceback.format_exc()}")
            return {"error": f"Analysis error: {str(e)}"}
    
    def _remove_repetition(self, text, n=3):
        """
        Enhanced method to remove repetitive sequences with better detection
        """
        words = text.split()
        if len(words) < n + 1:
            return text  # Not enough words for repetition

        new_words = []
        seen_sequences = defaultdict(list)
        
        for i in range(len(words)):
            # Check for duplicate sequences of varying lengths
            add_word = True
            
            for seq_len in range(n, max(n-2, 1), -1):  # Try different sequence lengths
                if i < seq_len:
                    continue
                    
                sequence = tuple(words[i-seq_len:i])
                seq_string = " ".join(sequence)
                
                # Check if this sequence has appeared before
                if seq_string in seen_sequences:
                    prev_positions = seen_sequences[seq_string]
                    
                    # Check if the previous occurrence was recent enough to count as repetition
                    if i - prev_positions[-1] <= seq_len * 2:
                        add_word = False
                        break
            
            if add_word:
                new_words.append(words[i])
                
            # Record this position for each sequence ending at this word
            for seq_len in range(1, min(n+1, i+1)):
                seq = tuple(words[i-seq_len+1:i+1])
                seq_str = " ".join(seq)
                seen_sequences[seq_str].append(i)
                
        return " ".join(new_words)
    
    async def refine_prompt(self):
        """Refine the current prompt based on analysis and additional information"""
        if not self.current_prompt:
            console.print(Panel("Please enter a prompt first (option 1).", border_style="red"))
            input("\nPress Enter to continue...")
            return

        console.clear()
        console.print(Panel("Refining Prompt...", border_style="cyan"))

        # If no analysis exists, perform it first
        if not self.current_analysis:
            console.print("[yellow]No analysis available. Performing analysis first...[/yellow]")
            self.current_analysis = await self.analyze_intent(self.current_prompt)

            if "error" in self.current_analysis:
                console.print(Panel(f"[red]Error: {self.current_analysis['error']}[/red]", border_style="red"))
                input("\nPress Enter to continue...")
                return

        # Get recommended template if available
        template = self.current_analysis.get("recommended_template", "general")

        # Format additional info for display
        additional_info_formatted = {}
        if isinstance(self.additional_info, dict):
            for question, answer in self.additional_info.items():
                additional_info_formatted[question] = answer
        else:
            additional_info_formatted = self.additional_info

        # Display current state before refinement
        console.print("[bold cyan]Current Information:[/bold cyan]")
        console.print(f"[blue]Task:[/blue] {self.current_analysis.get('task', 'Unknown')}")
        console.print(f"[blue]Recommended Template:[/blue] {template}")

        if additional_info_formatted:
            console.print("[blue]Additional Information:[/blue]")
            for question, answer in additional_info_formatted.items():
                console.print(f"- {question}: {answer}")

        # Identify missing information
        missing_info = await self.get_missing_info(self.current_prompt, self.current_analysis)

        if "error" in missing_info:
            console.print(Panel(f"[red]Error: {missing_info['error']}[/red]", border_style="red"))
            input("\nPress Enter to continue...")
            return

        # Back-and-forth interaction to clarify missing information
        if "missing_information" in missing_info and missing_info["missing_information"]:
            console.print("[bold cyan]Identifying Missing Information:[/bold cyan]")
            for item in missing_info["missing_information"]:
                question = item.get("question", "")
                importance = item.get("importance", "")
                example = item.get("example", "")

                console.print(f"[yellow]{question}[/yellow]")
                console.print(f"[dim]{importance}[/dim]")
                if example:
                    console.print(f"[blue]Example:[/blue] {example}")

                # Ask the user for clarification
                answer = Prompt.ask("Your answer (press Enter to skip)", default=example)
                if answer:
                    self.additional_info[question] = answer

        # Prepare the prompt for Gemini
        refine_prompt = f"""
You are an expert prompt engineer. Refine the following prompt to make it more effective, clear, and comprehensive.

INITIAL PROMPT:
"{self.current_prompt}"

PROMPT ANALYSIS:
- Task: {self.current_analysis.get('task', 'Unknown')}
- Main Topics: {', '.join(self.current_analysis.get('main_topics', []))}
- Desired Output Format: {self.current_analysis.get('output_format', 'Unknown')}
- Style and Tone: {self.current_analysis.get('style_and_tone', 'Unknown')}
- Target Audience: {self.current_analysis.get('audience', 'Unknown')}
- Constraints: {', '.join(self.current_analysis.get('constraints', []))}
- Additional Information: {self.additional_info}

INSTRUCTIONS:
1. Incorporate all relevant additional information.
2. Follow best practices for the {template} template.
3. Ensure the refined prompt is clear, specific, and provides all necessary context.
4. Address any ambiguities in the original prompt.
5. Return only the refined prompt text without explanations.
"""

        try:
            # Send the prompt to Gemini for refinement
            response = await self.ai_client.generate_content_async(refine_prompt, temperature=0.7)

            if response["status"] != "success":
                console.print(Panel(f"[red]Error: {response.get('error', 'Unknown error')}[/red]", border_style="red"))
                input("\nPress Enter to continue...")
                return

            # Extract the refined prompt
            self.refined_prompt = response["text"].strip()

            # Display the refined prompt
            console.print(Panel("Original Prompt", border_style="blue"))
            console.print(self.current_prompt)

            console.print(Panel("Refined Prompt", border_style="green"))
            console.print(self.refined_prompt)

            # Ask if the user wants to use the refined prompt
            if Confirm.ask("\nWould you like to use this refined prompt?"):
                self.current_prompt = self.refined_prompt
                self.current_analysis = None  # Reset analysis for the new prompt
                console.print("[green]Current prompt updated to the refined version.[/green]")

        except Exception as e:
            logger.error(f"Error during prompt refinement: {e}\n{traceback.format_exc()}")
            console.print(Panel(f"[red]An error occurred during refinement: {e}[/red]", border_style="red"))

        input("\nPress Enter to continue...")
    
    async def evaluate_prompt(self, prompt):
        """
        Evaluate the quality of a prompt using specific criteria
        """
        prompt_for_gemini = f"""
You are an expert prompt evaluator who specializes in assessing the quality and effectiveness of prompts for AI models. Evaluate the following prompt based on established criteria.

PROMPT TO EVALUATE:
"{prompt}"

Evaluate this prompt on the following criteria, rating each on a scale of 1-5 where 1 is poor and 5 is excellent:

1. Clarity: How clear and unambiguous are the instructions?
2. Specificity: How specific and detailed is the prompt?
3. Structure: How well-organized and logically structured is the prompt?
4. Completeness: How comprehensive is the prompt in providing necessary context?
5. Precision: How precisely does it define the expected output format/style?
6. Conciseness: How efficiently does it convey requirements without unnecessary text?

For each criterion, provide:
- Score (1-5)
- Brief justification (1-2 sentences)
- Specific improvement suggestion

Finally, provide an overall score (1-5) and 2-3 key recommendations to improve the prompt.

Format your response as valid JSON according to the following structure:
{{
  "criteria": {{
    "clarity": {{
      "score": 4,
      "justification": "Instructions are mostly clear with well-defined goals",
      "improvement": "Specify exactly what 'detailed analysis' means in this context"
    }},
    ...
  }},
  "overall": {{
    "score": 3.5,
    "recommendations": [
      "Add specific examples of the desired output format",
      "Clarify the target audience to better calibrate complexity"
    ]
  }}
}}
"""

        try:
            with console.status("Evaluating prompt quality..."):
                response = await self.ai_client.generate_content_async(prompt_for_gemini, temperature=0.7)
            
            if response["status"] != "success":
                logger.error(f"Prompt evaluation failed: {response.get('error', 'Unknown error')}")
                return {"error": response.get("error", "Unknown error")}
            
            # Parse the response
            result_text = response["text"].strip()
            # Remove any markdown code blocks if present
            result_text = re.sub(r'```json\s*|\s*```', '', result_text)
            
            try:
                evaluation = json.loads(result_text)
                return evaluation
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse evaluation JSON: {e}")
                logger.debug(f"Raw response: {result_text}")
                return {
                    "error": f"Invalid JSON response: {str(e)}",
                    "raw_response": result_text
                }
                
        except Exception as e:
            logger.error(f"Exception in prompt evaluation: {e}\n{traceback.format_exc()}")
            return {"error": f"Evaluation error: {str(e)}"}
    
    async def suggest_improvements(self, prompt, evaluation):
        """
        Suggest specific improvements based on prompt evaluation
        """
        # Extract evaluation criteria and scores
        try:
            criteria = evaluation.get("criteria", {})
            overall = evaluation.get("overall", {})
            
            # Find the lowest scoring criteria
            lowest_criteria = []
            lowest_score = 5
            
            for criterion, data in criteria.items():
                score = data.get("score", 5)
                if score < lowest_score:
                    lowest_score = score
                    lowest_criteria = [criterion]
                elif score == lowest_score:
                    lowest_criteria.append(criterion)
            
            prompt_for_gemini = f"""
You are an expert prompt engineer who specializes in improving prompts for AI models. You've evaluated a prompt and identified areas for improvement.

ORIGINAL PROMPT:
"{prompt}"

EVALUATION SUMMARY:
Overall score: {overall.get('score', 'N/A')}/5
Areas needing most improvement: {', '.join(lowest_criteria)}

Recommendations from evaluation:
{', '.join(overall.get('recommendations', ['No specific recommendations provided']))}

Your task is to provide THREE specific, actionable improvements to this prompt. Each improvement should:
1. Focus on a specific part of the prompt
2. Explain what exactly to change and why
3. Provide a concrete example of the improved version

Format your response as valid JSON with this structure:
{{
  "improvements": [
    {{
      "focus_area": "Clarity of task description",
      "issue": "The task description is vague and could be interpreted multiple ways",
      "recommendation": "Specify exactly what type of analysis is required",
      "example": "Analyze the following text to identify key themes, emotional tone, and rhetorical devices"
    }},
    ...
  ]
}}

Limit your response to only the most impactful improvements.
"""

            with console.status("Generating improvement suggestions..."):
                response = await self.ai_client.generate_content_async(prompt_for_gemini, temperature=0.7)
            
            if response["status"] != "success":
                logger.error(f"Improvement suggestion failed: {response.get('error', 'Unknown error')}")
                return {"error": response.get("error", "Unknown error")}
            
            # Parse the response
            result_text = response["text"].strip()
            # Remove any markdown code blocks if present
            result_text = re.sub(r'```json\s*|\s*```', '', result_text)
            
            try:
                suggestions = json.loads(result_text)
                return suggestions
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse suggestions JSON: {e}")
                logger.debug(f"Raw response: {result_text}")
                return {
                    "error": f"Invalid JSON response: {str(e)}",
                    "raw_response": result_text
                }
                
        except Exception as e:
            logger.error(f"Exception in improvement suggestions: {e}\n{traceback.format_exc()}")
            return {"error": f"Suggestion error: {str(e)}"}
    
    def get_prompt_history(self):
        """Return the history of prompt engineering sessions"""
        return self.history
    
    def save_history(self, filename="prompt_history.json"):
        """Save the prompt history to a file"""
        try:
            with open(filename, 'w') as f:
                json.dump(self.history, f, indent=2)
            return True, f"History saved to {filename}"
        except Exception as e:
            logger.error(f"Error saving history: {e}")
            return False, f"Error saving history: {str(e)}"
    
    def load_history(self, filename="prompt_history.json"):
        """Load prompt history from a file"""
        try:
            if os.path.exists(filename):
                with open(filename, 'r') as f:
                    self.history = json.load(f)
                return True, f"Loaded history from {filename}"
            return False, "History file not found"
        except Exception as e:
            logger.error(f"Error loading history: {e}")
            return False, f"Error loading history: {str(e)}"


class DatabaseManager:
    """Simple file-based database for storing prompt examples and templates"""
    def __init__(self, db_file="gil_database.json"):
        self.db_file = db_file
        self.db = self._load_database()
    
    def _load_database(self):
        """Load the database from file"""
        if os.path.exists(self.db_file):
            try:
                with open(self.db_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading database: {e}")
                return self._create_default_db()
        else:
            return self._create_default_db()
    
    def _create_default_db(self):
        """Create a default database structure"""
        return {
            "templates": {},
            "examples": {},
            "user_profiles": {},
            "settings": {
                "default_model": "gemini-2.0-flash",
                "default_temperature": 0.7
            }
        }
    
    def _save_database(self):
        """Save the database to file"""
        try:
            with open(self.db_file, 'w') as f:
                json.dump(self.db, f, indent=2)
            return True
        except Exception as e:
            logger.error(f"Error saving database: {e}")
            return False
    
    def add_template(self, name, template_data):
        """Add a template to the database"""
        self.db.setdefault("templates", {})[name] = template_data
        return self._save_database()
    
    def get_template(self, name):
        """Get a template from the database"""
        return self.db.get("templates", {}).get(name)
    
    def list_templates(self):
        """List all templates in the database"""
        return list(self.db.get("templates", {}).keys())
    
    def add_example(self, name, example_data):
        """Add an example to the database"""
        self.db.setdefault("examples", {})[name] = example_data
        return self._save_database()
    
    def get_example(self, name):
        """Get an example from the database"""
        return self.db.get("examples", {}).get(name)
    
    def list_examples(self):
        """List all examples in the database"""
        return list(self.db.get("examples", {}).keys())
    
    def update_settings(self, settings):
        """Update application settings"""
        self.db["settings"] = {**self.db.get("settings", {}), **settings}
        return self._save_database()
    
    def get_settings(self):
        """Get application settings"""
        return self.db.get("settings", {})


class GIL:
    """
    Enhanced GIL (Gemini-enhanced Interactive Prompt Engineering) system
    """
    def __init__(self, api_key=None, model_name="gemini-2.0-flash"):
        """Initialize the GIL system with all components"""
        # Initialize the AI client
        self.ai_client = AIModelClient(api_key, model_name)
        
        # Initialize NLP processor
        self.nlp_processor = NLPProcessor()
        
        # Initialize template manager
        self.template_manager = PromptTemplate()
        
        # Initialize prompt engineer
        self.prompt_engineer = PromptEngineer(
            self.ai_client, 
            self.nlp_processor, 
            self.template_manager
        )
        
        # Initialize database manager
        self.db_manager = DatabaseManager()
        
        # Load settings
        self.settings = self.db_manager.get_settings()
        
        # Application state
        self.running = True
        self.current_prompt = ""
        self.current_analysis = None
        self.additional_info = {}
        self.refined_prompt = ""
        self.current_evaluation = None
        
        # Initialize the event loop
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

    def _display_welcome(self):
        """Display welcome message and information"""
        console.clear()
        console.print(Panel(
            "[bold cyan]Welcome to GIL - Gemini-enhanced Interactive Prompt Engineer[/bold cyan]\n" +
            "[italic]A powerful tool for crafting effective prompts for AI models[/italic]",
            title="GIL v2.0",
            border_style="cyan"
        ))
        
        # Show model information
        model_info = self.ai_client.model_name
        if self.ai_client.demo_mode:
            model_info += " [red](DEMO MODE - Limited functionality)[/red]"
        
        console.print(f"[blue]Active Model:[/blue] {model_info}")
        
        # Show usage statistics if available
        stats = self.ai_client.get_usage_stats()
        if stats["requests"] > 0:
            console.print(f"[blue]Usage:[/blue] {stats['requests']} requests | " +
                          f"{stats['tokens']['total']} tokens | " +
                          f"Est. cost: ${stats['estimated_cost']:.4f}")
    
    def display_main_menu(self):
        """Display the enhanced main menu"""
        self._display_welcome()
        
        # Create a table for the menu options
        table = Table(show_header=False, box=None, padding=(0, 2, 0, 0))
        table.add_column("Option", style="cyan")
        table.add_column("Description")
        
        # Add menu options
        table.add_row("1", "Enter/Edit Prompt")
        table.add_row("2", "Analyze Current Prompt")
        table.add_row("3", "Refine Prompt")
        table.add_row("4", "Evaluate Prompt Quality")
        table.add_row("5", "Use Prompt Templates")
        table.add_row("6", "View Prompt History")
        table.add_row("7", "Model Settings")
        table.add_row("8", "Export Prompt/Analysis")
        table.add_row("9", "Exit")
        
        console.print(Panel(table, title="Main Menu", border_style="blue"))
        
        # Show current prompt if it exists
        if self.current_prompt:
            truncated = self.current_prompt[:100] + "..." if len(self.current_prompt) > 100 else self.current_prompt
            console.print(Panel(truncated, title="Current Prompt", border_style="green", padding=(1, 1)))
    
    async def get_user_prompt(self):
        """Get or edit the user's prompt"""
        console.clear()
        console.print(Panel("Enter or Edit Your Prompt", border_style="cyan"))
        
        if self.current_prompt:
            console.print("[blue]Current Prompt:[/blue]")
            console.print(Panel(self.current_prompt, padding=(1,1)))
            
            if Confirm.ask("Would you like to edit this prompt?"):
                # Offer multiline editing
                multiline = Confirm.ask("Would you like to enter a multiline prompt?")
                
                if multiline:
                    console.print("[yellow]Enter your prompt below. Press Ctrl+D (or Ctrl+Z on Windows) when finished.[/yellow]")
                    lines = []
                    try:
                        while True:
                            line = input()
                            lines.append(line)
                    except EOFError:
                        pass
                    self.current_prompt = "\n".join(lines)
                else:
                    self.current_prompt = Prompt.ask("Enter your prompt", default=self.current_prompt)
            
        else:
            # Offer multiline entry
            multiline = Confirm.ask("Would you like to enter a multiline prompt?")
            
            if multiline:
                console.print("[yellow]Enter your prompt below. Press Ctrl+D (or Ctrl+Z on Windows) when finished.[/yellow]")
                lines = []
                try:
                    while True:
                        line = input()
                        lines.append(line)
                except EOFError:
                        pass
                self.current_prompt = "\n".join(lines)
            else:
                self.current_prompt = Prompt.ask("Enter your prompt")
        
        # Reset analysis when prompt changes
        self.current_analysis = None
        self.additional_info = {}
        self.refined_prompt = ""
        self.current_evaluation = None
        
        console.print(Panel("Prompt updated successfully!", border_style="green"))
        input("\nPress Enter to continue...")
    
    async def analyze_prompt(self):
        """Analyze the current prompt using AI and NLP"""
        if not self.current_prompt:
            console.print(Panel("Please enter a prompt first (option 1).", border_style="red"))
            input("\nPress Enter to continue...")
            return
        
        console.clear()
        console.print(Panel("Analyzing Prompt...", border_style="cyan"))
        
        # Perform analysis
        self.current_analysis = await self.prompt_engineer.analyze_intent(self.current_prompt)
        
        if "error" in self.current_analysis:
            console.print(Panel(f"[red]Error: {self.current_analysis['error']}[/red]", border_style="red"))
            input("\nPress Enter to continue...")
            return
        
        # Display the analysis
        console.print(Panel("Prompt Analysis Results", border_style="green"))
        
        # Create a table for the analysis results
        table = Table(show_header=True, header_style="bold cyan", box=True)
        table.add_column("Attribute")
        table.add_column("Value")
        
        # Add analysis results to the table
        table.add_row("Task", self.current_analysis.get("task", "Unknown"))
        table.add_row("Main Topics", ", ".join(self.current_analysis.get("main_topics", ["Unknown"])))
        table.add_row("Output Format", self.current_analysis.get("output_format", "Unknown"))
        table.add_row("Style and Tone", self.current_analysis.get("style_and_tone", "Unknown"))
        table.add_row("Audience", self.current_analysis.get("audience", "Unknown"))
        table.add_row("Complexity Level", self.current_analysis.get("complexity_level", "Unknown"))
        table.add_row("Constraints", ", ".join(self.current_analysis.get("constraints", ["None"])))
        table.add_row("Implicit Assumptions", ", ".join(self.current_analysis.get("implicit_assumptions", ["None"])))
        table.add_row("Recommended Template", self.current_analysis.get("recommended_template", "general"))
        
        console.print(table)
        
        # Show analysis reasoning
        if "reasoning" in self.current_analysis:
            console.print(Panel(self.current_analysis["reasoning"], title="Analysis Reasoning", border_style="blue"))
        
        # Show local NLP analysis
        if "local_analysis" in self.current_analysis and self.current_analysis["local_analysis"].get("success", False):
            local = self.current_analysis["local_analysis"]
            
            console.print("\n[bold cyan]Local NLP Analysis:[/bold cyan]")
            if local.get("entities"):
                console.print("[blue]Entities:[/blue]", ", ".join([f"{e[0]} ({e[1]})" for e in local.get("entities", [])]))
            if local.get("key_verbs"):
                console.print("[blue]Key Verbs:[/blue]", ", ".join(local.get("key_verbs", [])))
            if local.get("categories"):
                console.print("[blue]Categories:[/blue]", ", ".join(local.get("categories", [])))
        
        # Allow editing the analysis
        if Confirm.ask("\nWould you like to modify this analysis?"):
            await self._edit_analysis()
        
        # Check for missing information
        if Confirm.ask("\nWould you like to identify missing information?"):
            await self._identify_missing_info()
        
        input("\nPress Enter to continue...")
    
    async def _edit_analysis(self):
        """Allow the user to edit the current analysis"""
        console.print(Panel("Edit Analysis", border_style="cyan"))
        
        edited = self.current_analysis.copy()
        
        # Edit main fields
        for field in ["task", "output_format", "style_and_tone", "audience", "complexity_level", "recommended_template"]:
            current = edited.get(field, "")
            edited[field] = Prompt.ask(f"Enter new value for '{field}'", default=current)
        
        # Edit list fields
        for field in ["main_topics", "constraints", "implicit_assumptions"]:
            current = ", ".join(edited.get(field, []))
            new_value = Prompt.ask(f"Enter new values for '{field}' (comma-separated)", default=current)
            edited[field] = [item.strip() for item in new_value.split(",") if item.strip()]
        
        self.current_analysis = edited
        console.print("[green]Analysis updated successfully![/green]")
    
    async def _identify_missing_info(self):
        """Identify missing information in the prompt"""
        console.print(Panel("Identifying Missing Information...", border_style="cyan"))
        
        # Get missing information from AI
        missing_info = await self.prompt_engineer.get_missing_info(self.current_prompt, self.current_analysis)
        
        if "error" in missing_info:
            console.print(Panel(f"[red]Error: {missing_info['error']}[/red]", border_style="red"))
            return
        
        # Display the missing information
        if "missing_information" in missing_info and missing_info["missing_information"]:
            console.print("[bold cyan]Missing Information:[/bold cyan]")
            
            for idx, item in enumerate(missing_info["missing_information"], 1):
                question = item.get("question", "")
                importance = item.get("importance", "")
                example = item.get("example", "")
                
                console.print(f"[bold]{idx}. {question}[/bold]")
                console.print(f"   [blue]Importance:[/blue] {importance}")
                console.print(f"   [blue]Example:[/blue] {example}")
                
                # Ask the user to provide the missing information
                answer = Prompt.ask(f"   Your answer", default=example)
                console.print()
                
                # Store the answer
                self.additional_info[question] = answer
        else:
            console.print("[green]No significant missing information identified.[/green]")
    
    async def refine_prompt(self):
        """Refine the current prompt based on analysis and additional information"""
        if not self.current_prompt:
            console.print(Panel("Please enter a prompt first (option 1).", border_style="red"))
            input("\nPress Enter to continue...")
            return

        console.clear()
        console.print(Panel("Refining Prompt...", border_style="cyan"))

        # If no analysis exists, perform it first
        if not self.current_analysis:
            console.print("[yellow]No analysis available. Performing analysis first...[/yellow]")
            self.current_analysis = await self.prompt_engineer.analyze_intent(self.current_prompt)

            if "error" in self.current_analysis:
                console.print(Panel(f"[red]Error: {self.current_analysis['error']}[/red]", border_style="red"))
                input("\nPress Enter to continue...")
                return

        # Get recommended template if available
        template = self.current_analysis.get("recommended_template", "general")

        # Format additional info for display
        additional_info_formatted = {}
        if isinstance(self.additional_info, dict):
            for question, answer in self.additional_info.items():
                additional_info_formatted[question] = answer
        else:
            additional_info_formatted = self.additional_info

        # Display current state before refinement
        console.print("[bold cyan]Current Information:[/bold cyan]")
        console.print(f"[blue]Task:[/blue] {self.current_analysis.get('task', 'Unknown')}")
        console.print(f"[blue]Recommended Template:[/blue] {template}")

        if additional_info_formatted:
            console.print("[blue]Additional Information:[/blue]")
            for question, answer in additional_info_formatted.items():
                console.print(f"- {question}: {answer}")

        # Identify missing information
        missing_info = await self.prompt_engineer.get_missing_info(self.current_prompt, self.current_analysis)

        if "error" in missing_info:
            console.print(Panel(f"[red]Error: {missing_info['error']}[/red]", border_style="red"))
            input("\nPress Enter to continue...")
            return

        # Back-and-forth interaction to clarify missing information
        if "missing_information" in missing_info and missing_info["missing_information"]:
            console.print("[bold cyan]Identifying Missing Information:[/bold cyan]")
            for item in missing_info["missing_information"]:
                question = item.get("question", "")
                importance = item.get("importance", "")
                example = item.get("example", "")

                console.print(f"[yellow]{question}[/yellow]")
                console.print(f"[dim]{importance}[/dim]")
                if example:
                    console.print(f"[blue]Example:[/blue] {example}")

                # Ask the user for clarification
                answer = Prompt.ask("Your answer (press Enter to skip)", default=example)
                if answer:
                    self.additional_info[question] = answer

        # Prepare the prompt for Gemini
        refine_prompt = f"""
You are an expert prompt engineer. Refine the following prompt to make it more effective, clear, and comprehensive.

INITIAL PROMPT:
"{self.current_prompt}"

PROMPT ANALYSIS:
- Task: {self.current_analysis.get('task', 'Unknown')}
- Main Topics: {', '.join(self.current_analysis.get('main_topics', []))}
- Desired Output Format: {self.current_analysis.get('output_format', 'Unknown')}
- Style and Tone: {self.current_analysis.get('style_and_tone', 'Unknown')}
- Target Audience: {self.current_analysis.get('audience', 'Unknown')}
- Constraints: {', '.join(self.current_analysis.get('constraints', []))}
- Additional Information: {self.additional_info}

INSTRUCTIONS:
1. Incorporate all relevant additional information.
2. Follow best practices for the {template} template.
3. Ensure the refined prompt is clear, specific, and provides all necessary context.
4. Address any ambiguities in the original prompt.
5. Return only the refined prompt text without explanations.
"""

        try:
            # Send the prompt to Gemini for refinement
            response = await self.ai_client.generate_content_async(refine_prompt, temperature=0.7)

            if response["status"] != "success":
                console.print(Panel(f"[red]Error: {response.get('error', 'Unknown error')}[/red]", border_style="red"))
                input("\nPress Enter to continue...")
                return

            # Extract the refined prompt
            self.refined_prompt = response["text"].strip()

            # Display the refined prompt
            console.print(Panel("Original Prompt", border_style="blue"))
            console.print(self.current_prompt)

            console.print(Panel("Refined Prompt", border_style="green"))
            console.print(self.refined_prompt)

            # Ask if the user wants to use the refined prompt
            if Confirm.ask("\nWould you like to use this refined prompt?"):
                self.current_prompt = self.refined_prompt
                self.current_analysis = None  # Reset analysis for the new prompt
                console.print("[green]Current prompt updated to the refined version.[/green]")

        except Exception as e:
            logger.error(f"Error during prompt refinement: {e}\n{traceback.format_exc()}")
            console.print(Panel(f"[red]An error occurred during refinement: {e}[/red]", border_style="red"))

        input("\nPress Enter to continue...")

    async def evaluate_prompt_quality(self):
        """Evaluate the quality of the current prompt"""
        if not self.current_prompt:
            console.print(Panel("Please enter a prompt first (option 1).", border_style="red"))
            input("\nPress Enter to continue...")
            return
        
        console.clear()
        console.print(Panel("Evaluating Prompt Quality...", border_style="cyan"))
        
        # Evaluate the prompt
        self.current_evaluation = await self.prompt_engineer.evaluate_prompt(self.current_prompt)
        
        if "error" in self.current_evaluation:
            console.print(Panel(f"[red]Error: {self.current_evaluation['error']}[/red]", border_style="red"))
            input("\nPress Enter to continue...")
            return
        
        # Display the evaluation
        criteria = self.current_evaluation.get("criteria", {})
        overall = self.current_evaluation.get("overall", {})
        
        # Create a table for the evaluation results
        table = Table(show_header=True, header_style="bold cyan", box=True)
        table.add_column("Criterion")
        table.add_column("Score")
        table.add_column("Justification")
        table.add_column("Improvement")
        
        # Add evaluation results to the table
        for criterion, data in criteria.items():
            score = data.get("score", "N/A")
            justification = data.get("justification", "")
            improvement = data.get("improvement", "")
            
            # Color-code scores
            score_text = f"{score}/5"
            if score <= 2:
                score_color = "[red]"
            elif score == 3:
                score_color = "[yellow]"
            else:
                score_color = "[green]"
                
            table.add_row(
                criterion.capitalize(),
                f"{score_color}{score_text}[/]",
                justification,
                improvement
            )
        
        console.print(table)
        
        # Display overall score and recommendations
        overall_score = overall.get("score", "N/A")
        score_color = "[red]" if overall_score <= 2 else "[yellow]" if overall_score == 3 else "[green]"
        
        console.print(f"\n[bold cyan]Overall Score:[/bold cyan] {score_color}{overall_score}/5[/]")
        
        if "recommendations" in overall:
            console.print("[bold cyan]Key Recommendations:[/bold cyan]")
            for idx, rec in enumerate(overall["recommendations"], 1):
                console.print(f"{idx}. {rec}")
        
        # Ask if user wants detailed improvement suggestions
        if Confirm.ask("\nWould you like to see detailed improvement suggestions?"):
            await self._show_improvement_suggestions()
        
        input("\nPress Enter to continue...")
    
    async def _show_improvement_suggestions(self):
        """Show detailed improvement suggestions for the current prompt"""
        console.print(Panel("Generating Improvement Suggestions...", border_style="cyan"))
        
        # Get improvement suggestions
        suggestions = await self.prompt_engineer.suggest_improvements(self.current_prompt, self.current_evaluation)
        
        if "error" in suggestions:
            console.print(Panel(f"[red]Error: {suggestions['error']}[/red]", border_style="red"))
            return
        
        # Display the suggestions
        if "improvements" in suggestions and suggestions["improvements"]:
            console.print("[bold cyan]Detailed Improvement Suggestions:[/bold cyan]")
            
            for idx, item in enumerate(suggestions["improvements"], 1):
                focus = item.get("focus_area", "")
                issue = item.get("issue", "")
                recommendation = item.get("recommendation", "")
                example = item.get("example", "")
                
                console.print(f"[bold]{idx}. {focus}[/bold]")
                console.print(f"   [blue]Issue:[/blue] {issue}")
                console.print(f"   [blue]Recommendation:[/blue] {recommendation}")
                console.print(f"   [blue]Example:[/blue] {example}")
                console.print()
        else:
            console.print("[yellow]No specific improvement suggestions available.[/yellow]")
    
    async def use_templates(self):
        """Use prompt templates to create or enhance prompts"""
        console.clear()
        console.print(Panel("Prompt Templates", border_style="cyan"))
        
        # List available templates
        templates = self.template_manager.list_templates()
        
        console.print("[bold cyan]Available Templates:[/bold cyan]")
        for idx, (name, description) in enumerate(templates, 1):
            console.print(f"{idx}. [blue]{name}[/blue] - {description}")
        
        # Get user choice
        choice = Prompt.ask(
            "\nSelect a template (number), or enter 'v' to view a template, or 'c' to cancel",
            default="c"
        )
        
        if choice.lower() == 'c':
            return
        
        elif choice.lower() == 'v':
            # View a template
            template_idx = Prompt.ask("Enter the template number to view", default="1")
            try:
                template_idx = int(template_idx) - 1
                if 0 <= template_idx < len(templates):
                    template_name = templates[template_idx][0]
                    template = self.template_manager.get_template(template_name)
                    
                    console.print(f"\n[bold cyan]Template: {template_name}[/bold cyan]")
                    console.print(f"[blue]Description:[/blue] {template['description']}")
                    console.print(f"[blue]Template:[/blue]\n{template['template']}")
                else:
                    console.print("[red]Invalid template number.[/red]")
            except ValueError:
                console.print("[red]Invalid input. Please enter a number.[/red]")
            input("\nPress Enter to continue...")
            return await self.use_templates()
        
        else:
            try:
                template_idx = int(choice) - 1
                if 0 <= template_idx < len(templates):
                    template_name = templates[template_idx][0]
                    template = self.template_manager.get_template(template_name)
                    
                    console.print(f"\n[bold cyan]Filling Template: {template_name}[/bold cyan]")
                    console.print(f"[blue]Description:[/blue] {template['description']}")
                    
                    # Extract placeholders from the template
                    placeholders = re.findall(r'\{(\w+)\}', template['template'])
                    filled_values = {}
                    
                    for placeholder in placeholders:
                        value = Prompt.ask(f"Enter value for '{placeholder}'", default="")
                        filled_values[placeholder] = value
                    
                    # Generate the filled template
                    filled_prompt = self.template_manager.fill_template(template_name, **filled_values)
                    console.print(Panel(filled_prompt, title="Generated Prompt", border_style="green"))
                    
                    # Ask if the user wants to use the generated prompt
                    if Confirm.ask("\nWould you like to use this generated prompt?"):
                        self.current_prompt = filled_prompt
                        self.current_analysis = None  # Reset analysis for the new prompt
                        console.print("[green]Current prompt updated to the generated version.[/green]")
                else:
                    console.print("[red]Invalid template number.[/red]")
            except ValueError:
                console.print("[red]Invalid input. Please enter a number.[/red]")
        
        input("\nPress Enter to continue...")
    
    def run(self):
        """Run the main loop of the GIL system"""
        while self.running:
            try:
                self.display_main_menu()
                choice = Prompt.ask("\nSelect an option", choices=[str(i) for i in range(1, 10)], default="9")
                
                if choice == "1":
                    self.loop.run_until_complete(self.get_user_prompt())
                elif choice == "2":
                    if not self.nlp_processor.nlp:
                        console.print("[red]NLP model not loaded. Please check your spaCy installation.[/red]")
                        input("\nPress Enter to continue...")
                        continue
                    self.loop.run_until_complete(self.analyze_prompt())
                elif choice == "3":
                    self.loop.run_until_complete(self.refine_prompt())
                elif choice == "4":
                    self.loop.run_until_complete(self.evaluate_prompt_quality())
                elif choice == "5":
                    self.loop.run_until_complete(self.use_templates())
                elif choice == "6":
                    self.view_prompt_history()
                elif choice == "7":
                    self.change_model_settings()
                elif choice == "8":
                    self.export_prompt_analysis()
                elif choice == "9":
                    self.running = False
                    console.print("[green]Exiting GIL. Goodbye![/green]")
            except Exception as e:
                logger.error(f"Error in main loop: {e}\n{traceback.format_exc()}")
                console.print(f"[red]An error occurred: {e}[/red]")
    
    def view_prompt_history(self):
        """View the history of prompts and refinements"""
        if not self.prompt_engineer.history:
            console.print("[yellow]No prompt history available.[/yellow]")
            input("\nPress Enter to continue...")
            return
        
        console.clear()
        console.print(Panel("Prompt History", border_style="cyan"))
        
        for idx, entry in enumerate(self.prompt_engineer.history, 1):
            console.print(f"[bold cyan]Session {idx}[/bold cyan]")
            console.print(f"[blue]Timestamp:[/blue] {time.ctime(entry['timestamp'])}")
            console.print(f"[blue]Initial Prompt:[/blue] {entry['initial_prompt']}")
            console.print(f"[blue]Refined Prompt:[/blue] {entry['refined_prompt']}")
            console.print()
        
        input("\nPress Enter to continue...")
    
    def change_model_settings(self):
        """Change the model settings"""
        console.clear()
        console.print(Panel("Model Settings", border_style="cyan"))
        
        current_model = self.ai_client.model_name
        console.print(f"[blue]Current Model:[/blue] {current_model}")
        
        new_model = Prompt.ask("Enter the new model name (or press Enter to keep current)", default=current_model)
        if new_model and new_model != current_model:
            if self.ai_client.change_model(new_model):
                console.print(f"[green]Model changed to {new_model} successfully.[/green]")
            else:
                console.print(f"[red]Failed to change model to {new_model}.[/red]")
        
        input("\nPress Enter to continue...")
    
    def export_prompt_analysis(self):
        """Export the current prompt and analysis to a file"""
        if not self.current_prompt:
            console.print("[red]No prompt available to export.[/red]")
            input("\nPress Enter to continue...")
            return
        
        filename = Prompt.ask("Enter the filename to save the prompt and analysis", default="prompt_analysis.json")
        try:
            # Ensure the directory exists
            os.makedirs(os.path.dirname("output/"+filename), exist_ok=True)
            
            data = {
                "prompt": self.current_prompt,
                "analysis": self.current_analysis,
                "refined_prompt": self.refined_prompt,
                "evaluation": self.current_evaluation
            }
            
            with open(filename, "w") as f:
                json.dump(data, f, indent=2)
            console.print(f"[green]Prompt and analysis exported to {filename} successfully.[/green]")
        except Exception as e:
            logger.error(f"Error exporting prompt and analysis: {e}")
            console.print(f"[red]Failed to export prompt and analysis: {e}[/red]")
        
        input("\nPress Enter to continue...")

def main():
    """Main entry point for the GIL system"""
    console.clear()
    console.print(Panel("[bold cyan]Starting GIL System...[/bold cyan]", border_style="cyan"))
    
    # Initialize GIL with optional API key and model name
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        console.print("[yellow]Warning: No API key found. Running in demo mode with limited functionality.[/yellow]")
    
    gil = GIL(api_key=api_key, model_name="gemini-2.0-flash")
    
    # Run the main loop
    try:
        gil.run()
    except KeyboardInterrupt:
        console.print("\n[red]Program interrupted by user. Exiting...[/red]")
    except Exception as e:
        logger.error(f"Unhandled exception: {e}\n{traceback.format_exc()}")
        console.print(f"[red]An unexpected error occurred: {e}[/red]")

if __name__ == "__main__":
    main()