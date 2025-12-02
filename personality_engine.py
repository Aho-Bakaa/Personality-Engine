"""
Personality Engine v3.0 - LLM-Driven Production Implementation
Complete refactor with OpenRouter integration for reasoning-driven extraction

Key improvements:
- LLM-based memory extraction 
- Reasoning-driven personality selection
- Semantic tone transformations
- Structured output with confidence scores
- Batch processing for efficiency
"""

import os
import json
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict
from enum import Enum
from datetime import datetime
from pathlib import Path
import re
import pandas as pd
from dotenv import load_dotenv
load_dotenv()
# ==================== CONFIGURATION ====================

class LLMConfig:
    API_KEY = os.getenv("OPENROUTER_API_KEY", "")
    MODEL = "openai/gpt-oss-20b:free"  
    BASE_URL = "https://openrouter.ai/api/v1"
    TIMEOUT = 60
    MAX_RETRIES = 3
    BATCH_SIZE = 5  

# ==================== ENUMS & DATA CLASSES ====================

class PersonalityArchetype(Enum):
    """Personality styles the agent can adopt"""
    MENTOR_CALM = "mentor_calm"
    WITTY_FRIEND = "witty_friend"
    THERAPIST = "therapist"
    TECHNICAL_EXPERT = "technical_expert"
    ENTHUSIAST = "enthusiast"
    PRAGMATIST = "pragmatist"

class EmotionalContext(Enum):
    """Current emotional state of user"""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    ANXIOUS = "anxious"
    FRUSTRATED = "frustrated"

@dataclass
class ToneProfile:
    """Represents tone characteristics of a personality"""
    conciseness: float = 0.5  
    formality: float = 0.5
    humor: float = 0.5        
    empathy: float = 0.5     
    structure: float = 0.5    
    technical_depth: float = 0.5 

    def to_dict(self):
        return asdict(self)

@dataclass
class UserPreference:
    """Structured representation of a user preference"""
    category: str
    preference: str
    anti_preference: Optional[str] = None
    frequency: int = 1
    evidence_ids: List[int] = field(default_factory=list)
    confidence_score: float = 0.0
    reasoning: str = ""

    def to_dict(self):
        return asdict(self)

@dataclass
class EmotionalPattern:
    """Structured representation of emotional patterns"""
    emotion: str
    valence: str 
    frequency: int = 1
    last_occurrence: Optional[str] = None
    triggers: List[str] = field(default_factory=list)
    evidence_ids: List[int] = field(default_factory=list)
    coping_strategies: List[str] = field(default_factory=list)
    reasoning: str = ""

    def to_dict(self):
        return asdict(self)

@dataclass
class RememberableFactum:
    """Structured representation of memorable facts"""
    fact_type: str
    content: str
    subcategory: str
    temporal_context: Optional[str] = None
    priority_level: str = "medium"
    evidence_ids: List[int] = field(default_factory=list)
    relationships: List[str] = field(default_factory=list)
    reasoning: str = ""

    def to_dict(self):
        return asdict(self)

@dataclass
class ExtractionResult:
    """Complete structured output from memory extraction"""
    preferences: List[UserPreference]
    emotional_patterns: List[EmotionalPattern]
    memorable_facts: List[RememberableFactum]
    extraction_timestamp: str
    total_conversations_analyzed: int
    llm_model_used: str = "openai/gpt-oss-20b:free"

    def to_dict(self):
        return {
            "preferences": [p.to_dict() for p in self.preferences],
            "emotional_patterns": [ep.to_dict() for ep in self.emotional_patterns],
            "facts": [f.to_dict() for f in self.memorable_facts],
            "metadata": {
                "extraction_timestamp": self.extraction_timestamp,
                "conversations_analyzed": self.total_conversations_analyzed,
                "llm_model": self.llm_model_used
            }
        }

    def to_json(self, filepath: Optional[str] = None) -> str:
        """Serialize to JSON string or file"""
        json_str = json.dumps(self.to_dict(), indent=2, default=str)
        if filepath:
            Path(filepath).write_text(json_str)
        return json_str

@dataclass
class PersonalitySelection:
    """Result of personality selection process"""
    personality: PersonalityArchetype
    confidence: float
    reasoning: str
    emotional_context: EmotionalContext
    recommended_tone: ToneProfile
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self):
        return {
            "personality": self.personality.value,
            "confidence": round(self.confidence, 2),
            "reasoning": self.reasoning,
            "emotional_context": self.emotional_context.value,
            "tone": self.recommended_tone.to_dict(),
            "metadata": self.metadata
        }

@dataclass
class TransformedResponse:
    """Output of personality transformation"""
    original_response: str
    transformed_response: str
    personality_used: PersonalityArchetype
    personality_confidence: float
    emotional_context: EmotionalContext
    tone_profile: ToneProfile
    reasoning: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self):
        return {
            "original": self.original_response,
            "transformed": self.transformed_response,
            "personality": self.personality_used.value,
            "confidence": round(self.personality_confidence, 2),
            "emotional_context": self.emotional_context.value,
            "tone": self.tone_profile.to_dict(),
            "reasoning": self.reasoning,
            "metadata": self.metadata
        }

# ==================== OPENROUTER CLIENT WRAPPER ====================

class OpenRouterClient:
    """Wrapper for OpenRouter API with retry logic and proper error handling"""

    def __init__(self, api_key: str = None):
        self.api_key = api_key or LLMConfig.API_KEY
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY not set. Set via environment variable or pass to constructor.")

        try:
            from openai import OpenAI
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=LLMConfig.BASE_URL
            )
        except ImportError:
            raise ImportError("anthropic library required: pip install anthropic")

        self.model = LLMConfig.MODEL

    def extract_json(self, prompt: str, schema_description: str = "") -> Dict[str, Any]:
        """Extract structured JSON from LLM response"""
        full_prompt = f"""{prompt}

CRITICAL: Return ONLY valid JSON (no markdown, no explanation).
{schema_description}"""

        for attempt in range(LLMConfig.MAX_RETRIES):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    max_tokens=4096,
                    messages=[{
                        "role": "user",
                        "content": full_prompt
                    }]
                )

                response_text = response.choices[0].message.content

                if "```json" in response_text:
                    response_text = response_text.split("```json")[1].split("```")[0]
                elif "```" in response_text:
                    response_text = response_text.split("```")[1].split("```")[0]

                return json.loads(response_text)

            except json.JSONDecodeError as e:
                if attempt == LLMConfig.MAX_RETRIES - 1:
                    raise ValueError(f"Failed to parse JSON after {LLMConfig.MAX_RETRIES} attempts: {e}")
                continue

            except Exception as e:
                if attempt == LLMConfig.MAX_RETRIES - 1:
                    raise
                continue

    def extract_text(self, prompt: str) -> str:
        """Extract plain text from LLM response"""
        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=2048,
            messages=[{
                "role": "user",
                "content": prompt
            }]
        )
        return response.choices[0].message.content.strip()

# ==================== MEMORY EXTRACTION ENGINE ====================

class MemoryExtractionEngine:
    """LLM-based memory extraction with reasoning"""

    def __init__(self, llm_client: OpenRouterClient):
        self.llm = llm_client

    def extract_all(self, conversations_df) -> ExtractionResult:
        """Execute complete LLM-driven extraction pipeline"""

        if pd is None:
            raise ImportError("pandas required for extract_all. Install: pip install pandas")

        # Process conversations in batches
        all_preferences = []
        all_emotions = []
        all_facts = []

        for batch_idx in range(0, len(conversations_df), LLMConfig.BATCH_SIZE):
            batch = conversations_df.iloc[batch_idx:batch_idx + LLMConfig.BATCH_SIZE]
            print(f"Processing batch {batch_idx // LLMConfig.BATCH_SIZE + 1}...")

            # Extract from this batch
            prefs, emotions, facts = self._extract_batch(batch)
            all_preferences.extend(prefs)
            all_emotions.extend(emotions)
            all_facts.extend(facts)

        # Deduplicate & consolidate
        preferences = self._consolidate_preferences(all_preferences)
        emotional_patterns = self._consolidate_emotions(all_emotions)
        memorable_facts = self._consolidate_facts(all_facts)

        return ExtractionResult(
            preferences=preferences,
            emotional_patterns=emotional_patterns,
            memorable_facts=memorable_facts,
            extraction_timestamp=datetime.now().isoformat(),
            total_conversations_analyzed=len(conversations_df),
            llm_model_used=LLMConfig.MODEL
        )

    def _extract_batch(self, batch_df) -> Tuple[List[UserPreference], List[EmotionalPattern], List[RememberableFactum]]:
        """Extract preferences, emotions, and facts from batch using LLM"""

        # Format batch as readable text
        conversations_text = self._format_conversations(batch_df)

        # Extraction prompt with structured schema
        extraction_prompt = f"""
{conversations_text}

Analyze these conversations and extract:

1. USER PREFERENCES: Things user likes/dislikes, communication style, tech preferences, values
2. EMOTIONAL PATTERNS: Recurring emotions, triggers, coping strategies, frequency
3. MEMORABLE FACTS: Personal info (name, job, location), achievements, relationships, important dates

For each item, provide:
- Clear statement of the preference/emotion/fact
- Which conversations provide evidence (by chat_id)
- Confidence score (0.0-1.0) based on frequency/clarity
- Brief reasoning why this was extracted

Return valid JSON with this structure:
{{
    "preferences": [
        {{
            "category": "string",
            "preference": "string",
            "anti_preference": "string or null",
            "frequency": number,
            "evidence_ids": [number],
            "confidence_score": number,
            "reasoning": "string"
        }}
    ],
    "emotional_patterns": [
        {{
            "emotion": "string",
            "valence": "positive|negative|neutral|mixed",
            "frequency": number,
            "last_occurrence": "ISO8601 timestamp or null",
            "triggers": ["string"],
            "evidence_ids": [number],
            "coping_strategies": ["string"],
            "reasoning": "string"
        }}
    ],
    "memorable_facts": [
        {{
            "fact_type": "string",
            "content": "string",
            "subcategory": "string",
            "temporal_context": "ISO8601 timestamp or null",
            "priority_level": "high|medium|low",
            "evidence_ids": [number],
            "relationships": ["string"],
            "reasoning": "string"
        }}
    ]
}}
"""

        extracted = self.llm.extract_json(extraction_prompt)

        # Parse results with validation
        preferences = []
        for p in extracted.get('preferences', []):
            try:
                preferences.append(UserPreference(**p))
            except Exception as e:
                print(f"Warning: Could not parse preference: {e}")
                continue

        emotions = []
        for e in extracted.get('emotional_patterns', []):
            try:
                emotions.append(EmotionalPattern(**e))
            except Exception as e:
                print(f"Warning: Could not parse emotion: {e}")
                continue

        facts = []
        for f in extracted.get('memorable_facts', []):
            try:
                facts.append(RememberableFactum(**f))
            except Exception as e:
                print(f"Warning: Could not parse fact: {e}")
                continue

        return preferences, emotions, facts

    def _format_conversations(self, batch_df) -> str:
        """Format batch of conversations for LLM analysis"""
        text = "CONVERSATIONS:\n\n"
        for idx, row in batch_df.iterrows():
            text += f"""[Chat ID: {row.get('chat_id', idx)}]
Timestamp: {row.get('time', 'N/A')}
User Query: {row.get('user_query', row.get('query', ''))}
Response: {row.get('response', '')}
---
"""
        return text

    def _consolidate_preferences(self, preferences: List[UserPreference]) -> List[UserPreference]:
        """Deduplicate and consolidate preferences"""
        preference_map = {}

        for pref in preferences:
            key = (pref.category, pref.preference)
            if key not in preference_map:
                preference_map[key] = pref
            else:
                # Merge frequencies
                existing = preference_map[key]
                existing.frequency += pref.frequency
                existing.evidence_ids = list(set(existing.evidence_ids + pref.evidence_ids))
                # Update confidence based on increased frequency
                existing.confidence_score = min(1.0, existing.confidence_score + 0.05)

        return sorted(list(preference_map.values()), 
                     key=lambda p: p.confidence_score, reverse=True)

    def _consolidate_emotions(self, emotions: List[EmotionalPattern]) -> List[EmotionalPattern]:
        """Deduplicate and consolidate emotional patterns"""
        emotion_map = {}

        for emotion in emotions:
            key = emotion.emotion
            if key not in emotion_map:
                emotion_map[key] = emotion
            else:
                existing = emotion_map[key]
                existing.frequency += emotion.frequency
                existing.evidence_ids = list(set(existing.evidence_ids + emotion.evidence_ids))
                existing.triggers = list(set(existing.triggers + emotion.triggers))
                existing.coping_strategies = list(set(existing.coping_strategies + emotion.coping_strategies))

        return sorted(list(emotion_map.values()), 
                     key=lambda e: e.frequency, reverse=True)

    def _consolidate_facts(self, facts: List[RememberableFactum]) -> List[RememberableFactum]:
        """Deduplicate and consolidate facts"""
        fact_map = {}

        for fact in facts:
            key = (fact.fact_type, fact.subcategory)
            if key not in fact_map:
                fact_map[key] = fact
            else:
                existing = fact_map[key]
                existing.evidence_ids = list(set(existing.evidence_ids + fact.evidence_ids))

        return sorted(list(fact_map.values()), 
                     key=lambda f: f.priority_level == 'high', reverse=True)

# ==================== PERSONALITY SELECTION ENGINE ====================

class PersonalitySelectionEngine:
    """LLM-based personality selection with reasoning"""

    def __init__(self, llm_client: OpenRouterClient):
        self.llm = llm_client

    def select_personality(self, memory: Dict[str, Any], user_query: str) -> PersonalitySelection:
        """Select best personality archetype with reasoning"""
        profile_text = self._format_user_profile(memory)

        selection_prompt = f"""
{profile_text}

CURRENT USER QUERY: "{user_query}"

Available personality archetypes:
1. MENTOR_CALM: Patient mentor who explains thoughtfully, high empathy, structured
2. WITTY_FRIEND: Casual friend who keeps things fun, high humor, concise
3. THERAPIST: Empathetic supporter who validates feelings, maximum empathy
4. TECHNICAL_EXPERT: Precise expert who values accuracy and technical depth
5. ENTHUSIAST: Energetic motivator who inspires action, high humor and empathy
6. PRAGMATIST: Direct, efficient problem-solver, maximum conciseness

Select the BEST matching personality based on:
- User's stated preferences and work style
- Current emotional patterns and stress levels
- Nature of the current query
- What personality would be most helpful RIGHT NOW

Return valid JSON:
{{
    "selected_personality": "technical_expert",
    "confidence": 0.92,
    "reasoning": "Clear explanation of why this personality was selected based on user profile and query",
    "emotional_context": "neutral|positive|negative|anxious|frustrated",
    "recommended_tone": {{
        "conciseness": 0.9,
        "formality": 0.8,
        "humor": 0.1,
        "empathy": 0.3,
        "structure": 0.85,
        "technical_depth": 0.95
    }},
    "metadata": {{
        "has_stress_patterns": boolean,
        "prefers_concise": boolean,
        "is_technical": boolean,
        "detection_reasoning": "Explain what signals led to this decision"
    }}
}}
"""

        result = self.llm.extract_json(selection_prompt)

        try:
            personality = PersonalityArchetype(result['selected_personality'])
            emotional_context = EmotionalContext(result['emotional_context'])
            tone = ToneProfile(**result['recommended_tone'])
        except (KeyError, ValueError) as e:
            raise ValueError(f"Invalid response format from personality selection: {e}")

        return PersonalitySelection(
            personality=personality,
            confidence=result['confidence'],
            reasoning=result['reasoning'],
            emotional_context=emotional_context,
            recommended_tone=tone,
            metadata=result.get('metadata', {})
        )

    def _format_user_profile(self, memory: Dict[str, Any]) -> str:
        """Format user profile for LLM analysis"""
        text = "USER PROFILE:\n\n"

        text += "TOP PREFERENCES:\n"
        prefs = memory.get('preferences', [])
        for pref in prefs[:5]:
            text += f"- {pref.get('category', '')}: {pref.get('preference', '')} (confidence: {pref.get('confidence_score', 0):.1f})\n"
        if not prefs:
            text += "- (None extracted)\n"

        text += "\nEMOTIONAL PATTERNS:\n"
        emotions = memory.get('emotional_patterns', [])
        for ep in emotions[:3]:
            text += f"- {ep.get('emotion', '')} (valence: {ep.get('valence', '')}, frequency: {ep.get('frequency', 1)})\n"
        if not emotions:
            text += "- (None extracted)\n"

        # Facts
        text += "\nKEY FACTS:\n"
        facts = memory.get('facts', [])
        for fact in facts[:5]:
            text += f"- {fact.get('content', '')} (type: {fact.get('type', '')}, priority: {fact.get('priority_level', '')})\n"
        if not facts:
            text += "- (None extracted)\n"

        return text

# ==================== TONE TRANSFORMATION ENGINE ====================

class ToneTransformationEngine:
    """LLM-based semantic tone transformation"""

    def __init__(self, llm_client: OpenRouterClient):
        self.llm = llm_client

    def transform_tone(self, response: str, tone_profile: ToneProfile, 
                       personality: PersonalityArchetype) -> str:
        """Transform response using semantic tone changes via LLM"""

        tone_desc = self._describe_tone(tone_profile, personality)

        tone_prompt = f"""
Transform this response to match the specified tone profile:

TONE PROFILE:
{tone_desc}

PERSONALITY ARCHETYPE: {personality.value}

ORIGINAL RESPONSE:
\"\"{response}\"\"

Requirements:
- Conciseness {int(tone_profile.conciseness * 100)}%: {"ultra-brief, minimal elaboration" if tone_profile.conciseness > 0.8 else "balanced verbosity" if tone_profile.conciseness > 0.3 else "detailed and thorough"}
- Formality {int(tone_profile.formality * 100)}%: {"very formal, proper grammar" if tone_profile.formality > 0.8 else "casual and conversational" if tone_profile.formality < 0.3 else "balanced tone"}
- Humor {int(tone_profile.humor * 100)}%: {"witty and funny" if tone_profile.humor > 0.8 else "serious and straightforward" if tone_profile.humor < 0.2 else "some light humor"}
- Empathy {int(tone_profile.empathy * 100)}%: {"highly empathetic, validate feelings" if tone_profile.empathy > 0.8 else "neutral, factual" if tone_profile.empathy < 0.3 else "balanced with some empathy"}
- Structure {int(tone_profile.structure * 100)}%: {"numbered/bulleted list" if tone_profile.structure > 0.8 else "flowing prose" if tone_profile.structure < 0.3 else "mixed format"}
- Technical depth {int(tone_profile.technical_depth * 100)}%: {"highly technical with jargon" if tone_profile.technical_depth > 0.8 else "simple, accessible language" if tone_profile.technical_depth < 0.3 else "balanced technical level"}

Return ONLY the transformed response. No explanations, no meta-commentary.
"""

        transformed = self.llm.extract_text(tone_prompt)
        return transformed

    def _describe_tone(self, tone: ToneProfile, personality: PersonalityArchetype) -> str:
        """Create human-readable tone description"""
        desc = f"""
- Conciseness: {tone.conciseness:.1f}/1.0 (0=verbose, 1=ultra-brief)
- Formality: {tone.formality:.1f}/1.0 (0=casual, 1=very formal)  
- Humor: {tone.humor:.1f}/1.0 (0=serious, 1=very funny)
- Empathy: {tone.empathy:.1f}/1.0 (0=detached, 1=highly empathetic)
- Structure: {tone.structure:.1f}/1.0 (0=flowing, 1=structured/numbered)
- Technical Depth: {tone.technical_depth:.1f}/1.0 (0=simple, 1=highly technical)

Personality Context: {personality.value}
"""
        return desc

# ==================== MAIN ORCHESTRATOR ====================

class PersonalityEngine:
    """Main orchestrator for LLM-driven personality transformation"""

    def __init__(self, api_key: str = None):
        self.llm_client = OpenRouterClient(api_key)
        self.memory_engine = MemoryExtractionEngine(self.llm_client)
        self.personality_engine = PersonalitySelectionEngine(self.llm_client)
        self.tone_engine = ToneTransformationEngine(self.llm_client)
        self.memory = None
        self.extraction_result = None

    def extract_memory(self, conversations_df) -> ExtractionResult:
        """Extract user memory from conversations"""
        result = self.memory_engine.extract_all(conversations_df)
        self.extraction_result = result
        self.memory = result.to_dict()

        print(f"\n✓ Memory Extraction Complete:")
        print(f"  - {len(result.preferences)} preferences")
        print(f"  - {len(result.emotional_patterns)} emotional patterns")
        print(f"  - {len(result.memorable_facts)} facts")
        print(f"  - Model: {result.llm_model_used}")

        return result

    def transform(self, user_query: str, base_response: str) -> TransformedResponse:
        """Transform response based on user profile and emotional context"""

        if self.memory is None:
            raise ValueError("No memory loaded. Call extract_memory() first.")

        # Select personality
        print(f"\nSelecting personality for query: {user_query[:50]}...")
        personality_selection = self.personality_engine.select_personality(
            self.memory, user_query
        )
        print(f"  ✓ Selected: {personality_selection.personality.value} ({personality_selection.confidence:.0%})")

        # Transform tone
        print(f"Transforming tone based on profile...")
        transformed = self.tone_engine.transform_tone(
            base_response,
            personality_selection.recommended_tone,
            personality_selection.personality
        )
        print(f"  ✓ Tone transformed")

        return TransformedResponse(
            original_response=base_response,
            transformed_response=transformed,
            personality_used=personality_selection.personality,
            personality_confidence=personality_selection.confidence,
            emotional_context=personality_selection.emotional_context,
            tone_profile=personality_selection.recommended_tone,
            reasoning=personality_selection.reasoning,
            metadata={
                "personality_reasoning": personality_selection.reasoning,
                "tone_adjustments": personality_selection.metadata
            }
        )

    def save_memory(self, filepath: str) -> str:
        """Save extracted memory to JSON file"""
        if self.extraction_result is None:
            raise ValueError("No extraction result to save. Call extract_memory() first.")
        return self.extraction_result.to_json(filepath)

# ==================== USAGE EXAMPLE ====================

if __name__ == "__main__":
    # Example usage
    print("Personality Engine v3.0 - OpenRouter LLM Integration")
    print("=" * 80)

    engine = PersonalityEngine()

    print("\n✓ Engine initialized with OpenRouter")
    print(f"  Model: {LLMConfig.MODEL}")
    print(f"  API Key: {'Set' if LLMConfig.API_KEY else 'NOT SET'}")