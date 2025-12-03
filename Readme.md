#  Personality Engine v3.0

> **LLM-driven personality transformation engine with intelligent CSV conversation processing using OpenRouter API.**

##  What is This?

Learns from CSV conversations to extract user insights, then adapts AI responses to match user personality & emotional state.

##  How It Works

## Layer 1: Memory Extraction
- Reads CSV: chat_id, time, user_query, response
- Batch processes 5 conversations at a time
- Sends to OpenRouter LLM for semantic extraction
- Returns: 9 preferences + 5 emotions + 6 facts

## Layer 2: Personality Selection
- Analyzes user profile + current query
- Selects from 6 personality types
- Returns: personality + confidence + tone profile

## Layer 3: Tone Transformation
- Takes base response
- Applies personality tone (6 dimensions)
- Returns: personality-adapted response

### Get API Key from Openrouter
