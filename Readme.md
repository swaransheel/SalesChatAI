# AI Sales Call Chatbot - Replit Configuration

## Overview

This is a Streamlit-based AI sales chatbot application built for LeadMate CRM. The application simulates a professional sales conversation, guiding prospects through a structured sales process using OpenAI's GPT models. The chatbot follows a five-stage sales methodology from introduction to deal closure.
## Demo Link
https://saleschatai-8dkqxyuuzit3pz8ctbapmw.streamlit.app/
## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
- **Framework**: Streamlit web application framework
- **Layout**: Centered layout with custom page configuration
- **UI Components**: Chat interface with conversation flow management
- **Styling**: Built-in Streamlit components with custom page title and icon

### Backend Architecture
- **Language**: Python
- **API Integration**: OpenAI GPT models for conversational AI
- **Session Management**: Streamlit's built-in session state management
- **Configuration**: Environment variable-based API key management

## Key Components

### 1. Main Application (`main.py`)
- **Purpose**: Core application entry point and conversation orchestration
- **Key Functions**:
  - OpenAI client initialization with API key validation
  - Conversation stage tracking and management
  - System prompt configuration for sales methodology

### 2. Conversation Flow Management
- **Five-Stage Sales Process**:
  1. Introduction & Rapport Building
  2. Lead Qualification  
  3. Solution Presentation
  4. Objection Handling
  5. Deal Closure
- **Stage Tracking**: Structured dictionary mapping for conversation progression

### 3. AI Integration Layer
- **Model**: OpenAI GPT (configurable model selection)
- **Prompt Engineering**: Detailed system prompt defining sales persona and methodology
- **Response Management**: Conversational, natural language responses (2-3 sentences)

## Data Flow

1. **User Input**: User messages captured through Streamlit chat interface
2. **Context Processing**: Messages combined with system prompt and conversation stage context
3. **AI Processing**: OpenAI API processes conversation context and generates responses
4. **Response Delivery**: AI responses displayed in chat interface
5. **Stage Progression**: Conversation stage tracked and updated based on interaction flow

## External Dependencies

### Required Packages
- **streamlit**: Web application framework for user interface
- **openai**: Official OpenAI Python client for GPT API integration

### Environment Variables
- **OPENAI_API_KEY**: Required for OpenAI API authentication
- Error handling implemented for missing API keys

### Third-Party Services
- **OpenAI API**: Primary AI service for conversation generation
- **Dependency**: Application requires active internet connection and valid OpenAI API access

## Deployment Strategy

### Platform
- **Target**: Replit cloud platform
- **Package Management**: Requirements.txt with automatic UPM installation
- **Runtime**: Python environment with Streamlit server

### Configuration Requirements
- Environment variable setup for OpenAI API key
- Automatic dependency installation via requirements.txt
- Streamlit app configuration for production deployment

### Scaling Considerations
- Stateless design allows for horizontal scaling
- Session management handled by Streamlit framework
- API rate limiting considerations for OpenAI integration

### Error Handling
- API key validation with graceful error messages
- Environment variable checking with application halt on missing credentials
- User-friendly error display through Streamlit interface
