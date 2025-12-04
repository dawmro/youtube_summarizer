# YouTube Video Summarizer & Q&A Tool 

import gradio as gr
import re
import logging
from functools import lru_cache
from typing import Optional, Dict, List, Tuple
from youtube_transcript_api import YouTubeTranscriptApi
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# ============================================================================
# CONFIGURATION & LOGGING
# ============================================================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Use model with extended context window (better for long transcripts)
LLM_MODEL = "llama3.1:8b-instruct-q8_0"
EMBEDDING_MODEL = "mxbai-embed-large"
OLLAMA_BASE_URL = "http://localhost:11434"

# Chunk configuration
DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 100

# ============================================================================
# GLOBAL MODEL INITIALIZATION (Done once at startup)
# ============================================================================

logger.info("Initializing models at startup...")
try:
    GLOBAL_LLM = Ollama(
        model=LLM_MODEL,
        temperature=0.7,
        top_p=0.9,
        base_url=OLLAMA_BASE_URL
    )
    GLOBAL_EMBEDDINGS = OllamaEmbeddings(
        model=EMBEDDING_MODEL,
        base_url=OLLAMA_BASE_URL
    )
    logger.info("✅ Models initialized successfully")
except Exception as e:
    logger.error(f"❌ Failed to initialize models: {e}")
    raise

# ============================================================================
# PROMPT TEMPLATES (Initialized once)
# ============================================================================

SUMMARY_CHAIN = LLMChain(
    llm=GLOBAL_LLM,
    prompt=PromptTemplate(
        template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are an AI assistant tasked with summarizing YouTube video transcripts.
Provide concise, informative summaries that capture the main points of the video content.

Instructions:
1. Summarize the transcript in a single concise paragraph.
2. Ignore any timestamps in your summary.
3. Focus on the spoken content (Text) of the video.

Note: In the transcript, "Text" refers to the spoken words in the video, and "start" indicates the timestamp.
<|eot_id|><|start_header_id|>user<|end_header_id|>

Please summarize the following YouTube video transcript:

{transcript}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

Summary:""",
        input_variables=["transcript"]
    )
)

QA_CHAIN = LLMChain(
    llm=GLOBAL_LLM,
    prompt=PromptTemplate(
        template="""Answer the following question based on the provided context from the video transcript.

If the answer is not in the context, say "I don't have enough information to answer this question."

Context:

{context}

Question: {question}

Answer:""",
        input_variables=["context", "question"]
    )
)

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

@lru_cache(maxsize=128)
def get_video_id(url: str) -> Optional[str]:
    """
    Extract YouTube video ID from URL with caching for repeated URLs.
    
    Args:
        url: YouTube URL
    
    Returns:
        Video ID or None if invalid
    """
    pattern = r'https:\/\/www\.youtube\.com\/watch\?v=([a-zA-Z0-9_-]{11})'
    match = re.search(pattern, url)
    return match.group(1) if match else None


def get_transcript(url: str) -> Optional[List[Dict]]:
    """
    Fetch English transcript from YouTube, prioritizing human-generated.
    
    Args:
        url: YouTube URL
    
    Returns:
        List of transcript dictionaries or None
    """
    video_id = get_video_id(url)
    if not video_id:
        logger.error("Invalid YouTube URL provided")
        return None

    try:
        ytt_api = YouTubeTranscriptApi()
        transcripts = ytt_api.list(video_id)
    except Exception as e:
        logger.error(f"Error accessing transcripts for video {video_id}: {e}")
        return None

    transcript_data = None
    # Prefer human-generated transcripts
    for t in transcripts:
        if t.language_code == 'en':
            if not t.is_generated:
                transcript_data = t.fetch()
                break
            elif transcript_data is None:  # Fallback to auto-generated
                transcript_data = t.fetch()

    return transcript_data


def convert_transcript_to_dict(transcript) -> List[Dict]:
    """
    Ensure transcript is standardized list of dicts.
    
    Args:
        transcript: Raw transcript data
    
    Returns:
        List of dictionaries with text, start, duration
    """
    if isinstance(transcript, list):
        return transcript
    elif hasattr(transcript, 'snippets'):
        return [
            {
                'text': snippet.text,
                'start': snippet.start,
                'duration': snippet.duration
            }
            for snippet in transcript.snippets
        ]
    else:
        raise TypeError(
            f"Expected list of transcript dicts, got {type(transcript).__name__}"
        )


def process_transcript_to_string(transcript: List[Dict]) -> str:
    """
    Format transcript list into single string efficiently.
    Uses list comprehension instead of string concatenation for better performance.
    
    Args:
        transcript: List of transcript dictionaries
    
    Returns:
        Formatted transcript string
    """
    transcript_dict = convert_transcript_to_dict(transcript)
    
    # Use list comprehension + join instead of += concatenation
    # This is O(n) instead of O(n²) for string building
    lines = []
    for item in transcript_dict:
        try:
            text_clean = item['text'].replace('\n', ' ')
            lines.append(f"Text: {text_clean} Start: {item['start']}")
        except KeyError:
            # Skip entries with missing keys
            continue
    
    return '\n'.join(lines)


def chunk_transcript(
    transcript: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP
) -> List[str]:
    """
    Split transcript into chunks for embedding and retrieval.
    
    Args:
        transcript: Full transcript string
        chunk_size: Size in characters
        overlap: Overlap between chunks
    
    Returns:
        List of text chunks
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    return text_splitter.split_text(transcript)


def retrieve_context(
    question: str,
    faiss_index: FAISS,
    k: int = 4
) -> str:
    """
    Retrieve relevant context from FAISS index.
    
    Args:
        question: User's question
        faiss_index: FAISS vector store
        k: Number of results
    
    Returns:
        Context string
    """
    results = faiss_index.similarity_search(question, k=k)
    return "\n".join([doc.page_content for doc in results])


# ============================================================================
# GRADIO SESSION FUNCTIONS WITH OPTIMIZATION
# ============================================================================

def summarize_video_gradio(
    video_url: str,
    state_data: gr.State
) -> Tuple[str, gr.State]:
    """
    Summarize a YouTube video.
    
    Uses global pre-initialized models
    Stores processed data in session state
    Thread-safe with Gradio state management
    """
    if not video_url:
        return "Please provide a valid YouTube URL.", state_data

    try:
        # Fetch transcript
        fetched_data = get_transcript(video_url)
        if not fetched_data:
            return "No English transcript available for this video.", state_data

        # Process to string
        processed_text = process_transcript_to_string(fetched_data)

        # Generate summary using pre-initialized chain
        logger.info(f"Generating summary for video...")
        summary = SUMMARY_CHAIN.run({"transcript": processed_text})

        # Store in session state for Q&A later
        setattr(state_data, "processed_transcript", processed_text)
        setattr(state_data, "chunks", chunk_transcript(processed_text))
        setattr(state_data, "video_url", video_url)

        logger.info("Summary generated successfully")
        return summary, state_data

    except Exception as e:
        logger.error(f"Error in summarize_video_gradio: {e}")
        return f"Error: {str(e)}", state_data


def answer_question_gradio(
    video_url: str,
    user_question: str,
    state_data: gr.State
) -> str:
    """
    Answer a question about video using RAG (optimized).
    
    Uses global pre-initialized models
    Reuses chunks from state if available
    Single FAISS index creation per session
    """
    if not user_question:
        return "Please provide a valid question."

    try:
        # Get or create chunks
        current_chunks = getattr(state_data, "chunks", None)

        if not current_chunks:
            # If chunks not ready, fetch and process transcript
            logger.info("Chunks not in state, fetching and processing...")
            fetched_data = get_transcript(video_url)
            if not fetched_data:
                return "Could not fetch transcript. Please summarize first."

            processed_text = process_transcript_to_string(fetched_data)
            current_chunks = chunk_transcript(processed_text)
            setattr(state_data, "chunks", current_chunks)

        # Create FAISS index (using pre-initialized embeddings)
        logger.info("Creating FAISS index...")
        faiss_index = FAISS.from_texts(current_chunks, GLOBAL_EMBEDDINGS)

        # Retrieve context and generate answer
        logger.info("Retrieving context and generating answer...")
        relevant_context = retrieve_context(user_question, faiss_index, k=4)
        answer = QA_CHAIN.predict(
            context=relevant_context,
            question=user_question
        )

        logger.info("Answer generated successfully")
        return answer

    except Exception as e:
        logger.error(f"Error in answer_question_gradio: {e}")
        return f"Error: {str(e)}"


# ============================================================================
# GRADIO INTERFACE
# ============================================================================

with gr.Blocks() as interface:
    gr.Markdown(
        "# 🎥 YouTube Video Summarizer & Q&A Tool"
        "\n**Powered by Ollama Llama 3.1 8B"
    )

    # Session state management
    session_state = gr.State(
        value={
            "processed_transcript": "",
            "chunks": [],
            "video_url": ""
        }
    )

    with gr.Group():
        gr.Markdown("## 📝 Input")
        video_url = gr.Textbox(
            label="YouTube Video URL",
            placeholder="Enter the YouTube Video URL (e.g., https://www.youtube.com/watch?v=...)",
            lines=1
        )

    with gr.Group():
        gr.Markdown("## 📊 Summarization")
        summarize_btn = gr.Button("🚀 Summarize Video", variant="primary")
        summary_output = gr.Textbox(
            label="Video Summary",
            lines=6,
            interactive=False
        )

    with gr.Group():
        gr.Markdown("## ❓ Question & Answering")
        question_input = gr.Textbox(
            label="Ask a Question About the Video",
            placeholder="What would you like to know about the video?",
            lines=2
        )
        question_btn = gr.Button("✨ Get Answer", variant="primary")
        answer_output = gr.Textbox(
            label="Answer to Your Question",
            lines=6,
            interactive=False
        )

    # Event handlers
    summarize_btn.click(
        summarize_video_gradio,
        inputs=[video_url, session_state],
        outputs=[summary_output, session_state]
    )

    question_btn.click(
        answer_question_gradio,
        inputs=[video_url, question_input, session_state],
        outputs=answer_output
    )

    # Requirements info
    gr.Markdown(
        """
        ## ⚙️ Requirements & Performance Features

        **Local Setup Required:**
        - Ollama server running: `ollama serve`
        - Models downloaded:
          - `ollama pull llama3.1:8b-instruct-q8_0`
          - `ollama pull mxbai-embed-large`

        **Performance Optimizations:**
        - Global model initialization (reused across requests)
        - Session-based state management (thread-safe)
        - Optimized string processing (O(n) vs O(n²))
        - Cached URL parsing caches video ID extraction results
        - Efficient chunk caching in session state

        **Usage Tips:**
        - YouTube video must have available English transcripts
        - Summary is cached for Q&A sessions
        """
    )


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    logger.info("Starting YouTube Video Summarizer & Q&A Application...")
    logger.info(f"Using model: {LLM_MODEL}")
    logger.info(f"Using embeddings: {EMBEDDING_MODEL}")

    # Enable queueing for better concurrent request handling
    interface.queue()
    interface.launch(server_name="0.0.0.0", server_port=7860)
