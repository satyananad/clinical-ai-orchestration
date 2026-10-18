"""
BACKWARD COMPATIBILITY SHIM FOR CONVERSATION SERVICE
---------------------------------------------------
Re-exports classes and functions from:
- conversation_service.py (Session memory & reference resolution)
- report_qa_service.py (Evidence-grounded RAG over reports)
"""

from .conversation_service import (
    SessionState,
    SESSIONS,
    get_or_create_session,
    is_contextual_query,
    handle_assistant_interaction
)

from .report_qa_service import (
    generate_report_summary,
    answer_report_question
)
