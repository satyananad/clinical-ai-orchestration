import React, { useState } from 'react';
import { sendAssistantChat, resetSession, AssistantResponse, Patient, ReportDetail } from '../services/api';

interface ChatMessage {
  sender: 'user' | 'assistant';
  content: string;
  isCustom?: boolean;
  candidates?: Patient[];
  citations?: string[];
  available_reports?: ReportDetail[];
}

interface ChatInterfaceProps {
  onReportLoaded: (report: ReportDetail, patient?: Patient) => void;
}

export const ChatInterface: React.FC<ChatInterfaceProps> = ({ onReportLoaded }) => {
  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      sender: 'assistant',
      content: 'Welcome, Doctor. You can speak or type commands such as:\n• "Open Priya\'s latest OPG report" (tests ambiguity handling)\n• "Open Priya Sharma\'s latest OPG report"\n• "Summarize her report" (tests follow-up conversation memory)'
    }
  ]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);

  const handleSend = async (queryText?: string) => {
    const q = queryText || input.trim();
    if (!q) return;

    const newMsgs: ChatMessage[] = [...messages, { sender: 'user', content: q }];
    setMessages(newMsgs);
    if (!queryText) setInput('');
    setLoading(true);

    try {
      const resp: AssistantResponse = await sendAssistantChat(q);

      if (resp.status === 'clarification_required' && resp.candidates) {
        setMessages([
          ...newMsgs,
          {
            sender: 'assistant',
            content: resp.message,
            candidates: resp.candidates
          }
        ]);
      } else if (resp.report) {
        onReportLoaded(resp.report, resp.patient);
        const ans = resp.answer || resp.message;
        setMessages([
          ...newMsgs,
          {
            sender: 'assistant',
            content: ans,
            citations: resp.citations,
            available_reports: resp.available_reports
          }
        ]);
      } else {
        setMessages([
          ...newMsgs,
          {
            sender: 'assistant',
            content: resp.answer || resp.message,
            citations: resp.citations,
            available_reports: resp.available_reports
          }
        ]);
      }
    } catch (err) {
      setMessages([...newMsgs, { sender: 'assistant', content: 'Error communicating with assistant backend.' }]);
    } finally {
      setLoading(false);
    }
  };

  const handleReset = async () => {
    await resetSession();
    setMessages([
      {
        sender: 'assistant',
        content: 'Conversation memory cleared. Session context has been reset.'
      }
    ]);
  };

  const simulateVoice = () => {
    setInput("Open Priya Sharma's latest OPG report");
  };

  return (
    <div className="bg-white rounded-2xl border border-slate-200 shadow-sm flex flex-col h-[560px]">
      <div className="p-4 border-b border-slate-100 flex items-center justify-between bg-slate-50 rounded-t-2xl">
        <div className="flex items-center space-x-2">
          <span className="text-blue-600 font-bold text-base">🎙️</span>
          <h2 className="font-semibold text-slate-900 text-sm">Voice & Conversational Assistant</h2>
        </div>
        <button 
          onClick={handleReset}
          className="text-xs text-slate-400 hover:text-rose-600 transition"
          title="Clear session context"
        >
          🔄 Reset Context
        </button>
      </div>

      <div className="flex-1 p-4 overflow-y-auto space-y-3 text-sm">
        {messages.map((m, idx) => (
          <div key={idx} className={`flex flex-col ${m.sender === 'user' ? 'items-end' : 'items-start'}`}>
            <div 
              className={`rounded-2xl p-3 max-w-[85%] text-xs shadow-sm ${
                m.sender === 'user' 
                  ? 'bg-blue-600 text-white' 
                  : 'bg-white border border-slate-200 text-slate-800'
              }`}
            >
              <p className={`font-semibold text-[10px] mb-0.5 ${m.sender === 'user' ? 'text-blue-100' : 'text-slate-400'}`}>
                {m.sender === 'user' ? 'Clinician' : 'Viscan Assistant'}
              </p>
              <div className="whitespace-pre-line">{m.content}</div>

              {m.candidates && m.candidates.length > 0 && (
                <div className="mt-2.5 space-y-1.5 border-t border-amber-200 pt-2">
                  <p className="text-[11px] font-bold text-amber-800">Matching patient records:</p>
                  {m.candidates.map(c => (
                    <button
                      key={c.patient_id}
                      onClick={() => handleSend(`Open ${c.patient_name} (${c.patient_id})`)}
                      className="w-full text-left bg-amber-50 hover:bg-amber-100 border border-amber-200 rounded-lg p-2 text-xs transition flex justify-between items-center text-slate-800"
                    >
                      <div>
                        <span className="font-bold text-blue-700">{c.patient_name}</span>
                        <span className="text-slate-500 ml-2">DOB: {c.date_of_birth} ({c.sex})</span>
                      </div>
                      <span className="text-[10px] text-slate-400 font-mono">{c.patient_id}</span>
                    </button>
                  ))}
                </div>
              )}

              {m.available_reports && m.available_reports.length > 0 && (
                <div className="mt-2.5 space-y-1.5 border-t border-blue-100 pt-2">
                  <p className="text-[11px] font-bold text-blue-900">Available Reports for this Patient ({m.available_reports.length}):</p>
                  <div className="flex flex-wrap gap-1.5">
                    {m.available_reports.map((rep) => (
                      <button
                        key={rep.report_id}
                        onClick={() => handleSend(`Open ${rep.report_type} report ${rep.report_id}`)}
                        className="bg-blue-50 hover:bg-blue-100 text-blue-800 border border-blue-200 rounded-lg px-2.5 py-1 text-[11px] font-medium transition flex items-center space-x-1"
                        title={`Switch to ${rep.report_type} (${rep.report_id})`}
                      >
                        <span>📄</span>
                        <span>{rep.report_type}</span>
                        <span className="text-slate-400 text-[10px]">({rep.report_id})</span>
                      </button>
                    ))}
                  </div>
                </div>
              )}

              {m.citations && m.citations.length > 0 && (
                <div className="mt-2 pt-2 border-t border-slate-200 text-[10px] text-slate-500 font-mono">
                  <span className="font-bold text-slate-700">Source Evidence Citations:</span>
                  {m.citations.map((c, i) => (
                    <div key={i} className="mt-0.5">• {c}</div>
                  ))}
                </div>
              )}
            </div>
          </div>
        ))}
      </div>

      <div className="p-3 border-t border-slate-100 bg-white rounded-b-2xl">
        <div className="flex items-center space-x-2">
          <input 
            type="text" 
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && handleSend()}
            placeholder="Speak or type request..." 
            className="flex-1 text-sm bg-slate-50 border border-slate-300 rounded-xl px-3 py-2.5 focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
          <button 
            onClick={simulateVoice}
            className="bg-slate-100 hover:bg-slate-200 text-slate-700 w-10 h-10 rounded-xl flex items-center justify-center transition"
            title="Simulate Voice Input"
          >
            🎤
          </button>
          <button 
            onClick={() => handleSend()}
            disabled={loading}
            className="bg-blue-600 hover:bg-blue-700 text-white w-10 h-10 rounded-xl flex items-center justify-center shadow-sm transition"
          >
            ➔
          </button>
        </div>
      </div>
    </div>
  );
};
