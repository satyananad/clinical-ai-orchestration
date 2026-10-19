import React, { useState, useRef } from 'react';
import { structureDictation, confirmDictation, uploadAudioDictation, DictationResponse } from '../services/api';
import { Mic, MicOff, Upload, CheckCircle2, AlertTriangle, FileText, Code2, Edit3, Plus, Trash2, ExternalLink } from 'lucide-react';

interface FindingEdit {
  anatomical_site: string;
  tooth_number?: string | null;
  category: string;
  observation: string;
  certainty: string;
  polarity: string;
  source_quote?: string;
}

export const DictationStudio: React.FC = () => {
  const [dictationText, setDictationText] = useState('');
  const [result, setResult] = useState<DictationResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [confirming, setConfirming] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [activeTab, setActiveTab] = useState<'visual' | 'json' | 'draft'>('visual');
  const [rawJsonText, setRawJsonText] = useState('');
  const [jsonError, setJsonError] = useState<string | null>(null);
  const [selectedPatient, setSelectedPatient] = useState<{ id: string; name: string }>({
    id: '',
    name: ''
  });

  // Audio file ref
  const fileInputRef = useRef<HTMLInputElement>(null);
  // Speech recognition ref
  const recognitionRef = useRef<any>(null);

  // 1. Process Text Dictation
  const handleProcess = async () => {
    if (!dictationText.trim()) return;
    setLoading(true);
    try {
      const data = await structureDictation(
        dictationText,
        selectedPatient.id || undefined,
        selectedPatient.name || undefined
      );
      setResult(data);
      setRawJsonText(JSON.stringify(data, null, 2));
      setJsonError(null);
    } catch (err) {
      alert('Failed to process clinical dictation.');
    } finally {
      setLoading(false);
    }
  };

  // 2. Microphone Audio Recording via Web Speech API
  const toggleSpeechRecording = () => {
    const SpeechRecognition = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
    if (!SpeechRecognition) {
      alert('Speech Recognition is not supported by this browser. Please use Chrome/Edge or upload an audio file.');
      return;
    }

    if (isRecording) {
      if (recognitionRef.current) {
        recognitionRef.current.stop();
      }
      setIsRecording(false);
      return;
    }

    try {
      const recognition = new SpeechRecognition();
      recognition.continuous = true;
      recognition.interimResults = true;
      recognition.lang = 'en-US';

      recognition.onstart = () => {
        setIsRecording(true);
      };

      recognition.onresult = (event: any) => {
        let currentTranscript = '';
        for (let i = 0; i < event.results.length; i++) {
          currentTranscript += event.results[i][0].transcript + ' ';
        }
        setDictationText(currentTranscript.trim());
      };

      recognition.onerror = (event: any) => {
        console.error('Speech recognition error:', event.error);
        setIsRecording(false);
      };

      recognition.onend = () => {
        setIsRecording(false);
      };

      recognitionRef.current = recognition;
      recognition.start();
    } catch (err) {
      console.error('Failed to start speech recognition:', err);
      setIsRecording(false);
    }
  };

  // 3. Audio File Upload
  const handleAudioFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setLoading(true);
    try {
      const data = await uploadAudioDictation(
        file,
        dictationText || undefined,
        selectedPatient.id || undefined,
        selectedPatient.name || undefined
      );
      setResult(data);
      setRawJsonText(JSON.stringify(data, null, 2));
      if (!dictationText && data.raw_dictation) {
        setDictationText(data.raw_dictation);
      }
    } catch (err) {
      alert('Failed to upload and process audio dictation file.');
    } finally {
      setLoading(false);
      if (fileInputRef.current) fileInputRef.current.value = '';
    }
  };

  // 4. Sample Presets
  const loadSample = (num: number) => {
    if (num === 1) {
      // Exact mandatory requirement example
      setDictationText("Lower right third molar is horizontally impacted with roots approximating the inferior alveolar nerve canal.");
    } else if (num === 2) {
      setDictationText("UR5 is root treated with a unilocular well defined apical radiolucency 8 by 7 by 6 mm. UR4 is root treated with an apical radiolucency 1 to 2 mm wide. Distal to UR8 is an unerupted supernumerary tooth with an enlarged follicle space, possibly a dentigerous cyst.");
    } else if (num === 3) {
      setDictationText("The temporomandibular joints and paranasal sinuses are outside the field of view. No abnormal calcifications are identified within the evaluated field.");
    }
  };

  // 5. Findings Editing Handlers
  const handleUpdateFinding = (index: number, field: keyof FindingEdit, value: any) => {
    if (!result) return;
    const updatedFindings = [...result.findings];
    updatedFindings[index] = {
      ...updatedFindings[index],
      [field]: value
    };
    const updatedResult = { ...result, findings: updatedFindings };
    setResult(updatedResult);
    setRawJsonText(JSON.stringify(updatedResult, null, 2));
  };

  const handleAddFinding = () => {
    if (!result) return;
    const newFinding: any = {
      anatomical_site: 'Dentition',
      tooth_number: null,
      category: 'General Observation',
      observation: 'New clinical observation',
      certainty: 'definite observation',
      polarity: 'present',
      source_quote: 'clinician added'
    };
    const updatedResult = {
      ...result,
      findings: [...result.findings, newFinding]
    };
    setResult(updatedResult);
    setRawJsonText(JSON.stringify(updatedResult, null, 2));
  };

  const handleDeleteFinding = (index: number) => {
    if (!result) return;
    const updatedFindings = result.findings.filter((_, idx) => idx !== index);
    const updatedResult = { ...result, findings: updatedFindings };
    setResult(updatedResult);
    setRawJsonText(JSON.stringify(updatedResult, null, 2));
  };

  // 6. Raw JSON Sync
  const handleRawJsonChange = (val: string) => {
    setRawJsonText(val);
    try {
      const parsed = JSON.parse(val);
      setResult(parsed);
      setJsonError(null);
    } catch (e: any) {
      setJsonError('Invalid JSON format: ' + e.message);
    }
  };

  // 7. Clinician Review & Sign-Off (Confirm Pipeline)
  const handleConfirmAndSignOff = async () => {
    if (!result) return;
    setConfirming(true);
    try {
      const payload = {
        dictation_json_path: result.dictation_json_path,
        patient_id: selectedPatient.id || result.patient_context?.patient_id,
        patient_name: selectedPatient.name || result.patient_context?.patient_name,
        findings: result.findings,
        impressions: result.impressions,
        recommendations: result.recommendations,
        raw_dictation: result.raw_dictation
      };
      const confirmed = await confirmDictation(payload);
      setResult(confirmed);
      setRawJsonText(JSON.stringify(confirmed, null, 2));
      alert('Draft report successfully reviewed and signed off by clinician.');
    } catch (err) {
      alert('Failed to sign off draft report.');
    } finally {
      setConfirming(false);
    }
  };

  const isConfirmed = result?.report_metadata?.status === 'CONFIRMED_BY_CLINICIAN';

  return (
    <div className="bg-white rounded-2xl border border-slate-200 shadow-sm p-6 space-y-5">
      {/* Header */}
      <div className="flex flex-wrap items-center justify-between border-b border-slate-100 pb-3 gap-2">
        <div className="flex items-center space-x-2">
          <span className="text-emerald-600 font-bold text-base">🩺</span>
          <div>
            <h2 className="font-bold text-slate-900 text-sm">Feature B: Clinical Dictation Studio</h2>
            <p className="text-[11px] text-slate-500">Audio & Speech-to-Text Ingestion • Editable Structured JSON • Clinician Review</p>
          </div>
        </div>
        <div className="flex items-center space-x-2">
          <span className="text-[11px] font-medium text-emerald-700 bg-emerald-50 px-2.5 py-1 rounded-full border border-emerald-200 flex items-center gap-1">
            <CheckCircle2 className="w-3 h-3 text-emerald-600" />
            Anti-Hallucination Guardrail Active
          </span>
        </div>
      </div>

      {/* Dictation Input Controls */}
      <div className="space-y-3">
        <div className="flex items-center justify-between">
          <label className="text-xs font-semibold text-slate-700">Audio / Speech-to-Text Transcript:</label>
          <div className="flex items-center space-x-2">
            {/* Live Microphone Dictation Button */}
            <button
              onClick={toggleSpeechRecording}
              className={`text-xs px-3 py-1.5 rounded-lg font-medium transition flex items-center space-x-1.5 ${
                isRecording 
                  ? 'bg-rose-600 text-white animate-pulse shadow-sm' 
                  : 'bg-emerald-50 hover:bg-emerald-100 text-emerald-800 border border-emerald-200'
              }`}
              title="Speak directly via microphone"
            >
              {isRecording ? <MicOff className="w-3.5 h-3.5" /> : <Mic className="w-3.5 h-3.5" />}
              <span>{isRecording ? 'Listening (Recording)...' : '🎙️ Record Audio Dictation'}</span>
            </button>

            {/* Audio File Upload Button */}
            <button
              onClick={() => fileInputRef.current?.click()}
              className="text-xs bg-slate-100 hover:bg-slate-200 text-slate-700 px-3 py-1.5 rounded-lg font-medium transition flex items-center space-x-1.5 border border-slate-200"
              title="Upload audio file (.wav, .mp3, .m4a, .webm)"
            >
              <Upload className="w-3.5 h-3.5 text-slate-500" />
              <span>Upload Audio</span>
            </button>
            <input
              type="file"
              ref={fileInputRef}
              onChange={handleAudioFileUpload}
              accept="audio/*,.wav,.mp3,.m4a,.webm"
              className="hidden"
            />
          </div>
        </div>

        <textarea
          rows={3}
          value={dictationText}
          onChange={(e) => setDictationText(e.target.value)}
          placeholder="Dictate or type clinical findings, e.g. 'Lower right third molar is horizontally impacted with roots approximating the inferior alveolar nerve canal.'"
          className="w-full text-xs bg-slate-50 border border-slate-300 rounded-xl p-3 focus:outline-none focus:ring-2 focus:ring-emerald-500 font-mono shadow-inner"
        />

        {/* Quick Requirement Preset Samples */}
        <div className="flex flex-wrap items-center justify-between gap-2">
          <div className="flex flex-wrap items-center gap-1.5">
            <span className="text-[11px] text-slate-500 font-medium mr-1">Presets:</span>
            <button 
              onClick={() => loadSample(1)}
              className="text-[11px] bg-slate-100 hover:bg-emerald-100 text-slate-700 hover:text-emerald-900 px-2.5 py-1 rounded-md border border-slate-200 transition"
            >
              Mandatory Ex: Lower right 3rd molar
            </button>
            <button 
              onClick={() => loadSample(2)}
              className="text-[11px] bg-slate-100 hover:bg-emerald-100 text-slate-700 hover:text-emerald-900 px-2.5 py-1 rounded-md border border-slate-200 transition"
            >
              CBCT CBCT: UR5, UR4, Follicle
            </button>
            <button 
              onClick={() => loadSample(3)}
              className="text-[11px] bg-slate-100 hover:bg-emerald-100 text-slate-700 hover:text-emerald-900 px-2.5 py-1 rounded-md border border-slate-200 transition"
            >
              Regional: TMJ & Sinuses
            </button>
          </div>

          <button
            onClick={handleProcess}
            disabled={loading || !dictationText.trim()}
            className="text-xs bg-emerald-600 hover:bg-emerald-700 disabled:opacity-50 text-white px-4 py-2 rounded-xl font-semibold shadow-sm transition flex items-center space-x-1"
          >
            <span>{loading ? 'Synthesizing...' : '✨ Structure Dictation Draft'}</span>
          </button>
        </div>
      </div>

      {/* Structured Output & Clinical Review Section */}
      {result && (
        <div className="border-t border-slate-200 pt-5 space-y-4">
          {/* Review Required Banner */}
          <div className={`p-4 rounded-xl border flex flex-wrap items-center justify-between gap-3 ${
            isConfirmed 
              ? 'bg-emerald-50 border-emerald-300 text-emerald-950' 
              : 'bg-amber-50 border-amber-300 text-amber-950'
          }`}>
            <div className="flex items-start space-x-3">
              {isConfirmed ? (
                <CheckCircle2 className="w-5 h-5 text-emerald-600 shrink-0 mt-0.5" />
              ) : (
                <AlertTriangle className="w-5 h-5 text-amber-600 shrink-0 mt-0.5" />
              )}
              <div>
                <p className="text-xs font-bold">
                  {isConfirmed ? 'CLINICALLY CONFIRMED REPORT' : '⚠️ DRAFT REPORT — REQUIRES CLINICIAN REVIEW'}
                </p>
                <p className="text-[11px] mt-0.5 opacity-90">
                  {isConfirmed 
                    ? `Approved by attending clinician at ${result.report_metadata?.confirmed_at || 'record verification'}.`
                    : 'Unverified AI synthesis from clinical dictation. Attending clinician must review and sign off before medical record entry.'}
                </p>
              </div>
            </div>

            <div className="flex items-center space-x-2">
              {/* Patient Selector */}
              <select
                value={selectedPatient.id}
                onChange={(e) => {
                  const val = e.target.value;
                  if (val === 'PAT-1001') setSelectedPatient({ id: 'PAT-1001', name: 'Priya Sharma' });
                  else if (val === 'PAT-1004') setSelectedPatient({ id: 'PAT-1004', name: 'Priya Sharmila' });
                  else if (val === 'PAT-1008') setSelectedPatient({ id: 'PAT-1008', name: 'Vikram Malhotra' });
                  else if (val === 'PAT-1002') setSelectedPatient({ id: 'PAT-1002', name: 'Rahul Verma' });
                  else setSelectedPatient({ id: '', name: '' });
                }}
                className="text-xs bg-white border border-slate-300 rounded-lg px-2.5 py-1.5 focus:outline-none focus:ring-1 focus:ring-blue-500 font-medium text-slate-700"
              >
                <option value="">Patient: Unassigned / Pending Selection</option>
                <option value="PAT-1001">Priya Sharma (PAT-1001)</option>
                <option value="PAT-1004">Priya Sharmila (PAT-1004)</option>
                <option value="PAT-1008">Vikram Malhotra (PAT-1008)</option>
                <option value="PAT-1002">Rahul Verma (PAT-1002)</option>
              </select>

              {/* Confirm & Sign Off Button */}
              {!isConfirmed ? (
                <button
                  onClick={handleConfirmAndSignOff}
                  disabled={confirming}
                  className="text-xs bg-emerald-600 hover:bg-emerald-700 text-white px-3.5 py-1.5 rounded-lg font-bold shadow-xs transition flex items-center space-x-1"
                >
                  <CheckCircle2 className="w-3.5 h-3.5" />
                  <span>{confirming ? 'Signing Off...' : 'Sign Off & Approve Draft'}</span>
                </button>
              ) : (
                <div className="flex items-center space-x-1.5 bg-emerald-100 text-emerald-800 border border-emerald-300 px-3 py-1.5 rounded-lg text-xs font-bold">
                  <CheckCircle2 className="w-4 h-4 text-emerald-600" />
                  <span>Digitally Signed: Dr. R. Mehta, MDS (SEALED)</span>
                </div>
              )}
            </div>
          </div>

          {/* View Mode Tabs */}
          <div className="flex items-center justify-between border-b border-slate-200 pb-1">
            <div className="flex space-x-2">
              <button
                onClick={() => setActiveTab('visual')}
                className={`text-xs px-3 py-1.5 rounded-lg font-semibold transition flex items-center space-x-1.5 ${
                  activeTab === 'visual' ? 'bg-slate-900 text-white' : 'text-slate-600 hover:bg-slate-100'
                }`}
              >
                <Edit3 className="w-3.5 h-3.5" />
                <span>Editable Findings ({result.findings.length})</span>
              </button>
              <button
                onClick={() => setActiveTab('json')}
                className={`text-xs px-3 py-1.5 rounded-lg font-semibold transition flex items-center space-x-1.5 ${
                  activeTab === 'json' ? 'bg-slate-900 text-white' : 'text-slate-600 hover:bg-slate-100'
                }`}
              >
                <Code2 className="w-3.5 h-3.5" />
                <span>Editable Structured JSON</span>
              </button>
              <button
                onClick={() => setActiveTab('draft')}
                className={`text-xs px-3 py-1.5 rounded-lg font-semibold transition flex items-center space-x-1.5 ${
                  activeTab === 'draft' ? 'bg-slate-900 text-white' : 'text-slate-600 hover:bg-slate-100'
                }`}
              >
                <FileText className="w-3.5 h-3.5" />
                <span>Readable Draft Report</span>
              </button>
            </div>

            {result.dictation_pdf_path && (
              <a
                href={`/patient-pdf/${result.dictation_pdf_path.split('/').pop()}`}
                target="_blank"
                rel="noreferrer"
                className="text-xs text-blue-600 hover:text-blue-800 font-medium flex items-center space-x-1"
              >
                <span>View Diagnostic PDF</span>
                <ExternalLink className="w-3 h-3" />
              </a>
            )}
          </div>

          {/* Tab 1: Visual Findings Editor */}
          {activeTab === 'visual' && (
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <p className="text-xs text-slate-500">Edit observations, tooth numbering, and diagnostic certainty:</p>
                <button
                  onClick={handleAddFinding}
                  className="text-[11px] bg-slate-100 hover:bg-slate-200 text-slate-700 px-2.5 py-1 rounded-md font-medium transition flex items-center space-x-1"
                >
                  <Plus className="w-3 h-3" />
                  <span>Add Finding</span>
                </button>
              </div>

              <div className="space-y-2.5">
                {result.findings.map((f, idx) => (
                  <div key={idx} className="bg-slate-50 border border-slate-200 rounded-xl p-3.5 space-y-2.5">
                    <div className="flex items-center justify-between">
                      <span className="text-[11px] font-bold text-slate-700">Finding #{idx + 1}</span>
                      <button
                        onClick={() => handleDeleteFinding(idx)}
                        className="text-slate-400 hover:text-rose-600 transition"
                        title="Delete finding"
                      >
                        <Trash2 className="w-3.5 h-3.5" />
                      </button>
                    </div>

                    <div className="grid grid-cols-1 md:grid-cols-3 gap-2">
                      <div>
                        <label className="text-[10px] font-semibold text-slate-600 block mb-0.5">Anatomical Site:</label>
                        <input
                          type="text"
                          value={f.anatomical_site}
                          onChange={(e) => handleUpdateFinding(idx, 'anatomical_site', e.target.value)}
                          className="w-full text-xs bg-white border border-slate-300 rounded-md px-2 py-1 font-medium text-slate-800"
                        />
                      </div>
                      <div>
                        <label className="text-[10px] font-semibold text-slate-600 block mb-0.5">Tooth # (FDI/Universal):</label>
                        <input
                          type="text"
                          value={f.tooth_number || ''}
                          placeholder="e.g. 48 or null"
                          onChange={(e) => handleUpdateFinding(idx, 'tooth_number', e.target.value || null)}
                          className="w-full text-xs bg-white border border-slate-300 rounded-md px-2 py-1 text-slate-800"
                        />
                      </div>
                      <div>
                        <label className="text-[10px] font-semibold text-slate-600 block mb-0.5">Diagnostic Certainty:</label>
                        <select
                          value={f.certainty}
                          onChange={(e) => handleUpdateFinding(idx, 'certainty', e.target.value)}
                          className="w-full text-xs bg-white border border-slate-300 rounded-md px-2 py-1 text-slate-800 font-medium"
                        >
                          <option value="definite observation">definite observation</option>
                          <option value="definite observation (etiology uncertain)">definite observation (etiology uncertain)</option>
                          <option value="suspected / differential diagnosis">suspected / differential diagnosis</option>
                          <option value="negative / normal finding">negative / normal finding</option>
                          <option value="unclear / clinician review required">unclear / clinician review required</option>
                        </select>
                      </div>
                    </div>

                    <div>
                      <label className="text-[10px] font-semibold text-slate-600 block mb-0.5">Observation Statement:</label>
                      <textarea
                        rows={2}
                        value={f.observation}
                        onChange={(e) => handleUpdateFinding(idx, 'observation', e.target.value)}
                        className="w-full text-xs bg-white border border-slate-300 rounded-md p-2 text-slate-800"
                      />
                    </div>
                  </div>
                ))}
              </div>

              {/* Impressions summary */}
              <div className="bg-blue-50/60 border border-blue-200/70 rounded-xl p-3.5 space-y-1.5">
                <p className="text-xs font-bold text-blue-900">Radiologist Impression:</p>
                <ul className="list-disc list-inside text-xs text-blue-950 space-y-1">
                  {result.impressions.map((imp, idx) => (
                    <li key={idx}>{imp}</li>
                  ))}
                </ul>
              </div>
            </div>
          )}

          {/* Tab 2: Raw Editable JSON */}
          {activeTab === 'json' && (
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-xs text-slate-500">Edit JSON directly. Modifications sync across the report:</span>
                {jsonError ? (
                  <span className="text-[11px] text-rose-600 font-semibold">{jsonError}</span>
                ) : (
                  <span className="text-[11px] text-emerald-600 font-semibold">✓ Valid JSON</span>
                )}
              </div>
              <textarea
                rows={12}
                value={rawJsonText}
                onChange={(e) => handleRawJsonChange(e.target.value)}
                className="w-full bg-slate-900 text-emerald-400 p-3.5 rounded-xl text-[11px] font-mono focus:outline-none focus:ring-2 focus:ring-emerald-500"
              />
            </div>
          )}

          {/* Tab 3: Readable Draft Report */}
          {activeTab === 'draft' && (
            <div className="bg-slate-50 border border-slate-200 p-4 rounded-xl text-xs text-slate-800 whitespace-pre-wrap font-mono leading-relaxed max-h-96 overflow-y-auto shadow-inner">
              {result.narrative_draft}
            </div>
          )}
        </div>
      )}
    </div>
  );
};
