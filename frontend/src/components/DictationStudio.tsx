import React, { useState } from 'react';
import { structureDictation, DictationResponse } from '../services/api';

export const DictationStudio: React.FC = () => {
  const [dictationText, setDictationText] = useState('');
  const [result, setResult] = useState<DictationResponse | null>(null);
  const [loading, setLoading] = useState(false);

  const handleProcess = async () => {
    if (!dictationText.trim()) return;
    setLoading(true);
    try {
      const data = await structureDictation(dictationText);
      setResult(data);
    } catch (err) {
      alert('Failed to process clinical dictation.');
    } finally {
      setLoading(false);
    }
  };

  const loadSample = (num: number) => {
    if (num === 1) {
      setDictationText("Lower right third molar is horizontally impacted with roots approximating the inferior alveolar nerve canal. Mild pericoronal radiolucency noted around the crown.");
    } else if (num === 2) {
      setDictationText("UR5 is root treated with a unilocular well defined apical radiolucency 8 by 7 by 6 mm. UR4 is root treated with an apical radiolucency 1 to 2 mm wide. Distal to UR8 is an unerupted supernumerary tooth with an enlarged follicle space, possibly a dentigerous cyst.");
    } else if (num === 3) {
      setDictationText("Tooth #29 shows apical rarefying and sclerosing osteitis. TMJs show bilateral condylar flattening consistent with degenerative joint changes. Left palatine tonsillolith noted incidentally.");
    }
  };

  return (
    <div className="bg-white rounded-2xl border border-slate-200 shadow-sm p-6">
      <div className="flex items-center justify-between border-b border-slate-100 pb-3 mb-4">
        <div className="flex items-center space-x-2">
          <span className="text-emerald-600 font-bold text-base">🩺</span>
          <h2 className="font-bold text-slate-900 text-sm">Feature B: Spoken Clinical Dictation Studio</h2>
        </div>
        <span className="text-xs font-medium text-emerald-600 bg-emerald-50 px-2.5 py-0.5 rounded-full border border-emerald-200">
          Anti-Hallucination Guardrail Active
        </span>
      </div>

      <div className="space-y-3">
        <label className="block text-xs font-medium text-slate-700">Doctor Dictation Transcript:</label>
        <textarea
          rows={3}
          value={dictationText}
          onChange={(e) => setDictationText(e.target.value)}
          placeholder="e.g. Lower right third molar is horizontally impacted with roots approximating the inferior alveolar nerve canal."
          className="w-full text-xs bg-slate-50 border border-slate-300 rounded-xl p-3 focus:outline-none focus:ring-2 focus:ring-emerald-500 font-mono"
        />

        <div className="flex flex-wrap items-center justify-between gap-2">
          <div className="flex items-center space-x-2">
            <button 
              onClick={() => loadSample(1)}
              className="text-[11px] bg-slate-100 hover:bg-slate-200 text-slate-700 px-2.5 py-1 rounded-md transition"
            >
              Sample 1: Impacted 48
            </button>
            <button 
              onClick={() => loadSample(2)}
              className="text-[11px] bg-slate-100 hover:bg-slate-200 text-slate-700 px-2.5 py-1 rounded-md transition"
            >
              Sample 2: Harley St CBCT (UR5, UR4)
            </button>
            <button 
              onClick={() => loadSample(3)}
              className="text-[11px] bg-slate-100 hover:bg-slate-200 text-slate-700 px-2.5 py-1 rounded-md transition"
            >
              Sample 3: Tooth #29 & TMJ
            </button>
          </div>

          <button
            onClick={handleProcess}
            disabled={loading}
            className="text-xs bg-emerald-600 hover:bg-emerald-700 text-white px-4 py-2 rounded-xl font-semibold shadow-sm transition flex items-center"
          >
            {loading ? 'Synthesizing...' : '✨ Synthesize Structured Draft'}
          </button>
        </div>
      </div>

      {result && (
        <div className="mt-5 border-t border-slate-100 pt-4">
          <div className="bg-amber-50 border-l-4 border-amber-500 p-3 rounded-r-lg mb-4">
            <p className="text-xs font-bold text-amber-900 flex items-center">
              ⚠️ DRAFT REPORT - REQUIRES CLINICIAN REVIEW
            </p>
            <p className="text-[11px] text-amber-800 mt-0.5">
              Unverified AI synthesis from voice dictation. Treating clinician must review and sign off.
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <h4 className="text-xs font-bold text-slate-900 mb-1">Editable Structured JSON:</h4>
              <pre className="bg-slate-900 text-emerald-400 p-3 rounded-xl text-[11px] font-mono max-h-56 overflow-y-auto">
                {JSON.stringify(result, null, 2)}
              </pre>
            </div>
            <div>
              <h4 className="text-xs font-bold text-slate-900 mb-1">Readable Draft Report:</h4>
              <div className="bg-slate-50 border border-slate-200 p-3 rounded-xl text-xs text-slate-800 max-h-56 overflow-y-auto whitespace-pre-wrap font-sans">
                {result.narrative_draft}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
