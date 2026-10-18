import React from 'react';
import { ReportDetail, Patient } from '../services/api';

interface ReportViewerProps {
  report: ReportDetail | null;
  patient: Patient | null;
}

export const ReportViewer: React.FC<ReportViewerProps> = ({ report, patient }) => {
  if (!report) {
    return (
      <div className="bg-white rounded-2xl border border-slate-200 shadow-sm p-6 flex flex-col">
        <div className="flex items-center justify-between border-b border-slate-100 pb-4 mb-4">
          <div className="flex items-center space-x-2">
            <span className="px-2.5 py-0.5 rounded-full text-xs font-semibold bg-slate-100 text-slate-600">Pending</span>
            <h2 className="font-bold text-slate-900 text-base">Select or Request a Report</h2>
          </div>
        </div>
        <div className="p-8 bg-slate-50 border border-dashed border-slate-200 rounded-xl text-center text-slate-400 text-xs">
          <p className="text-2xl mb-2">📋</p>
          <p>No report loaded. Use the conversational assistant to request a patient report (e.g. <em>"Open Priya Sharma's latest OPG report"</em>).</p>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-white rounded-2xl border border-slate-200 shadow-sm p-6 flex flex-col">
      <div className="flex items-center justify-between border-b border-slate-100 pb-4 mb-4">
        <div>
          <div className="flex items-center space-x-2">
            <span className="px-2.5 py-0.5 rounded-full text-xs font-semibold bg-blue-100 text-blue-800">
              {report.report_type}
            </span>
            <h2 className="font-bold text-slate-900 text-base">
              {report.report_type} Diagnostic Report
            </h2>
          </div>
          <p className="text-xs text-slate-500 mt-1">
            Patient: <span className="font-medium text-slate-700">{patient?.patient_name || 'Verified Patient'}</span> ({report.patient_id}) | Date: {report.report_date} | Report ID: {report.report_id}
          </p>
        </div>
        {report.pdf_filename && (
          <a 
            href={`http://localhost:8000/patient-pdf/${report.pdf_filename}`} 
            target="_blank" 
            rel="noreferrer"
            className="inline-flex items-center text-xs font-medium bg-slate-900 text-white px-3 py-2 rounded-lg hover:bg-slate-800 transition shadow-sm"
          >
            📄 View Diagnostic PDF
          </a>
        )}
      </div>

      <div className="space-y-4 text-sm text-slate-700">
        <div className="bg-blue-50/60 p-3 rounded-xl border border-blue-100 text-xs">
          <span className="font-bold text-blue-900">Clinical Indication:</span> {report.clinical_indication || 'Baseline clinical assessment'}
        </div>

        <div>
          <h4 className="font-bold text-xs text-slate-900 uppercase tracking-wide mb-2">Findings:</h4>
          <div className="space-y-2">
            {report.findings.map((f, i) => (
              <div key={i} className="bg-slate-50 p-2.5 rounded-lg border border-slate-200 text-xs">
                <div className="flex justify-between font-semibold text-slate-800">
                  <span>{f.region}</span>
                  <span className="text-[10px] text-blue-600 bg-blue-50 px-1.5 py-0.5 rounded border border-blue-100">
                    {f.certainty || 'descriptive'}
                  </span>
                </div>
                <p className="text-slate-600 mt-1">{f.observation}</p>
              </div>
            ))}
          </div>
        </div>

        <div className="pt-2 border-t border-slate-100">
          <h4 className="font-bold text-xs text-slate-900 uppercase tracking-wide mb-1">Radiologist Impression:</h4>
          <ul className="list-disc list-inside text-xs text-slate-700 space-y-1">
            {report.impression.map((imp, i) => (
              <li key={i}>{imp}</li>
            ))}
          </ul>
        </div>

        {report.recommendation && (
          <div className="p-2.5 bg-emerald-50 border border-emerald-200 rounded-lg text-xs text-emerald-900">
            <span className="font-bold">Recommendation:</span> {report.recommendation}
          </div>
        )}
      </div>
    </div>
  );
};
