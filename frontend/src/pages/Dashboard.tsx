import React, { useState } from 'react';
import { Header } from '../components/Header';
import { ChatInterface } from '../components/ChatInterface';
import { PatientSearch } from '../components/PatientSearch';
import { ReportViewer } from '../components/ReportViewer';
import { DictationStudio } from '../components/DictationStudio';
import { ReportDetail, Patient, getPatientReports } from '../services/api';

export const Dashboard: React.FC = () => {
  const [activeReport, setActiveReport] = useState<ReportDetail | null>(null);
  const [activePatient, setActivePatient] = useState<Patient | null>(null);

  const handlePatientSelected = async (patient: Patient) => {
    setActivePatient(patient);
    try {
      const reports = await getPatientReports(patient.patient_id);
      if (reports.length > 0) {
        setActiveReport(reports[0]);
      }
    } catch (err) {
      console.error(err);
    }
  };

  const handleReportLoaded = (report: ReportDetail, patient?: Patient) => {
    setActiveReport(report);
    if (patient) {
      setActivePatient(patient);
    }
  };

  return (
    <div className="bg-slate-50 min-h-screen flex flex-col font-sans">
      <Header />
      <main className="flex-1 max-w-7xl w-full mx-auto p-4 sm:p-6 lg:p-8 grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left Column: Assistant & Quick Patient Search */}
        <div className="lg:col-span-5 flex flex-col space-y-6">
          <ChatInterface onReportLoaded={handleReportLoaded} />
          <PatientSearch onSelectPatient={handlePatientSelected} />
        </div>

        {/* Right Column: Report Viewer & Dictation Studio */}
        <div className="lg:col-span-7 flex flex-col space-y-6">
          <ReportViewer report={activeReport} patient={activePatient} />
          <DictationStudio />
        </div>
      </main>
    </div>
  );
};
