import React, { useState, useEffect } from 'react';
import { searchPatients, Patient } from '../services/api';

interface PatientSearchProps {
  onSelectPatient: (patient: Patient) => void;
}

export const PatientSearch: React.FC<PatientSearchProps> = ({ onSelectPatient }) => {
  const [query, setQuery] = useState('');
  const [patients, setPatients] = useState<Patient[]>([]);
  const [loading, setLoading] = useState(false);

  const fetchPatients = async (searchTerm?: string) => {
    setLoading(true);
    try {
      const data = await searchPatients(searchTerm);
      setPatients(data);
    } catch (e) {
      console.error(e);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchPatients();
  }, []);

  return (
    <div className="bg-white rounded-2xl border border-slate-200 shadow-sm p-4">
      <h3 className="font-semibold text-slate-900 text-sm mb-3 flex items-center">
        👥 Patient Quick Search
      </h3>
      <div className="flex space-x-2 mb-3">
        <input 
          type="text" 
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && fetchPatients(query)}
          placeholder="Search patient name (e.g. Priya, Vikram)..." 
          className="flex-1 text-xs bg-slate-50 border border-slate-300 rounded-lg px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
        />
        <button 
          onClick={() => fetchPatients(query)} 
          className="text-xs bg-slate-800 hover:bg-slate-900 text-white px-3 py-2 rounded-lg font-medium transition"
        >
          {loading ? '...' : 'Search'}
        </button>
      </div>
      <div className="max-h-48 overflow-y-auto space-y-1.5 text-xs">
        {patients.map(p => (
          <div 
            key={p.patient_id}
            onClick={() => onSelectPatient(p)}
            className="flex items-center justify-between p-2 bg-slate-50 hover:bg-blue-50 border border-slate-200 rounded-lg cursor-pointer transition"
          >
            <div>
              <span className="font-bold text-slate-800">{p.patient_name}</span>
              <span className="text-slate-400 ml-1">({p.sex}, {p.date_of_birth})</span>
            </div>
            <span className="text-blue-600 font-mono text-[10px]">{p.patient_id}</span>
          </div>
        ))}
      </div>
    </div>
  );
};
