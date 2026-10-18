import React from 'react';

export const Header: React.FC = () => {
  return (
    <header className="bg-white border-b border-slate-200 sticky top-0 z-50 shadow-sm">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
        <div className="flex items-center space-x-3">
          <div className="w-10 h-10 rounded-xl bg-blue-600 flex items-center justify-center text-white font-bold text-xl shadow-md">
            🦷
          </div>
          <div>
            <h1 className="font-bold text-lg text-slate-900 tracking-tight leading-none">Viscan AI</h1>
            <p className="text-xs text-blue-600 font-medium">Voice-First Dental Radiology Assistant</p>
          </div>
        </div>
        <div className="flex items-center space-x-4">
          <span className="inline-flex items-center px-2.5 py-1 rounded-full text-xs font-medium bg-emerald-100 text-emerald-800 border border-emerald-200">
            <span className="w-2 h-2 mr-1.5 rounded-full bg-emerald-500 animate-pulse"></span>
            FastAPI Backend (8000)
          </span>
          <a 
            href="http://localhost:8000/docs" 
            target="_blank" 
            rel="noreferrer"
            className="text-xs text-slate-600 hover:text-blue-600 font-medium border border-slate-300 rounded-lg px-3 py-1.5 transition"
          >
            Swagger API Docs
          </a>
        </div>
      </div>
    </header>
  );
};
