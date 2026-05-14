import { useState, useRef } from 'react';
import { FileSpreadsheet, Upload, Download, X, CheckCircle2, AlertTriangle } from 'lucide-react';
import DiagnosisBadge from '../components/DiagnosisBadge';
import LoadingSpinner from '../components/LoadingSpinner';
import { predictBatch } from '../api';

export default function BatchPredict() {
  const [file, setFile] = useState(null);
  const [dragActive, setDragActive] = useState(false);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [sortField, setSortField] = useState(null);
  const [sortAsc, setSortAsc] = useState(true);
  const inputRef = useRef(null);

  const handleDrag = (e) => { e.preventDefault(); e.stopPropagation(); setDragActive(e.type === 'dragenter' || e.type === 'dragover'); };
  const handleDrop = (e) => { e.preventDefault(); e.stopPropagation(); setDragActive(false); if (e.dataTransfer.files?.[0]) { setFile(e.dataTransfer.files[0]); setResult(null); setError(null); } };
  const handleFileChange = (e) => { if (e.target.files?.[0]) { setFile(e.target.files[0]); setResult(null); setError(null); } };

  const handleSubmit = async () => {
    if (!file) return;
    setLoading(true); setError(null); setResult(null);
    try { const res = await predictBatch(file); setResult(res); }
    catch (err) { setError(err.message); }
    finally { setLoading(false); }
  };

  const handleDownload = () => {
    if (!result) return;
    const header = 'Row,Diagnosis,Confidence\n';
    const rows = result.results.map(r => `${r.row},${r.diagnosis},${r.confidence}`).join('\n');
    const blob = new Blob([header + rows], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a'); a.href = url; a.download = 'predictions.csv'; a.click();
    URL.revokeObjectURL(url);
  };

  const handleSort = (field) => {
    if (sortField === field) setSortAsc(!sortAsc);
    else { setSortField(field); setSortAsc(true); }
  };

  const sortedResults = result ? [...result.results].sort((a, b) => {
    if (!sortField) return 0;
    const va = a[sortField], vb = b[sortField];
    if (typeof va === 'number') return sortAsc ? va - vb : vb - va;
    return sortAsc ? String(va).localeCompare(String(vb)) : String(vb).localeCompare(String(va));
  }) : [];

  return (
    <div className="page-container gradient-mesh min-h-screen">
      <div className="mb-8 animate-fade-in">
        <div className="flex items-center gap-3 mb-2">
          <div className="w-12 h-12 rounded-2xl bg-gradient-to-br from-indigo-500 to-purple-500 flex items-center justify-center shadow-lg shadow-indigo-500/20">
            <FileSpreadsheet className="w-6 h-6 text-white" />
          </div>
          <div>
            <h1 className="page-title">Batch Prediction</h1>
            <p className="page-subtitle mb-0">Upload a CSV file with multiple patients</p>
          </div>
        </div>
      </div>

      {/* Upload Area */}
      <div className="glass-card p-6 mb-6 animate-fade-in">
        <div
          onDragEnter={handleDrag} onDragLeave={handleDrag} onDragOver={handleDrag} onDrop={handleDrop}
          onClick={() => inputRef.current?.click()}
          className={`border-2 border-dashed rounded-2xl p-10 text-center cursor-pointer transition-all duration-300
            ${dragActive ? 'border-medical-teal bg-medical-teal/5' : 'border-white/10 hover:border-white/20 hover:bg-white/[0.02]'}
            ${file ? 'border-emerald-500/30 bg-emerald-500/5' : ''}`}
        >
          <input ref={inputRef} type="file" accept=".csv" onChange={handleFileChange} className="hidden" />
          {file ? (
            <div className="flex flex-col items-center gap-3">
              <CheckCircle2 className="w-10 h-10 text-emerald-400" />
              <div>
                <p className="text-white font-medium">{file.name}</p>
                <p className="text-slate-500 text-sm">{(file.size / 1024).toFixed(1)} KB</p>
              </div>
              <button onClick={(e) => { e.stopPropagation(); setFile(null); setResult(null); }} className="text-xs text-slate-500 hover:text-rose-400 flex items-center gap-1 transition-colors">
                <X className="w-3 h-3" /> Remove
              </button>
            </div>
          ) : (
            <div className="flex flex-col items-center gap-3">
              <Upload className={`w-10 h-10 ${dragActive ? 'text-medical-teal' : 'text-slate-600'}`} />
              <div>
                <p className="text-white font-medium">Drop your CSV file here</p>
                <p className="text-slate-500 text-sm">or click to browse — 30 columns, no header</p>
              </div>
            </div>
          )}
        </div>

        <div className="flex gap-3 mt-4">
          <button onClick={handleSubmit} disabled={!file || loading} className="btn-primary flex items-center gap-2 disabled:opacity-40 disabled:cursor-not-allowed">
            <FileSpreadsheet className="w-4 h-4" />{loading ? 'Processing...' : 'Run Batch Prediction'}
          </button>
        </div>
      </div>

      {loading && <LoadingSpinner message="Processing CSV..." submessage="Analyzing all patient rows" />}
      {error && <div className="glass-card p-4 mb-6 bg-rose-500/5 border-rose-500/20 animate-fade-in"><div className="flex items-center gap-2 text-rose-400"><AlertTriangle className="w-5 h-5" /><p className="text-sm">{error}</p></div></div>}

      {result && !loading && (
        <div className="space-y-6 animate-slide-up">
          {/* Summary Cards */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="glass-card p-5 text-center"><p className="text-3xl font-bold text-white">{result.total}</p><p className="text-sm text-slate-400 mt-1">Total Patients</p></div>
            <div className="glass-card p-5 text-center border-rose-500/10"><p className="text-3xl font-bold text-rose-400">{result.malignant}</p><p className="text-sm text-slate-400 mt-1">Malignant</p></div>
            <div className="glass-card p-5 text-center border-emerald-500/10"><p className="text-3xl font-bold text-emerald-400">{result.benign}</p><p className="text-sm text-slate-400 mt-1">Benign</p></div>
          </div>

          {/* Results Table */}
          <div className="glass-card p-6">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold text-white">Results</h3>
              <button onClick={handleDownload} className="btn-secondary text-sm !px-4 !py-2 flex items-center gap-2"><Download className="w-4 h-4" />Download CSV</button>
            </div>
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="border-b border-white/5">
                    {['row','diagnosis','confidence'].map(f => (
                      <th key={f} onClick={() => handleSort(f)} className="text-left text-xs font-medium text-slate-500 uppercase tracking-wider pb-3 pr-4 cursor-pointer hover:text-slate-300 transition-colors">
                        {f === 'row' ? 'Row #' : f.charAt(0).toUpperCase() + f.slice(1)}
                        {sortField === f && <span className="ml-1">{sortAsc ? '↑' : '↓'}</span>}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody className="divide-y divide-white/5">
                  {sortedResults.map(r => (
                    <tr key={r.row} className="hover:bg-white/[0.02] transition-colors">
                      <td className="py-3 pr-4 text-sm text-slate-300">{r.row}</td>
                      <td className="py-3 pr-4"><DiagnosisBadge diagnosis={r.diagnosis} size="sm" /></td>
                      <td className="py-3 text-sm text-slate-300">{(r.confidence * 100).toFixed(2)}%</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
