import { useState, useEffect } from 'react';
import { History as HistoryIcon, ChevronDown, ChevronUp, Filter, RefreshCw } from 'lucide-react';
import DiagnosisBadge from '../components/DiagnosisBadge';
import LoadingSpinner from '../components/LoadingSpinner';
import { getHistory } from '../api';

const FEATURES_PER_ROW = 5;

export default function HistoryPage() {
  const [records, setRecords] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [filter, setFilter] = useState('all');
  const [expandedRow, setExpandedRow] = useState(null);
  const [page, setPage] = useState(1);
  const perPage = 10;

  const fetchData = async () => {
    setLoading(true); setError(null);
    try { const data = await getHistory(); setRecords(data.records || []); }
    catch (err) { setError(err.message); }
    finally { setLoading(false); }
  };

  useEffect(() => { fetchData(); }, []);

  const filtered = records.filter(r => filter === 'all' || r.diagnosis === filter);
  const totalPages = Math.max(1, Math.ceil(filtered.length / perPage));
  const paginated = filtered.slice((page - 1) * perPage, page * perPage);

  return (
    <div className="page-container gradient-mesh min-h-screen">
      <div className="mb-8 animate-fade-in">
        <div className="flex items-center gap-3 mb-2">
          <div className="w-12 h-12 rounded-2xl bg-gradient-to-br from-amber-500 to-orange-500 flex items-center justify-center shadow-lg shadow-amber-500/20">
            <HistoryIcon className="w-6 h-6 text-white" />
          </div>
          <div>
            <h1 className="page-title">Prediction History</h1>
            <p className="page-subtitle mb-0">All saved prediction records</p>
          </div>
        </div>
      </div>

      {/* Toolbar */}
      <div className="glass-card p-4 mb-6 flex flex-wrap items-center justify-between gap-4 animate-fade-in">
        <div className="flex items-center gap-2">
          <Filter className="w-4 h-4 text-slate-500" />
          {['all', 'Malignant', 'Benign'].map(f => (
            <button key={f} onClick={() => { setFilter(f); setPage(1); }}
              className={`px-3 py-1.5 rounded-lg text-sm font-medium transition-all
                ${filter === f ? 'bg-medical-teal/15 text-medical-teal-light border border-medical-teal/20' : 'text-slate-400 hover:text-white hover:bg-white/5 border border-transparent'}`}>
              {f === 'all' ? 'All' : f}
            </button>
          ))}
          <span className="text-xs text-slate-600 ml-2">{filtered.length} records</span>
        </div>
        <button onClick={fetchData} className="btn-secondary text-sm !px-3 !py-1.5 flex items-center gap-1.5">
          <RefreshCw className="w-3.5 h-3.5" />Refresh
        </button>
      </div>

      {loading && <LoadingSpinner message="Loading history..." />}
      {error && (
        <div className="glass-card p-6 text-center animate-fade-in">
          <HistoryIcon className="w-12 h-12 text-slate-700 mx-auto mb-3" />
          <p className="text-slate-400 mb-1">Could not load history</p>
          <p className="text-slate-600 text-sm">{error}</p>
        </div>
      )}

      {!loading && !error && records.length === 0 && (
        <div className="glass-card p-12 text-center animate-fade-in">
          <HistoryIcon className="w-16 h-16 text-slate-700 mx-auto mb-4" />
          <p className="text-slate-400 text-lg">No prediction history</p>
          <p className="text-slate-600 text-sm mt-1">Make predictions to see them saved here (requires Supabase)</p>
        </div>
      )}

      {!loading && !error && filtered.length > 0 && (
        <div className="glass-card overflow-hidden animate-slide-up">
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="border-b border-white/5">
                  <th className="text-left text-xs font-medium text-slate-500 uppercase tracking-wider p-4">#</th>
                  <th className="text-left text-xs font-medium text-slate-500 uppercase tracking-wider p-4">Patient</th>
                  <th className="text-left text-xs font-medium text-slate-500 uppercase tracking-wider p-4">Diagnosis</th>
                  <th className="text-left text-xs font-medium text-slate-500 uppercase tracking-wider p-4">Confidence</th>
                  <th className="text-left text-xs font-medium text-slate-500 uppercase tracking-wider p-4">Date</th>
                  <th className="text-left text-xs font-medium text-slate-500 uppercase tracking-wider p-4">Details</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-white/5">
                {paginated.map((record, i) => {
                  const idx = (page - 1) * perPage + i + 1;
                  const isExpanded = expandedRow === record.id;
                  const features = record.input_features || {};
                  const featureEntries = Object.entries(features);
                  return (
                    <>
                      <tr key={record.id || idx} className="hover:bg-white/[0.02] transition-colors">
                        <td className="p-4 text-sm text-slate-500">{idx}</td>
                        <td className="p-4 text-sm text-slate-300">{record.patient_label || '—'}</td>
                        <td className="p-4"><DiagnosisBadge diagnosis={record.diagnosis} size="sm" /></td>
                        <td className="p-4 text-sm text-slate-300">{(record.confidence * 100).toFixed(2)}%</td>
                        <td className="p-4 text-sm text-slate-500">{new Date(record.created_at).toLocaleString()}</td>
                        <td className="p-4">
                          <button onClick={() => setExpandedRow(isExpanded ? null : record.id)}
                            className="text-slate-500 hover:text-medical-teal transition-colors">
                            {isExpanded ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
                          </button>
                        </td>
                      </tr>
                      {isExpanded && featureEntries.length > 0 && (
                        <tr key={`${record.id}-details`}>
                          <td colSpan={6} className="p-4 bg-navy-950/50">
                            <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-5 gap-3">
                              {featureEntries.map(([k, v]) => (
                                <div key={k} className="text-xs">
                                  <span className="text-slate-600">{k.replace(/_/g, ' ')}</span>
                                  <p className="text-slate-300 font-mono">{typeof v === 'number' ? v.toFixed(4) : v}</p>
                                </div>
                              ))}
                            </div>
                          </td>
                        </tr>
                      )}
                    </>
                  );
                })}
              </tbody>
            </table>
          </div>

          {/* Pagination */}
          {totalPages > 1 && (
            <div className="flex items-center justify-between p-4 border-t border-white/5">
              <p className="text-sm text-slate-500">Page {page} of {totalPages}</p>
              <div className="flex gap-2">
                <button onClick={() => setPage(p => Math.max(1, p - 1))} disabled={page === 1}
                  className="btn-secondary text-sm !px-3 !py-1.5 disabled:opacity-30">Previous</button>
                <button onClick={() => setPage(p => Math.min(totalPages, p + 1))} disabled={page === totalPages}
                  className="btn-secondary text-sm !px-3 !py-1.5 disabled:opacity-30">Next</button>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
