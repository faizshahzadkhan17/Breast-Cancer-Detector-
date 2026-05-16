import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Activity,
  Users,
  ShieldAlert,
  ShieldCheck,
  ArrowRight,
  Zap,
  TrendingUp,
} from 'lucide-react';
import StatCard from '../components/StatCard';
import DiagnosisBadge from '../components/DiagnosisBadge';
import { getHistory, healthCheck } from '../api';

export default function Dashboard() {
  const navigate = useNavigate();
  const [stats, setStats] = useState({ total: 0, malignant: 0, benign: 0 });
  const [recentPredictions, setRecentPredictions] = useState([]);
  const [serverStatus, setServerStatus] = useState('checking');

  useEffect(() => {
    // Check server health
    healthCheck()
      .then(() => setServerStatus('online'))
      .catch(() => setServerStatus('offline'));

    // Fetch history for stats
    getHistory()
      .then((data) => {
        const records = data.records || [];
        const malignant = records.filter((r) => r.diagnosis === 'Malignant').length;
        setStats({
          total: records.length,
          malignant,
          benign: records.length - malignant,
        });
        setRecentPredictions(records.slice(0, 5));
      })
      .catch(() => {
        // Supabase not configured — show zeros
      });
  }, []);

  return (
    <div className="page-container gradient-mesh min-h-screen">
      {/* Header */}
      <div className="mb-10 animate-fade-in">
        <div className="flex items-center gap-3 mb-4">
          <div className="w-12 h-12 rounded-2xl bg-gradient-to-br from-medical-teal to-teal-400 flex items-center justify-center shadow-lg shadow-medical-teal/20">
            <Activity className="w-6 h-6 text-white" />
          </div>
          <div>
            <h1 className="page-title">Dashboard</h1>
            <p className="page-subtitle mb-0">Breast Cancer Detection Overview</p>
          </div>
        </div>

        {/* Server status */}
        <div className={`inline-flex items-center gap-2 px-3 py-1.5 rounded-full text-xs font-medium
          ${serverStatus === 'online'
            ? 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/20'
            : serverStatus === 'offline'
              ? 'bg-rose-500/10 text-rose-400 border border-rose-500/20'
              : 'bg-yellow-500/10 text-yellow-400 border border-yellow-500/20'
          }`}
        >
          <span className={`w-2 h-2 rounded-full ${
            serverStatus === 'online' ? 'bg-emerald-400 animate-pulse' :
            serverStatus === 'offline' ? 'bg-rose-400' : 'bg-yellow-400 animate-pulse'
          }`} />
          Server {serverStatus === 'checking' ? 'Connecting...' : serverStatus}
        </div>
      </div>

      {/* Stat Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-5 mb-10 stagger-children">
        <StatCard
          icon={Users}
          label="Total Predictions"
          value={stats.total}
          subtitle="All time"
          gradient="teal"
        />
        <StatCard
          icon={ShieldAlert}
          label="Malignant"
          value={stats.malignant}
          subtitle={stats.total ? `${Math.round((stats.malignant / stats.total) * 100)}% of total` : '—'}
          gradient="rose"
        />
        <StatCard
          icon={ShieldCheck}
          label="Benign"
          value={stats.benign}
          subtitle={stats.total ? `${Math.round((stats.benign / stats.total) * 100)}% of total` : '—'}
          gradient="emerald"
        />
        <StatCard
          icon={TrendingUp}
          label="Model Accuracy"
          value="97%"
          subtitle="Logistic Regression"
          gradient="indigo"
        />
      </div>

      {/* Quick Actions */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-5 mb-10">
        <button
          onClick={() => navigate('/predict')}
          className="glass-card-hover p-6 text-left group"
        >
          <div className="flex items-center justify-between">
            <div>
              <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-medical-teal to-teal-400 flex items-center justify-center mb-4">
                <Zap className="w-5 h-5 text-white" />
              </div>
              <h3 className="text-lg font-semibold text-white mb-1">Single Prediction</h3>
              <p className="text-sm text-slate-400">Enter 30 features for instant diagnosis</p>
            </div>
            <ArrowRight className="w-5 h-5 text-slate-600 group-hover:text-medical-teal group-hover:translate-x-1 transition-all" />
          </div>
        </button>

        <button
          onClick={() => navigate('/batch')}
          className="glass-card-hover p-6 text-left group"
        >
          <div className="flex items-center justify-between">
            <div>
              <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-indigo-500 to-purple-500 flex items-center justify-center mb-4">
                <Users className="w-5 h-5 text-white" />
              </div>
              <h3 className="text-lg font-semibold text-white mb-1">Batch Prediction</h3>
              <p className="text-sm text-slate-400">Upload CSV for multiple patients at once</p>
            </div>
            <ArrowRight className="w-5 h-5 text-slate-600 group-hover:text-indigo-400 group-hover:translate-x-1 transition-all" />
          </div>
        </button>
      </div>

      {/* Recent Predictions Table */}
      <div className="glass-card p-6 animate-slide-up">
        <div className="flex items-center justify-between mb-5">
          <h2 className="text-lg font-semibold text-white">Recent Predictions</h2>
          {recentPredictions.length > 0 && (
            <button
              onClick={() => navigate('/history')}
              className="text-sm text-medical-teal hover:text-medical-teal-light transition-colors flex items-center gap-1"
            >
              View all <ArrowRight className="w-4 h-4" />
            </button>
          )}
        </div>

        {recentPredictions.length === 0 ? (
          <div className="text-center py-12">
            <Activity className="w-12 h-12 text-slate-700 mx-auto mb-3" />
            <p className="text-slate-500">No predictions yet</p>
            <p className="text-slate-600 text-sm mt-1">
              {serverStatus === 'offline'
                ? 'Start the backend server to begin'
                : 'Make your first prediction to see results here'}
            </p>
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="border-b border-white/5">
                  <th className="text-left text-xs font-medium text-slate-500 uppercase tracking-wider pb-3 pr-4">Patient</th>
                  <th className="text-left text-xs font-medium text-slate-500 uppercase tracking-wider pb-3 pr-4">Diagnosis</th>
                  <th className="text-left text-xs font-medium text-slate-500 uppercase tracking-wider pb-3 pr-4">Confidence</th>
                  <th className="text-left text-xs font-medium text-slate-500 uppercase tracking-wider pb-3">Date</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-white/5">
                {recentPredictions.map((record, i) => (
                  <tr key={record.id || i} className="hover:bg-white/[0.02] transition-colors">
                    <td className="py-3 pr-4 text-sm text-slate-300">
                      {record.patient_label || `Patient ${i + 1}`}
                    </td>
                    <td className="py-3 pr-4">
                      <DiagnosisBadge diagnosis={record.diagnosis} size="sm" />
                    </td>
                    <td className="py-3 pr-4 text-sm text-slate-300">
                      {(record.confidence * 100).toFixed(1)}%
                    </td>
                    <td className="py-3 text-sm text-slate-500">
                      {new Date(record.created_at).toLocaleDateString()}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}
