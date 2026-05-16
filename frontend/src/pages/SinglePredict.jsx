import { useState } from 'react';
import { Search, RotateCcw, Sparkles } from 'lucide-react';
import FeatureInput from '../components/FeatureInput';
import ConfidenceRing from '../components/ConfidenceRing';
import DiagnosisBadge from '../components/DiagnosisBadge';
import LoadingSpinner from '../components/LoadingSpinner';
import { predictSingle } from '../api';

const FEATURE_NAMES = [
  'radius_mean','texture_mean','perimeter_mean','area_mean','smoothness_mean',
  'compactness_mean','concavity_mean','concave_points_mean','symmetry_mean','fractal_dimension_mean',
  'radius_se','texture_se','perimeter_se','area_se','smoothness_se',
  'compactness_se','concavity_se','concave_points_se','symmetry_se','fractal_dimension_se',
  'radius_worst','texture_worst','perimeter_worst','area_worst','smoothness_worst',
  'compactness_worst','concavity_worst','concave_points_worst','symmetry_worst','fractal_dimension_worst',
];

const GROUPS = {
  'Mean Values': FEATURE_NAMES.slice(0, 10),
  'Standard Error': FEATURE_NAMES.slice(10, 20),
  'Worst Values': FEATURE_NAMES.slice(20, 30),
};

const SAMPLES = {
  'Malignant #1': [17.99,10.38,122.8,1001,0.1184,0.2776,0.3001,0.1471,0.2419,0.07871,1.095,0.9053,8.589,153.4,0.006399,0.04904,0.05373,0.01587,0.03003,0.006193,25.38,17.33,184.6,2019,0.1622,0.6656,0.7119,0.2654,0.4601,0.1189],
  'Benign #1': [13.08,15.71,85.63,520,0.1075,0.127,0.04568,0.0311,0.1967,0.06811,0.1852,0.7477,1.383,14.67,0.004097,0.01898,0.01698,0.00649,0.01678,0.002425,14.5,20.49,96.09,630.5,0.1312,0.2776,0.189,0.07283,0.3184,0.08183],
};

export default function SinglePredict() {
  const [features, setFeatures] = useState(Object.fromEntries(FEATURE_NAMES.map(f => [f, ''])));
  const [patientLabel, setPatientLabel] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const handleChange = (name, value) => setFeatures(prev => ({ ...prev, [name]: value }));

  const handleReset = () => { setFeatures(Object.fromEntries(FEATURE_NAMES.map(f => [f, '']))); setPatientLabel(''); setResult(null); setError(null); };

  const handleLoadSample = (key) => {
    const vals = SAMPLES[key];
    const nf = {}; FEATURE_NAMES.forEach((n, i) => { nf[n] = vals[i].toString(); });
    setFeatures(nf); setPatientLabel(key); setResult(null); setError(null);
  };

  const handleSubmit = async (e) => {
    e.preventDefault(); setError(null); setResult(null);
    const values = FEATURE_NAMES.map(f => parseFloat(features[f]));
    if (values.some(isNaN)) { setError('Please fill in all 30 features with valid numbers.'); return; }
    setLoading(true);
    try { const res = await predictSingle(values, patientLabel || null); setResult(res); }
    catch (err) { setError(err.message); }
    finally { setLoading(false); }
  };

  return (
    <div className="page-container gradient-mesh min-h-screen">
      <div className="mb-8 animate-fade-in">
        <div className="flex items-center gap-3 mb-2">
          <div className="w-12 h-12 rounded-2xl bg-gradient-to-br from-medical-teal to-teal-400 flex items-center justify-center shadow-lg shadow-medical-teal/20">
            <Search className="w-6 h-6 text-white" />
          </div>
          <div>
            <h1 className="page-title">Single Prediction</h1>
            <p className="page-subtitle mb-0">Enter 30 cell nucleus measurements</p>
          </div>
        </div>
      </div>

      <div className="glass-card p-4 mb-6 animate-fade-in">
        <div className="flex flex-wrap items-center gap-3">
          <span className="text-sm text-slate-400 flex items-center gap-1.5">
            <Sparkles className="w-4 h-4 text-medical-teal" /> Quick load:
          </span>
          {Object.keys(SAMPLES).map(name => (
            <button key={name} type="button" onClick={() => handleLoadSample(name)} className="btn-secondary text-xs !px-3 !py-1.5">{name}</button>
          ))}
        </div>
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
        <form onSubmit={handleSubmit} className="xl:col-span-2 space-y-6">
          <div className="glass-card p-5">
            <label htmlFor="patient-label" className="text-sm font-medium text-slate-300 mb-2 block">Patient Label (optional)</label>
            <input id="patient-label" type="text" value={patientLabel} onChange={e => setPatientLabel(e.target.value)} placeholder="e.g. Patient #1234" className="input-field" />
          </div>

          {Object.entries(GROUPS).map(([groupName, groupFeatures]) => (
            <div key={groupName} className="glass-card p-5">
              <h3 className="text-base font-semibold text-white mb-4 flex items-center gap-2">
                <span className="w-1.5 h-1.5 rounded-full bg-medical-teal" />{groupName}
              </h3>
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
                {groupFeatures.map(name => (
                  <FeatureInput key={name} name={name} value={features[name]} onChange={handleChange} index={FEATURE_NAMES.indexOf(name)} />
                ))}
              </div>
            </div>
          ))}

          <div className="flex gap-3">
            <button type="submit" disabled={loading} className="btn-primary flex items-center gap-2">
              <Search className="w-4 h-4" />{loading ? 'Predicting...' : 'Predict Diagnosis'}
            </button>
            <button type="button" onClick={handleReset} className="btn-secondary flex items-center gap-2">
              <RotateCcw className="w-4 h-4" />Reset
            </button>
          </div>
        </form>

        <div className="xl:col-span-1">
          <div className="glass-card p-6 sticky top-6">
            <h3 className="text-lg font-semibold text-white mb-5">Prediction Result</h3>
            {loading && <LoadingSpinner message="Analyzing features..." submessage="Running through ML model" />}
            {error && <div className="p-4 rounded-xl bg-rose-500/10 border border-rose-500/20 animate-fade-in"><p className="text-rose-400 text-sm">{error}</p></div>}
            {result && !loading && (
              <div className="flex flex-col items-center text-center animate-slide-up">
                <ConfidenceRing confidence={result.confidence} size={160} />
                <div className="mt-5"><DiagnosisBadge diagnosis={result.diagnosis} size="lg" /></div>
                <div className="mt-4 space-y-1">
                  <p className="text-sm text-slate-400">Confidence: <span className="text-white font-semibold">{(result.confidence * 100).toFixed(2)}%</span></p>
                  {result.patient_label && <p className="text-sm text-slate-500">{result.patient_label}</p>}
                </div>
              </div>
            )}
            {!result && !loading && !error && (
              <div className="text-center py-8">
                <Search className="w-12 h-12 text-slate-700 mx-auto mb-3" />
                <p className="text-slate-500 text-sm">Fill in features and click Predict</p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
