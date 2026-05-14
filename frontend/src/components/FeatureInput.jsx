import { Info } from 'lucide-react';
import { useState } from 'react';

const featureDescriptions = {
  radius_mean: 'Mean of distances from center to points on the perimeter',
  texture_mean: 'Standard deviation of gray-scale values',
  perimeter_mean: 'Mean size of the core tumor perimeter',
  area_mean: 'Mean area of the core tumor',
  smoothness_mean: 'Mean of local variation in radius lengths',
  compactness_mean: 'Mean of perimeter² / area − 1.0',
  concavity_mean: 'Mean severity of concave portions of the contour',
  concave_points_mean: 'Mean number of concave portions of the contour',
  symmetry_mean: 'Mean symmetry of the cell nucleus',
  fractal_dimension_mean: 'Mean "coastline approximation" − 1',
  radius_se: 'Standard error of radius',
  texture_se: 'Standard error of texture',
  perimeter_se: 'Standard error of perimeter',
  area_se: 'Standard error of area',
  smoothness_se: 'Standard error of smoothness',
  compactness_se: 'Standard error of compactness',
  concavity_se: 'Standard error of concavity',
  concave_points_se: 'Standard error of concave points',
  symmetry_se: 'Standard error of symmetry',
  fractal_dimension_se: 'Standard error of fractal dimension',
  radius_worst: 'Worst (largest) radius value',
  texture_worst: 'Worst (largest) texture value',
  perimeter_worst: 'Worst (largest) perimeter value',
  area_worst: 'Worst (largest) area value',
  smoothness_worst: 'Worst (largest) smoothness value',
  compactness_worst: 'Worst (largest) compactness value',
  concavity_worst: 'Worst (largest) concavity value',
  concave_points_worst: 'Worst (largest) concave points value',
  symmetry_worst: 'Worst (largest) symmetry value',
  fractal_dimension_worst: 'Worst (largest) fractal dimension value',
};

export default function FeatureInput({ name, value, onChange, index }) {
  const [showTooltip, setShowTooltip] = useState(false);
  const displayName = name
    .replace(/_/g, ' ')
    .replace(/\b\w/g, (c) => c.toUpperCase());

  return (
    <div className="relative group">
      <div className="flex items-center justify-between mb-1.5">
        <label
          htmlFor={`feature-${name}`}
          className="text-sm font-medium text-slate-300 flex items-center gap-1.5"
        >
          <span className="text-xs text-slate-600 font-mono w-5">{(index + 1).toString().padStart(2, '0')}</span>
          {displayName}
        </label>
        <div className="relative">
          <button
            type="button"
            className="text-slate-600 hover:text-medical-teal transition-colors"
            onMouseEnter={() => setShowTooltip(true)}
            onMouseLeave={() => setShowTooltip(false)}
            aria-label={`Info about ${displayName}`}
          >
            <Info className="w-3.5 h-3.5" />
          </button>
          {showTooltip && (
            <div className="absolute bottom-full right-0 mb-2 w-56 p-2.5 rounded-lg
              bg-navy-900 border border-white/10 text-xs text-slate-300 shadow-xl z-50
              animate-fade-in">
              {featureDescriptions[name] || 'No description available'}
              <div className="absolute -bottom-1 right-3 w-2 h-2 bg-navy-900 border-r border-b border-white/10 rotate-45" />
            </div>
          )}
        </div>
      </div>
      <input
        id={`feature-${name}`}
        type="number"
        step="any"
        value={value}
        onChange={(e) => onChange(name, e.target.value)}
        placeholder="0.00"
        className="input-field text-sm"
      />
    </div>
  );
}
