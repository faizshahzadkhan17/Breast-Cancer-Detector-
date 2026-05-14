export default function ConfidenceRing({ confidence, size = 140, strokeWidth = 10 }) {
  const radius = (size - strokeWidth) / 2;
  const circumference = 2 * Math.PI * radius;
  const progress = confidence * circumference;
  const percentage = Math.round(confidence * 100);

  const isMalignant = confidence >= 0.5;
  const color = isMalignant ? '#e11d48' : '#10b981';
  const bgColor = isMalignant ? 'rgba(225, 29, 72, 0.1)' : 'rgba(16, 185, 129, 0.1)';
  const glowColor = isMalignant ? 'rgba(225, 29, 72, 0.3)' : 'rgba(16, 185, 129, 0.3)';

  return (
    <div className="relative inline-flex items-center justify-center">
      <svg width={size} height={size} className="-rotate-90">
        {/* Background circle */}
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="none"
          stroke="rgba(255,255,255,0.05)"
          strokeWidth={strokeWidth}
        />
        {/* Progress arc */}
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="none"
          stroke={color}
          strokeWidth={strokeWidth}
          strokeDasharray={circumference}
          strokeDashoffset={circumference - progress}
          strokeLinecap="round"
          className="transition-all duration-1000 ease-out"
          style={{
            filter: `drop-shadow(0 0 8px ${glowColor})`,
          }}
        />
      </svg>
      {/* Center text */}
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        <span className="text-3xl font-bold" style={{ color }}>
          {percentage}%
        </span>
        <span className="text-xs text-slate-400 mt-1">
          {isMalignant ? 'Malignancy' : 'Benign'}
        </span>
      </div>
    </div>
  );
}
