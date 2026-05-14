export default function StatCard({ icon: Icon, label, value, subtitle, gradient, delay = 0 }) {
  const gradients = {
    teal: 'from-medical-teal to-teal-400',
    rose: 'from-rose-500 to-pink-500',
    emerald: 'from-emerald-500 to-green-400',
    indigo: 'from-indigo-500 to-purple-500',
  };

  const glowColors = {
    teal: 'rgba(13, 148, 136, 0.2)',
    rose: 'rgba(225, 29, 72, 0.2)',
    emerald: 'rgba(16, 185, 129, 0.2)',
    indigo: 'rgba(99, 102, 241, 0.2)',
  };

  return (
    <div
      className="glass-card-hover p-6 relative overflow-hidden"
      style={{ animationDelay: `${delay}ms` }}
    >
      {/* Background glow */}
      <div
        className="absolute -top-10 -right-10 w-32 h-32 rounded-full blur-3xl opacity-50"
        style={{ background: glowColors[gradient] || glowColors.teal }}
      />

      <div className="relative flex items-start justify-between">
        <div>
          <p className="text-sm text-slate-400 font-medium mb-1">{label}</p>
          <p className="text-3xl font-bold text-white mb-1">{value}</p>
          {subtitle && <p className="text-sm text-slate-500">{subtitle}</p>}
        </div>
        <div className={`w-12 h-12 rounded-xl bg-gradient-to-br ${gradients[gradient] || gradients.teal}
          flex items-center justify-center shadow-lg`}
        >
          <Icon className="w-6 h-6 text-white" />
        </div>
      </div>
    </div>
  );
}
