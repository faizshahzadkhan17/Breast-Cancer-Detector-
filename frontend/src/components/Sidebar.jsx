import { NavLink } from 'react-router-dom';
import {
  LayoutDashboard,
  Search,
  FileSpreadsheet,
  History,
  ChevronLeft,
  ChevronRight,
  Activity,
} from 'lucide-react';

const navItems = [
  { to: '/', icon: LayoutDashboard, label: 'Dashboard' },
  { to: '/predict', icon: Search, label: 'Single Predict' },
  { to: '/batch', icon: FileSpreadsheet, label: 'Batch Predict' },
  { to: '/history', icon: History, label: 'History' },
];

export default function Sidebar({ collapsed, onToggle }) {
  return (
    <aside
      className={`fixed top-0 left-0 h-screen z-40 transition-all duration-300 ease-in-out
        ${collapsed ? 'w-20' : 'w-64'}
        bg-navy-900/80 backdrop-blur-xl border-r border-white/5 flex flex-col`}
    >
      {/* Logo */}
      <div className="flex items-center gap-3 px-5 py-6 border-b border-white/5">
        <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-medical-teal to-teal-400 flex items-center justify-center flex-shrink-0">
          <Activity className="w-5 h-5 text-white" />
        </div>
        {!collapsed && (
          <div className="animate-fade-in">
            <h1 className="text-sm font-bold text-white leading-tight">Breast Cancer</h1>
            <p className="text-xs text-medical-teal font-medium">Detector</p>
          </div>
        )}
      </div>

      {/* Navigation */}
      <nav className="flex-1 py-4 px-3 space-y-1">
        {navItems.map(({ to, icon: Icon, label }) => (
          <NavLink
            key={to}
            to={to}
            className={({ isActive }) =>
              `flex items-center gap-3 px-3 py-3 rounded-xl transition-all duration-200 group
              ${isActive
                ? 'bg-medical-teal/15 text-medical-teal-light border border-medical-teal/20'
                : 'text-slate-400 hover:text-white hover:bg-white/5 border border-transparent'
              }`
            }
          >
            <Icon className="w-5 h-5 flex-shrink-0" />
            {!collapsed && (
              <span className="text-sm font-medium animate-fade-in">{label}</span>
            )}
          </NavLink>
        ))}
      </nav>

      {/* Collapse Toggle */}
      <div className="p-3 border-t border-white/5">
        <button
          onClick={onToggle}
          className="w-full flex items-center justify-center gap-2 px-3 py-2.5 rounded-xl
            text-slate-500 hover:text-white hover:bg-white/5 transition-all duration-200"
          aria-label={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
        >
          {collapsed ? <ChevronRight className="w-5 h-5" /> : <ChevronLeft className="w-5 h-5" />}
          {!collapsed && <span className="text-sm">Collapse</span>}
        </button>
      </div>
    </aside>
  );
}
