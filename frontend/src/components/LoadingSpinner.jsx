import { Loader2 } from 'lucide-react';

export default function LoadingSpinner({ message = 'Processing...', submessage = null }) {
  return (
    <div className="flex flex-col items-center justify-center py-12 animate-fade-in">
      <div className="relative">
        {/* Outer glow ring */}
        <div className="absolute inset-0 w-16 h-16 rounded-full bg-medical-teal/20 blur-xl animate-pulse-slow" />
        {/* Spinning icon */}
        <Loader2 className="w-16 h-16 text-medical-teal animate-spin" />
      </div>
      <p className="text-white font-medium mt-6">{message}</p>
      {submessage && (
        <p className="text-slate-500 text-sm mt-1">{submessage}</p>
      )}
    </div>
  );
}
