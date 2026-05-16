import { ShieldAlert, ShieldCheck } from 'lucide-react';

export default function DiagnosisBadge({ diagnosis, size = 'md' }) {
  const isMalignant = diagnosis === 'Malignant';

  const sizes = {
    sm: 'px-2.5 py-0.5 text-xs',
    md: 'px-3 py-1 text-sm',
    lg: 'px-4 py-1.5 text-base',
  };

  return (
    <span className={`${isMalignant ? 'badge-malignant' : 'badge-benign'} ${sizes[size]} gap-1.5`}>
      {isMalignant
        ? <ShieldAlert className="w-3.5 h-3.5" />
        : <ShieldCheck className="w-3.5 h-3.5" />
      }
      {diagnosis}
    </span>
  );
}
