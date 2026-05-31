import { X, Scale } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { useCompare } from '@/contexts/CompareContext';
import { useRouter } from '@/lib/router';

export function CompareTray() {
  const { items, removeFromCompare, clearCompare } = useCompare();
  const { navigate } = useRouter();

  if (items.length === 0) return null;

  return (
    <div className="pointer-events-auto w-full px-4 pb-2 pt-0">
      <div
        className="max-w-2xl mx-auto bg-slate-900/98 border border-slate-700 rounded-2xl px-4 py-3 shadow-2xl animate-in slide-in-from-bottom-2 duration-300"
        role="region"
        aria-label="자격증 비교 트레이"
      >
        <div className="flex items-center gap-3 flex-wrap sm:flex-nowrap">
          {/* Icon + Cert chips */}
          <div className="flex items-center gap-2 flex-1 min-w-0 flex-wrap">
            <Scale className="w-4 h-4 text-blue-400 shrink-0" aria-hidden />
            {items.map(item => (
              <div
                key={item.id}
                className="flex items-center gap-1 pl-2.5 pr-1.5 py-1 bg-slate-800 border border-slate-700 rounded-lg text-xs text-slate-200 font-medium"
              >
                <span className="truncate max-w-[130px] sm:max-w-[160px]">{item.name}</span>
                <button
                  type="button"
                  onClick={() => removeFromCompare(item.id)}
                  aria-label={`${item.name} 비교 제거`}
                  className="text-slate-500 hover:text-slate-200 transition-colors ml-0.5 p-2 rounded focus-ring"
                >
                  <X className="w-3 h-3" />
                </button>
              </div>
            ))}
            {items.length < 3 && (
              <span className="text-[11px] text-slate-600 font-medium shrink-0">
                {3 - items.length}개 더 추가 가능
              </span>
            )}
          </div>

          {/* Actions */}
          <div className="flex items-center gap-3 shrink-0">
            <button
              type="button"
              onClick={clearCompare}
              className="text-[11px] text-slate-600 hover:text-slate-400 transition-colors font-medium rounded focus-ring px-1 py-0.5"
            >
              초기화
            </button>
            <Button
              onClick={() => {
                const ids = items.map(i => i.id).join(',');
                navigate(`/certs/compare?ids=${ids}`);
              }}
              disabled={items.length < 2}
              className="h-8 px-4 bg-blue-600 hover:bg-blue-700 text-white text-xs font-bold rounded-xl disabled:opacity-40 transition-colors"
            >
              비교하기
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
}
