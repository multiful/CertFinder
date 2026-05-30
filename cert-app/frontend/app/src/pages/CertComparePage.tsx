import { useState, useEffect, useMemo } from 'react';
import {
  ChevronLeft,
  Scale,
  X,
  TrendingUp,
  Users,
  Award,
  BookOpen,
  Briefcase,
  ShieldCheck,
  Zap,
  Plus,
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Skeleton } from '@/components/ui/skeleton';
import { getCertificationDetail } from '@/lib/api';
import { useRouter } from '@/lib/router';
import { useCompare } from '@/contexts/CompareContext';
import type { QualificationDetail } from '@/types';

export function CertComparePage() {
  const { navigate } = useRouter();
  const { items, removeFromCompare, addToCompare, isInCompare } = useCompare();

  const ids = useMemo(() => {
    const params = new URLSearchParams(window.location.search);
    const raw = params.get('ids') || '';
    return raw
      .split(',')
      .map(s => parseInt(s.trim(), 10))
      .filter(n => !isNaN(n) && n > 0);
  }, []);

  const [certs, setCerts] = useState<(QualificationDetail | null)[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (ids.length === 0) { setLoading(false); return; }
    setLoading(true);
    Promise.all(ids.map(id => getCertificationDetail(id, null)))
      .then(results => setCerts(results))
      .finally(() => setLoading(false));
  }, [ids.join(',')]);

  const validCerts = useMemo(() => certs.filter(Boolean) as QualificationDetail[], [certs]);

  const avgPassRate = (cert: QualificationDetail) => {
    if (!cert.stats || cert.stats.length === 0) return null;
    const rates = cert.stats.map(s => s.pass_rate).filter(r => r != null) as number[];
    if (rates.length === 0) return null;
    return rates.reduce((a, b) => a + b, 0) / rates.length;
  };

  const getBest = (key: 'latest_pass_rate' | 'avg_difficulty', direction: 'max' | 'min') => {
    if (validCerts.length < 2) return new Set<number>();
    const values = validCerts.map(c => {
      if (key === 'avg_difficulty') return c.avg_difficulty;
      return c.latest_pass_rate;
    });
    const numVals = values.filter(v => v != null) as number[];
    if (numVals.length < 2) return new Set<number>();
    const best = direction === 'max' ? Math.max(...numVals) : Math.min(...numVals);
    return new Set(
      validCerts
        .filter((c, i) => values[i] === best)
        .map(c => c.qual_id)
    );
  };

  const getBestAvg = () => {
    if (validCerts.length < 2) return new Set<number>();
    const values = validCerts.map(avgPassRate);
    const numVals = values.filter(v => v != null) as number[];
    if (numVals.length < 2) return new Set<number>();
    const best = Math.max(...numVals);
    return new Set(
      validCerts
        .filter((_, i) => values[i] != null && Math.abs((values[i] as number) - best) < 0.01)
        .map(c => c.qual_id)
    );
  };

  const bestPassRate = useMemo(() => getBest('latest_pass_rate', 'max'), [validCerts]);
  const bestDifficulty = useMemo(() => getBest('avg_difficulty', 'min'), [validCerts]);
  const bestAvgPassRate = useMemo(() => getBestAvg(), [validCerts]);

  const cellClass = (isBest: boolean) =>
    isBest
      ? 'bg-blue-500/10 rounded-lg px-3 py-1.5 inline-block text-blue-300 font-bold'
      : '';

  const passRateColor = (rate: number | null) => {
    if (rate == null) return 'text-slate-500';
    if (rate > 70) return 'text-emerald-400';
    if (rate >= 30) return 'text-amber-400';
    return 'text-rose-500';
  };

  const difficultyColor = (diff: number | null) => {
    if (diff == null) return 'text-slate-500';
    if (diff >= 7.5) return 'text-rose-400';
    if (diff >= 5.0) return 'text-amber-400';
    return 'text-emerald-400';
  };

  // --- Empty state ---
  if (!loading && ids.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center min-h-[60vh] space-y-6 text-center">
        <div className="p-8 bg-slate-900 rounded-full">
          <Scale className="w-14 h-14 text-slate-700" />
        </div>
        <div className="space-y-2">
          <h1 className="text-2xl font-bold text-white">비교할 자격증을 선택하세요</h1>
          <p className="text-slate-500 max-w-sm">자격증 목록이나 상세 페이지에서 최대 3개를 선택하면 이 페이지에서 나란히 비교할 수 있습니다.</p>
        </div>
        <Button onClick={() => navigate('/certs')} className="bg-blue-600 hover:bg-blue-700 text-white rounded-xl">
          자격증 탐색하러 가기
        </Button>
      </div>
    );
  }

  // --- Loading ---
  if (loading) {
    return (
      <div className="space-y-8 pb-40">
        <Skeleton className="h-10 w-64 bg-slate-900 rounded-xl" />
        <div className="overflow-x-auto">
          <div className="grid gap-4 min-w-[560px]" style={{ gridTemplateColumns: `160px repeat(${ids.length}, 1fr)` }}>
            {Array.from({ length: (ids.length + 1) * 10 }).map((_, i) => (
              <Skeleton key={i} className="h-10 bg-slate-900 rounded-lg" />
            ))}
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-8 pb-40 max-w-5xl mx-auto">
      {/* Page header */}
      <div className="space-y-4">
        <Button
          variant="ghost"
          onClick={() => navigate('/certs')}
          className="text-slate-500 hover:text-white -ml-4 flex items-center gap-2"
        >
          <ChevronLeft className="w-4 h-4" /> 자격증 목록
        </Button>
        <div className="flex items-end justify-between gap-4 flex-wrap">
          <div className="space-y-2">
            <Badge variant="outline" className="border-blue-500/30 text-blue-400 px-3 py-1 flex items-center gap-1.5 w-fit">
              <Scale className="w-3 h-3" /> 자격증 비교
            </Badge>
            <h1 className="text-3xl font-bold text-white tracking-tight">
              {validCerts.length}개 자격증 비교 분석
            </h1>
            <p className="text-slate-500 text-sm">
              파란색 셀은 해당 항목에서 가장 유리한 값입니다.
            </p>
          </div>
          {validCerts.length < 3 && (
            <Button
              onClick={() => navigate('/certs')}
              variant="outline"
              className="border-slate-700 text-slate-400 hover:text-white rounded-xl flex items-center gap-2"
            >
              <Plus className="w-4 h-4" /> 자격증 추가
            </Button>
          )}
        </div>
      </div>

      {/* Mobile scroll hint */}
      <p className="sm:hidden text-[11px] text-slate-600 flex items-center gap-1.5 -mb-4">
        <span aria-hidden>←</span> 좌우로 스크롤하여 비교 <span aria-hidden>→</span>
      </p>

      {/* Comparison table */}
      <div className="overflow-x-auto rounded-2xl border border-slate-800" tabIndex={0} aria-label="자격증 비교 표 (좌우 스크롤 가능)">
        <table className="w-full border-collapse" style={{ minWidth: `${160 + validCerts.length * 220}px` }}>
          {/* Cert name header */}
          <thead>
            <tr className="border-b border-slate-800">
              <th
                className="sticky left-0 z-20 bg-slate-950 w-40 min-w-[160px] px-6 py-5 text-left text-xs font-bold text-slate-600 uppercase tracking-wider"
                scope="col"
              />
              {validCerts.map(cert => (
                <th
                  key={cert.qual_id}
                  className="px-6 py-5 text-left min-w-[200px] bg-slate-900/50"
                  scope="col"
                >
                  <div className="space-y-2">
                    <button
                      type="button"
                      onClick={() => navigate(`/certs/${cert.qual_id}`)}
                      className="text-base font-bold text-white hover:text-blue-400 transition-colors text-left leading-snug rounded focus-visible:outline-none focus-visible:ring-[3px] focus-visible:ring-blue-500/50"
                    >
                      {cert.qual_name}
                    </button>
                    <div className="flex items-center gap-2 flex-wrap">
                      <Badge className="bg-slate-800 text-slate-400 border-none text-[10px] px-1.5 py-0">
                        {cert.qual_type || '—'}
                      </Badge>
                      {cert.is_active ? (
                        <span className="text-[10px] text-emerald-500 font-bold">시행중</span>
                      ) : (
                        <span className="text-[10px] text-slate-500 font-bold">종료</span>
                      )}
                    </div>
                    <button
                      type="button"
                      onClick={() => removeFromCompare(cert.qual_id)}
                      className="flex items-center gap-1 text-[11px] text-slate-600 hover:text-slate-400 transition-colors font-medium rounded focus-visible:outline-none focus-visible:ring-[3px] focus-visible:ring-blue-500/50"
                      aria-label={`${cert.qual_name} 비교 제거`}
                    >
                      <X className="w-3 h-3" /> 제거
                    </button>
                  </div>
                </th>
              ))}
            </tr>
          </thead>

          <tbody>
            {/* ── 합격률 섹션 ── */}
            <SectionHeader label="합격률" />

            {/* 최근 합격률 */}
            <tr className="border-b border-slate-800/60 hover:bg-slate-900/20 transition-colors">
              <RowLabel icon={<Zap className="w-3.5 h-3.5 text-slate-500" />} label="최근 합격률" />
              {validCerts.map(cert => {
                const rate = cert.latest_pass_rate;
                const best = bestPassRate.has(cert.qual_id) && rate != null;
                return (
                  <td key={cert.qual_id} className="px-6 py-4">
                    {rate != null ? (
                      <div className={`space-y-1.5 ${best ? 'bg-blue-500/10 rounded-lg px-3 py-2 -mx-3' : ''}`}>
                        <div className={`text-base font-black ${passRateColor(rate)}`}>
                          {rate}%
                        </div>
                        <div className="h-1.5 bg-slate-800 rounded-full overflow-hidden max-w-[80px]">
                          <div
                            className={`h-full rounded-full ${rate > 70 ? 'bg-emerald-500' : rate >= 30 ? 'bg-amber-500' : 'bg-rose-500'}`}
                            style={{ width: `${Math.min(rate, 100)}%` }}
                          />
                        </div>
                      </div>
                    ) : (
                      <span className="text-slate-600 text-sm">정보 없음</span>
                    )}
                  </td>
                );
              })}
            </tr>

            {/* 평균 합격률 */}
            <tr className="border-b border-slate-800/60 bg-slate-900/20 hover:bg-slate-900/30 transition-colors">
              <RowLabel icon={<TrendingUp className="w-3.5 h-3.5 text-slate-500" />} label="평균 합격률" />
              {validCerts.map(cert => {
                const avg = avgPassRate(cert);
                const best = bestAvgPassRate.has(cert.qual_id) && avg != null;
                return (
                  <td key={cert.qual_id} className="px-6 py-4">
                    {avg != null ? (
                      <span className={`text-sm font-bold ${passRateColor(avg)} ${best ? 'bg-blue-500/10 rounded-lg px-2 py-1 -mx-2' : ''}`}>
                        {avg.toFixed(1)}%
                      </span>
                    ) : (
                      <span className="text-slate-600 text-sm">정보 없음</span>
                    )}
                    {avg != null && (
                      <span className="text-[10px] text-slate-600 ml-1.5">({cert.stats?.length || 0}회차 평균)</span>
                    )}
                  </td>
                );
              })}
            </tr>

            {/* ── 난이도 섹션 ── */}
            <SectionHeader label="난이도" />

            {/* 평균 난이도 */}
            <tr className="border-b border-slate-800/60 hover:bg-slate-900/20 transition-colors">
              <RowLabel icon={<Award className="w-3.5 h-3.5 text-slate-500" />} label="평균 난이도" />
              {validCerts.map(cert => {
                const diff = cert.avg_difficulty;
                const best = bestDifficulty.has(cert.qual_id) && diff != null;
                return (
                  <td key={cert.qual_id} className="px-6 py-4">
                    {diff != null ? (
                      <div className={`inline-flex items-center gap-1.5 ${best ? 'bg-blue-500/10 rounded-lg px-2 py-1 -mx-2' : ''}`}>
                        <span className={`text-base font-black ${difficultyColor(diff)}`}>
                          {diff.toFixed(1)}
                        </span>
                        <span className="text-[11px] text-slate-600">/10</span>
                        {best && <span className="text-[10px] text-blue-400 font-bold">가장 쉬움</span>}
                      </div>
                    ) : (
                      <span className="text-slate-600 text-sm">정보 없음</span>
                    )}
                  </td>
                );
              })}
            </tr>

            {/* ── 응시 정보 ── */}
            <SectionHeader label="응시 정보" />

            {/* 누적 응시자 */}
            <tr className="border-b border-slate-800/60 bg-slate-900/20 hover:bg-slate-900/30 transition-colors">
              <RowLabel icon={<Users className="w-3.5 h-3.5 text-slate-500" />} label="누적 응시자" />
              {validCerts.map(cert => (
                <td key={cert.qual_id} className="px-6 py-4">
                  <span className="text-sm font-bold text-slate-300 tabular-nums">
                    {cert.total_candidates ? cert.total_candidates.toLocaleString('ko-KR') + '명' : '정보 없음'}
                  </span>
                </td>
              ))}
            </tr>

            {/* 시험 구성 */}
            <tr className="border-b border-slate-800/60 hover:bg-slate-900/20 transition-colors">
              <RowLabel icon={<BookOpen className="w-3.5 h-3.5 text-slate-500" />} label="시험 구성" />
              {validCerts.map(cert => {
                const parts: string[] = [];
                if ((cert.written_cnt || 0) > 0) parts.push(`필기 ${cert.written_cnt}과목`);
                if ((cert.practical_cnt || 0) > 0) parts.push(`실기 ${cert.practical_cnt}과목`);
                if ((cert.interview_cnt || 0) > 0) parts.push(`면접 ${cert.interview_cnt}과목`);
                return (
                  <td key={cert.qual_id} className="px-6 py-4">
                    {parts.length > 0 ? (
                      <div className="space-y-0.5">
                        {parts.map(p => (
                          <div key={p} className="text-sm text-slate-300">{p}</div>
                        ))}
                      </div>
                    ) : (
                      <span className="text-slate-600 text-sm">정보 없음</span>
                    )}
                  </td>
                );
              })}
            </tr>

            {/* ── 자격 정보 ── */}
            <SectionHeader label="자격 분류" />

            {/* 자격 등급 */}
            <tr className="border-b border-slate-800/60 bg-slate-900/20 hover:bg-slate-900/30 transition-colors">
              <RowLabel icon={<ShieldCheck className="w-3.5 h-3.5 text-slate-500" />} label="자격 등급" />
              {validCerts.map(cert => (
                <td key={cert.qual_id} className="px-6 py-4">
                  <span className="text-sm text-slate-300 font-medium">
                    {cert.grade_code || '등급 없음'}
                  </span>
                </td>
              ))}
            </tr>

            {/* 직무 분류 */}
            <tr className="border-b border-slate-800/60 hover:bg-slate-900/20 transition-colors">
              <RowLabel icon={<Award className="w-3.5 h-3.5 text-slate-500" />} label="직무 분류 (NCS)" title="국가직무능력표준(NCS) 기준 직무 분류 체계" />
              {validCerts.map(cert => (
                <td key={cert.qual_id} className="px-6 py-4">
                  <span className="text-sm text-slate-300">
                    {[cert.ncs_large, cert.main_field].filter(Boolean).join(' › ') || '정보 없음'}
                  </span>
                </td>
              ))}
            </tr>

            {/* 관리 기관 */}
            <tr className="hover:bg-slate-900/20 transition-colors">
              <RowLabel icon={<ShieldCheck className="w-3.5 h-3.5 text-slate-500" />} label="관리 기관" />
              {validCerts.map(cert => (
                <td key={cert.qual_id} className="px-6 py-4">
                  <span className="text-sm text-slate-400 leading-snug">{cert.managing_body || '정보 없음'}</span>
                </td>
              ))}
            </tr>

            {/* 관련 직무 */}
            <SectionHeader label="커리어" />
            <tr className="border-t border-slate-800/60 hover:bg-slate-900/20 transition-colors">
              <RowLabel icon={<Briefcase className="w-3.5 h-3.5 text-slate-500" />} label="관련 직무" />
              {validCerts.map(cert => {
                const count = cert.jobs?.length || 0;
                return (
                  <td key={cert.qual_id} className="px-6 py-4">
                    {count > 0 ? (
                      <button
                        type="button"
                        onClick={() => navigate(`/certs/${cert.qual_id}#jobs`)}
                        className="text-sm font-bold text-blue-400 hover:text-blue-300 transition-colors rounded focus-visible:outline-none focus-visible:ring-[3px] focus-visible:ring-blue-500/50"
                      >
                        {count}개 직무 보기
                      </button>
                    ) : (
                      <span className="text-slate-600 text-sm">데이터 없음</span>
                    )}
                  </td>
                );
              })}
            </tr>
          </tbody>
        </table>
      </div>

      {/* Quick-add to compare if fewer than 3 */}
      {validCerts.length < 3 && (
        <div className="flex items-center gap-4 pt-4 border-t border-slate-800/50">
          <p className="text-sm text-slate-500">
            최대 3개까지 비교할 수 있습니다. 지금 {validCerts.length}개를 비교 중입니다.
          </p>
          <Button
            onClick={() => navigate('/certs')}
            variant="outline"
            className="border-slate-700 text-slate-400 hover:text-white rounded-xl text-sm"
          >
            <Plus className="w-4 h-4 mr-1.5" /> 추가 선택
          </Button>
        </div>
      )}
    </div>
  );
}

function SectionHeader({ label }: { label: string }) {
  return (
    <tr>
      <td
        colSpan={100}
        className="sticky left-0 px-6 py-2 text-[10px] font-black text-slate-600 uppercase tracking-[0.08em] bg-slate-950 border-b border-slate-800/60"
      >
        {label}
      </td>
    </tr>
  );
}

function RowLabel({
  icon,
  label,
  title,
}: {
  icon: React.ReactNode;
  label: string;
  title?: string;
}) {
  return (
    <td
      className="sticky left-0 z-10 bg-slate-950 px-6 py-4 text-xs font-bold text-slate-500 whitespace-nowrap border-r border-slate-800/40"
      title={title}
    >
      <div className="flex items-center gap-2">
        {icon}
        {label}
      </div>
    </td>
  );
}
