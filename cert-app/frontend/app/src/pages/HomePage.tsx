import { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import {
  Search,
  TrendingUp,
  ArrowRight,
  CheckCircle2,
  ChevronRight,
  Sparkles,
  AlertCircle
} from 'lucide-react';
import { Card, CardContent } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { useRouter } from '@/lib/router';
import { getTrendingCerts, getCertificationsCatalogTotal, FALLBACK_CERT_CATALOG_TOTAL } from '@/lib/api';
import type { TrendingQualification } from '@/types';
import { toast } from 'sonner';

export function HomePage() {
  const router = useRouter();
  const [search, setSearch] = useState('');
  const [trendingCerts, setTrendingCerts] = useState<TrendingQualification[]>([]);
  const [loading, setLoading] = useState(false);
  const [hasRequestedTrending, setHasRequestedTrending] = useState(false);
  const [trendingError, setTrendingError] = useState(false);
  const [certCatalogTotal, setCertCatalogTotal] = useState(FALLBACK_CERT_CATALOG_TOTAL);
  const trendingSectionRef = useRef<HTMLElement>(null);
  const fetchStartedRef = useRef(false);

  useEffect(() => {
    getCertificationsCatalogTotal().then(setCertCatalogTotal).catch(() => {});
  }, []);

  const fetchTrendingData = useCallback(async () => {
    if (fetchStartedRef.current && !trendingError) return;
    if (trendingError) fetchStartedRef.current = false;
    fetchStartedRef.current = true;
    setHasRequestedTrending(true);
    setTrendingError(false);
    setLoading(true);
    try {
      const res = await getTrendingCerts(6);
      setTrendingCerts(res.items ?? []);
    } catch (error) {
      console.error('Failed to fetch trending data:', error);
      setTrendingError(true);
      toast.error('인기 자격증을 불러오지 못했습니다.');
    } finally {
      setLoading(false);
    }
  }, [trendingError]);

  useEffect(() => {
    const el = trendingSectionRef.current;
    if (!el) return;
    const observer = new IntersectionObserver(
      (entries) => {
        if (entries[0]?.isIntersecting) fetchTrendingData();
      },
      { rootMargin: '120px', threshold: 0 }
    );
    observer.observe(el);
    return () => observer.disconnect();
  }, [fetchTrendingData]);

  /** API score는 조회·검색 가중치 누적값(또는 응시자 수 fallback)이라 %가 아님 → 목록 내 1위 대비 상대 지표로만 표시 */
  const trendingRelativePctById = useMemo(() => {
    const valid = trendingCerts.map((c) => (Number.isFinite(c.score) && c.score >= 0 ? c.score : 0));
    const max = valid.length ? Math.max(...valid) : 0;
    const map = new Map<number, number>();
    trendingCerts.forEach((c, i) => {
      const s = valid[i] ?? 0;
      map.set(c.qual_id, max > 0 ? Math.round((s / max) * 100) : 0);
    });
    return map;
  }, [trendingCerts]);

  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault();
    if (search.trim()) {
      router.navigate(`/certs?q=${encodeURIComponent(search)}`);
    } else {
      router.navigate('/certs');
    }
  };

  return (
    <div className="flex flex-col gap-20 pb-40 overflow-hidden">
      {/* Hero Section */}
      <section className="relative min-h-[85vh] flex items-center pt-20">
        {/* Background Decorative Elements */}
        <div
          className="absolute top-0 left-1/2 -translate-x-1/2 w-full h-full -z-10 pointer-events-none"
          style={{
            background: 'radial-gradient(ellipse 55% 45% at 28% 32%, oklch(0.5 0.09 248 / 0.07) 0%, transparent 70%), radial-gradient(ellipse 50% 40% at 72% 68%, oklch(0.5 0.07 275 / 0.06) 0%, transparent 70%)'
          }}
        />

        <div className="grid lg:grid-cols-2 gap-16 items-center">
          <div className="space-y-8 text-center lg:text-left">
            <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-blue-500/10 border border-blue-500/20 text-blue-400 text-sm font-medium animate-fade-in">
              <Sparkles className="w-4 h-4" />
              <span>국가자격 데이터, 하이브리드 AI 추천</span>
            </div>

            <h1 className="font-extrabold text-white leading-[1.1] [letter-spacing:-0.03em] [font-size:clamp(3rem,7vw,4.5rem)]">
              데이터로 설계하는<br />
              <span className="text-blue-400">
                당신의 커리어 경로
              </span>
            </h1>

            <p className="text-slate-400 text-lg lg:text-xl max-w-xl mx-auto lg:mx-0 leading-relaxed font-medium">
              합격률, 난이도, 직무 매칭은 DB 통계로, 전공 맞춤 추천은{' '}
              <span className="text-white font-semibold">AI 분석 엔진</span>
              으로 제공합니다.
            </p>

            <div className="flex flex-col gap-3 justify-center lg:justify-start pt-4">
              <form onSubmit={handleSearch} className="relative group max-w-md w-full">
                <div className="absolute -inset-1 bg-gradient-to-r from-blue-600 to-indigo-600 rounded-2xl blur opacity-25 group-hover:opacity-50 transition duration-1000 group-hover:duration-200" />
                <div className="relative">
                  <label htmlFor="home-cert-search" className="sr-only">자격증 검색</label>
                  <Search className="absolute left-4 top-1/2 -translate-y-1/2 text-slate-500 w-5 h-5" />
                  <input
                    id="home-cert-search"
                    name="q"
                    type="search"
                    enterKeyHint="search"
                    placeholder="관심 있는 자격증을 검색하세요..."
                    value={search}
                    onChange={(e) => setSearch(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key !== 'Enter') return;
                      e.preventDefault();
                      handleSearch(e as unknown as React.FormEvent);
                    }}
                    className="w-full h-14 pl-12 pr-4 bg-slate-900 border border-slate-800 rounded-xl text-white placeholder:text-slate-500 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-[colors,box-shadow]"
                  />
                </div>
              </form>
              <div className="flex items-center gap-6 pl-1">
                <button
                  type="button"
                  onClick={() => router.navigate('/recommendation')}
                  className="group text-left"
                >
                  <span className="flex items-center gap-1 text-sm text-slate-400 group-hover:text-blue-400 font-medium transition-colors">
                    전공 추천 <ChevronRight className="w-3.5 h-3.5" />
                  </span>
                  <span className="block text-[11px] text-slate-600 group-hover:text-slate-500 transition-colors leading-tight mt-0.5">학과별 자격증 DB 매핑</span>
                </button>
                <button
                  type="button"
                  onClick={() => router.navigate('/ai-recommendations')}
                  className="group text-left"
                >
                  <span className="flex items-center gap-1 text-sm text-blue-400 group-hover:text-blue-300 font-semibold transition-colors">
                    AI 추천 <ChevronRight className="w-3.5 h-3.5" />
                  </span>
                  <span className="block text-[11px] text-slate-600 group-hover:text-slate-500 transition-colors leading-tight mt-0.5">커리어 목표 기반 로드맵</span>
                </button>
              </div>
            </div>

            <div className="flex items-center justify-center lg:justify-start gap-8 pt-8 text-slate-400 text-sm font-medium">
              <div className="flex items-center gap-2">
                <CheckCircle2 className="w-4 h-4 text-emerald-500" /> {certCatalogTotal.toLocaleString('ko-KR')}개 자격증 데이터
              </div>
              <div className="flex items-center gap-2">
                <CheckCircle2 className="w-4 h-4 text-emerald-500" /> AI 분석 기반 맞춤 추천
              </div>
            </div>
          </div>

          <div className="relative hidden lg:block">
            <div className="relative z-10 bg-slate-950 rounded-2xl border border-slate-800/60 overflow-hidden">
              {/* DB 규모 헤더 */}
              <div className="px-8 pt-8 pb-6 border-b border-slate-800/40">
                <p className="text-[10px] font-bold text-slate-600 uppercase tracking-[0.12em] mb-2">총 분석 종목</p>
                <p className="text-5xl font-black text-white tabular-nums tracking-tight leading-none">
                  {certCatalogTotal.toLocaleString('ko-KR')}
                </p>
                <p className="text-sm text-slate-500 font-medium mt-2">개 국가기술자격</p>
              </div>

              {/* 샘플 자격증 행 — 실제 데이터 형태 미리보기 */}
              {([
                { name: '정보처리기사', type: '기사', passRate: 26.2, diff: 7.4 },
                { name: '한국사능력검정 1급', type: '국가공인', passRate: 81.4, diff: 3.1 },
                { name: '빅데이터분석기사', type: '기사', passRate: 34.7, diff: 6.2 },
              ] as const).map((cert, i, arr) => (
                <div key={cert.name} className={`flex items-center gap-4 px-8 py-4 ${i < arr.length - 1 ? 'border-b border-slate-800/40' : ''}`}>
                  <div className="flex-1 min-w-0">
                    <p className="text-sm font-bold text-white truncate">{cert.name}</p>
                    <p className="text-[11px] text-slate-600 mt-0.5 font-medium">{cert.type}</p>
                  </div>
                  <div className="shrink-0 text-right">
                    <p className={`text-sm font-bold tabular-nums ${cert.passRate > 70 ? 'text-emerald-400' : cert.passRate >= 30 ? 'text-amber-400' : 'text-rose-400'}`}>
                      합격 {cert.passRate}%
                    </p>
                    <p className={`text-[11px] font-medium tabular-nums ${cert.diff >= 7 ? 'text-rose-400' : cert.diff >= 5 ? 'text-amber-400' : 'text-emerald-400'}`}>
                      난이도 {cert.diff}/10
                    </p>
                  </div>
                </div>
              ))}

              {/* 푸터 */}
              <div className="px-8 py-3 bg-slate-900/30 border-t border-slate-800/40 flex items-center justify-between">
                <span className="text-[11px] text-slate-600">450+ 직무 · 10년+ 합격률 기록</span>
                <span className="text-[11px] text-slate-500 font-medium">실제 데이터 예시</span>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Features Section — numbered spec-sheet layout */}
      <section>
        <div className="flex flex-col sm:flex-row sm:items-end justify-between gap-6 mb-10">
          <div className="space-y-3">
            <Badge variant="outline" className="border-blue-500/30 text-blue-400 px-4 py-1">핵심 기능</Badge>
            <h2 className="text-3xl md:text-4xl font-bold text-white tracking-tight">데이터 기반 경로 설계 3단계</h2>
          </div>
          <p className="text-sm text-slate-500 font-medium max-w-xs leading-relaxed">국가자격 데이터와 AI 분석 엔진이 결합된 세 가지 핵심 서비스입니다.</p>
        </div>

        <div className="divide-y divide-slate-800/50">
          {([
            {
              num: '01',
              title: '전공별 자격증 추천',
              tag: 'DB 매핑',
              desc: '학과–자격증 데이터베이스 매핑과 합격률 통계를 바탕으로 전공별 최적 자격증 목록을 제공합니다.',
              link: '/recommendations',
              stat: '전공 DB',
              statVal: '연동됨',
            },
            {
              num: '02',
              title: 'AI 커리어 로드맵',
              tag: '하이브리드 AI',
              desc: '커리어 목표와 현재 역량을 입력하면 복합 AI 검색 엔진이 최적 취득 순서와 이유를 제시합니다.',
              link: '/ai-recommendations',
              stat: 'AI 엔진',
              statVal: 'RAG 기반',
            },
            {
              num: '03',
              title: '직무·진로 매칭',
              tag: '연봉·전망',
              desc: '자격증 취득 후 진입 가능한 직업의 채용 수요와 연봉 분포 데이터를 직무 단위로 분석합니다.',
              link: '/jobs',
              stat: '직무 분류',
              statVal: '450개+',
            },
          ] as const).map((f) => (
            <button
              key={f.num}
              type="button"
              onClick={() => router.navigate(f.link)}
              className="w-full flex items-start gap-6 sm:gap-10 py-8 group text-left hover:bg-slate-900/30 transition-colors duration-200 px-5 -mx-5 rounded-2xl focus-ring focus-visible:ring-offset-2 focus-visible:ring-offset-slate-950"
            >
              <span className="text-[10px] font-mono text-slate-700 font-bold tracking-[0.12em] pt-1.5 w-6 shrink-0 tabular-nums select-none">
                {f.num}
              </span>
              <div className="flex-1 grid sm:grid-cols-[1fr_auto] items-center gap-4 min-w-0">
                <div className="space-y-2 min-w-0">
                  <div className="flex items-center gap-3 flex-wrap">
                    <h3 className="text-xl font-bold text-white group-hover:text-blue-400 transition-colors duration-200 leading-tight">
                      {f.title}
                    </h3>
                    <span className="text-[10px] font-bold text-slate-600 border border-slate-800 group-hover:border-slate-700 px-2 py-0.5 rounded-full tracking-wide transition-colors duration-200">
                      {f.tag}
                    </span>
                  </div>
                  <p className="text-sm text-slate-500 max-w-xl leading-relaxed">{f.desc}</p>
                </div>
                <div className="flex items-center gap-5 shrink-0">
                  <div className="hidden sm:block text-right">
                    <p className="text-[9px] font-bold text-slate-700 uppercase tracking-[0.1em] mb-0.5">{f.stat}</p>
                    <p className="text-sm font-black text-slate-400 tabular-nums">{f.statVal}</p>
                  </div>
                  <ArrowRight className="w-5 h-5 text-slate-700 group-hover:text-blue-400 group-hover:translate-x-1 transition-[colors,transform] duration-200 shrink-0" />
                </div>
              </div>
            </button>
          ))}
        </div>
      </section>

      {/* Top Certs Section - 데이터는 이 섹션이 뷰포트에 들어올 때 로드 */}
      <section ref={trendingSectionRef} className="bg-slate-900/30 py-24 border-y border-slate-800/50">
        <div className="container mx-auto px-6">
          <div className="flex flex-col md:flex-row justify-between items-end gap-6 mb-12">
            <div className="space-y-4">
              <Badge variant="outline" className="border-blue-500/30 text-blue-400 px-4 py-1">최근 트렌드</Badge>
              <h2 className="text-3xl md:text-4xl font-bold text-white">최근 주목받는 자격증</h2>
              <p className="text-slate-400">실시간 데이터가 반영된 최신 인기 트렌드를 확인하세요.</p>
            </div>
            <Button
              variant="ghost"
              onClick={() => router.navigate('/certs')}
              className="text-blue-400 hover:text-blue-300 hover:bg-blue-400/5 flex items-center gap-2"
            >
              전체 보기 <ChevronRight className="w-4 h-4" />
            </Button>
          </div>

          {loading ? (
            <div className="divide-y divide-slate-800/40">
              {[1, 2, 3, 4, 5, 6].map(i => (
                <div key={i} className="flex items-center gap-5 py-5">
                  <div className="w-9 h-6 bg-slate-900 rounded animate-pulse shrink-0" />
                  <div className="flex-1 space-y-2">
                    <div className="h-4 bg-slate-900 rounded animate-pulse w-3/4" />
                    <div className="h-3 bg-slate-900 rounded animate-pulse w-1/3" />
                  </div>
                  <div className="w-10 h-4 bg-slate-900 rounded animate-pulse shrink-0" />
                </div>
              ))}
            </div>
          ) : trendingCerts.length > 0 ? (
            <div className="divide-y divide-slate-800/40" role="list">
              {trendingCerts.map((cert, index) => (
                <div
                  key={cert.qual_id}
                  role="listitem"
                  onClick={() => router.navigate(`/certs/${cert.qual_id}`)}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' || e.key === ' ') {
                      e.preventDefault();
                      router.navigate(`/certs/${cert.qual_id}`);
                    }
                  }}
                  tabIndex={0}
                  aria-label={cert.qual_name}
                  style={{ animationDelay: `${index * 40}ms` }}
                  className="group flex items-center gap-5 py-5 px-4 -mx-4 rounded-xl cursor-pointer hover:bg-slate-900/60 transition-colors duration-150 animate-in fade-in slide-in-from-bottom-2 duration-400 focus-visible:ring-[3px] focus-visible:ring-blue-500/50 focus-visible:outline-none"
                >
                  <span className="text-2xl font-black tabular-nums text-slate-800 group-hover:text-slate-700 w-9 shrink-0 leading-none select-none transition-colors">
                    {String(index + 1).padStart(2, '0')}
                  </span>
                  <div className="flex-1 min-w-0">
                    <h3 className="text-base font-bold text-white group-hover:text-blue-400 transition-colors truncate">
                      {cert.qual_name}
                    </h3>
                    <div className="flex items-center gap-2 mt-1">
                      <Badge className="bg-slate-800/80 text-slate-400 border-none text-[10px] px-1.5 py-0 h-4">{cert.qual_type}</Badge>
                      {cert.main_field && (
                        <span className="text-[11px] text-slate-600 font-medium truncate">{cert.main_field}</span>
                      )}
                    </div>
                  </div>
                  <div
                    className="flex items-center gap-1.5 shrink-0 text-blue-400 tabular-nums"
                    title="이 목록 1위 대비 상대 인기도"
                  >
                    <TrendingUp className="w-3.5 h-3.5" aria-hidden />
                    <span className="text-sm font-bold">{trendingRelativePctById.get(cert.qual_id) ?? 0}%</span>
                  </div>
                  <ChevronRight className="w-4 h-4 text-slate-700 group-hover:text-blue-400 transition-colors shrink-0" />
                </div>
              ))}
            </div>
          ) : trendingError ? (
            <Card className="bg-red-500/5 border-red-500/20 py-12">
              <CardContent className="flex flex-col items-center justify-center space-y-4">
                <AlertCircle className="w-12 h-12 text-red-500 opacity-50" />
                <div className="text-center">
                  <h3 className="text-lg font-bold text-white">인기 자격증을 불러오지 못했습니다</h3>
                  <p className="text-slate-400 text-sm mt-1">네트워크를 확인한 뒤 다시 시도해 주세요.</p>
                </div>
                <Button onClick={() => { setTrendingError(false); fetchStartedRef.current = false; fetchTrendingData(); }} variant="outline" className="border-red-500/30 text-red-400">
                  다시 시도
                </Button>
              </CardContent>
            </Card>
          ) : (
            <div className="py-12 text-center text-slate-500 bg-slate-900/20 rounded-2xl border border-dashed border-slate-800">
              {hasRequestedTrending
                ? '데이터 집계 중입니다... 자격증을 검색하거나 상세 페이지를 조회해보세요!'
                : '스크롤하면 최근 주목받는 자격증을 불러옵니다.'}
            </div>
          )}
        </div>
      </section>

      {/* Guide Section */}
      <section>
        <div className="space-y-10">
          <div className="space-y-3">
            <Badge variant="outline" className="border-blue-500/30 text-blue-400 px-4 py-1">활용 가이드</Badge>
            <h2 className="text-3xl md:text-4xl font-bold text-white">이렇게 활용하세요</h2>
          </div>

          <div className="grid md:grid-cols-2 gap-x-0 gap-y-0 max-w-4xl mx-auto">
            {([
              {
                num: '01',
                title: '합격률로 난이도 파악',
                body: '종목마다 합격률이 크게 다릅니다. 연도별·회차별 추이를 차트로 확인하면 준비 기간과 전략을 현실적으로 계획할 수 있습니다.',
              },
              {
                num: '02',
                title: '전공 연결 자격증 탐색',
                body: '전공명을 입력하면 해당 학과에서 주로 취득하는 국가자격증 목록이 표시됩니다. 합격률 통계와 결합해 우선순위를 제안합니다.',
              },
              {
                num: '03',
                title: 'AI 커리어 로드맵',
                body: '"클라우드 보안 분야에서 일하고 싶다"처럼 목표를 입력하면 AI 엔진이 직무 연관성, 합격률, 취득 이력을 종합해 자격증 순서를 제시합니다.',
              },
              {
                num: '04',
                title: '직무·연봉 전망 확인',
                body: '자격증 취득 후 진입 가능한 직무의 전망과 연봉을 워크넷·커리어넷 기반 데이터로 확인합니다. 직무 역량 레이더 차트도 제공합니다.',
              },
            ] as const).map((item) => (
              <div key={item.num} className="flex gap-5 py-8 border-b border-slate-800/50 last:border-0 md:[&:nth-child(odd)]:border-r md:[&:nth-child(odd)]:pr-12 md:[&:nth-child(even)]:pl-12">
                <span className="text-[10px] font-mono font-bold text-slate-700 tracking-[0.12em] pt-1 shrink-0 w-5 select-none">{item.num}</span>
                <div className="space-y-2">
                  <h3 className="text-base font-bold text-white">{item.title}</h3>
                  <p className="text-sm text-slate-500 leading-relaxed">{item.body}</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section>
        <div className="relative rounded-[2rem] overflow-hidden bg-slate-900 border border-slate-800 p-12 md:p-20 text-center">
          {/* 상단 강조선 */}
          <div className="absolute top-0 left-0 right-0 h-px bg-gradient-to-r from-transparent via-blue-500/40 to-transparent" />
          <div className="relative z-10 space-y-8 max-w-3xl mx-auto">
            <Badge variant="outline" className="border-blue-500/30 text-blue-400 px-4 py-1">지금 시작하기</Badge>
            <h2 className="text-3xl md:text-5xl font-extrabold text-white leading-tight">
              커리어의 다음 단계를<br />지금 바로 설계해 보세요
            </h2>
            <p className="text-slate-400 text-lg">
              DB 통계 기반 전공 추천과 AI 맞춤 추천을 함께 쓸 수 있습니다.
              회원가입 없이 대부분의 기능을 무료로 이용할 수 있습니다.
            </p>
            <div className="flex flex-col sm:flex-row items-center gap-5 justify-center pt-4">
              <Button
                onClick={() => router.navigate('/ai-recommendations')}
                size="lg"
                className="bg-blue-600 hover:bg-blue-700 text-white text-lg font-bold rounded-xl h-14 px-10 shadow-xl shadow-blue-500/20"
              >
                AI 자격증 추천 시작
              </Button>
              <div className="flex items-center gap-6 text-sm">
                <button
                  type="button"
                  onClick={() => router.navigate('/recommendation')}
                  className="text-slate-500 hover:text-white transition-colors flex items-center gap-1 font-medium"
                >
                  전공 추천 <ChevronRight className="w-3.5 h-3.5" />
                </button>
                <button
                  type="button"
                  onClick={() => router.navigate('/jobs')}
                  className="text-slate-500 hover:text-white transition-colors flex items-center gap-1 font-medium"
                >
                  직업 전망 <ChevronRight className="w-3.5 h-3.5" />
                </button>
              </div>
            </div>
          </div>
        </div>
      </section>
    </div>
  );
}
