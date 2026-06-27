
import React, { useState, useMemo, useEffect } from 'react';
import {
    Sparkles,
    Tag,
    GraduationCap,
    BrainCircuit,
    MessageSquare,
    ChevronRight,
    ChevronDown,
    Info,
    LogIn,
    RefreshCw,
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Skeleton } from '@/components/ui/skeleton';
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from '@/components/ui/accordion';
import { getHybridRecommendations, getAvailableMajors } from '@/lib/api';
import { useRouter } from '@/lib/router';
import { useAuth } from '@/hooks/useAuth';
import type { HybridRecommendationResponse } from '@/types';
import { toast } from 'sonner';

const LOADING_MSGS = [
    '전공 DB에서 관련 자격증 후보를 추출하는 중입니다...',
    'AI 엔진이 전공-자격증 적합도를 계산하는 중입니다...',
    '커리어 목표와 자격증 간 의미적 유사도를 분석하는 중입니다...',
    '합격률 · 난이도 통계를 반영해 후보를 재정렬하는 중입니다...',
    '최종 추천 목록을 구성하는 중입니다...',
] as const;

const sampleMajors = [
    '컴퓨터공학', '정보통신공학', '전자공학', '전기공학', '기계공학',
    '건축학', '경영학', '회계학', '의학', '간호학', '데이터사이언스'
];

const AI_CACHE_KEY = 'ai-rec-cache';

type AiRecCachePayload = {
    userId: string | null;
    major?: string;
    interest?: string;
    results?: HybridRecommendationResponse;
};
/** 로그인 사용자 하이브리드 추천 API limit (백엔드 le=20). 늘려도 RAG 비용은 거의 동일. */
const HYBRID_RECOMMEND_LIMIT = 15;


const POPULAR_MAJORS = ['컴퓨터공학', '경영학', '전기공학', '간호학', '기계공학', '데이터사이언스'];


export function AiRecommendationPage() {
    const [major, setMajor] = useState('');
    const [interest, setInterest] = useState('');
    const [inputValue, setInputValue] = useState('');
    const [showSuggestions, setShowSuggestions] = useState(false);
    /** Enter로 '정확히 일치'하는 전공만 목록에 남길 때 true (입력 변경 시 해제) */
    const [majorExactMode, setMajorExactMode] = useState(false);
    const [loading, setLoading] = useState(false);
    const [results, setResults] = useState<HybridRecommendationResponse | null>(null);
    const [majorError, setMajorError] = useState<string | null>(null);
    const [submitError, setSubmitError] = useState<string | null>(null);
    const { navigate } = useRouter();
    const { token, user, loading: authLoading } = useAuth();
    const profileMajor = (user as any)?.user_metadata?.detail_major as string | undefined;

    const [availableMajors, setAvailableMajors] = useState<string[]>([]);

    // 전공별 샘플 미리보기 상태
    const [selectedPreviewMajor, setSelectedPreviewMajor] = useState<string | null>(null);
    const [previewLoading, setPreviewLoading] = useState(false);
    const [previewCache, setPreviewCache] = useState<Map<string, any[]>>(new Map());

    const handlePreviewMajor = async (m: string) => {
        setSelectedPreviewMajor(m);
        if (previewCache.has(m)) return;
        setPreviewLoading(true);
        try {
            const res = await getHybridRecommendations(m, '', 3, null);
            setPreviewCache(prev => new Map(prev).set(m, res.results));
        } catch {
            // 미리보기 실패 시 조용히 무시
        } finally {
            setPreviewLoading(false);
        }
    };

    const handleFillFromPreview = () => {
        if (!selectedPreviewMajor) return;
        setMajor(selectedPreviewMajor);
        setInputValue(selectedPreviewMajor);
        window.scrollTo({ top: 0, behavior: 'smooth' });
    };

    // sessionStorage 복원은 user.id(로그인 전환) 기준만 — 매 입력마다 돌면 캐시가 입력을 덮어씀
    // authLoading 중에는 실행하지 않음 — 로드 전에 user=null로 캐시를 잘못 삭제하는 버그 방지
    useEffect(() => {
        if (authLoading) return;
        const currentUid = user?.id ?? null;
        let payload: AiRecCachePayload | null = null;
        try {
            const raw = sessionStorage.getItem(AI_CACHE_KEY);
            if (raw) {
                const parsed = JSON.parse(raw) as Partial<AiRecCachePayload> & {
                    major?: string;
                    interest?: string;
                    results?: HybridRecommendationResponse;
                };
                const owner =
                    parsed.userId === undefined || parsed.userId === ''
                        ? null
                        : String(parsed.userId);

                const cacheValidForCurrentSession =
                    (currentUid == null && owner == null) ||
                    (currentUid != null && owner === currentUid);

                if (!cacheValidForCurrentSession) {
                    sessionStorage.removeItem(AI_CACHE_KEY);
                    if (currentUid != null) {
                        setInterest('');
                        setResults(null);
                        if (profileMajor) {
                            setMajor(profileMajor);
                            setInputValue(profileMajor);
                        } else {
                            setMajor('');
                            setInputValue('');
                        }
                    } else {
                        setMajor('');
                        setInputValue('');
                        setInterest('');
                        setResults(null);
                    }
                    return;
                }

                payload = {
                    userId: owner,
                    major: parsed.major,
                    interest: parsed.interest,
                    results: parsed.results,
                };
            }
        } catch {
            sessionStorage.removeItem(AI_CACHE_KEY);
        }

        if (payload) {
            const { major: m, interest: i, results: r } = payload;
            if (m) {
                setMajor(m);
                setInputValue(m);
            }
            if (i) setInterest(i);
            if (r) setResults(r);
        }
        // invalid 분기에서 profileMajor 참조 — 의도적으로 [user?.id, authLoading]만 의존 (profileMajor 넣으면 캐시 재복원 위험)
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [user?.id, authLoading]);

    // 로그인 + JWT에 전공이 늦게 붙는 경우·캐시 없이 진입 시 전공 칸만 자동 채움 (이미 입력 있으면 유지)
    useEffect(() => {
        if (!user?.id || !profileMajor) return;
        setMajor((m) => (m.trim() ? m : profileMajor));
        setInputValue((v) => (v.trim() ? v : profileMajor));
    }, [user?.id, profileMajor]);

    React.useEffect(() => {
        getAvailableMajors().then(res => setAvailableMajors(res.majors)).catch(() => { });
    }, []);

    useEffect(() => {
        if (!loading) { setLoadingMsgIdx(0); return; }
        const interval = setInterval(() => {
            setLoadingMsgIdx(prev => (prev + 1) % LOADING_MSGS.length);
        }, 3000);
        return () => clearInterval(interval);
    }, [loading]);

    useEffect(() => {
        if (!results) { setBarsVisible(false); return; }
        const t = setTimeout(() => setBarsVisible(true), 200);
        return () => clearTimeout(t);
    }, [results]);

    const filteredMajors = useMemo(() => {
        const list = availableMajors.length > 0 ? availableMajors : sampleMajors;
        const t = inputValue.trim();
        if (!t) return list.slice(0, 24);
        if (majorExactMode) return list.filter(m => m === t);
        return list.filter(m => m.includes(t));
    }, [availableMajors, inputValue, majorExactMode]);

    const handleRecommend = async () => {
        if (!major) {
            setMajorError('전공을 선택하거나 입력해주세요.');
            toast.error('전공을 선택하거나 입력해주세요.');
            return;
        }
        setMajorError(null);
        setSubmitError(null);
        setLoading(true);
        try {
            const res = await getHybridRecommendations(major, interest, HYBRID_RECOMMEND_LIMIT, token);
            setResults(res);
            // 결과를 sessionStorage에 캐싱 → 뒤로가기 시 재호출 없이 복원
            try {
                const cache: AiRecCachePayload = {
                    userId: user?.id ?? null,
                    major,
                    interest,
                    results: res,
                };
                sessionStorage.setItem(AI_CACHE_KEY, JSON.stringify(cache));
            } catch {
                // storage 용량 초과 등 무시
            }
        } catch (err: any) {
            console.error(err);
            setSubmitError('추천 결과를 가져오는데 실패했습니다. 잠시 후 다시 시도해 주세요.');
            toast.error('추천 결과를 가져오는데 실패했습니다.');
        } finally {
            setLoading(false);
        }
    };

    const handleReset = () => {
        // 로그인 + 프로필 전공이 있는 경우, 전공은 고정하고 나머지만 리셋
        if (profileMajor) {
            setMajor(profileMajor);
            setInputValue(profileMajor);
        } else {
            setMajor('');
            setInputValue('');
        }
        setInterest('');
        setResults(null);
        setSubmitError(null);
        sessionStorage.removeItem(AI_CACHE_KEY);
    };

    const [highlightedIdx, setHighlightedIdx] = useState(-1);
    const [loadingMsgIdx, setLoadingMsgIdx] = useState(0);
    const [barsVisible, setBarsVisible] = useState(false);
    const [expandedReasons, setExpandedReasons] = useState<Set<number>>(new Set());

    const toggleReason = (e: React.MouseEvent, qualId: number) => {
        e.stopPropagation();
        setExpandedReasons(prev => {
            const next = new Set(prev);
            next.has(qualId) ? next.delete(qualId) : next.add(qualId);
            return next;
        });
    };

    const navigateToCert = (qualId: number) => {
        navigate(`/certs/${qualId}`);
    };

    return (
        <div className="max-w-6xl mx-auto space-y-12 pb-20">
            {/* Screen reader live region */}
            <div aria-live="polite" aria-atomic="true" className="sr-only">
                {loading && '자격증 추천을 분석하는 중입니다. 잠시 기다려 주세요.'}
                {results && !loading && `${results.major} 전공 AI 추천 완료. ${results.results.length}개의 자격증을 찾았습니다.`}
            </div>
            {/* Hero Section */}
            <div className="relative rounded-3xl bg-slate-900 border border-slate-800 p-8 md:p-12">
                <div
                  className="absolute inset-0 rounded-3xl pointer-events-none"
                  style={{ background: 'radial-gradient(ellipse 60% 60% at 100% 0%, oklch(0.5 0.09 248 / 0.07) 0%, transparent 60%), radial-gradient(ellipse 55% 55% at 0% 100%, oklch(0.5 0.07 275 / 0.06) 0%, transparent 60%)' }}
                />

                <div className="relative z-10 flex flex-col md:flex-row items-center gap-10">
                    <div className="flex-1 space-y-6">
                        <Badge variant="outline" className="bg-blue-500/10 text-blue-400 border-blue-500/20 px-3 py-1">
                            <BrainCircuit className="w-4 h-4 mr-2" />
                            AI 자격증 추천
                        </Badge>
                        <h1 className="text-4xl md:text-5xl font-extrabold text-white leading-tight">
                            관심사와 전공을 <br />
                            하나의 로드맵으로.
                        </h1>
                        <p className="text-slate-400 text-lg max-w-xl">
                            전공, 관심사, 프로필을 반영해 자격 후보를 고릅니다.
                        </p>
                    </div>

                    <div className="w-full max-w-md bg-slate-950/60 backdrop-blur-md border border-slate-800 p-6 rounded-2xl shadow-inner relative z-30">
                        <div className="space-y-5">
                            <div className="space-y-2 relative">
                                <label htmlFor="major-input" className="text-sm font-bold text-slate-300 flex items-center gap-2">
                                    <GraduationCap className="w-4 h-4 text-blue-400" />
                                    나의 전공
                                </label>
                                <Input
                                    id="major-input"
                                    name="major"
                                    placeholder="예: 컴퓨터공학, 경영학"
                                    value={inputValue}
                                    role="combobox"
                                    aria-haspopup="listbox"
                                    aria-expanded={showSuggestions && filteredMajors.length > 0}
                                    aria-controls="major-listbox"
                                    aria-autocomplete="list"
                                    aria-activedescendant={highlightedIdx >= 0 ? `major-option-${highlightedIdx}` : undefined}
                                    onChange={(e) => {
                                        setMajorExactMode(false);
                                        setInputValue(e.target.value);
                                        setMajor(e.target.value);
                                        setShowSuggestions(true);
                                        setMajorError(null);
                                        setHighlightedIdx(-1);
                                    }}
                                    onKeyDown={(e) => {
                                        if (showSuggestions && filteredMajors.length > 0) {
                                            if (e.key === 'ArrowDown') {
                                                e.preventDefault();
                                                setHighlightedIdx(prev => Math.min(prev + 1, filteredMajors.length - 1));
                                                return;
                                            }
                                            if (e.key === 'ArrowUp') {
                                                e.preventDefault();
                                                setHighlightedIdx(prev => Math.max(prev - 1, -1));
                                                return;
                                            }
                                            if (e.key === 'Enter' && highlightedIdx >= 0) {
                                                e.preventDefault();
                                                const m = filteredMajors[highlightedIdx]!;
                                                setMajor(m);
                                                setInputValue(m);
                                                setMajorExactMode(false);
                                                setShowSuggestions(false);
                                                setHighlightedIdx(-1);
                                                setMajorError(null);
                                                return;
                                            }
                                        }
                                        if (e.key === 'Escape') {
                                            setMajorExactMode(false);
                                            setShowSuggestions(false);
                                            setHighlightedIdx(-1);
                                            return;
                                        }
                                        if (e.key !== 'Enter' || profileMajor) return;
                                        const t = inputValue.trim();
                                        if (!t) return;
                                        e.preventDefault();
                                        const list = availableMajors.length > 0 ? availableMajors : sampleMajors;
                                        const exact = list.filter(m => m === t);
                                        if (exact.length === 1) {
                                            setMajor(exact[0]!);
                                            setInputValue(exact[0]!);
                                            setMajorExactMode(false);
                                            setShowSuggestions(false);
                                            setHighlightedIdx(-1);
                                            return;
                                        }
                                        setMajorExactMode(true);
                                        setShowSuggestions(true);
                                    }}
                                    onFocus={() => setShowSuggestions(true)}
                                    readOnly={!!profileMajor}
                                    className={`bg-slate-900/80 h-12 focus:ring-blue-500/20 text-white ${majorError ? 'border-red-500/60 focus-visible:ring-red-500/30' : 'border-slate-700'}`}
                                    aria-invalid={!!majorError}
                                    aria-describedby={majorError ? 'major-input-error' : undefined}
                                />
                                {majorError && (
                                    <p id="major-input-error" role="alert" className="text-xs text-red-400 font-medium px-1">{majorError}</p>
                                )}
                                {showSuggestions && (
                                    <div
                                        id="major-listbox"
                                        role="listbox"
                                        aria-label="전공 목록"
                                        className="absolute top-full left-0 right-0 mt-1 bg-slate-900 border border-slate-700 rounded-xl shadow-[0_20px_50px_rgba(0,0,0,0.5)] z-[100] max-h-[min(22rem,70vh)] overflow-y-auto overflow-x-hidden overscroll-contain"
                                    >
                                        {filteredMajors.length > 0 ? (
                                            filteredMajors.map((m, idx) => (
                                                <div
                                                    key={`${m}__${idx}`}
                                                    id={`major-option-${idx}`}
                                                    role="option"
                                                    aria-selected={highlightedIdx === idx}
                                                    className={`px-4 py-3 cursor-pointer text-sm text-slate-300 transition-colors border-b border-slate-800 last:border-0 break-words ${
                                                        highlightedIdx === idx ? 'bg-slate-800 text-white' : 'hover:bg-slate-800'
                                                    }`}
                                                    onClick={() => {
                                                        setMajor(m);
                                                        setInputValue(m);
                                                        setMajorExactMode(false);
                                                        setShowSuggestions(false);
                                                        setMajorError(null);
                                                        setHighlightedIdx(-1);
                                                    }}
                                                >
                                                    {m}
                                                </div>
                                            ))
                                        ) : (
                                            <div className="px-4 py-3 text-sm text-slate-500">
                                                {majorExactMode
                                                    ? '입력과 정확히 같은 이름의 전공이 없습니다. Esc로 전체 검색으로 돌아가세요.'
                                                    : '일치하는 전공이 없습니다.'}
                                            </div>
                                        )}
                                        {!profileMajor && (
                                            <p className="px-4 py-2 text-[10px] text-slate-600 border-t border-slate-800 bg-slate-950/40">
                                                {majorExactMode
                                                    ? 'Esc: 포함 검색으로 / ↑↓ 방향키로 선택'
                                                    : '↑↓ 방향키로 이동, Enter로 선택, Esc로 닫기'}
                                            </p>
                                        )}
                                    </div>
                                )}
                            </div>

                            <div className="space-y-3">
                                <label htmlFor="career-interest" className="text-sm font-bold text-slate-300 flex items-center gap-2">
                                    <MessageSquare className="w-4 h-4 text-blue-400" />
                                    어떤 일을 하고 싶나요? (커리어 목표)
                                </label>
                                <textarea
                                    id="career-interest"
                                    name="interest"
                                    placeholder="예: 클라우드 보안 환경에서 일하고 싶어요. 데이터 분석을 금융에 적용하고 싶습니다."
                                    value={interest}
                                    onChange={(e) => setInterest(e.target.value)}
                                    maxLength={500}
                                    aria-describedby="career-interest-hint"
                                    className="w-full bg-slate-900/80 border-slate-700 rounded-lg p-3 text-sm h-28 focus:ring-2 focus:ring-blue-500/50 border outline-none text-white focus:border-blue-500 transition-colors placeholder:text-slate-600 shadow-inner resize-none"
                                />
                                {interest.length > 400 && (
                                    <p className={`text-[10px] text-right tabular-nums ${interest.length > 480 ? 'text-rose-400' : 'text-slate-500'}`}>
                                        {500 - interest.length}자 남음
                                    </p>
                                )}
                                <p id="career-interest-hint" className="text-[11px] text-slate-500 leading-relaxed">
                                    AI는 입력한 전공, 커리어 목표뿐 아니라 마이페이지에 저장된
                                    <span className="font-semibold text-slate-300"> 학년, 학과, 관심 자격증, 취득 자격증, 난이도</span>
                                    를 함께 고려해 현재 레벨에 맞는 자격증 난이도를 추천합니다.
                                </p>
                            </div>

                            <Button
                                onClick={handleRecommend}
                                disabled={loading}
                                className="w-full h-12 bg-blue-600 hover:bg-blue-700 text-white font-bold text-lg rounded-xl shadow-lg shadow-blue-900/30"
                            >
                                {loading ? <Skeleton className="w-5 h-5 bg-white/30 rounded-full animate-pulse" /> : "AI 분석"}
                            </Button>
                            {submitError && (
                                <p className="text-xs text-red-400 font-medium text-center py-1" role="alert">
                                    {submitError}
                                </p>
                            )}
                        </div>
                    </div>
                </div>
            </div>

            {/* Results Section */}
            {loading && (
                <div className="space-y-6">
                    <div className="rounded-2xl border border-slate-800 bg-slate-900/40 px-4 py-5 text-center space-y-3">
                        <p
                            key={loadingMsgIdx}
                            className="text-sm text-slate-300 font-medium animate-in fade-in duration-500"
                        >
                            {LOADING_MSGS[loadingMsgIdx]}
                        </p>
                        <p className="text-xs text-slate-500">
                            비로그인 미리보기는 후보 탐색을 가볍게 해 더 빠르게 응답합니다. 통상 약 5~25초입니다.
                        </p>
                        <div className="relative h-1.5 w-full max-w-md mx-auto rounded-full bg-slate-800 overflow-hidden">
                            {/* 막대 너비 f를 바꾸면 index.css @keyframes aiRecIndeterminate 100% translateX% = (1-f)/f×100 (막대 기준) 로 맞출 것 */}
                            <div
                                className="absolute left-0 top-0 h-full w-[36%] rounded-full bg-gradient-to-r from-blue-500 to-indigo-500 will-change-transform"
                                style={{
                                    animation:
                                        'aiRecIndeterminate 1.35s linear infinite alternate',
                                }}
                            />
                        </div>
                    </div>
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                        {[1, 2, 3, 4, 5, 6].map(i => (
                            <Skeleton key={i} className="h-64 rounded-2xl bg-slate-900/50" />
                        ))}
                    </div>
                </div>
            )}

            {results && (
                <div className="space-y-8">
                    <div className="flex items-center justify-between flex-wrap gap-4">
                        <div className="space-y-1">
                            <h2 className="text-2xl font-bold text-white flex items-center gap-3 flex-wrap">
                                <Sparkles className="w-6 h-6 text-yellow-500" />
                                분석 결과
                                <Badge className="bg-amber-500/10 text-amber-400 border-amber-500/20 text-[10px] font-normal">
                                    AI 분석 완료
                                </Badge>
                            </h2>
                            <p className="text-slate-400">
                                {results.major} 전공과 {results.interest ? `"${results.interest}"` : "시스템 데이터"}를 결합한 추천입니다.
                                {results.guest_limited && (
                                    <span className="ml-2 text-amber-400 font-medium text-xs">
                                        (비로그인 미리보기: 상위 3개만 표시)
                                    </span>
                                )}
                            </p>
                        </div>
                        <Button
                            variant="outline"
                            size="sm"
                            onClick={handleReset}
                            className="border-slate-700 text-slate-400 hover:text-white hover:bg-slate-800 rounded-xl"
                        >
                            <RefreshCw className="w-4 h-4 mr-2" />
                            다시 검색
                        </Button>
                    </div>

                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                        {results.results.map((res, idx) => (
                            <Card
                                key={res.qual_id}
                                onClick={() => navigateToCert(res.qual_id)}
                                onKeyDown={(e) => e.key === 'Enter' && navigateToCert(res.qual_id)}
                                role="button"
                                tabIndex={0}
                                style={{ animationDelay: `${idx * 40}ms` }}
                                className="bg-slate-900/40 border-slate-800 hover:border-blue-500/40 hover:bg-slate-900 transition-colors cursor-pointer group rounded-2xl overflow-hidden shadow-sm hover:shadow-blue-500/10 animate-in fade-in slide-in-from-bottom-3 duration-500 focus-visible:ring-2 focus-visible:ring-blue-500/50 focus-visible:outline-none"
                            >
                                <div className="h-2 bg-blue-600/20 group-hover:bg-blue-600 transition-colors" />
                                <CardHeader className="pb-2">
                                    <div className="flex justify-between items-start">
                                        <div className="flex items-center gap-2">
                                            <div className="w-10 h-10 rounded-lg bg-slate-950 flex items-center justify-center text-blue-400 font-bold border border-slate-800">
                                                {idx + 1}
                                            </div>
                                            {res.llm_reason && (
                                                <Badge className="bg-yellow-500/10 text-yellow-400 border-yellow-500/20 text-[9px] px-1.5 py-0">
                                                    ✦ AI 생성
                                                </Badge>
                                            )}
                                        </div>
                                        <div className="text-right shrink-0" title="전공 연관성과 커리어 목표 일치도를 결합한 하이브리드 점수">
                                            <p className="text-[10px] font-bold text-slate-600 uppercase tracking-[0.1em] leading-none mb-1">AI 점수</p>
                                            <p className="text-2xl font-black tabular-nums text-white leading-none">
                                                {Math.min(100, Math.round((res.hybrid_score ?? 0) * 100))}<span className="text-sm text-slate-500 font-bold ml-0.5">%</span>
                                            </p>
                                        </div>
                                    </div>
                                    <CardTitle className="text-xl font-bold text-white mt-3 group-hover:text-blue-400 transition-colors line-clamp-2">
                                        {res.qual_name}
                                    </CardTitle>
                                    {(res.hybrid_score ?? 0) < 0.3 && (
                                        <span className="text-[10px] text-slate-500 font-medium">연관성이 낮습니다</span>
                                    )}
                                    {res.pass_rate != null && (
                                        <div className="flex items-center gap-2 mt-1">
                                            <span className="text-[11px] text-slate-500">최근합격률</span>
                                            <div className="flex-1 h-1.5 bg-slate-800 rounded-full overflow-hidden max-w-[80px]">
                                                <div
                                                    className={`h-full w-full rounded-full transition-transform duration-700 ease-out origin-left ${
                                                        res.pass_rate > 70 ? 'bg-emerald-500' :
                                                        res.pass_rate >= 30 ? 'bg-amber-500' : 'bg-rose-500'
                                                    }`}
                                                    style={{ transform: `scaleX(${barsVisible ? Math.min(res.pass_rate, 100) / 100 : 0})` }}
                                                />
                                            </div>
                                            <span className={`text-[11px] font-bold ${
                                                res.pass_rate > 70 ? 'text-emerald-400' :
                                                res.pass_rate >= 30 ? 'text-amber-400' : 'text-rose-500'
                                            }`}>
                                                {res.pass_rate.toFixed(1)}%
                                            </span>
                                        </div>
                                    )}
                                </CardHeader>
                                <CardContent className="space-y-4">
                                    <div
                                        className="flex items-start gap-2 bg-slate-950/50 p-3 rounded-xl border border-slate-800 cursor-pointer hover:border-blue-500/40 hover:bg-slate-950/80 transition-colors focus-visible:ring-2 focus-visible:ring-blue-500/50 focus-visible:outline-none rounded-xl"
                                        onClick={(e) => toggleReason(e, res.qual_id)}
                                        onKeyDown={(e) => {
                                            if (e.key === 'Enter' || e.key === ' ') {
                                                e.stopPropagation();
                                                e.preventDefault();
                                                setExpandedReasons(prev => {
                                                    const next = new Set(prev);
                                                    next.has(res.qual_id) ? next.delete(res.qual_id) : next.add(res.qual_id);
                                                    return next;
                                                });
                                            }
                                        }}
                                        role="button"
                                        tabIndex={0}
                                        aria-expanded={expandedReasons.has(res.qual_id)}
                                        aria-label={expandedReasons.has(res.qual_id) ? '추천 이유 접기' : '추천 이유 더보기'}
                                    >
                                        <Info className="w-4 h-4 text-blue-400/60 mt-0.5 flex-shrink-0" />
                                        <div className="flex-1 min-w-0">
                                            <p className={`text-sm text-slate-400 leading-relaxed italic ${expandedReasons.has(res.qual_id) ? '' : 'line-clamp-3'}`}>
                                                {(res.reason && !res.reason.startsWith('dataset:')) ? res.reason : "귀하의 전공 역량과 관심사를 고려하여 추천된 자격증입니다."}
                                            </p>
                                            <span className="mt-1.5 flex items-center gap-1 text-[11px] font-semibold text-slate-500">
                                                {expandedReasons.has(res.qual_id) ? (
                                                    <>
                                                        <ChevronDown className="w-3 h-3 rotate-180" />
                                                        접기
                                                    </>
                                                ) : (
                                                    <>
                                                        <ChevronDown className="w-3 h-3" />
                                                        더보기
                                                    </>
                                                )}
                                            </span>
                                        </div>
                                    </div>

                                    <div className="flex items-center justify-between text-[11px] font-bold text-slate-600 pt-2">
                                        <span className="flex items-center gap-1">
                                            <Tag className="w-3 h-3 text-slate-600" />
                                            전공 연관성
                                        </span>
                                        <div className="w-24 h-1 bg-slate-800 rounded-full overflow-hidden">
                                            <div
                                                className="h-full w-full bg-blue-500 transition-transform duration-700 ease-out origin-left"
                                                style={{
                                                    transform: `scaleX(${barsVisible ? (() => {
                                                        const norm = res.major_score_normalized ?? Math.min(1, Math.max(0, res.major_score / 10));
                                                        return (20 + 80 * Math.min(1, Math.max(0, norm))) / 100;
                                                    })() : 0})`,
                                                }}
                                            />
                                        </div>
                                    </div>
                                    <div className="flex items-center justify-between text-[11px] font-bold text-slate-600">
                                        <span className="flex items-center gap-1">
                                            <BrainCircuit className="w-3 h-3 text-slate-600" />
                                            관심도 일치
                                        </span>
                                        <div className="w-24 h-1 bg-slate-800 rounded-full overflow-hidden">
                                            <div
                                                className="h-full w-full bg-emerald-500 transition-transform duration-700 ease-out origin-left"
                                                style={{
                                                    transform: `scaleX(${barsVisible ? (() => {
                                                        const norm = res.semantic_score_normalized ?? Math.min(1, Math.max(0, res.semantic_similarity ?? 0));
                                                        return (20 + 80 * Math.min(1, Math.max(0, norm))) / 100;
                                                    })() : 0})`,
                                                }}
                                            />
                                        </div>
                                    </div>

                                    <div className="pt-2 flex justify-end">
                                        <Button variant="ghost" size="sm" className="text-blue-400 group-hover:translate-x-1 transition-transform p-0 hover:bg-transparent">
                                            상세보기 <ChevronRight className="w-4 h-4" />
                                        </Button>
                                    </div>
                                </CardContent>
                            </Card>
                        ))}
                    </div>

                    {/* 비로그인 — 인라인 회원가입 CTA */}
                    {results.guest_limited && (
                        <div className="border border-dashed border-blue-500/20 rounded-2xl px-6 py-8 flex flex-col sm:flex-row items-center gap-6 bg-blue-500/5">
                            <div className="flex-1 space-y-1 text-center sm:text-left">
                                <p className="text-base font-bold text-white">상위 3개 결과를 확인했습니다.</p>
                                <p className="text-sm text-slate-400 leading-relaxed">
                                    로그인하면 맞춤형 추천 결과{' '}
                                    <span className="text-white font-semibold">최대 {HYBRID_RECOMMEND_LIMIT}개</span>와
                                    취득 이력 기반 난이도 조정을 이용할 수 있습니다.
                                </p>
                            </div>
                            <Button
                                onClick={() => navigate('/auth/login')}
                                className="shrink-0 bg-blue-600 hover:bg-blue-700 text-white font-bold rounded-xl px-6 h-11"
                            >
                                <LogIn className="w-4 h-4 mr-2" />
                                무료 회원가입
                            </Button>
                        </div>
                    )}
                </div>
            )}

            {/* AI 시스템 패널 + 전공별 샘플 미리보기 */}
            {!results && !loading && (
                <div className="space-y-10 pt-8 border-t border-slate-800/50">

                    {/* ── 1. 입력 가이드 ── */}
                    <Accordion type="single" collapsible className="w-full">
                        <AccordionItem value="guide" className="border border-slate-800 rounded-2xl overflow-hidden bg-slate-900/50">
                            <AccordionTrigger className="px-5 py-4 hover:no-underline hover:bg-slate-800/30 [&>svg]:text-slate-500">
                                <div className="flex items-center gap-2">
                                    <MessageSquare className="w-4 h-4 text-blue-400" />
                                    <span className="text-sm font-bold text-slate-300">좋은 추천을 받으려면</span>
                                </div>
                            </AccordionTrigger>
                            <AccordionContent className="px-5 pb-5">
                                <div className="grid grid-cols-1 md:grid-cols-3 gap-4 pt-2">
                                    {([
                                        {
                                            step: '01',
                                            label: '전공명은 구체적으로',
                                            desc: "'컴퓨터공학', '간호학'처럼 정확한 학과명을 입력할수록 관련 자격증이 더 정밀하게 매칭됩니다.",
                                        },
                                        {
                                            step: '02',
                                            label: '커리어 목표를 함께 입력',
                                            desc: "관심사란에 '데이터 분석 취업 준비', '공무원 준비' 등 목표를 쓸수록 AI 적합도가 올라갑니다.",
                                        },
                                        {
                                            step: '03',
                                            label: '취득 자격증은 자동 제외',
                                            desc: '로그인 후 마이페이지에서 취득 자격증을 등록하면 추천 목록에서 자동으로 빠집니다.',
                                        },
                                    ] as const).map((item) => (
                                        <div key={item.step} className="bg-slate-950/40 border border-slate-800 rounded-xl p-4 space-y-2">
                                            <p className="text-2xl font-black text-slate-700">{item.step}</p>
                                            <h4 className="text-sm font-bold text-white">{item.label}</h4>
                                            <p className="text-xs text-slate-500 leading-relaxed">{item.desc}</p>
                                        </div>
                                    ))}
                                </div>
                            </AccordionContent>
                        </AccordionItem>
                    </Accordion>

                    {/* ── 2. 전공별 AI 추천 미리보기 ── */}
                    <div className="space-y-5">
                        <div className="flex items-center justify-between flex-wrap gap-3">
                            <div className="flex items-center gap-2">
                                <Sparkles className="w-4 h-4 text-yellow-400" />
                                <h3 className="text-sm font-bold text-slate-400">전공별 AI 추천 미리보기</h3>
                            </div>
                            <p className="text-xs text-slate-600">탭을 클릭하면 실제 AI가 분석합니다</p>
                        </div>

                        {/* 전공 탭 */}
                        <div className="flex flex-wrap gap-2">
                            {POPULAR_MAJORS.map((m) => (
                                <button
                                    key={m}
                                    onClick={() => handlePreviewMajor(m)}
                                    className={`px-4 py-2 rounded-xl text-sm font-semibold border transition-all ${
                                        selectedPreviewMajor === m
                                            ? 'bg-blue-600 border-blue-500 text-white shadow-lg shadow-blue-900/30'
                                            : 'bg-slate-900/50 border-slate-800 text-slate-400 hover:border-blue-500/40 hover:text-slate-200'
                                    }`}
                                >
                                    {m}
                                    {previewCache.has(m) && selectedPreviewMajor !== m && (
                                        <span className="ml-1.5 w-1.5 h-1.5 rounded-full bg-emerald-400 inline-block" />
                                    )}
                                </button>
                            ))}
                        </div>

                        {/* 미리보기 결과 */}
                        {selectedPreviewMajor && (
                            <div className="space-y-4 animate-in fade-in slide-in-from-bottom-2 duration-300">
                                {previewLoading ? (
                                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                                        {[0, 1, 2].map(i => (
                                            <div key={i} className="bg-slate-900/40 border border-slate-800 rounded-2xl p-5 space-y-3">
                                                <div className="flex justify-between">
                                                    <div className="w-8 h-8 rounded-lg bg-slate-800 animate-pulse" />
                                                    <div className="h-5 w-20 bg-slate-800 rounded-full animate-pulse" />
                                                </div>
                                                <div className="h-5 w-full bg-slate-800 rounded animate-pulse" />
                                                <div className="h-5 w-3/4 bg-slate-800 rounded animate-pulse" />
                                                <div className="h-12 bg-slate-950/50 rounded-xl animate-pulse" />
                                            </div>
                                        ))}
                                    </div>
                                ) : (previewCache.get(selectedPreviewMajor) || []).length > 0 ? (
                                    <>
                                        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                                            {(previewCache.get(selectedPreviewMajor) || []).map((res: any, idx: number) => (
                                                <div
                                                    key={res.qual_id}
                                                    onClick={() => navigate(`/certs/${res.qual_id}`)}
                                                    onKeyDown={(e) => e.key === 'Enter' && navigate(`/certs/${res.qual_id}`)}
                                                    role="button"
                                                    tabIndex={0}
                                                    className="group bg-slate-900/40 border border-slate-800 hover:border-blue-500/40 hover:bg-slate-900 rounded-2xl p-5 cursor-pointer transition-colors space-y-3 focus-visible:ring-2 focus-visible:ring-blue-500/50 focus-visible:outline-none"
                                                >
                                                    <div className="flex items-center justify-between">
                                                        <div className="w-8 h-8 rounded-lg bg-slate-950 border border-slate-800 flex items-center justify-center text-xs font-bold text-slate-400">
                                                            {idx + 1}
                                                        </div>
                                                        <Badge
                                                            className="bg-blue-500/10 text-blue-400 border-blue-500/20 text-[10px]"
                                                            title="전공 연관성과 커리어 목표 일치도를 결합한 하이브리드 점수"
                                                        >
                                                            AI 적합도 {Math.min(100, Math.round((res.hybrid_score ?? 0) * 100))}%
                                                        </Badge>
                                                    </div>
                                                    <p className="text-sm font-bold text-white group-hover:text-blue-300 transition-colors line-clamp-2 leading-snug">
                                                        {res.qual_name}
                                                    </p>
                                                    {(res.hybrid_score ?? 0) < 0.3 && (
                                                        <span className="text-[10px] text-slate-500 font-medium">연관성이 낮습니다</span>
                                                    )}
                                                    {res.reason && !res.reason.startsWith('dataset:') && (
                                                        <p className="text-[11px] text-slate-500 leading-relaxed line-clamp-2 italic">
                                                            {res.reason}
                                                        </p>
                                                    )}
                                                    {/* 미니 스코어 바: 정규화 값 우선 */}
                                                    <div className="space-y-1.5 pt-1">
                                                        <div className="flex items-center gap-2">
                                                            <div className="w-2 h-2 rounded-sm bg-blue-500 shrink-0" />
                                                            <div className="h-1 flex-1 bg-slate-800 rounded-full overflow-hidden">
                                                                <div
                                                                    className="h-full bg-blue-500 rounded-full transition-all"
                                                                    style={{
                                                                        width: `${Math.min(100, (res.major_score_normalized ?? Math.min(1, (res.major_score ?? 0) / 10)) * 100)}%`,
                                                                    }}
                                                                />
                                                            </div>
                                                            <span className="text-[10px] text-slate-600 w-8 shrink-0">전공</span>
                                                        </div>
                                                        <div className="flex items-center gap-2">
                                                            <div className="w-2 h-2 rounded-sm bg-emerald-500 shrink-0" />
                                                            <div className="h-1 flex-1 bg-slate-800 rounded-full overflow-hidden">
                                                                <div
                                                                    className="h-full bg-emerald-500 rounded-full transition-all"
                                                                    style={{
                                                                        width: `${Math.min(100, (res.semantic_score_normalized ?? Math.min(1, res.semantic_similarity ?? 0)) * 100)}%`,
                                                                    }}
                                                                />
                                                            </div>
                                                            <span className="text-[10px] text-slate-600 w-8 shrink-0">관심</span>
                                                        </div>
                                                    </div>
                                                </div>
                                            ))}
                                        </div>
                                        <div className="flex items-center gap-3 pt-1">
                                            <Button
                                                onClick={handleFillFromPreview}
                                                className="bg-blue-600 hover:bg-blue-700 text-white font-bold rounded-xl text-sm h-10 px-5"
                                            >
                                                <GraduationCap className="w-4 h-4 mr-2" />
                                                {selectedPreviewMajor} 전공으로 상세 분석하기
                                            </Button>
                                            <p className="text-xs text-slate-600">커리어 목표를 추가하면 더 정확해집니다</p>
                                        </div>
                                    </>
                                ) : (
                                    <p className="text-sm text-slate-500 text-center py-8">미리보기 결과를 불러오지 못했습니다.</p>
                                )}
                            </div>
                        )}

                        {!selectedPreviewMajor && (
                            <div className="flex items-center justify-center h-24 rounded-2xl border border-dashed border-slate-800 text-slate-600 text-sm">
                                위 전공 탭을 클릭하면 AI가 실시간으로 추천 결과를 미리 보여드립니다
                            </div>
                        )}
                    </div>
                </div>
            )}
        </div>
    );
}
