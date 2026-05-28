import { Database, BrainCircuit, TrendingUp, CheckCircle2, Users, ChevronLeft, Award, Target, Search } from 'lucide-react';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { useRouter } from '@/lib/router';

const FAQ = [
  {
    q: 'CertFinder의 자격증 데이터는 어디서 가져오나요?',
    a: '한국산업인력공단(Q-Net)과 국가자격정보 공개 DB를 기반으로 수집 및 정제한 데이터를 사용합니다. 합격률, 응시자 수, 시험 회차별 통계는 공개된 국가통계를 반영하며, 주기적으로 업데이트됩니다.',
  },
  {
    q: 'AI 추천 결과는 어떻게 만들어지나요?',
    a: '입력한 전공과 커리어 목표를 기반으로 BM25 키워드 검색, 의미론적 벡터 검색(pgvector), 맥락 기반 추론을 조합한 하이브리드 RAG 엔진이 자격증 후보를 선별합니다. 단순 키워드 매칭이 아니라 직무 연관성, 합격률, 난이도를 함께 고려해 순위를 결정합니다.',
  },
  {
    q: '합격률 데이터는 얼마나 최신인가요?',
    a: '시험 회차별 합격률은 공개된 국가통계 기준으로 수집됩니다. 자격증마다 최신 데이터 시점이 다를 수 있으며, 상세 페이지의 "최근 합격률" 항목에 기준 연도·회차가 표시됩니다.',
  },
  {
    q: '회원가입 없이도 사용할 수 있나요?',
    a: '네. 자격증 검색, 합격률·난이도 조회, 전공 기반 추천, 직무 분석 등 대부분의 기능은 로그인 없이 이용 가능합니다. 로그인하면 취득 자격증 등록, AI 추천 결과 전체 보기, 관심 자격증 저장 기능을 추가로 사용할 수 있습니다.',
  },
  {
    q: '직무·직업 전망 데이터는 어디서 오나요?',
    a: '워크넷(고용24), 커리어넷, 국가기술자격 통계를 바탕으로 구성했습니다. 직무 역량 레이더 차트의 수치는 이들 데이터를 알고리즘으로 정규화한 상대적 지표이며, 개인의 역량과는 다를 수 있습니다.',
  },
  {
    q: '추천 결과가 실제 취업이나 시험 합격을 보장하나요?',
    a: '아닙니다. CertFinder의 모든 분석과 추천은 통계 데이터와 AI 알고리즘에 기반한 참고 정보입니다. 최종 취득 목표 선정은 개인의 상황, 전공, 목표 직무를 직접 검토하여 결정하시기 바랍니다.',
  },
];

const DATA_SOURCES = [
  { name: 'Q-Net (한국산업인력공단)', desc: '자격증 정보, 시험 일정, 합격률 공식 데이터' },
  { name: '국가자격정보 공개 DB', desc: '전체 자격종목 목록, NCS 분류, 등급 체계' },
  { name: '워크넷 (고용24)', desc: '직무별 채용 동향, 직업 전망 정보' },
  { name: '커리어넷', desc: '직업 특성, 핵심 역량, 교육 경로 데이터' },
  { name: '국가기술자격 통계연보', desc: '연도별 응시자 수, 합격자 수, 합격률 통계' },
];

export function AboutPage() {
  const router = useRouter();

  return (
    <div className="max-w-4xl mx-auto space-y-20 pb-20">

      {/* Back */}
      <Button
        variant="ghost"
        onClick={() => router.navigate('/')}
        className="text-slate-500 hover:text-white -ml-4 flex items-center gap-2"
      >
        <ChevronLeft className="w-4 h-4" /> 홈으로
      </Button>

      {/* Hero */}
      <section className="space-y-6">
        <Badge className="bg-blue-500/10 text-blue-400 border-blue-500/20">서비스 소개</Badge>
        <h1 className="text-4xl md:text-5xl font-black text-white leading-tight tracking-tight">
          CertFinder란 무엇인가요?
        </h1>
        <p className="text-slate-400 text-lg leading-relaxed font-medium max-w-2xl">
          CertFinder는 대한민국 국가기술자격 전체 데이터를 한 곳에 모아, 전공과 커리어 목표에 맞는
          자격증 경로를 AI로 제안하는 정보 분석 서비스입니다.
          막연했던 자격증 선택을 합격률, 난이도, 직무 연관성이라는 실제 데이터를 기반으로 결정할 수 있도록 돕습니다.
        </p>
      </section>

      {/* How it works */}
      <section className="space-y-10">
        <h2 className="text-2xl font-bold text-white">어떻게 동작하나요?</h2>
        <div className="grid md:grid-cols-3 gap-6">
          {[
            {
              step: '01',
              icon: Database,
              title: '국가 데이터 수집',
              desc: 'Q-Net, 워크넷, 커리어넷 등 공공 데이터를 수집·정제해 자격증별 합격률, 시험 구조, 직무 연관성 데이터를 구축합니다.',
            },
            {
              step: '02',
              icon: BrainCircuit,
              title: '하이브리드 AI 분석',
              desc: '전공과 커리어 목표를 키워드 검색, 의미론적 벡터 분석, 맥락 추론 세 채널로 분석해 가장 관련도 높은 자격증을 순위화합니다.',
            },
            {
              step: '03',
              icon: TrendingUp,
              title: '맞춤 추천 제공',
              desc: '합격률, 난이도, 취득 자격증 이력, 학년을 모두 고려한 개인화 추천 목록을 제공합니다. 이미 취득한 자격증은 자동으로 제외됩니다.',
            },
          ].map((item) => (
            <div key={item.step} className="bg-slate-900/40 border border-slate-800 rounded-2xl p-6 space-y-4">
              <div className="flex items-center gap-3">
                <span className="text-2xl font-black text-slate-700">{item.step}</span>
                <item.icon className="w-5 h-5 text-blue-400" />
              </div>
              <h3 className="text-lg font-bold text-white">{item.title}</h3>
              <p className="text-sm text-slate-400 leading-relaxed font-medium">{item.desc}</p>
            </div>
          ))}
        </div>
      </section>

      {/* Core features */}
      <section className="space-y-8">
        <h2 className="text-2xl font-bold text-white">주요 기능</h2>
        <div className="space-y-4">
          {[
            {
              icon: Search,
              title: '자격증 통합 검색',
              desc: '1,100여 종의 국가기술자격을 이름, 발행기관, 분야별로 검색하고, 합격률·난이도 기준으로 정렬할 수 있습니다. 관심 자격증을 즐겨찾기로 저장해 비교할 수도 있습니다.',
            },
            {
              icon: Target,
              title: '전공 기반 자격증 추천',
              desc: '전공명을 입력하면 국가 DB의 전공-자격증 매핑 데이터와 합격률 통계를 결합해 적합한 자격증 목록을 제공합니다. 전산·공학·경영·의료 등 100여 개 전공을 지원합니다.',
            },
            {
              icon: BrainCircuit,
              title: 'AI 하이브리드 추천',
              desc: '전공과 커리어 목표를 자유 문장으로 입력하면 BM25 키워드 검색과 pgvector 의미론적 검색을 결합한 하이브리드 RAG 엔진이 실시간으로 최적의 자격증 후보를 분석합니다.',
            },
            {
              icon: Award,
              title: '자격증 상세 통계',
              desc: '연도별·회차별 합격률 변화 추이, 응시자 수, 난이도 등급을 차트와 데이터 테이블로 확인할 수 있습니다. 시험 구조(필기/실기/면접)와 관련 직무도 함께 제공합니다.',
            },
            {
              icon: Users,
              title: '직무·진로 분석',
              desc: '자격증과 연관된 직무의 직업 전망, 초임 연봉, 핵심 역량을 레이더 차트로 시각화합니다. 워크넷과 커리어넷 데이터를 기반으로 객관적인 직업 전망 정보를 제공합니다.',
            },
          ].map((f) => (
            <div key={f.title} className="flex gap-5 p-5 bg-slate-900/30 border border-slate-800/60 rounded-2xl">
              <div className="shrink-0 w-10 h-10 rounded-xl bg-blue-500/10 border border-blue-500/20 flex items-center justify-center">
                <f.icon className="w-5 h-5 text-blue-400" />
              </div>
              <div className="space-y-1.5">
                <h3 className="font-bold text-white">{f.title}</h3>
                <p className="text-sm text-slate-400 leading-relaxed font-medium">{f.desc}</p>
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* Data sources */}
      <section className="space-y-8">
        <h2 className="text-2xl font-bold text-white">데이터 출처</h2>
        <p className="text-slate-400 font-medium leading-relaxed">
          CertFinder의 모든 자격증 정보, 합격률, 직무 데이터는 대한민국 공공 데이터 포털 및
          국가 기관의 공개 데이터를 기반으로 합니다. 상업적 재가공이나 미인증 출처 데이터를 사용하지 않습니다.
        </p>
        <div className="grid sm:grid-cols-2 gap-4">
          {DATA_SOURCES.map((src) => (
            <div key={src.name} className="flex items-start gap-3 p-4 bg-slate-900/40 border border-slate-800 rounded-xl">
              <CheckCircle2 className="w-4 h-4 text-emerald-400 mt-0.5 shrink-0" />
              <div>
                <p className="text-sm font-bold text-white">{src.name}</p>
                <p className="text-xs text-slate-500 mt-0.5 font-medium">{src.desc}</p>
              </div>
            </div>
          ))}
        </div>
        <p className="text-xs text-slate-600 font-medium">
          * 데이터 갱신 주기는 항목별로 상이하며, 상세 페이지에 기준 시점이 표시됩니다.
          공식 시험 일정 및 접수 정보는 반드시 Q-Net(www.q-net.or.kr) 공식 사이트를 확인하시기 바랍니다.
        </p>
      </section>

      {/* FAQ */}
      <section className="space-y-8">
        <h2 className="text-2xl font-bold text-white">자주 묻는 질문</h2>
        <div className="space-y-4">
          {FAQ.map((item, i) => (
            <div key={i} className="p-6 bg-slate-900/40 border border-slate-800 rounded-2xl space-y-3">
              <h3 className="font-bold text-white flex items-start gap-2">
                <span className="text-blue-400 shrink-0">Q.</span>
                {item.q}
              </h3>
              <p className="text-sm text-slate-400 leading-relaxed font-medium pl-6">
                {item.a}
              </p>
            </div>
          ))}
        </div>
      </section>

      {/* CTA */}
      <section className="bg-slate-900/50 border border-slate-800 rounded-3xl p-10 text-center space-y-6">
        <h2 className="text-2xl font-bold text-white">지금 바로 시작해보세요</h2>
        <p className="text-slate-400 font-medium max-w-md mx-auto">
          회원가입 없이도 자격증 검색, 합격률 분석, AI 추천 기능을 무료로 이용할 수 있습니다.
        </p>
        <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
          <Button
            onClick={() => router.navigate('/ai-recommendations')}
            className="bg-blue-600 hover:bg-blue-700 text-white font-bold px-8 h-12 rounded-xl"
          >
            AI 자격증 추천 시작
          </Button>
          <Button
            variant="ghost"
            onClick={() => router.navigate('/certs')}
            className="text-slate-400 hover:text-white font-medium"
          >
            자격증 탐색하기
          </Button>
        </div>
      </section>

      {/* Disclaimer */}
      <section className="text-xs text-slate-600 leading-relaxed font-medium border-t border-slate-800 pt-8 space-y-2">
        <p>
          CertFinder는 국가자격 정보를 참고 목적으로 제공하는 비공식 서비스입니다.
          시험 접수, 합격자 발표, 자격증 발급 등 공식 업무는 반드시 Q-Net, 해당 시행 기관의 공식 사이트를 이용하시기 바랍니다.
        </p>
        <p>
          제공되는 합격률, 난이도, 직무 전망 데이터는 참고용이며, 개인의 취득 결과를 보장하지 않습니다.
          AI 추천 결과는 통계 기반의 제안일 뿐이며 전문 직업 상담을 대체하지 않습니다.
        </p>
      </section>

    </div>
  );
}
