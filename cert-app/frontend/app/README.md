# CertFinder Frontend

**React 19 + TypeScript + Vite 기반 국가자격증 통합 분석 플랫폼 UI**

---

## 기술 스택

| 항목 | 내용 |
|------|------|
| Framework | React 19, Vite 7 |
| Language | TypeScript 5.9 |
| Styling | Tailwind CSS 3, Shadcn UI (Radix UI) |
| Server State | TanStack Query v5 (staleTime 1분~1시간) |
| Routing | Custom Client-side Router (`src/lib/router.tsx`) |
| Auth | Supabase JS (`@supabase/supabase-js`) |
| 성능 모니터링 | Vercel Speed Insights (`@vercel/speed-insights/react`) |

---

## 프로젝트 구조

```text
src/
├── App.tsx              # 루트 — 라우팅·SpeedInsights 마운트
├── main.tsx             # 진입점 — QueryClientProvider
├── components/
│   ├── layout/          # Header(UserMenu), Layout, Sidebar
│   ├── common/          # CertLogo 등 공통 요소
│   └── ui/              # shadcn/ui 원자 컴포넌트
├── pages/               # 도메인 페이지
│   ├── HomePage         # 메인 (트렌딩·배너)
│   ├── CertListPage     # 자격증 탐색 (필터·페이지네이션)
│   ├── CertDetailPage   # 자격증 상세 (통계·합격률)
│   ├── RecommendationPage   # 전공 기반 추천
│   ├── AiRecommendationPage # AI(RAG) 자연어 추천
│   ├── JobListPage / JobDetailPage
│   ├── MyPage           # 즐겨찾기·취득 자격증·XP·티어
│   ├── PrivacyPolicyPage / TermsOfServicePage
│   └── ContactPage
├── hooks/
│   ├── useAuth.ts       # 세션·로그인·로그아웃 (Supabase)
│   ├── useCerts.ts      # 자격증 목록·상세·필터 (TanStack Query)
│   ├── useRecommendations.ts
│   ├── useMajors.ts / usePopularMajors.ts
│   └── ...
├── lib/
│   ├── api.ts           # fetch 래퍼 (재시도·Mock Fallback)
│   ├── router.tsx       # 경량 SPA 라우터
│   ├── queryKeys.ts     # TanStack Query 키 팩토리
│   └── supabase.ts      # Supabase 클라이언트 싱글톤
└── types/               # 전역 인터페이스 (Cert, Job, Recommendation 등)
```

---

## 주요 기능

- **Vercel Speed Insights**: Core Web Vitals(LCP·FID·CLS) 자동 수집. `<SpeedInsights />` 컴포넌트가 `App.tsx` 최하단에 마운트됨. 로컬에서는 no-op, Vercel 배포 시 자동 활성화.
- **Lazy 코드 스플리팅**: 모든 페이지 `React.lazy` + `Suspense` — 초기 번들 최소화.
- **커스텀 SPA 라우터**: `popstate` + `RouterContext.subscribe` 기반. React Router 미사용으로 번들 경량화.
- **다크 모드 고정**: 글로벌 `dark` 클래스 (Tailwind dark 변수 전체 활성).
- **접근성**: 페이지 전환 시 `<main id="main-content">` 헤딩 포커스 복원.
- **XP·티어 시스템**: 취득 자격증 난이도 기반 XP 합산 → 9단계 레벨(Bronze~Diamond).

---

## 설치 및 실행

```bash
# 의존성 설치
npm install

# 환경 변수 설정
cp .env.example .env
# VITE_API_BASE_URL=https://api.certfinder.cloud
# VITE_SUPABASE_URL=...
# VITE_SUPABASE_ANON_KEY=...

# 개발 서버
npm run dev

# 프로덕션 빌드
npm run build
```

---

## 환경 변수

| 변수명 | 설명 |
|--------|------|
| `VITE_API_BASE_URL` | 백엔드 API 베이스 URL |
| `VITE_SUPABASE_URL` | Supabase 프로젝트 URL |
| `VITE_SUPABASE_ANON_KEY` | Supabase anon key |

---

## 배포

- **호스팅**: Vercel (main 브랜치 자동 배포)
- **Speed Insights**: Vercel 프로젝트 대시보드 → Speed Insights 탭에서 Core Web Vitals 확인
- **환경 변수**: Vercel 프로젝트 설정 → Environment Variables
