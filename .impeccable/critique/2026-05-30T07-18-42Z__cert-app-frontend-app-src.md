---
target: CertFinder src (all pages)
total_score: 24
p0_count: 0
p1_count: 2
p2_count: 3
timestamp: 2026-05-30T07-18-42Z
slug: cert-app-frontend-app-src
---
## Design Health Score

| # | Heuristic | Score | Key Issue |
|---|-----------|-------|-----------|
| 1 | Visibility of System Status | 3 | Skeleton loading excellent; "운영 중" pulse on hero widget is ambiguous |
| 2 | Match System / Real World | 3 | Korean copy natural; "하이브리드 RAG" and "NCS 분류" appear without inline definition |
| 3 | User Control and Freedom | 2 | No active-filter summary on cert list; AI rec error state lost after toast dismisses |
| 4 | Consistency and Standards | 3 | Signal-blue disciplined; bookmark button breaks single-accent rule with amber-filled active state |
| 5 | Error Prevention | 2 | AI rec textarea has no character limit; no autosave message; major validation is post-submit only |
| 6 | Recognition Rather Than Recall | 3 | Trending keywords visible; sort direction toggle is icon-only invented affordance |
| 7 | Flexibility and Efficiency | 2 | No keyboard shortcuts; URL state sync is undiscoverable |
| 8 | Aesthetic and Minimalist Design | 3 | "실시간 분석 활성" badge is pure noise; floating Star/Award icons on hero are decoration |
| 9 | Error Recovery | 2 | AI rec error state is stored but never rendered; toast-only feedback means persistent failure is invisible |
| 10 | Help and Documentation | 1 | Near-zero contextual help; no tooltips on invented affordances |
| **Total** | | **24/40** | **Acceptable — significant improvements needed** |

## Anti-Patterns Verdict

**LLM assessment:** Macro design intent is sound. Several AI-tell details remain: gradient text on logo wordmark (Layout.tsx:69, :183) directly violates DESIGN.md's prohibition. Floating decorative icons (animate-bounce-slow Star, animate-float Award, HomePage.tsx:225-231) are pure decoration. "실시간 분석 활성" badge (CertListPage.tsx:496) is semantic decoration that never changes state. Repeated body copy in AiRecommendationPage hero (lines 259+262) says identical things.

**Deterministic scan:** Detector unavailable (bundled detector not found). Manual inspection: gradient text ×2 (Layout.tsx:69, :183), decorative motion ×2 (HomePage.tsx:225, :228), icon-only invented affordance ×1 (CertListPage.tsx:481), repeated copy ×1 (AiRecommendationPage.tsx:259+262), swallowed error state ×1 (AiRecommendationPage.tsx:58).

## What's Working

1. Loading state execution is excellent — skeletons, IntersectionObserver lazy load for trending certs, indeterminate progress bar with Korean contextual copy during AI analysis.
2. CertDetailPage information architecture — four-tab structure maps cleanly to decision stages; sticky tab bar, chart stage toggles, and radar chart are all purposeful density.
3. Signal blue discipline on interactive elements — single-accent rule holds on 80%+ of surfaces.

## Priority Issues

**[P1] Gradient text on CertFinder wordmark — every page**
- What: `bg-gradient-to-r from-white to-slate-400 bg-clip-text text-transparent` on logo text, Layout.tsx:69 (header) and Layout.tsx:183 (footer). Direct DESIGN.md violation.
- Why it matters: Wordmark appears on every page — system-level violation.
- Fix: Replace with `text-white` (header) and `text-white/80` (footer).

**[P1] AI recommendation error state is invisible after toast dismisses**
- What: `const [, setError] = useState<Error | null>(null)` at AiRecommendationPage.tsx:58 — state is stored but display variable is discarded. Toast fires, then no persistent failure indicator.
- Why it matters: Users in a 5–25 second loading flow cannot tell whether to wait or retry after a failure.
- Fix: Render inline error message below submit button when error state is set.

**[P2] Floating decorative icons on homepage hero**
- What: `animate-bounce-slow` Star and `animate-float` Award at HomePage.tsx:225-231.
- Why it matters: The most category-reflex element on the page — generic Tailwind landing page template signature.
- Fix: Remove both entirely.

**[P2] Sort direction toggle is icon-only invented affordance**
- What: CertListPage.tsx:481-491 — TrendingUp icon rotated 180°, no label, no accessible tooltip.
- Fix: Add text label `오름차순` / `내림차순` or a Tooltip component.

**[P2] Redundant copy in AiRecommendationPage hero**
- What: AiRecommendationPage.tsx:259-263 — two consecutive sentences say identical things. Delete the second.

## Persona Red Flags

**Jordan (First-Timer):** Types "컴퓨터공학" in homepage search → gets 0 results on cert list → no suggestion to try 전공 추천 or AI 추천. Two recommendation nav links with no differentiation visible to a first-timer.

**Sam (Accessibility):** Major suggestion dropdown (AiRecommendationPage.tsx:318) is a custom div-based listbox with no ARIA roles — no role="listbox", no role="option", no aria-expanded, no keyboard navigation within the list.

**Riley (Stress Tester):** Backend 5XX on AI rec → toast dismisses after 4s → form shows no persistent error state. User cannot tell if request failed or is still pending.

**취준생 (Primary persona):** CertDetailPage 관련 직무 tab shows "분석 중" empty state for certs with no job data — no alternative path, no ETA. User is dead-ended.

## Minor Observations

- CertDetailPage.tsx:454: `font-bold text-slate-500` duplicated in className.
- AiRecommendationPage.tsx:254: `<span className="text-white">` wrapping text that is already white — dead markup.
- HomePage.tsx:239: "Core Modules" badge is English on a Korean-first surface.
- AI rec pre-submit section renders guide cards + preview placeholder simultaneously — larger surface than it earns before first interaction.

## Questions to Consider

- "The '실시간 분석 활성' badge in cert list — if removing it changes nothing functionally, what stops you from deleting it today?"
- "The two recommendation entry points ('전공 추천' and 'AI 추천') have no visible functional description — does a first-time user know they are different products?"
- "The cert detail empty job state says '분석 중' — is it actually loading, or does the data simply not exist for that cert? If the latter, the copy is misleading."
