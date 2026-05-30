---
target: CertFinder src (all pages) — post-fix run
total_score: 28
p0_count: 0
p1_count: 1
p2_count: 3
timestamp: 2026-05-30T07-46-39Z
slug: cert-app-frontend-app-src
---
## Design Health Score

| # | Heuristic | Score | Key Issue |
|---|-----------|-------|-----------|
| 1 | Visibility of System Status | 3 | Rotating loading messages excellent; "운영 중" pulse still ambiguous |
| 2 | Match Between System / Real World | 3 | Nav sub-labels help; "하이브리드 RAG" still unexplained |
| 3 | User Control and Freedom | 3 | Jordan rescue path added; persistent error state; no undo/filter summary |
| 4 | Consistency and Standards | 3 | Logo gradient fixed; auth modal gradient text surviving in UserMenu:464+468; amber CTA button |
| 5 | Error Prevention | 2 | No character limit on interest textarea |
| 6 | Recognition Rather Than Recall | 3 | Sort labeled; nav sub-labels distinguish rec types |
| 7 | Flexibility and Efficiency | 2 | No keyboard shortcuts; URL sync undiscoverable |
| 8 | Aesthetic and Minimalist Design | 4 | Floating icons, "실시간 분석 활성" badge, redundant copy, dead span all removed |
| 9 | Error Recovery | 3 | Persistent inline AI rec error; honest job empty state with rescue path |
| 10 | Help and Documentation | 2 | Nav sub-labels, Jordan rescue, rotating messages; no contextual help system |
| **Total** | | **28/40** | **Good — address weak areas, solid foundation** |

## Anti-Patterns Verdict

LLM assessment: Main tells removed. "Precision instrument" read now defensible on most surfaces. Two remaining violations: UserMenu.tsx:464+468 gradient text in auth modal dialog titles (same pattern removed from logo); CertDetailPage.tsx:390 amber full-button background on bookmarked state (breaks single-accent rule).

Deterministic scan: Detector unavailable. Manual: gradient text ×2 (UserMenu.tsx:464, :468); amber CTA ×1 (CertDetailPage.tsx:390).

## What's Working

1. Rotating loading messages: product-specific step descriptions replace generic "loading..." copy. 5-step cycle with fade transition.
2. Aesthetic noise removal is thorough: floating icons, theatrical badge, redundant copy, dead span all gone.
3. Bar fill animations calibrated: 200ms delay + 700ms ease-out reads as gauges settling.

## Priority Issues

**[P1] Gradient text in auth modal (UserMenu.tsx:464+468)**
- What: Dialog titles "새로운 시작" and "환영합니다" use bg-clip-text text-transparent with gradient backgrounds.
- Fix: Replace both spans with plain text-white.

**[P2] Bookmark CTA uses amber full-button background**
- What: CertDetailPage.tsx:390 — bg-amber-500 when bookmarked. Amber is success/semantic only.
- Fix: Keep button bg-blue-600 always; express state via icon fill + label text only.

**[P2] Nav height inconsistency from sub-labels**
- What: Items with description are taller than items without; active-state border box is different sizes per item.
- Fix: Either pad all items to same height, or move descriptions to Tooltip (cleaner).

**[P2] "운영 중" pulse semantically empty**
- What: HomePage.tsx:189 — always-green pulse; no degraded state exists.
- Fix: Remove or replace with a real dynamic metric (last update time, catalog count).

## Minor Observations

- CertDetailPage.tsx:454: duplicate font-bold text-slate-500 className still not cleaned.
- bookmarkPulse keyframe uses arbitrary animation syntax in JSX; works but bypasses Tailwind purge scan.
- Mobile nav active description inherits blue color from parent — reads as lighter blue, acceptable.
