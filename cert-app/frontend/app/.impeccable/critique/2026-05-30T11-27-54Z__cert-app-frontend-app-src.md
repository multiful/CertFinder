---
target: CertFinder src — final polish run
total_score: 32
p0_count: 0
p1_count: 0
p2_count: 2
p3_count: 2
timestamp: 2026-05-30T11-27-54Z
slug: cert-app-frontend-app-src
---
## Design Health Score

| # | Heuristic | Score | Key Issue |
|---|-----------|-------|-----------|
| 1 | Visibility of System Status | 4 | Rotating loading messages excellent; live cert count replaces ambiguous pulse; persistent error state |
| 2 | Match Between System / Real World | 3 | "복합 AI 검색" replaces "하이브리드 RAG"; "NCS 분류" still unexplained |
| 3 | User Control and Freedom | 3 | Jordan rescue; persistent error; job rescue; no active-filter summary |
| 4 | Consistency and Standards | 4 | Gradient text eliminated everywhere; amber CTA fixed; nav height consistent; duplicates cleaned |
| 5 | Error Prevention | 3 | textarea maxLength=500 + char count; still post-submit validation for major |
| 6 | Recognition Rather Than Recall | 3 | Sort labeled; nav sub-labels; AI 적합도 tooltip |
| 7 | Flexibility and Efficiency | 2 | No keyboard shortcuts; URL sync undiscoverable |
| 8 | Aesthetic and Minimalist Design | 4 | All noise removed; delight additions purposeful |
| 9 | Error Recovery | 3 | Persistent AI rec error; honest job empty + AI link; Jordan path |
| 10 | Help and Documentation | 3 | AI 적합도 tooltip; textarea char hint; nav sub-labels; rotating messages teach system |
| **Total** | | **32/40** | **Good — approaching excellent** |

## Anti-Patterns Verdict

Clean pass. Gradient text = 0 (verified scan). Floating decorative animations = 0. Theater badges = 0. Redundant copy = 0. Amber CTA = 0. MyPage tier emoji is deliberate/contained. No AI tells on primary surfaces.

## What's Working

1. AI loading (rotating 5-step messages) is product-specific and solves the 5-25s wait.
2. Keyboard accessibility is now consistent: role=button + tabIndex + onKeyDown + focus-visible ring across all clickable non-button elements. Major dropdown is WCAG 2.2 ARIA 1.2 combobox compliant.
3. Color system is internally consistent: signal blue in one role only; no amber CTAs, no gradient text, no second accent.

## Remaining Priority Issues

**[P2] "NCS 분류" unexplained domain term — CertDetailPage info tab**
Fix: Replace with "직무 분류 (NCS)" + title tooltip explaining NCS.

**[P2] No keyboard shortcut for search — Layout.tsx**
Fix: Cmd+K / Ctrl+K in Layout useEffect that focuses search input or navigates to /certs.

**[P3] No active-filter summary in cert list**
Fix: Chip row above results showing active filters with individual close buttons.

**[P3] No recent/saved search history**
Fix: Persist last N queries in localStorage, surface in search dropdown.

## Positive findings

- transition-[opacity,transform] on ArrowRight: precise and correct.
- focus-visible ring pattern is uniform across all newly accessible elements.
- prefers-reduced-motion covers card-hover-effect and animate-in.
- MyPage blur: transition-colors duration-700 — still large blur but no longer animating all properties.
