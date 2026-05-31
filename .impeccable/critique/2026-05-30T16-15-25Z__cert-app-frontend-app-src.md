---
target: CertFinder src (all pages)
total_score: 28
p0_count: 0
p1_count: 2
timestamp: 2026-05-30T16-15-25Z
slug: cert-app-frontend-app-src
---
## Design Health Score

| # | Heuristic | Score | Key Issue |
|---|-----------|-------|-----------|
| 1 | Visibility of System Status | 3 | Trending lazy-load indicator buried; cycling AI messages and skeleton states excellent |
| 2 | Match System / Real World | 3 | "RAG 기반", "하이브리드 AI" unexplained; English tier names in Korean product |
| 3 | User Control and Freedom | 3 | No cancel during AI processing; no undo link on bookmark toast |
| 4 | Consistency and Standards | 3 | JobListPage auto-searches on type; CertListPage requires explicit submit |
| 5 | Error Prevention | 3 | Good overall; AI textarea has no length guidance |
| 6 | Recognition Rather Than Recall | 3 | Mobile filter collapsed — active state invisible without tapping |
| 7 | Flexibility and Efficiency | 3 | Good: ⌘K, URL-synced state, profile autofill; missing: bulk bookmark actions |
| 8 | Aesthetic and Minimalist Design | 2 | Filter panel: 14+ simultaneous choices; AI result cards: 3 competing bars per card |
| 9 | Error Recovery | 3 | Specific messages with retry; AI submit error lacks actionable guidance |
| 10 | Help and Documentation | 2 | "AI 적합도 X%" unexplained; RAG undefined; NCS 분류 no tooltip; tier undocumented |
| **Total** | | **28/40** | **Good** |

## Anti-Patterns Verdict

Not AI-generated on first read. The numbered spec-sheet layout, data-authority pass rate treatment, and Korean-first legibility distinguish it. Two borderline concerns: (1) Homepage right-panel hero metric template (big number + 2 supporting stats); (2) uniform AI result card architecture across 15 cards.

Manual scan (detector not bundled) found: `bg-black/60` on JobListPage input (pure black, violates color system); `shadow-xl shadow-blue-900/30` on JobList search button at rest (non-hero resting shadow); search auto-search vs explicit-submit inconsistency between JobListPage and CertListPage.

## Priority Issues

**[P1] Login wall fires at peak engagement on AI recommendation page**
After investing effort filling in major + career goal, guests see blurred/truncated results. This interrupts at the emotional peak.
Fix: Show 3 complete results to guests. Position inline CTA after card 3: "나머지 12개 결과를 보려면 회원가입 (무료)." Convert from gate to gain.
Command: /impeccable craft AI-recommendation guest-to-auth conversion

**[P1] CertListPage filter panel — 14+ simultaneous decision points**
Filter panel shows all 14+ choices at once (search types, field select, sort, type chips, favorites, pass-rate filter, acquired filter). Working Memory cap is 4; at 14+ users hit analysis paralysis.
Fix: Collapse to primary: search input + type chips. Move field, sort, pass-rate, acquired behind "필터 더 보기" toggle. Show active count badge.
Command: /impeccable distill CertListPage filter panel

**[P2] CertDetail page has no forward path**
After consuming 4 tabs of data, users have no contextual next action. Page ends at job data + footer.
Fix: Add 3-item next-steps section: AI roadmap (pre-filled), similar certs (filtered), related jobs.
Command: /impeccable craft cert-detail next-steps footer

**[P2] AI result cards — three bars compete, primary score buried**
Three equal-weight progress bars (합격률, 전공 연관성, 관심도 일치) per card. AI 적합도 % is a small badge, not the hero number.
Fix: Promote AI 적합도 as single large number per card. Collapse component bars to expandable "점수 구성 보기."
Command: /impeccable distill AiRecommendationPage result-cards

**[P2] Two recommendation pathways confuse first-time users**
"전공 추천" vs "AI 추천" — distinction not clear from nav labels. First-timers may pick at random.
Fix: Rename to "학과별 DB 추천" and "AI 커리어 추천." Surface sub-descriptions in mobile nav.
Command: /impeccable clarify navigation recommendation labels

## Persona Red Flags

**Jordan (Confused First-Timer)**: Hits 14+ filter options with no start hint. Reads cert detail page but doesn't know which tab answers "is this hard." After reading CertDetail, no forward path — uses browser back. Reaches AI page, types major + goal, gets partial results behind login gate — leaves without registering.

**Casey (Distracted Mobile User)**: Mobile filter accordion collapsed — can't see active filters. Search type toggle hidden inside accordion. Sticky tab bar on CertDetail stacks with sticky header — reduces content area on mobile. Compare tray + cookie banner both appear at the bottom on first visit — two stacked overlapping banners.

**지혜 (Project-Specific: Engineering Junior)**: Grammar concatenation bug "전기기사은(는)" erodes trust. "AI 적합도 67%" has no reference scale. Trending certs are not major-specific — can't see what EE peers are taking.

## Minor Observations

- Grammar bug in certOverview: `${cert.qual_name}은(는)` always appends without consonant check — "전기기사은(는)" → should be "전기기사는"
- JobListPage input uses `bg-black/60` (pure black) instead of system `bg-slate-900`
- JobListPage search button has `shadow-xl shadow-blue-900/30` at rest — non-hero resting shadow
- Share button on CertDetail has no state change during async clipboard copy
- Nav sub-descriptions (DB 매핑, 커리어 목표) visible on desktop only — not shown in mobile nav
- CertDetail sticky tab bar `top-[88px]` may misalign on mobile browser chrome variations

## Questions to Consider

1. The two recommendation pathways represent different product philosophies. Is the current distinction intentional enough for users to self-select correctly, or should one be the entry point?
2. What if a new visitor's first experience was a three-question guided filter instead of 14 simultaneous choices? Would that damage returning-user experience, or could both coexist?
3. CertDetail tells users everything about the cert but nothing about how to start. A single "이 자격증 준비 로드맵 만들기" CTA might be the most valuable sentence on the page.
