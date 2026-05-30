---
target: CertFinder src — post-harden-distill-optimize-extract-polish
total_score: 33
p0_count: 0
p1_count: 0
p2_count: 3
p3_count: 2
timestamp: 2026-05-30T13-03-41Z
slug: cert-app-frontend-app-src
---
## Design Health Score

| # | Heuristic | Score | Key Finding |
|---|-----------|-------|-------------|
| 1 | Visibility of System Status | 4 | 로딩/토스트/비교 트레이/aria-live 전 페이지 일관 |
| 2 | Match System / Real World | 3 | 히어로 가짜 메트릭 제거; "연도별" 여전히 약간 추상적 |
| 3 | User Control and Freedom | 3 | viewMode 저장됨; 북마크 undo 없음 |
| 4 | Consistency and Standards | 4 | .focus-ring 단일 소스; 번호형 레이아웃 통일 |
| 5 | Error Prevention | 3 | 비교 3개 제한, 낙관적 업데이트+롤백 |
| 6 | Recognition Rather Than Recall | 3 | "비교"/"비교중" 텍스트 레이블; NCS ? 툴팁 |
| 7 | Flexibility and Efficiency | 3 | viewMode 저장 ↑; Cmd+K 유일한 단축키 |
| 8 | Aesthetic and Minimalist Design | 4 | 히어로 카드 정제; blur 제거; Guide 번호형 |
| 9 | Error Recovery | 3 | 전 페이지 재시도 버튼; 구체적 에러 메시지 |
| 10 | Help and Documentation | 3 | NCS 툴팁; nav 서브라벨; 비교 설명 텍스트 |
| **Total** | | **33/40** | **Good — approaching Excellent** |

## Anti-Patterns Verdict

통과. Hero metric: 해소. 동일 카드 그리드: 해소. blur 제거.
잔존: CTA 섹션 blue gradient (기능적, 가장 generic한 요소).

## Priority Issues

**[P2] "연도별" — 히어로 stat 값이 서술어**
- Location: HomePage.tsx hero 우측 stat
- Fix: 실제 연도 범위 또는 삭제

**[P2] sessionStorage → localStorage 전환**
- Location: CompareContext.tsx COMPARE_KEY
- Fix: 탭 닫힘 후에도 비교 선택 유지

**[P2] Guide 섹션 gap 과다**
- Location: HomePage.tsx guide grid gap-x-16 + pr-16 = 128px
- Fix: gap-x-0 또는 pr-8

**[P3] 100개 즐겨찾기 미가상화**
**[P3] CTA 섹션 generic gradient**

## Persona Red Flags

**Jordan**: "연도별" stat 불명확
**Sam**: role="button" on article 중복 (pre-existing)
**Jun**: sessionStorage 비교 탭 닫힘 손실

## Minor Observations

- "연도별"과 "450+" 동일 weight이지만 성격 다름
- .pass-high/.pass-mid/.pass-low 유틸리티 미사용 상태
- viewMode useState 선언 위치 최상단으로 정리 권장
