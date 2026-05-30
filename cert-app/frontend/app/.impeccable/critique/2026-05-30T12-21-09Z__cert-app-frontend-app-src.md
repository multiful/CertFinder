---
target: CertFinder src — post-bolder-harden-adapt-polish
total_score: 32
p0_count: 0
p1_count: 2
p2_count: 3
p3_count: 2
timestamp: 2026-05-30T12-21-09Z
slug: cert-app-frontend-app-src
---
## Design Health Score

| # | Heuristic | Score | Key Finding |
|---|-----------|-------|-------------|
| 1 | Visibility of System Status | 4 | 스켈레톤, aria-live, 토스트, 비교 트레이 slide-up 완비 |
| 2 | Match System / Real World | 3 | "NCS 분류" 미설명; 나머지 도메인 용어 자연스러움 |
| 3 | User Control and Freedom | 3 | 필터 칩 개별 X, 비교 초기화; 북마크 undo 없음 |
| 4 | Consistency and Standards | 4 | ring-[3px] 통일 완료, signal blue 단일 액센트 유지 |
| 5 | Error Prevention | 3 | 비교 3개 제한 + disabled, 북마크 낙관적 업데이트+롤백 |
| 6 | Recognition Rather Than Recall | 3 | Scale 아이콘 label 없음; 필터 상태 요약 칩 좋음 |
| 7 | Flexibility and Efficiency | 2 | Cmd+K만 있음, 나머지 단축키 없음, list view preference 미저장 |
| 8 | Aesthetic and Minimalist Design | 4 | 피처 섹션 재설계, 장식 없음, 계층 명확 |
| 9 | Error Recovery | 3 | 네트워크 에러 재시도, 구체적 토스트, 비교 empty state 명확 |
| 10 | Help and Documentation | 3 | nav 서브라벨, 트렌딩 상대 지표 툴팁, NCS 미설명 |
| **Total** | | **32/40** | **Good** |

## Anti-Patterns Verdict

대체로 통과. 잔존 위험:
1. "서비스 커버리지" hero 카드 — hero-metric template 경계선
2. Guide 섹션 — CheckCircle + h3 + p 동일 구조 4개

## Priority Issues

**[P1] Scale 아이콘 레이블 없음 — 비교 진입점 불명확**
- Location: CertListPage.tsx 그리드 카드, 리스트 뷰
- Fix: hover tooltip 또는 항상 노출 텍스트 레이블 "비교"

**[P1] NCS 분류 미설명**
- Location: CertComparePage.tsx 직무 분류 행, CertDetailPage
- Fix: info 아이콘 + hover tooltip으로 NCS 설명

**[P2] 홈 피처 오른쪽 "서비스 커버리지" 카드 — hero metric 잔존**
- Location: HomePage.tsx:184-229
- Fix: 단일 stat 또는 텍스트로 흡수

**[P2] 필터 패널 모바일 전개 — 항상 노출**
- Location: CertListPage.tsx:367
- Fix: md 미만에서 접힌 "필터 열기" 버튼

**[P2] 리스트 뷰 선호 미저장**
- Location: CertListPage.tsx:111
- Fix: localStorage로 viewMode 저장

## Persona Red Flags

**Jordan**: Scale 아이콘 목적 불명확, NCS 이해 불가, 비교 선택 유지 불확실

**Sam**: 피처 번호 "01" aria-hidden 없음, article+role=button 중첩

**Jun (수험생)**: 비교 버튼 32px 탭 실수, 필터 패널 화면 점유, sessionStorage 탭 닫힘 초기화

## Minor Observations

- article[role=list] 안 article[aria-label]+Card[role=button] 의미론적 혼재
- CTA 섹션 white text on blue-600: 4.5:1 AA borderline
- 피처 섹션 "RAG 기반" Jordan에게 불명확
