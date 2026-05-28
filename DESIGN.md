---
name: CertFinder
description: Korea's national certification discovery and AI career path recommendation platform
colors:
  surface-base: "#020617"
  surface-raised: "#0f172a"
  surface-elevated: "#1e293b"
  surface-subtle: "#334155"
  signal-blue: "#2563eb"
  signal-blue-bright: "#3b82f6"
  signal-blue-text: "#60a5fa"
  accent-indigo: "#818cf8"
  success-emerald: "#34d399"
  text-primary: "#f1f5f9"
  text-secondary: "#94a3b8"
  text-muted: "#64748b"
  border-default: "#1e293b"
  border-subtle: "#0f172a"
typography:
  display:
    fontFamily: "'Outfit', 'Noto Sans KR', sans-serif"
    fontSize: "clamp(3rem, 7vw, 4.5rem)"
    fontWeight: 800
    lineHeight: 1.1
    letterSpacing: "-0.03em"
  headline:
    fontFamily: "'Outfit', 'Noto Sans KR', sans-serif"
    fontSize: "clamp(1.875rem, 4vw, 2.25rem)"
    fontWeight: 700
    lineHeight: 1.2
    letterSpacing: "-0.01em"
  title:
    fontFamily: "'Outfit', 'Noto Sans KR', sans-serif"
    fontSize: "1.25rem"
    fontWeight: 700
    lineHeight: 1.3
    letterSpacing: "normal"
  body:
    fontFamily: "'Outfit', 'Noto Sans KR', sans-serif"
    fontSize: "0.9375rem"
    fontWeight: 400
    lineHeight: 1.625
    letterSpacing: "normal"
  label:
    fontFamily: "'Outfit', 'Noto Sans KR', sans-serif"
    fontSize: "0.75rem"
    fontWeight: 700
    lineHeight: 1.4
    letterSpacing: "0.05em"
rounded:
  sm: "6px"
  md: "8px"
  lg: "10px"
  xl: "12px"
  2xl: "16px"
  3xl: "24px"
  full: "9999px"
spacing:
  xs: "4px"
  sm: "8px"
  md: "16px"
  lg: "24px"
  xl: "32px"
  2xl: "48px"
  3xl: "64px"
components:
  button-primary:
    backgroundColor: "{colors.signal-blue}"
    textColor: "{colors.text-primary}"
    rounded: "{rounded.xl}"
    padding: "0 32px"
    height: "36px"
  button-primary-hover:
    backgroundColor: "#1d4ed8"
    textColor: "{colors.text-primary}"
    rounded: "{rounded.xl}"
    padding: "0 32px"
  button-ghost:
    backgroundColor: "transparent"
    textColor: "{colors.text-secondary}"
    rounded: "{rounded.lg}"
    padding: "8px 16px"
  button-ghost-hover:
    backgroundColor: "{colors.surface-elevated}"
    textColor: "{colors.text-primary}"
    rounded: "{rounded.lg}"
    padding: "8px 16px"
  nav-item-default:
    backgroundColor: "transparent"
    textColor: "{colors.text-secondary}"
    rounded: "{rounded.lg}"
    padding: "8px 16px"
  nav-item-active:
    backgroundColor: "#1d4ed81a"
    textColor: "{colors.signal-blue-text}"
    rounded: "{rounded.lg}"
    padding: "8px 16px"
  input-search:
    backgroundColor: "{colors.surface-raised}"
    textColor: "{colors.text-primary}"
    rounded: "{rounded.xl}"
    padding: "0 16px 0 48px"
    height: "56px"
  card-feature:
    backgroundColor: "#0f172a66"
    textColor: "{colors.text-primary}"
    rounded: "{rounded.xl}"
    padding: "32px"
  badge-blue:
    backgroundColor: "#2563eb1a"
    textColor: "{colors.signal-blue-text}"
    rounded: "{rounded.full}"
    padding: "4px 16px"
---

# Design System: CertFinder

## 1. Overview

**Creative North Star: "The Precision Instrument"**

CertFinder is a tool for people who need to make consequential decisions about their career. The design reflects that weight. Like a high-end analytical instrument, every element is calibrated to function; decoration is a signal failure. The surface is dark not for aesthetic coolness but because this is a focused workspace, and focused work happens in low ambient distraction. Blue appears once, as a signal color, exactly as a gauge needle moves into a specific zone.

The system is built on a single restraint: one accent, used for one purpose. The electric blue (`#2563eb` / `#3b82f6`) is reserved for actionable elements and active states. When it appears, the user's attention follows. Its rarity is not a design quirk; it is the entire point. Neutral mid-tones carry the data. Typography does the hierarchy. The blue fires only when the user needs to act.

What this system explicitly rejects: the cluttered portal aesthetic of Q-Net and HRD Korea; the generic SaaS dashboard feel of shadcn defaults left unmodified; gamification excess that reduces a professional tool to a language-learning app; neon and glassmorphism that look distinctive but communicate nothing. The goal is a premium professional tool that does not need to announce itself as premium. It earns that read through restraint, density control, and typographic authority.

**Key Characteristics:**
- Dark, near-black base (slate-950) with cool blue-indigo tint in the neutrals
- Single signal color (electric blue) with strict surface coverage discipline
- Dual-font system: Outfit for Latin display power, Noto Sans KR for Korean density
- Tight radius vocabulary: components lean geometric, not soft
- Data surfaces are content; chrome is minimal
- Flat by default, elevation only on hover and focus states

## 2. Colors: The Signal Palette

One surface family, one signal color, one success state. Nothing is named by hue; everything is named by role.

### Primary
- **Signal Blue** (`#2563eb` / `#3b82f6` for text/glows): The sole action color. CTA buttons, active navigation states, focus rings, interactive progress indicators. Never used decoratively. On any given screen, Signal Blue should cover no more than 10% of visible surface area. Its appearance means: act here.
- **Signal Blue Text** (`#60a5fa`): The legible form of the accent, used on dark surfaces for text labels, active nav items, inline emphasis. Lighter than the action blue; distinguishes labeled state from interactive affordance.

### Secondary
- **Accent Indigo** (`#818cf8`): Secondary depth in gradient contexts (logo wordmark, AI-feature icons). Not used as a standalone action color. Appears only alongside Signal Blue in multi-stop gradients. Prohibited as a solo accent.

### Neutral
- **Surface Base** (`#020617`): Page background. The deepest layer. Cool-tinted near-black, not true black.
- **Surface Raised** (`#0f172a`): Card and panel backgrounds. Primary content containers.
- **Surface Elevated** (`#1e293b`): Hover surfaces, input fills, active overlays. The visible step above Raised.
- **Surface Subtle** (`#334155`): Dividers, disabled states, subtle border accents.
- **Text Primary** (`#f1f5f9`): Headings, prominent labels, all high-emphasis content.
- **Text Secondary** (`#94a3b8`): Body copy, descriptions, secondary metadata. The most-used text role.
- **Text Muted** (`#64748b`): Timestamps, helper text, least-critical information. Never used for interactive labels.
- **Border Default** (`#1e293b`): Standard card/panel borders. Matches Surface Elevated for optical consistency.

### Tertiary
- **Success Emerald** (`#34d399`): Confirmation states, checkmarks, acquisition badges. One role only. Not a design accent.

### Named Rules

**The Signal Rule.** Signal Blue is the only accent color. Indigo lives only inside gradients. Emerald lives only in success states. Purple appears only in the hero headline gradient and nowhere else. If you're about to apply color that doesn't fall into one of these four functions, the answer is a neutral.

**The No True Black Rule.** Surface Base is `#020617`, not `#000000`. Every neutral carries the cool blue-indigo tint of the palette. Pure black and pure white are prohibited; they read as errors in this system.

## 3. Typography

**Display Font:** Outfit (Google Fonts, weights 400–800)
**Korean Body Font:** Noto Sans KR (Google Fonts, weights 300, 400, 500, 700, 900)
**Stack:** `'Outfit', 'Noto Sans KR', sans-serif` — unified across all roles

**Character:** Outfit brings geometric precision and contemporary weight contrast to Latin text; Noto Sans KR provides the stroke density required for Korean characters at small sizes. Together they read as a single deliberate system rather than two fonts in conflict. The pairing rewards the extrabold weights: weight contrast is the primary hierarchy signal, not size alone.

### Hierarchy

- **Display** (800 weight, `clamp(3rem, 7vw, 4.5rem)`, line-height 1.1, tracking −0.03em): Hero headlines only. One per page. Korean and Latin both work at this scale; let the content determine line breaks, never force them.
- **Headline** (700 weight, `clamp(1.875rem, 4vw, 2.25rem)`, line-height 1.2, tracking −0.01em): Section titles, page headings. Tight tracking keeps authority; do not loosen.
- **Title** (700 weight, `1.25rem` / 20px, line-height 1.3): Card headings, list item titles, sidebar section labels.
- **Body** (400 weight, `0.9375rem` / 15px base, line-height 1.625): All reading copy. Maximum line length 65ch on wide layouts. Korean body uses 500 weight at the same size for optical equivalence with Latin 400.
- **Label** (700 weight, `0.75rem` / 12px, tracking 0.05em, often uppercase for English): Badges, tags, micro-labels, nav items, data column headers. Never apply label tracking to Korean; it breaks syllable spacing.

### Named Rules

**The Korean Density Rule.** Korean characters are visually heavier than Latin at the same weight. When mixing Korean and Latin in the same string, let Outfit render the Latin and Noto Sans KR handle Korean; do not force one font to cover both. At body size, use `font-weight: 500` for Korean-dominant paragraphs to maintain visual parity with Latin 400.

**The Weight-First Rule.** Hierarchy is weight before size. Two items at the same size but different weights (700 vs 400) read as different levels. Do not create hierarchy by size alone; always pair a size step with a weight step.

## 4. Elevation

This system is flat by default. Surfaces are differentiated by background tone (`surface-base` → `surface-raised` → `surface-elevated`), not by shadow depth. Shadows are a response to state (hover, focus, lift), not a baseline property. A card at rest has a border and a slightly lighter background than the page; it does not have a drop shadow.

The one exception: the primary CTA button on the hero carries `box-shadow: 0 10px 30px -10px rgba(37, 99, 235, 0.3)` — a blue-tinted ambient shadow that visually anchors the most important action on the page. This is the only shadow allowed at rest.

### Shadow Vocabulary

- **Hover Lift** (`0 20px 40px -20px rgba(59, 130, 246, 0.3)`): Applied to interactive cards on hover alongside `translateY(-8px) scale(1.02)`. The blue tint confirms the hover is in the Signal Blue domain.
- **CTA Anchor** (`0 10px 30px -10px rgba(37, 99, 235, 0.3)`): On the primary button at rest in the hero. Unique exception; do not replicate on other buttons.
- **Modal / Overlay** (`0 25px 60px -10px rgba(0, 0, 0, 0.5)`): Full-screen overlays and drawers only.

### Named Rules

**The Flat-By-Default Rule.** If a shadow exists at rest, it is wrong. Tonal layering (surface-raised on surface-base) is the resting elevation system. Shadows appear only as a response to interaction: hover on cards, focus on inputs, lift on modals.

## 5. Components

### Buttons

Buttons are geometric and direct. Primary buttons are action anchors, not decorative elements.

- **Shape:** Rounded corners (`rounded-xl`, 12px). Not pill-shaped; not sharp. The 12px radius reads as intentional without being soft.
- **Primary** (`bg-[#2563eb]`, white text, `px-8 h-9` default / `px-8 h-14` for hero CTAs): Signal Blue fill, white label. The most visible interactive element. Used for navigation-forward actions and form submissions.
- **Primary hover:** `bg-[#1d4ed8]` (blue-700), `scale(1.02)` on hero variant.
- **Focus visible:** `ring-[3px] ring-blue-500/50` — visible ring, never invisible.
- **Ghost:** Transparent background, `text-slate-400`, `hover:text-white hover:bg-slate-800/50`. Used for secondary actions and navigation within sections.
- **Destructive:** `bg-destructive` (deep red). Reserved strictly for irreversible delete actions.
- **Disabled:** `opacity-50 pointer-events-none`. No style variation; opacity alone signals unavailability.

### Chips / Badges

- **Blue badge:** `bg-blue-600/10 border border-blue-500/30 text-blue-400`, `rounded-full`, `px-4 py-1`, label weight (700, 12px, tracking 0.05em). Used for section labels like "Core Modules", "Recent Trends".
- **Indigo badge:** Same structure, `border-indigo-500/30 text-indigo-400`. Used for AI-adjacent labels.
- **State badge (acquisition):** `bg-emerald-500/10 text-emerald-400 border-emerald-500/20`. Marks already-acquired certifications in card overlays.

### Cards / Containers

- **Corner style:** `rounded-xl` (12px) standard. `rounded-2xl` (16px) for larger feature cards and modals.
- **Background:** `bg-slate-900/40` at rest; `bg-slate-900/80` on hover. The opacity transition provides a tangible hover state without color change.
- **Shadow strategy:** None at rest. Hover lift shadow (see Elevation section) is triggered by `.card-hover-effect`.
- **Border:** `border border-slate-800` standard. On hover, the border color shifts to the card's accent color at 50% opacity (e.g. `hover:border-blue-500/50`).
- **Internal padding:** `p-8` (32px) for feature cards. `p-6` (24px) for data cards. `p-4` (16px) for compact list items.
- **Nested panels:** Inner panels within cards use `bg-slate-900/50 rounded-xl border border-slate-800`. No deeper nesting.

### Inputs / Fields

- **Default style:** `bg-slate-900 border border-slate-800 rounded-xl text-white placeholder:text-slate-500`. The border matches the card border system; inputs read as part of the same surface family.
- **Hero search:** `h-14` (56px), `pl-12` for icon offset, `pr-4`. Wraps in a gradient-glow container on hover (`from-blue-600 to-indigo-600 blur opacity-25 → opacity-50`).
- **Standard input:** `h-9` (36px), `rounded-md` (8px), `border-input`. Used inside forms.
- **Focus:** `ring-[3px] ring-blue-500/50 border-transparent`. The ring replaces the border; do not show both.
- **Error state:** `border-destructive ring-destructive/20`. Red ring, not blue.
- **Disabled:** `opacity-50 cursor-not-allowed`.

### Navigation

- **Header:** Sticky, `bg-slate-950/80 backdrop-blur-md border-b border-slate-800`. Height `h-16` (64px). The blur creates separation from content without a hard shadow.
- **Nav items (default):** `text-slate-400 hover:text-white hover:bg-slate-800/50 rounded-lg px-4 py-2 text-sm font-bold`. Ghost style; minimal footprint.
- **Nav items (active):** `bg-blue-600/10 text-blue-400 border border-blue-500/20`. The tinted background + border combination makes the active state unambiguous without requiring a colored underline or left stripe.
- **Mobile nav:** Full-width, `bg-slate-950`, same item structure. Animates in with `slide-in-from-top duration-200`.
- **Logo:** Wordmark in gradient `from-white to-slate-400`, `font-black tracking-tighter`. The gradient keeps the logo premium without introducing a new color. Logo icon has a subtle `from-blue-600 to-indigo-600` blur glow on hover.

### Signature Component: AI Recommendation Badge

The inline badge on AI-powered features uses `bg-blue-500/10 border border-blue-500/20 rounded-full px-3 py-1 text-blue-400 text-sm font-medium` with a `<Sparkles>` icon at 16px. This is the only component that may use an icon alongside a label at this size. It signals "AI-powered" without announcing it loudly.

## 6. Do's and Don'ts

### Do:
- **Do** use `#2563eb` as the only action accent. One signal color. Any screen with two different accent colors has failed this rule.
- **Do** keep cards flat at rest (`border border-slate-800`, no shadow). Trigger the hover lift only on interactive cards with cursor pointer.
- **Do** use weight contrast (700 vs 400) as the primary hierarchy signal before adjusting size.
- **Do** set Korean body text at `font-weight: 500` when in paragraphs; Latin body uses 400. The optical weight must match.
- **Do** constrain body copy to 65ch maximum line length on wide layouts.
- **Do** use `rounded-xl` (12px) for cards and primary buttons; use `rounded-full` for badges and pills only.
- **Do** show the focus ring at full opacity: `ring-[3px] ring-blue-500/50`. Never hide it or reduce opacity below 50%.
- **Do** use tonal layering (surface-base → surface-raised → surface-elevated) as the resting elevation system.

### Don't:
- **Don't** use gradient text (`background-clip: text` + gradient). The hero headline gradient was a prior implementation choice; it is prohibited going forward. Use solid `text-white` or `text-blue-400` instead.
- **Don't** use glassmorphism as a default card style. The `.glass` utility (`backdrop-blur + rgba background`) is available only for the header and contextual overlays. Never on regular content cards.
- **Don't** use side-stripe borders (`border-left` or `border-right` greater than 1px as a colored accent). This is the government-portal anti-pattern. Use full borders, tinted backgrounds, or nothing.
- **Don't** make the UI look like a generic SaaS dashboard: shadcn defaults unmodified, cream/white backgrounds, identical icon+heading+text card grids. Every component must be deliberately configured against those defaults.
- **Don't** make it look like a government portal (Q-Net, HRD Korea): no dense tables without spacing, no low-contrast gray-on-gray text, no 2000s-era layout density.
- **Don't** add gamification patterns that compete with data: no conspicuous XP bars in primary views, no animated confetti, no badge grids as primary UI.
- **Don't** use neon borders, glowing outlines, or heavy glassmorphism that pushes toward crypto/cyberpunk aesthetics.
- **Don't** put shadows on cards at rest. If a card has a shadow at rest, remove it.
- **Don't** use `#000000` or `#ffffff` directly. Every neutral must carry the cool blue-indigo tint of the palette.
- **Don't** use label tracking (`letter-spacing: 0.05em`) on Korean text. It breaks syllable grouping and reads as a mistake.
