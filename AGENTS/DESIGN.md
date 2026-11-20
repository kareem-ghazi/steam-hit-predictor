---
name: GameMetrics
colors:
  surface: '#0b1326'
  surface-dim: '#0b1326'
  surface-bright: '#31394d'
  surface-container-lowest: '#060e20'
  surface-container-low: '#131b2e'
  surface-container: '#171f33'
  surface-container-high: '#222a3d'
  surface-container-highest: '#2d3449'
  on-surface: '#dae2fd'
  on-surface-variant: '#bfc8d0'
  inverse-surface: '#dae2fd'
  inverse-on-surface: '#283044'
  outline: '#89929a'
  outline-variant: '#3f484f'
  surface-tint: '#82cfff'
  primary: '#a1d9ff'
  on-primary: '#00344b'
  primary-container: '#66c0f4'
  on-primary-container: '#004d6c'
  inverse-primary: '#00658d'
  secondary: '#accae6'
  on-secondary: '#143349'
  secondary-container: '#2f4c63'
  on-secondary-container: '#9ebcd7'
  tertiary: '#5debae'
  on-tertiary: '#003824'
  tertiary-container: '#39ce94'
  on-tertiary-container: '#005337'
  error: '#ffb4ab'
  on-error: '#690005'
  error-container: '#93000a'
  on-error-container: '#ffdad6'
  primary-fixed: '#c6e7ff'
  primary-fixed-dim: '#82cfff'
  on-primary-fixed: '#001e2d'
  on-primary-fixed-variant: '#004c6b'
  secondary-fixed: '#cce5ff'
  secondary-fixed-dim: '#accae6'
  on-secondary-fixed: '#001e31'
  on-secondary-fixed-variant: '#2d4a61'
  tertiary-fixed: '#6ffbbe'
  tertiary-fixed-dim: '#4edea3'
  on-tertiary-fixed: '#002113'
  on-tertiary-fixed-variant: '#005236'
  background: '#0b1326'
  on-background: '#dae2fd'
  surface-variant: '#2d3449'
typography:
  display-hero:
    fontFamily: Inter
    fontSize: 48px
    fontWeight: '800'
    lineHeight: 56px
    letterSpacing: -0.02em
  display-hero-mobile:
    fontFamily: Inter
    fontSize: 32px
    fontWeight: '800'
    lineHeight: 40px
    letterSpacing: -0.02em
  headline-lg:
    fontFamily: Inter
    fontSize: 32px
    fontWeight: '700'
    lineHeight: 40px
    letterSpacing: -0.015em
  headline-lg-mobile:
    fontFamily: Inter
    fontSize: 24px
    fontWeight: '700'
    lineHeight: 32px
    letterSpacing: -0.01em
  headline-md:
    fontFamily: Inter
    fontSize: 20px
    fontWeight: '600'
    lineHeight: 28px
    letterSpacing: -0.01em
  headline-sm:
    fontFamily: Inter
    fontSize: 16px
    fontWeight: '600'
    lineHeight: 24px
    letterSpacing: 0em
  body-lg:
    fontFamily: Inter
    fontSize: 16px
    fontWeight: '400'
    lineHeight: 26px
    letterSpacing: 0em
  body-md:
    fontFamily: Inter
    fontSize: 14px
    fontWeight: '400'
    lineHeight: 22px
    letterSpacing: 0em
  body-sm:
    fontFamily: Inter
    fontSize: 12px
    fontWeight: '400'
    lineHeight: 18px
    letterSpacing: 0.01em
  label-lg:
    fontFamily: JetBrains Mono
    fontSize: 14px
    fontWeight: '500'
    lineHeight: 20px
    letterSpacing: 0.02em
  label-md:
    fontFamily: JetBrains Mono
    fontSize: 12px
    fontWeight: '500'
    lineHeight: 16px
    letterSpacing: 0.03em
  label-sm:
    fontFamily: JetBrains Mono
    fontSize: 10px
    fontWeight: '500'
    lineHeight: 14px
    letterSpacing: 0.04em
  data-metric:
    fontFamily: JetBrains Mono
    fontSize: 24px
    fontWeight: '700'
    lineHeight: 30px
    letterSpacing: -0.02em
rounded:
  sm: 0.125rem
  DEFAULT: 0.25rem
  md: 0.375rem
  lg: 0.5rem
  xl: 0.75rem
  full: 9999px
spacing:
  gutter: 1.25rem
  gutter-mobile: 0.75rem
  gutter-desktop: 1.5rem
  margin: 1.5rem
  margin-mobile: 1rem
  margin-desktop: 2.5rem
  space-xs: 0.25rem
  space-sm: 0.5rem
  space-md: 1rem
  space-lg: 1.5rem
  space-xl: 2.25rem
---

## Brand & Style

This design system establishes an atmospheric, authoritative, high-density analytics interface tailored for indie studios, AAA publishers, and venture analysts forecasting game commercial viability on Steam. 

### Brand Personality & Emotional Impact
- **Predictive Authority:** Conveys mathematical rigor, telemetry-grade precision, and high confidence through clear numerical metrics and dense visual hierarchy.
- **Steam Ecosystem Native:** Evokes the familiar industrial, dark-mode heritage of desktop gaming clients while elevating it with high-fidelity SaaS craftsmanship.
- **Cinematic Technicality:** Deep oceanic slates combined with luminous cyan accents foster an immersive command-center feel, instilling calm focus during volatile launch cycles.

### Design Movement: Atmospheric Skeuomorphic-Glass Hybrid
The visual language merges modern glassmorphism with Steam-inspired tactile depth:
- Multi-stop linear gradients across cards mimic brushed anodized metal and deep viewport glass.
- Subtly glowing inner borders emulate back-lit CRT/HUD terminals.
- High-contrast telemetry tags and monospaced data readouts contrast against deep background canvases.

## Colors

The palette is engineered for prolonged data analysis sessions under low-light conditions, prioritizing data contrast and optical glow without inducing eye fatigue.

### Palette Architecture
- **Base Canvas (`#0f172a` to `#171a21`):** Deep slate foundation grounding all panels and workspaces.
- **Surfaces & Layers (`#1b2838`, `#1e232d`, `#2a475e`):** Multi-tiered blue-grays providing structural layering without stark white separation lines.
- **Steam Cyan Primary (`#66c0f4`, hover `#4196c6`):** The signature focal accent for primary calls-to-action, active indicators, and high-priority selections.
- **Secondary Atmospheric Blue (`#2a475e`):** Structural anchor used for secondary button fills, passive borders, and table headers.

### Telemetry & Sentiment Semantic Status Tiers
- **Breakout / Overwhelmingly Positive (>=80% Score):** `#10b981` (Emerald Base) / `#4ade80` (Luminous Highlight).
- **Sustained / Very Positive (65–79% Score):** `#38bdf8` (Vibrant Sky Cyan).
- **Mixed / Volatile (45–64% Score):** `#f59e0b` (Amber Glow).
- **Underperforming / Flop / Negative (<45% Score):** `#f43f5e` (Rose Coral).

### Functional Borders & Glows
- Passive dividers: `rgba(51, 65, 85, 0.6)`.
- Interactive hover borders: `rgba(102, 192, 244, 0.35)`.
- Critical metric backlights: soft radial gradients using status hues at 8–12% opacity.

## Typography

The typographic system relies on a dual-engine structure: **Inter** handles narrative clarity, navigation, and executive summaries, while **JetBrains Mono** provides precision-grade legibility for dense technical telemetry, Steam AppIDs, confidence intervals, CCU (concurrent users), and algorithm loss readouts.

### Application Guidelines
- **Metric Values & Telemetry:** All tabular figures, probability distributions, pricing models, and Steam IDs must strictly use `JetBrains Mono` with tabular lining figures enabled (`font-variant-numeric: tabular-nums`).
- **Hierarchy Pairing:** Section titles pair a crisp `Inter` headline (`headline-md`) with an adjacent `label-sm` or `label-md` badge tracking model latency or batch dataset freshness.
- **Optical Balance:** Uppercase tracking for all `label-*` roles must stay between `+0.02em` and `+0.04em` to preserve legibility across low-contrast deep slate panels.

## Layout & Spacing

The layout model is built upon a high-density, 12-column fluid grid optimized for data visualization dashboards, predictive trend graphs, and side-by-side Steam game comparisons.

### Grid & Viewport Scaling
- **Desktop (>= 1280px):** 12 columns, max-width `1600px`, `1.5rem` (`24px`) gutters, and `2.5rem` (`40px`) outer page margins. Panels can span 3, 4, 6, 8, or 12 columns.
- **Tablet (768px – 1279px):** 8 columns, `1.25rem` (`20px`) gutters, `1.5rem` (`24px`) margins. Two-column card grids collapse to single columns or balanced horizontal cards.
- **Mobile (< 768px):** 4 columns, `0.75rem` (`12px`) gutters, `1rem` (`16px`) outer canvas margins. Multi-metric tables transform into swipeable cards or horizontally scrollable data lists.

### Spacing Principles
- Component inner paddings prioritize compact scanability: `space-sm` (`8px`) for telemetry pills and filter chips; `space-md` (`16px`) for standard cards; `space-lg` (`24px`) for primary predictive display hero containers.
- Element gaps use strict multiples of 4px to maintain rigid alignment across charts, radar graphs, and companion leaderboards.

## Elevation & Depth

Visual hierarchy uses a layered, ambient atmospheric model inspired by Steam's client chrome and high-end workstation consoles.

### Elevation Levels

1. **Canvas (Ground Floor - `#0f172a`):** Pure background layer with faint radial gradient accents (`rgba(27, 40, 56, 0.4)`) positioned in screen corners to eliminate optical flatness.
2. **Surface Tier 1 (`#1b2838` with `rgba(51, 65, 85, 0.4)` border):** Base cards, data tables, and passive module containers. No drop shadow; relies strictly on subtle borders and solid backplates.
3. **Surface Tier 2 (Floating Cards & Hover States - `#1e232d`):** Lifted data components, active prediction cards, and flyout menus. Uses a double-layered ambient shadow:
   - Ambient: `0 8px 24px -4px rgba(0, 0, 0, 0.45)`
   - Rim: `0 0 0 1px rgba(102, 192, 244, 0.2)`
4. **Surface Tier 3 (Glass Header & Critical Overlays):** Sticky glass navigation and modal panels. 
   - Background: `rgba(23, 26, 33, 0.75)` with `backdrop-filter: blur(12px) saturate(160%)`.
   - Rim light: Top inner edge highlighted with a 1px border gradient from `rgba(102, 192, 244, 0.35)` fading down to `transparent`.
   - Shadow: `0 16px 36px -8px rgba(0, 0, 0, 0.75)`.

### Depth Atmosphere
Never use harsh, opaque solid drop shadows. All elevation relies on tinted deep-slate blurs combined with micro-contrast edge highlights to simulate rear-illuminated display glass.

## Shapes

The design system employs a **Soft** shape language (`roundedness: 1`), giving UI components a calibrated, industrial-software geometry that avoids aggressive brutalist sharp edges while resisting consumer-app pill bubbles.

### Radius Scale
- **Base Components (`4px` / `0.25rem`):** Checkboxes, code chips, tags, dropdown triggers, and data grid rows.
- **Cards & Modals (`8px` / `0.5rem` - `rounded-lg`):** Telemetry containers, AI forecast modules, and Steam game hero previews.
- **Special Elements (`12px` / `0.75rem` - `rounded-xl`):** Flyout tooltips, multi-tab segmented containers, and modal dialog bodies.
- **Pill Exceptions (`9999px`):** Status indicator dots, segmented filter switches, and telemetry status badges (e.g., "Overwhelmingly Positive" badge tags) to preserve instant visual classification across dense tables.

## Components

### Buttons
- **Primary Action (Predict / Benchmark / Export):** Background gradient transitioning from `#4196c6` to `#66c0f4`, white text (`Inter` SemiBold), subtle top inner highlight `rgba(255, 255, 255, 0.25)`, hover glow `0 0 16px rgba(102, 192, 244, 0.4)`. Corner radius `4px`.
- **Secondary Action (Steam Sync / Filter Reset):** Deep blue-gray fill (`#2a475e`), 1px border in `rgba(102, 192, 244, 0.25)`, text in `#66c0f4`. On hover, surface shifts to `#335875` with crisp border brightening.
- **Ghost / Utility:** Transparent fill with `rgba(255, 255, 255, 0.7)` text; hover reveals `rgba(255, 255, 255, 0.06)` background.

### Segmented Tabs & Navigators
- Framed in a single continuous container (`#171a21`) with `rounded-xl` borders and `4px` internal padding.
- Active segment: `#2a475e` backplate, illuminated cyan text, and a micro 2px Steam Cyan accent bar anchored at the bottom edge.
- Inactive segments: Text in `#94a3b8`, transitions to `#e2e8f0` upon hover.

### ML Hit-Prediction Cards
- Composed of three functional zones:
  1. **Header:** Title, AppID in `JetBrains Mono` (`label-sm`), publisher tag, and release timeframe.
  2. **Predictive Gauge:** Multi-point radar graph or a segmented score bar featuring semantic color fill (e.g., `#10b981` for Hit probability >=80%).
  3. **Telemetry Matrix:** 3-column micro-grid displaying Steam CCU projections, wishlist velocity multiplier, and model confidence rating with standard deviation.
- Background uses a subtle diagonal surface gradient from `#1b2838` to `#171a21` with a `1px` border of `rgba(51, 65, 85, 0.5)`.

### Telemetry Metric Pills & Badges
- Compact capsules (`rounded-full`) with a `space-xs` vertical by `space-sm` horizontal padding.
- Monospaced numerical values accompanied by a pulsating `6px` status dot indicating realtime Steam pipeline sync.
- Status variants map directly to sentiment colors:
  - *Overwhelmingly Positive:* Text `#4ade80`, background `rgba(16, 185, 129, 0.12)`, border `rgba(74, 222, 128, 0.3)`.
  - *Mixed:* Text `#f59e0b`, background `rgba(245, 158, 11, 0.12)`, border `rgba(245, 158, 11, 0.3)`.
  - *Negative:* Text `#f43f5e`, background `rgba(244, 63, 94, 0.12)`, border `rgba(244, 63, 94, 0.3)`.

### Form Controls & Inputs
- **Search & Filter Inputs:** Fill `#101726` with `1px` border `#334155`. Inactive placeholder `#64748b`.
- **Focus State:** Border transitions to `#66c0f4` paired with an external `0 0 0 3px rgba(102, 192, 244, 0.18)` glow ring.
- **Checkboxes & Radios:** `16px` squared with `4px` radius; checked state fills with `#66c0f4` bearing a stark `#0f172a` check glyph.

### Steam-Inspired Glass Header
- Height fixed at `64px`, sticky top edge with `z-index: 50`.
- Integrated global Steam AppID quick-switcher, algorithmic dataset sync indicator, and user studio quota meter.
- Gradient border underline blending from `transparent` through `rgba(102, 192, 244, 0.3)` to `transparent`.