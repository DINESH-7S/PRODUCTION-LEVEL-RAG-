# InsightEngine design language

Structure and rules adapted from the Linear DESIGN.md analysis (VoltAgent/awesome-design-md, MIT).
No Linear logos, imagery or proprietary fonts are used. Type falls back to Geist / system sans.
The accent is emerald instead of lavender to avoid the default "AI purple" look.

Reading: dark technical product UI for developers and researchers. Dials: variance 4, motion 3, density 6.

## Colour tokens

| Token | Value | Use |
|---|---|---|
| canvas | `#0a0b0d` | page background (never pure black) |
| surface-1 | `#101215` | panels, cards, sidebar |
| surface-2 | `#15181c` | inputs, hover |
| hairline | `#22262c` | 1px borders |
| hairline-strong | `#323740` | focus-adjacent borders |
| ink | `#eef0f2` | primary text |
| ink-muted | `#b4bac3` | body text |
| ink-subtle | `#8a919c` | captions (4.5:1 on canvas) |
| accent | `#3ecf8e` | the single chromatic accent: primary buttons, active tab, links, local-retrieval tag |
| on-accent | `#04130c` | text on accent buttons |
| warn | `#f2b95c` | web-fallback tag only |

One accent, used identically everywhere. No gradients on text, no glows.

## Type

- Sans: Geist, falling back to system UI. Mono: Geist Mono, falling back to ui-monospace.
- Display 32/1.15, weight 600, tracking -0.03em. Section titles 20/1.3, weight 600. Body 15/1.55. Caption 13/1.4.
- Scores, page numbers and doc ids are set in mono.

## Shape and spacing

- Radius scale: 6px (tags, inputs), 8px (buttons), 12px (panels). Nothing else.
- Spacing: 4, 8, 12, 16, 24, 32, 48.
- Panels use a 1px hairline and no drop shadow. Group with space and dividers before adding a card.

## Components

- Primary button: accent fill, on-accent text, 8px radius. Hover lightens, active moves 1px down.
- Secondary button: surface-1 fill, hairline border, ink text.
- Tags: 6px radius, tinted fill, hairline border. Local = accent, web fallback = warn.
- Passage card: surface-1, hairline border, 2px accent left rule, mono meta line.
- Tabs: text only, accent underline on the active tab.

## Rules

- No emoji in the UI. Plain text labels.
- No em-dashes in visible copy.
- Focus rings stay visible (2px accent outline).
- Text contrast is at least WCAG AA on every surface.
- Motion: 150ms ease on colour and transform only; honour `prefers-reduced-motion`.
