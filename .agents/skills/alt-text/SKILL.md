---
name: alt-text
description: |
  Write alt text for images. Defines what constitutes "good" alt text. Use when adding images to a documents. a11y is for LLMs as well as humans.
---

Good alt text is what you'd say about the image if you were reading aloud to a blind friend: a brief, well-judged remark that lets them follow along, not a clinical pixel inventory. Consider:

- Context: The same image needs different alt text depending on what it's illustrating. If the image context isn't clear from the conversation, ask about it.
- Length: Aim for one sentence (~125 characters); two if the image is complex and the complexity matters. Longer alt text derails the article, and some TTS systems truncate or paraphrase it badly.

## Patterns by image type

**Photographs.** Subject + action. Include named people/places/objects when they matter to the surrounding text; omit when they don't.

**Screenshots.** Describe the UI state.

**Diagrams/architecture.** Name components and relationships in reading order. High level first, then details.

**Charts/plots.** Lead with the takeaway. Include specific values only if they matter.

**Memes/reaction images.** Describe what's happening in a way that conveys the joke or feeling.

**Decorative images.** Use `alt=""` (no alt text). Mention this if you suspect it applies.

**Functional images (buttons, icons).** Describe the function, not the appearance, unless the appearance is the point.

## Antipatterns

- ~~Mechanical inventories of colors, positions, and objects~~
- ~~Duplicating the caption~~
- ~~Generic placeholders: "Chart", "Diagram", "Photo"~~
- ~~Paragraph-length alt text — use a short alt + longer prose description nearby~~

## Captions

Captions serve a different purpose entirely: they assume the content of the image is already accessible to the reader. The caption should make a _statement_ about it.

## Process

1. Gather context — from the surrounding document, from the conversation history, or by asking.
2. Open the image and take a moment to understand it.
3. Compose a concise description that captures the purpose of the image in context.
4. Add the alt text to the image in the document (e.g. Markdown), or present it to the user.
