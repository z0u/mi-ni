---
name: writing
description: |
  Writing style for composing text. Use for any prose: Markdown, GitHub issues & PRs, proposals, and technical writing such as academic papers. Improves collaboration effectiveness.
---

Write with a clear, correct, and understated style: considered but conversational, precise but not stuffy. Use the first person and voice uncertainty.

Characteristics:

- Semantically-appropriate punctuation, including colons, semicolons, and the various dashes. Use the Oxford comma. Use double quotation marks for quotes, and single quotation marks for quotes within quotes. Do not use "smart quotes".
- Use em dashes sparingly (approx. one per page). Prefer commas or parentheses for asides, and semicolons to connect independent clauses.
- "I think...", "It seems...": hedge thoughtfully to signal confidence, but not to the point of being vague or non-committal.
- Varied sentence rhythm.
- American spelling to match the convention in scientific literature.
- High readability, with a Flesch-Kincaid grade level of around 10-12.
- Assume an intelligent audience and use precise language. See _Concision_ for more on this.
- Alt text for all images: aids vision-impaired people and LLMs alike.

Anti-patterns:

- ~~Business jargon and bombast~~. Avoid buzzwords, corporate-speak, and military and baseball metaphors.
- ~~All the lists~~. Mostly use paragraphs, but use lists sparingly when they are the clearest way to present the information.
- ~~Heavy-handed transitions~~. Avoid "Furthermore", "In conclusion", "The honest answer is", etc. Just continue the thought.
- ~~Excessive use of em dashes~~. Prefer other punctuation.
- ~~Excessive use of bold and italic text~~. List items should not be bolded. 1-2 callouts (bold) per page or section; italics only for references & borrowed words, or when it's truly unobvious which phrase should be emphasized (usually the reader can infer without it).
- ~~Contrastive conclusions~~. "is a feature, not a confound", etc. Instead of `A, ~A`, just say `A`.

### Concision and clarity

We write to communicate. We respect the intelligence of our readers, but we also respect their time. We aim for high readability because it improves our chances of conveying complex technical ideas. We accept that writing takes time, and spend that time to make the writing as clear and concise as possible. We spend the effort so our audience doesn't have to.

## Workflow

Start by planning in your own words. Then, revise the draft to align with this style and tone. Use the characteristics and anti-patterns as a guide. Finally, review the draft for clarity, correctness, and consistency.

First drafts are invariably too verbose, so always edit them down after writing them. You should use tools to help: 1. Write in a file and save it. Include markers if the document has many sections. 2. Count length: `wc -wml file` for the whole file, and `.agents/skills/writing/scripts/block-wc file` for per-block word counts (sorted so the verbose blocks surface first) to see where to cut. 3. Edit the file to reduce length, aiming for > 20% reduction. Stop when you can't remove more without losing load-bearing information. 4. Remove the markers with a tool, then confirm none survived (e.g. `rg 'R:|/R' file` returns nothing).

Example section/block markers:

```md
<!--R:short-description-->

The text.

<!--/R-->
```

When in doubt about how to phrase something, err on the side of clarity and simplicity. Avoid jargon and complex sentence structures unless they are necessary to convey the technical content accurately.
