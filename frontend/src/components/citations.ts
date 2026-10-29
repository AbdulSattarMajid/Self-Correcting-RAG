import type { Plugin } from "unified";

type MarkdownNode = {
  type: string;
  value?: string;
  url?: string;
  children?: MarkdownNode[];
};

const citationPattern = /\[([^\]\n]+?\.pdf,\s*pages?\s+[\d,\s-]+)\]|\[E\d+(?:\s*[-,]\s*E?\d+)*\]/gi;

function transformText(node: MarkdownNode): void {
  if (!node.children) return;

  for (let index = 0; index < node.children.length; index += 1) {
    const child = node.children[index];
    if (child.type !== "text" || !child.value) {
      transformText(child);
      continue;
    }

    let cursor = 0;
    const replacement: MarkdownNode[] = [];
    citationPattern.lastIndex = 0;
    for (const match of child.value.matchAll(citationPattern)) {
      const start = match.index ?? 0;
      if (start > cursor) replacement.push({ type: "text", value: child.value.slice(cursor, start) });
      if (match[1]) {
        replacement.push({
          type: "link",
          url: `https://citation.local/${encodeURIComponent(match[1])}`,
          children: [{ type: "text", value: match[1] }],
        });
      }
      cursor = start + match[0].length;
    }
    if (cursor > 0) {
      if (cursor < child.value.length) replacement.push({ type: "text", value: child.value.slice(cursor) });
      node.children.splice(index, 1, ...replacement);
      index += replacement.length - 1;
    }
  }
}

export const remarkCitationChips: Plugin = () => (tree) => {
  transformText(tree as MarkdownNode);
};
